"""
MOE_CUR_mixtral.py

CUR Decomposition-based Compression for Mixtral-8x7B MoE Model

This script implements CUR decomposition as an alternative to SVD for compressing
Mixture of Experts models. Unlike SVD which creates new orthogonal bases, CUR
uses actual columns and rows from the original weight matrices, preserving
interpretability and working with "real expert data".

CUR Decomposition: W ≈ C * U * R
- C: Selected columns from W (captures output space)
- U: Connecting matrix (typically pseudoinverse-based)
- R: Selected rows from W (captures input space)

Key difference from SVD approach:
- SVD shares V matrix across experts (synthetic basis)
- CUR shares R matrix across experts (real data from actual weight matrices)
"""

import json
import os
import sys
import argparse
import copy
import warnings
import time
from datetime import datetime
from tqdm import tqdm
from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.mixtral.modeling_mixtral import (
    MixtralAttention,
    MixtralSparseMoeBlock,
    MixtralConfig,
    MixtralDecoderLayer,
    MixtralRMSNorm,
    MixtralRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv
)
from datasets import load_dataset
from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model

warnings.filterwarnings('ignore')

# ============================================================================
# CUR Decomposition Algorithms
# ============================================================================

def compute_leverage_scores(matrix: torch.Tensor, k: int, mode: str = 'column') -> torch.Tensor:
    """
    Compute leverage scores for column or row selection in CUR decomposition.
    
    For matrix A (m x n):
    - Row leverage: squared row norms of top-k left singular vectors U_k
    - Column leverage: squared column norms of top-k right singular vectors V_k
    
    Args:
        matrix: Input matrix (m x n)
        k: Number of singular vectors to use
        mode: 'row' or 'column'
    
    Returns:
        Leverage scores (length m for rows, length n for columns)
    """
    try:
        # Compute full SVD
        U, S, Vh = torch.linalg.svd(matrix.float(), full_matrices=False)
        k = min(k, S.shape[0])
        
        if mode == 'row':
            # Row leverage: use left singular vectors U
            Uk = U[:, :k]  # (m x k)
            scores = (Uk ** 2).sum(dim=1)  # length m
        elif mode == 'column':
            # Column leverage: use right singular vectors V (rows of Vh)
            Vk = Vh[:k, :]  # (k x n)
            scores = (Vk ** 2).sum(dim=0)  # length n
        else:
            raise ValueError("mode must be 'row' or 'column'")
        
        # Normalize to sum to k (standard CUR normalization)
        scores = scores * (k / scores.sum())
        
    except Exception as e:
        print(f"Warning: SVD failed in leverage score computation: {e}")
        # Fallback: use squared norms
        if mode == 'row':
            scores = torch.sum(matrix ** 2, dim=1)
        else:  # column
            scores = torch.sum(matrix ** 2, dim=0)
    
    return scores


def select_columns_rows(matrix: torch.Tensor, 
                       num_cols: int, 
                       num_rows: int,
                       selection_method: str = 'leverage') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Select columns and rows from matrix for CUR decomposition.
    
    CUR preserves actual matrix structure by selecting real columns/rows,
    unlike SVD which creates synthetic bases. This maintains interpretability
    and works with "real expert data" as emphasized by the mentor.
    
    Args:
        matrix: Original weight matrix (m x n)
        num_cols: Number of columns to select
        num_rows: Number of rows to select
        selection_method: 'leverage', 'norm', or 'random'
    
    Returns:
        C: Selected columns (m x num_cols) - actual columns from matrix
        R: Selected rows (num_rows x n) - actual rows from matrix
        col_indices: Indices of selected columns
        row_indices: Indices of selected rows
    """
    m, n = matrix.shape
    
    # Ensure we don't select more than available
    num_cols = min(num_cols, n)
    num_rows = min(num_rows, m)
    
    if selection_method == 'leverage':
        # Compute leverage scores based on SVD (measures importance in principal subspace)
        # This is the theoretically optimal way to select columns/rows for CUR
        try:
            col_scores = compute_leverage_scores(matrix, k=min(num_cols * 2, min(m, n)), mode='column')
            row_scores = compute_leverage_scores(matrix, k=min(num_rows * 2, min(m, n)), mode='row')
            
            # Validate leverage scores
            col_valid = not (torch.isnan(col_scores).any() or torch.isinf(col_scores).any())
            row_valid = not (torch.isnan(row_scores).any() or torch.isinf(row_scores).any())
            
            if not col_valid or not row_valid:
                print("Warning: Invalid leverage scores detected, falling back to norm-based selection")
                selection_method = 'norm'
            else:
                # ROBUST LEVERAGE-BASED SELECTION (Deterministic Top-K)
                # Instead of random sampling, select top-k by leverage score
                # This preserves theoretical benefits while being numerically stable
                
                # Ensure scores are positive
                col_scores = torch.clamp(col_scores, min=1e-10)
                row_scores = torch.clamp(row_scores, min=1e-10)
                
                # Normalize to sum to 1 (proper probability distribution)
                col_scores = col_scores / col_scores.sum()
                row_scores = row_scores / row_scores.sum()
                
                # Select top-k indices by leverage score (deterministic)
                # This selects the most "important" columns/rows based on principal subspace
                _, col_indices = torch.topk(col_scores, num_cols)
                _, row_indices = torch.topk(row_scores, num_rows)
                
                # Sort indices for consistency (important for reproducibility)
                col_indices, _ = torch.sort(col_indices)
                row_indices, _ = torch.sort(row_indices)
                
                # Convert to proper device and dtype
                col_indices = col_indices.long()
                row_indices = row_indices.long()
                
        except Exception as e:
            print(f"Warning: Leverage score computation failed ({e}), falling back to norm-based selection")
            selection_method = 'norm'
    
    if selection_method == 'norm':
        # Norm-based selection: Select columns/rows with highest magnitude
        # This is a stable heuristic that works well for neural network weights
        # High-norm features are often important, making this a good fallback
        
        col_norms = torch.norm(matrix, dim=0)  # [n] - column norms
        row_norms = torch.norm(matrix, dim=1)  # [m] - row norms
        
        # Get top-k indices by norm
        _, col_indices = torch.topk(col_norms, num_cols)
        _, row_indices = torch.topk(row_norms, num_rows)
        
        # Sort indices for consistency
        col_indices, _ = torch.sort(col_indices)
        row_indices, _ = torch.sort(row_indices)
        
    elif selection_method == 'random':
        # Uniform random sampling (baseline for comparison)
        # Useful for ablation studies to measure benefit of leverage/norm selection
        col_indices = torch.randperm(n, device=matrix.device)[:num_cols]
        row_indices = torch.randperm(m, device=matrix.device)[:num_rows]
        
        # Sort for consistency
        col_indices, _ = torch.sort(col_indices)
        row_indices, _ = torch.sort(row_indices)
    
    # Extract selected columns and rows
    # CRITICAL: These are ACTUAL columns/rows from the original matrix
    # This is what makes CUR different from SVD - we preserve real data structure
    C = matrix[:, col_indices]  # [m, num_cols] - actual columns from W
    R = matrix[row_indices, :]  # [num_rows, n] - actual rows from W
    
    return C, R, col_indices, row_indices


def select_rows_with_leverage(matrix: torch.Tensor, k: int, selection_method: str = 'leverage') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select rows from matrix for shared R in CUR-MoE.
    
    This is used to compute a global R matrix from stacked expert weights.
    
    Args:
        matrix: Weight matrix (m x n)
        k: Number of rows to select
        selection_method: 'leverage', 'norm', or 'random'
    
    Returns:
        R: Selected rows (k x n)
        row_indices: Indices of selected rows
    """
    m, n = matrix.shape
    k = min(k, m)
    
    if selection_method == 'leverage':
        try:
            # Use row leverage scores
            scores = compute_leverage_scores(matrix, k=min(2*k, min(m, n)), mode='row')
            scores = torch.clamp(scores, min=1e-12)
            scores = scores / scores.sum()
            
            # Select top-k by leverage score
            _, row_indices = torch.topk(scores, k)
            row_indices, _ = torch.sort(row_indices)
        except Exception as e:
            print(f"Warning: Leverage scoring failed ({e}), using norm-based")
            selection_method = 'norm'
    
    if selection_method == 'norm':
        # Select rows with highest norms
        row_norms = torch.norm(matrix, dim=1)
        _, row_indices = torch.topk(row_norms, k)
        row_indices, _ = torch.sort(row_indices)
    elif selection_method == 'random':
        row_indices = torch.randperm(m, device=matrix.device)[:k]
        row_indices, _ = torch.sort(row_indices)
    
    # Extract selected rows
    R = matrix[row_indices, :]  # (k, n)
    
    return R, row_indices


def fit_CU_given_R(W: torch.Tensor, R_shared: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fit C and U matrices given a fixed shared R matrix.
    
    For W ≈ C * U * R_shared, we compute:
    - X = W @ R_shared^T @ (R_shared @ R_shared^T)^{-1}
    - Then set C = X, U = I
    
    This ensures W ≈ C * U * R_shared holds.
    
    Args:
        W: Weight matrix to approximate (m x n)
        R_shared: Fixed shared R matrix (k x n)
    
    Returns:
        C: Column matrix (m x k)
        U: Connecting matrix (k x k), set to identity
    """
    k = R_shared.shape[0]
    
    # Compute Gram matrix: G = R @ R^T
    G = R_shared @ R_shared.t()  # (k, k)
    
    # Add regularization for stability
    G_reg = G + 1e-6 * torch.eye(k, device=G.device, dtype=G.dtype)
    
    try:
        # Compute inverse
        G_inv = torch.linalg.inv(G_reg)
    except:
        # Fallback to pseudoinverse
        G_inv = torch.linalg.pinv(G_reg)
    
    # Project W onto row-span of R: C = W @ R^T @ G^{-1}
    C = W @ R_shared.t() @ G_inv  # (m, k)
    
    # Use identity for U (can be refined later with additional factorization)
    U = torch.eye(k, device=W.device, dtype=W.dtype)
    
    return C, U


def compute_connecting_matrix(W_intersect: torch.Tensor, reg: float = 1e-6) -> torch.Tensor:
    """
    Compute the connecting matrix U in CUR decomposition.
    
    For W ≈ C * U * R, we compute U = pinv(W_intersect)
    where W_intersect = W[row_indices, col_indices]
    
    Args:
        W_intersect: Intersection matrix (k x k)
        reg: Regularization strength for numerical stability
    
    Returns:
        U: Connecting matrix (k x k)
    """
    k = W_intersect.shape[0]
    
    # Add regularization for numerical stability
    W_reg = W_intersect.float() + reg * torch.eye(
        k, device=W_intersect.device, dtype=torch.float32
    )
    
    try:
        # U = pseudoinverse of intersection matrix
        U = torch.linalg.pinv(W_reg)
    except Exception as e:
        print(f"Warning: Pseudoinverse failed ({e}), using identity")
        U = torch.eye(k, device=W_intersect.device, dtype=torch.float32)
    
    return U.to(W_intersect.dtype)


def cur_decomposition(W: torch.Tensor, 
                     rank: int,
                     selection_method: str = 'leverage') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform CUR decomposition on weight matrix W.
    
    Args:
        W: Weight matrix (m x n)
        rank: Target rank (number of columns/rows to select)
        selection_method: Method for selecting columns/rows
    
    Returns:
        C: Selected columns (m x rank)
        U: Connecting matrix (rank x rank)
        R: Selected rows (rank x n)
    """
    m, n = W.shape
    
    # Ensure rank is valid
    rank = max(1, min(rank, min(m, n)))
    
    # Select columns and rows
    C, R, col_indices, row_indices = select_columns_rows(
        W, num_cols=rank, num_rows=rank, selection_method=selection_method
    )
    
    # Extract intersection matrix using proper indexing
    # W_intersect = W[row_indices, :][:, col_indices]
    row_indices_np = row_indices.cpu().numpy() if row_indices.is_cuda else row_indices.numpy()
    col_indices_np = col_indices.cpu().numpy() if col_indices.is_cuda else col_indices.numpy()
    
    # Use np.ix_ for safe advanced indexing
    W_np = W.cpu().numpy() if W.is_cuda else W.numpy()
    W_intersect_np = W_np[np.ix_(row_indices_np, col_indices_np)]
    W_intersect = torch.from_numpy(W_intersect_np).to(W.device, W.dtype)
    
    # Compute connecting matrix
    U = compute_connecting_matrix(W_intersect)
    
    return C, U, R


# ============================================================================
# CUR-Compressed Layer Modules
# ============================================================================

class CUR_Linear(nn.Module):
    """
    Linear layer compressed using CUR decomposition.
    
    Original: y = W*x + b where W is (out_features x in_features)
    CUR: W ≈ C*U*R, so y ≈ C*(U*(R*x)) + b
    
    Forward pass: x -> R*x -> U*(R*x) -> C*(U*(R*x)) + b
    """
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # C: (out_features x rank) - selected columns from original W
        self.C = nn.Linear(rank, out_features, bias=False)
        
        # U: (rank x rank) - connecting matrix
        self.U = nn.Linear(rank, rank, bias=False)
        
        # R: (rank x in_features) - selected rows from original W  
        self.R = nn.Linear(in_features, rank, bias=False)
        
        # Original bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply CUR decomposition: C * U * R * x
        out = self.R(x)      # (batch, ..., rank)
        out = self.U(out)    # (batch, ..., rank)
        out = self.C(out)    # (batch, ..., out_features)
        
        if self.bias is not None:
            out = out + self.bias
        
        return out


class CUR_MixtralAttention(nn.Module):
    """
    Mixtral attention mechanism with CUR-compressed projection matrices.
    
    Compresses q_proj, k_proj, v_proj, and o_proj using CUR decomposition.
    """
    def __init__(self, config: MixtralConfig, layer_idx: int, rank_ratio: float = 0.5):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        # Calculate ranks for CUR decomposition
        # Rank = (m*n*ratio) / (m+n) to approximately achieve the compression ratio
        def calc_rank(in_dim, out_dim, ratio):
            return max(1, int((in_dim * out_dim * ratio) / (in_dim + out_dim)))
        
        self.q_rank = calc_rank(self.hidden_size, self.num_heads * self.head_dim, rank_ratio)
        self.k_rank = calc_rank(self.hidden_size, self.num_key_value_heads * self.head_dim, rank_ratio)
        self.v_rank = calc_rank(self.hidden_size, self.num_key_value_heads * self.head_dim, rank_ratio)
        self.o_rank = calc_rank(self.num_heads * self.head_dim, self.hidden_size, rank_ratio)
        
        # CUR-compressed projections
        self.q_proj = CUR_Linear(self.hidden_size, self.num_heads * self.head_dim, self.q_rank)
        self.k_proj = CUR_Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, self.k_rank)
        self.v_proj = CUR_Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, self.v_rank)
        self.o_proj = CUR_Linear(self.num_heads * self.head_dim, self.hidden_size, self.o_rank)
        
        self.rotary_emb = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()
        
        # Apply CUR-compressed projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Rotary positional embeddings
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        if past_key_value is not None:
            # Reuse k, v from cache
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        # Repeat k/v heads if num_key_value_heads < num_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # Attention computation
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / np.sqrt(self.head_dim)
        
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value


class MixtralCURExpert(nn.Module):
    """
    Single Mixtral expert with CUR-compressed weights.
    
    Each expert has three projections: gate_proj, up_proj, down_proj
    All are compressed using CUR decomposition.
    """
    def __init__(self, config: MixtralConfig, rank_w1: int, rank_w2: int, rank_w3: int,
                 shared_R_w1: nn.Module = None, shared_R_w2: nn.Module = None, 
                 shared_R_w3: nn.Module = None):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        
        # Store ranks
        self.rank_w1 = rank_w1
        self.rank_w2 = rank_w2
        self.rank_w3 = rank_w3
        
        # w1 (gate_proj): hidden_dim -> ffn_dim
        if shared_R_w1 is not None:
            self.w1_R = shared_R_w1  # Shared across experts
        else:
            self.w1_R = nn.Linear(self.hidden_dim, rank_w1, bias=False)
        self.w1_U = nn.Linear(rank_w1, rank_w1, bias=False)
        self.w1_C = nn.Linear(rank_w1, self.ffn_dim, bias=False)
        
        # w3 (up_proj): hidden_dim -> ffn_dim
        if shared_R_w3 is not None:
            self.w3_R = shared_R_w3  # Shared across experts
        else:
            self.w3_R = nn.Linear(self.hidden_dim, rank_w3, bias=False)
        self.w3_U = nn.Linear(rank_w3, rank_w3, bias=False)
        self.w3_C = nn.Linear(rank_w3, self.ffn_dim, bias=False)
        
        # w2 (down_proj): ffn_dim -> hidden_dim
        if shared_R_w2 is not None:
            self.w2_R = shared_R_w2  # Shared across experts
        else:
            self.w2_R = nn.Linear(self.ffn_dim, rank_w2, bias=False)
        self.w2_U = nn.Linear(rank_w2, rank_w2, bias=False)
        self.w2_C = nn.Linear(rank_w2, self.hidden_dim, bias=False)
        
        self.act_fn = F.silu
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Expert computation: down_proj(silu(gate_proj(x)) * up_proj(x))
        
        # gate_proj with CUR
        gate = self.w1_R(hidden_states)
        gate = self.w1_U(gate)
        gate = self.w1_C(gate)
        gate = self.act_fn(gate)
        
        # up_proj with CUR
        up = self.w3_R(hidden_states)
        up = self.w3_U(up)
        up = self.w3_C(up)
        
        # Element-wise multiplication
        intermediate = gate * up
        
        # down_proj with CUR
        out = self.w2_R(intermediate)
        out = self.w2_U(out)
        out = self.w2_C(out)
        
        return out


class CUR_MixtralSparseMoeBlock(nn.Module):
    """
    Mixtral Sparse MoE block with CUR-compressed experts.
    
    Key innovation: Share R matrices across experts (similar to sharing V in SVD).
    R matrices capture input space transformations and are shared,
    while C and U matrices remain expert-specific.
    """
    def __init__(self, config: MixtralConfig, rank_ratio: float = 0.5):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        
        # Calculate ranks for each projection
        def calc_rank(in_dim, out_dim, ratio):
            return max(1, int((in_dim * out_dim * ratio) / (in_dim + out_dim)))
        
        self.rank_w1 = calc_rank(self.hidden_dim, self.ffn_dim, rank_ratio)
        self.rank_w2 = calc_rank(self.ffn_dim, self.hidden_dim, rank_ratio)
        self.rank_w3 = calc_rank(self.hidden_dim, self.ffn_dim, rank_ratio)
        
        # Shared R matrices across all experts
        # This is the key difference from SVD: we share actual row data, not synthetic bases
        self.shared_R_w1 = nn.Linear(self.hidden_dim, self.rank_w1, bias=False)
        self.shared_R_w3 = nn.Linear(self.hidden_dim, self.rank_w3, bias=False)
        self.shared_R_w2 = nn.Linear(self.ffn_dim, self.rank_w2, bias=False)
        
        # Initialize shared R matrices to zero (will be filled during compression)
        nn.init.zeros_(self.shared_R_w1.weight)
        nn.init.zeros_(self.shared_R_w3.weight)
        nn.init.zeros_(self.shared_R_w2.weight)
        
        # Create experts with shared R matrices
        self.experts = nn.ModuleList([
            MixtralCURExpert(
                config, 
                self.rank_w1, 
                self.rank_w2, 
                self.rank_w3,
                self.shared_R_w1,
                self.shared_R_w2,
                self.shared_R_w3
            )
            for _ in range(self.num_experts)
        ])
        
        # Router (gating network) - not compressed
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        # Router logits
        router_logits = self.gate(hidden_states)
        
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        
        # Expert computation
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            
            if top_x.shape[0] == 0:
                continue
            
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()
            
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]
            
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


# ============================================================================
# Data Loading Utilities
# ============================================================================

def get_calib_data(dataset_name: str, tokenizer, nsamples: int = 256, seqlen: int = 2048, seed: int = 3):
    """Load calibration data for profiling."""
    cache_file = f"cache/{dataset_name}_{nsamples}_{seqlen}_{seed}.pt"
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if not os.path.exists("cache"):
        os.makedirs("cache")
    
    if os.path.exists(cache_file):
        print(f"Loading cached calibration data from {cache_file}")
        return torch.load(cache_file)
    
    print(f"Preparing calibration data from {dataset_name}...")
    
    if dataset_name == "wikitext2":
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        tot_text = "\n\n".join(traindata["text"])
    elif dataset_name == "c4":
        traindata = load_dataset("json", data_files="utils/c4-train.json")['train']
        tot_text = "\n\n".join(traindata["text"])
    elif dataset_name == "ptb":
        traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
        tot_text = "\n\n".join(traindata["sentence"])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    import random
    random.seed(seed)
    
    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, len(tot_text) - seqlen * 10)
        j = i + seqlen * 10
        trainenc = tokenizer(tot_text[i:j], return_tensors="pt")
        if trainenc.input_ids.shape[1] < seqlen:
            continue
        inp = trainenc.input_ids[:, :seqlen]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    
    torch.save(traindataset, cache_file)
    return traindataset


def get_test_loader(dataset_name: str, tokenizer, seq_len: int = 2048, batch_size: int = 4):
    """Get test data loader for evaluation."""
    
    if dataset_name == "wikitext2":
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt").input_ids[0]
    elif dataset_name == "ptb":
        testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')
        testenc = tokenizer("\n\n".join(testdata["sentence"]), return_tensors="pt").input_ids[0]
    elif dataset_name == "c4":
        testdata = load_dataset("json", data_files="utils/c4-validation.json")['train']
        testenc = tokenizer("\n\n".join(testdata[:2000]["text"]), return_tensors="pt").input_ids[0]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    nsamples = testenc.numel() // seq_len
    test_ids_batch = []
    for i in range(nsamples):
        batch = testenc[(i * seq_len):((i + 1) * seq_len)]
        test_ids_batch.append(batch)
    
    test_ids_batch = torch.stack(test_ids_batch)
    
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, tensors):
            self.tensors = tensors
        def __getitem__(self, index):
            return self.tensors[index]
        def __len__(self):
            return len(self.tensors)
    
    test_dataset = SimpleDataset(test_ids_batch)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader


# ============================================================================
# Profiling and Whitening
# ============================================================================

def find_layers(module, layers=[nn.Linear], name=''):
    """Recursively find all layers of specified types in a module."""
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


@torch.no_grad()
def profile_model_activations(model, calib_data, device='cuda', layers_to_profile=None):
    """
    Profile model to collect activation statistics for whitening.
    
    Returns scaling matrices that will be used to preprocess weights before CUR.
    """
    print("Profiling model activations for whitening...")
    
    layers = model.model.layers
    if layers_to_profile is None:
        layers_to_profile = list(range(len(layers)))
    
    model.eval()
    
    # Get the device of the model's first parameter (don't force move)
    first_param = next(model.parameters())
    if first_param.device.type == 'meta':
        # Model is on meta device, need to materialize it properly
        print("Model is on meta device, loading with proper device map...")
        device = torch.device(device)
    else:
        device = first_param.device
    
    # Don't move embedding/norm layers if they're already placed by device_map
    # Just check they're accessible
    embed_tokens = model.model.embed_tokens
    norm_layer = model.model.norm
    
    # Data structures for collecting statistics
    profiling_data = {}
    for layer_idx in layers_to_profile:
        profiling_data[layer_idx] = {}
    
    # Forward hook to capture inputs
    def make_hook(layer_idx, name):
        def hook(module, input, output):
            if layer_idx not in profiling_data:
                return
            inp = input[0].detach()
            if name not in profiling_data[layer_idx]:
                profiling_data[layer_idx] = {}
            if name not in profiling_data[layer_idx]:
                profiling_data[layer_idx][name] = []
            # Store on CPU to avoid OOM
            profiling_data[layer_idx][name].append(inp.cpu())
        return hook
    
    # Register hooks on the layers we want to profile
    handles = []
    for layer_idx in layers_to_profile:
        layer = layers[layer_idx]
        subset = find_layers(layer)
        for name, module in subset.items():
            full_name = f"layer{layer_idx}.{name}"
            handle = module.register_forward_hook(make_hook(layer_idx, name))
            handles.append(handle)
    
    # Run forward passes
    print("Running forward passes to collect activation statistics...")
    for batch_idx, batch in enumerate(tqdm(calib_data, desc="Profiling")):
        try:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            
            # Move to the device where the model actually is
            # The model's device_map will handle the rest
            if hasattr(model, 'device'):
                input_ids = input_ids.to(model.device)
                attention_mask = attention_mask.to(model.device)
            else:
                # Find the device of the first layer
                first_layer_device = next(layers[0].parameters()).device
                input_ids = input_ids.to(first_layer_device)
                attention_mask = attention_mask.to(first_layer_device)
            
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM in batch {batch_idx}, skipping...")
                torch.cuda.empty_cache()
                continue
            else:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Compute scaling matrices from collected activations
    print("Computing scaling matrices...")
    scaling_matrices = {}
    
    for layer_idx in layers_to_profile:
        scaling_matrices[layer_idx] = {}
        
        if layer_idx not in profiling_data:
            continue
            
        for name, inputs_list in profiling_data[layer_idx].items():
            if len(inputs_list) == 0:
                continue
            
            try:
                # Concatenate all inputs
                all_inputs = torch.cat(inputs_list, dim=0)  # (total_samples, seq_len, hidden_dim)
                all_inputs = all_inputs.reshape(-1, all_inputs.shape[-1])  # (N, hidden_dim)
                
                # Compute covariance matrix: E[x * x^T]
                mean = all_inputs.mean(dim=0, keepdim=True)
                centered = all_inputs - mean
                
                # Covariance matrix
                cov = (centered.t() @ centered) / (centered.shape[0] - 1)
                
                # Add small diagonal for numerical stability
                cov = cov + 1e-6 * torch.eye(cov.shape[0])
                
                try:
                    # Cholesky decomposition: cov = L * L^T
                    scaling_matrix = torch.linalg.cholesky(cov)
                    scaling_matrices[layer_idx][name] = scaling_matrix
                except Exception as e:
                    print(f"Warning: Cholesky failed for layer {layer_idx}, {name}: {e}")
                    # Fallback: use identity
                    scaling_matrices[layer_idx][name] = torch.eye(cov.shape[0])
                    
            except Exception as e:
                print(f"Warning: Could not compute scaling matrix for {layer_idx}.{name}: {e}")
                continue
    
    # Clear profiling data to free memory
    del profiling_data
    torch.cuda.empty_cache()
    
    print(f"Profiling complete. Collected scaling matrices for {len(scaling_matrices)} layers.")
    return scaling_matrices

# ============================================================================
# CUR Compression Pipeline
# ============================================================================

@torch.no_grad()
def compress_model_with_cur(model, scaling_matrices, rank_ratio=0.5, 
                           layers_to_compress=None, device='cuda',
                           selection_method='leverage'):
    """
    Apply CUR decomposition to compress model weights.
    
    This replaces original layers with CUR-compressed versions.
    """
    print(f"Applying CUR compression with rank_ratio={rank_ratio}...")
    
    layers = model.model.layers
    if layers_to_compress is None:
        layers_to_compress = list(range(len(layers)))
    
    # Determine actual device to use
    device = torch.device(device)
    
    for layer_idx in tqdm(layers_to_compress, desc="Compressing layers"):
        layer = layers[layer_idx]
        
        # Move layer to device if it's not already there
        # Check if parameters are on meta device
        try:
            first_param = next(layer.parameters())
            if first_param.device.type != 'meta':
                layer_device = first_param.device
            else:
                # Need to materialize from meta
                print(f"Warning: Layer {layer_idx} is on meta device, this may cause issues")
                layer_device = device
        except StopIteration:
            layer_device = device
        
        # Get all linear layers in this layer
        subset = find_layers(layer)
        
        # ====================================================================
        # Compress Attention
        # ====================================================================
        
        # Create CUR attention module
        cur_attn = CUR_MixtralAttention(model.config, layer_idx, rank_ratio)
        cur_attn = cur_attn.to(layer_device)
        
        # Copy rotary embeddings from original attention (avoids version compatibility issues)
        if hasattr(layer.self_attn, 'rotary_emb'):
            cur_attn.rotary_emb = layer.self_attn.rotary_emb
        
        # Compress q_proj, k_proj, v_proj, o_proj
        attn_proj_names = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj']
        
        for proj_name in attn_proj_names:
            if proj_name not in subset:
                continue
            
            orig_layer = subset[proj_name]
            
            # Get weight data (handle meta device)
            if orig_layer.weight.device.type == 'meta':
                print(f"Warning: {proj_name} is on meta device, skipping...")
                continue
                
            W = orig_layer.weight.data.clone().to(layer_device)
            
            # Apply whitening if available
            if layer_idx in scaling_matrices and proj_name in scaling_matrices[layer_idx]:
                scaling_matrix = scaling_matrices[layer_idx][proj_name].to(layer_device)
                try:
                    scaling_matrix_inv = torch.linalg.inv(scaling_matrix)
                    W = W @ scaling_matrix_inv
                except:
                    print(f"Warning: Could not apply whitening for {proj_name}")
            
            # Determine target module
            if 'q_proj' in proj_name:
                target = cur_attn.q_proj
            elif 'k_proj' in proj_name:
                target = cur_attn.k_proj
            elif 'v_proj' in proj_name:
                target = cur_attn.v_proj
            elif 'o_proj' in proj_name:
                target = cur_attn.o_proj
            else:
                continue
            
            # Perform CUR decomposition
            rank = target.rank
            C, U, R = cur_decomposition(W, rank, selection_method)
            
            # Assign to CUR module (NO TRANSPOSES - shapes already match)
            # C: (out_features, rank) - matches nn.Linear(rank, out_features).weight
            # U: (rank, rank) - matches nn.Linear(rank, rank).weight
            # R: (rank, in_features) - matches nn.Linear(in_features, rank).weight
            target.C.weight.data = C.clone()  # (out, rank)
            target.U.weight.data = U.clone()  # (rank, rank)
            target.R.weight.data = R.clone()  # (rank, in)
            
            # Copy bias if exists
            if orig_layer.bias is not None and target.bias is not None:
                target.bias.data = orig_layer.bias.data.clone()
        
        # Replace attention
        layer.self_attn = cur_attn
        
        # ====================================================================
        # Compress MoE Block with Shared R
        # ====================================================================
        
        if hasattr(layer, 'block_sparse_moe'):
            print(f"Compressing MoE block in layer {layer_idx} with shared R matrices...")
            
            # Create CUR MoE module
            cur_moe = CUR_MixtralSparseMoeBlock(model.config, rank_ratio)
            cur_moe = cur_moe.to(layer_device)
            
            # Copy router weights (not compressed)
            if hasattr(layer.block_sparse_moe, 'gate'):
                if layer.block_sparse_moe.gate.weight.device.type != 'meta':
                    cur_moe.gate.weight.data = layer.block_sparse_moe.gate.weight.data.clone()
            
            # ----------------------------------------------------------------
            # Stage 1: Compute Global Shared R matrices from ALL experts
            # ----------------------------------------------------------------
            print("  Computing global shared R from stacked expert weights...")
            
            # Collect all expert weights (with whitening applied)
            all_w1_weights = []
            all_w2_weights = []
            all_w3_weights = []
            
            for expert_idx in range(model.config.num_local_experts):
                orig_expert = layer.block_sparse_moe.experts[expert_idx]
                
                # w1 (gate_proj): [ffn_dim, hidden_dim]
                if orig_expert.w1.weight.device.type != 'meta':
                    W_w1 = orig_expert.w1.weight.data.clone().to(layer_device)
                    proj_name = f'block_sparse_moe.experts.{expert_idx}.w1'
                    if layer_idx in scaling_matrices and proj_name in scaling_matrices[layer_idx]:
                        scaling_matrix = scaling_matrices[layer_idx][proj_name].to(layer_device)
                        try:
                            W_w1 = W_w1 @ torch.linalg.inv(scaling_matrix)
                        except:
                            pass
                    all_w1_weights.append(W_w1)
                
                # w2 (down_proj): [hidden_dim, ffn_dim]
                if orig_expert.w2.weight.device.type != 'meta':
                    W_w2 = orig_expert.w2.weight.data.clone().to(layer_device)
                    proj_name = f'block_sparse_moe.experts.{expert_idx}.w2'
                    if layer_idx in scaling_matrices and proj_name in scaling_matrices[layer_idx]:
                        scaling_matrix = scaling_matrices[layer_idx][proj_name].to(layer_device)
                        try:
                            W_w2 = W_w2 @ torch.linalg.inv(scaling_matrix)
                        except:
                            pass
                    all_w2_weights.append(W_w2)
                
                # w3 (up_proj): [ffn_dim, hidden_dim]
                if orig_expert.w3.weight.device.type != 'meta':
                    W_w3 = orig_expert.w3.weight.data.clone().to(layer_device)
                    proj_name = f'block_sparse_moe.experts.{expert_idx}.w3'
                    if layer_idx in scaling_matrices and proj_name in scaling_matrices[layer_idx]:
                        scaling_matrix = scaling_matrices[layer_idx][proj_name].to(layer_device)
                        try:
                            W_w3 = W_w3 @ torch.linalg.inv(scaling_matrix)
                        except:
                            pass
                    all_w3_weights.append(W_w3)
            
            # Stack weights from all experts and compute shared R
            if len(all_w1_weights) > 0:
                stacked_w1 = torch.cat(all_w1_weights, dim=0)  # (num_experts * ffn_dim, hidden_dim)
                global_R_w1, _ = select_rows_with_leverage(
                    stacked_w1,
                    k=cur_moe.experts[0].rank_w1,
                    selection_method=selection_method
                )
                cur_moe.shared_R_w1.weight.data = global_R_w1.clone()
                print(f"  Shared R_w1 shape: {global_R_w1.shape}")
            
            if len(all_w2_weights) > 0:
                stacked_w2 = torch.cat(all_w2_weights, dim=0)  # (num_experts * hidden_dim, ffn_dim)
                global_R_w2, _ = select_rows_with_leverage(
                    stacked_w2,
                    k=cur_moe.experts[0].rank_w2,
                    selection_method=selection_method
                )
                cur_moe.shared_R_w2.weight.data = global_R_w2.clone()
                print(f"  Shared R_w2 shape: {global_R_w2.shape}")
            
            if len(all_w3_weights) > 0:
                stacked_w3 = torch.cat(all_w3_weights, dim=0)  # (num_experts * ffn_dim, hidden_dim)
                global_R_w3, _ = select_rows_with_leverage(
                    stacked_w3,
                    k=cur_moe.experts[0].rank_w3,
                    selection_method=selection_method
                )
                cur_moe.shared_R_w3.weight.data = global_R_w3.clone()
                print(f"  Shared R_w3 shape: {global_R_w3.shape}")
            
            # ----------------------------------------------------------------
            # Stage 2: Fit expert-specific C and U given shared R
            # ----------------------------------------------------------------
            print("  Fitting expert-specific C and U given shared R...")
            
            for expert_idx in range(model.config.num_local_experts):
                cur_expert = cur_moe.experts[expert_idx]
                
                # Fit w1: C_w1, U_w1 given shared R_w1
                if expert_idx < len(all_w1_weights):
                    W_w1 = all_w1_weights[expert_idx]
                    C_w1, U_w1 = fit_CU_given_R(W_w1, global_R_w1)
                    cur_expert.w1_C.weight.data = C_w1.clone()
                    cur_expert.w1_U.weight.data = U_w1.clone()
                
                # Fit w2: C_w2, U_w2 given shared R_w2
                if expert_idx < len(all_w2_weights):
                    W_w2 = all_w2_weights[expert_idx]
                    C_w2, U_w2 = fit_CU_given_R(W_w2, global_R_w2)
                    cur_expert.w2_C.weight.data = C_w2.clone()
                    cur_expert.w2_U.weight.data = U_w2.clone()
                
                # Fit w3: C_w3, U_w3 given shared R_w3
                if expert_idx < len(all_w3_weights):
                    W_w3 = all_w3_weights[expert_idx]
                    C_w3, U_w3 = fit_CU_given_R(W_w3, global_R_w3)
                    cur_expert.w3_C.weight.data = C_w3.clone()
                    cur_expert.w3_U.weight.data = U_w3.clone()
            
            # Replace MoE block
            layer.block_sparse_moe = cur_moe
            print(f"  MoE block compression complete for layer {layer_idx}")
        
        # Clear CUDA cache after each layer
        torch.cuda.empty_cache()
    
    print("CUR compression completed!")
    return model


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate_perplexity(model, tokenizer, dataset_name='wikitext2', 
                       seq_len=2048, batch_size=4, device='cuda'):
    """Evaluate model perplexity on a dataset."""
    print(f"Evaluating perplexity on {dataset_name}...")
    
    model.eval()
    test_loader = get_test_loader(dataset_name, tokenizer, seq_len, batch_size)
    
    nlls = []
    n_samples = 0
    
    for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Evaluating {dataset_name}")):
        try:
            input_ids = batch.to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, labels=input_ids)
            neg_log_likelihood = outputs.loss
            
            nlls.append(neg_log_likelihood)
            n_samples += 1
            
        except Exception as e:
            print(f"Error in evaluation batch {batch_idx}: {e}")
            continue
    
    if len(nlls) == 0:
        return float('inf')
    
    # Calculate perplexity
    avg_nll = torch.stack(nlls).mean()
    ppl = torch.exp(avg_nll).item()
    
    return ppl


def calculate_compression_ratio(model, layers_compressed):
    """Calculate actual compression ratio achieved."""
    total_params = 0
    compressed_params = 0
    
    for layer_idx in layers_compressed:
        layer = model.model.layers[layer_idx]
        
        # Attention
        if isinstance(layer.self_attn, CUR_MixtralAttention):
            attn = layer.self_attn
            # CUR parameters: C (out x rank) + U (rank x rank) + R (rank x in)
            for proj in [attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj]:
                compressed_params += proj.C.weight.numel() + proj.U.weight.numel() + proj.R.weight.numel()
                total_params += proj.C.out_features * proj.R.in_features
        
        # MoE
        if isinstance(layer.block_sparse_moe, CUR_MixtralSparseMoeBlock):
            moe = layer.block_sparse_moe
            # Shared R matrices
            compressed_params += moe.shared_R_w1.weight.numel()
            compressed_params += moe.shared_R_w3.weight.numel()
            compressed_params += moe.shared_R_w2.weight.numel()
            
            # Per-expert C and U matrices
            for expert in moe.experts:
                compressed_params += expert.w1_C.weight.numel() + expert.w1_U.weight.numel()
                compressed_params += expert.w3_C.weight.numel() + expert.w3_U.weight.numel()
                compressed_params += expert.w2_C.weight.numel() + expert.w2_U.weight.numel()
                
                total_params += moe.ffn_dim * moe.hidden_dim * 3  # w1, w2, w3
    
    ratio = compressed_params / total_params if total_params > 0 else 1.0
    return ratio, compressed_params, total_params


# ============================================================================
# Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='CUR Decomposition for Mixtral-8x7B Compression')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='mistralai/Mixtral-8x7B-v0.1',
                       help='Mixtral model name or path')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for computation')
    
    # Compression arguments
    parser.add_argument('--rank_ratio', type=float, default=0.5,
                       help='Compression ratio for CUR (0-1). Lower = more compression.')
    parser.add_argument('--layers_to_compress', type=str, default='all',
                       help='Layers to compress: "all" or comma-separated indices (e.g., "0,1,2")')
    parser.add_argument('--selection_method', type=str, default='leverage',
                       choices=['leverage', 'norm', 'random'],
                       help='Method for selecting columns/rows in CUR')
    
    # Data arguments
    parser.add_argument('--calib_dataset', type=str, default='wikitext2',
                       choices=['wikitext2', 'c4', 'ptb'],
                       help='Dataset for calibration')
    parser.add_argument('--eval_dataset', type=str, default='wikitext2',
                       choices=['wikitext2', 'c4', 'ptb'],
                       help='Dataset for evaluation')
    parser.add_argument('--nsamples', type=int, default=256,
                       help='Number of calibration samples')
    parser.add_argument('--seqlen', type=int, default=2048,
                       help='Sequence length')
    parser.add_argument('--seed', type=int, default=3,
                       help='Random seed')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for evaluation')
    
    # Execution arguments
    parser.add_argument('--skip_profiling', action='store_true',
                       help='Skip profiling step (use if scaling matrices already saved)')
    parser.add_argument('--profiling_path', type=str, default=None,
                       help='Path to load/save profiling matrices')
    parser.add_argument('--save_model', type=str, default=None,
                       help='Path to save compressed model')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only evaluate, do not compress')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 80)
    print("CUR-MOE: CUR Decomposition for Mixtral-8x7B Compression")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Compression ratio: {args.rank_ratio}")
    print(f"Selection method: {args.selection_method}")
    print(f"Calibration dataset: {args.calib_dataset} ({args.nsamples} samples)")
    print(f"Evaluation dataset: {args.eval_dataset}")
    print("=" * 80)
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
    
    print(f"Model loaded: {model.config.num_hidden_layers} layers, "
          f"{model.config.num_local_experts} experts per layer")
    
    # Determine layers to compress
    if args.layers_to_compress == 'all':
        layers_to_compress = list(range(model.config.num_hidden_layers))
    else:
        layers_to_compress = [int(x) for x in args.layers_to_compress.split(',')]
    
    print(f"Layers to compress: {layers_to_compress}")
    
    if not args.eval_only:
        # Step 1: Profiling
        if args.skip_profiling and args.profiling_path and os.path.exists(args.profiling_path):
            print(f"Loading profiling matrices from {args.profiling_path}")
            scaling_matrices = torch.load(args.profiling_path)
        else:
            calib_data = get_calib_data(args.calib_dataset, tokenizer, args.nsamples, args.seqlen, args.seed)
            scaling_matrices = profile_model_activations(
                model, calib_data, args.device, layers_to_compress
            )
            
            if args.profiling_path:
                os.makedirs(os.path.dirname(args.profiling_path), exist_ok=True)
                torch.save(scaling_matrices, args.profiling_path)
                print(f"Profiling matrices saved to {args.profiling_path}")
        
        # Step 2: CUR Compression
        start_time = time.time()
        model = compress_model_with_cur(
            model, 
            scaling_matrices,
            rank_ratio=args.rank_ratio,
            layers_to_compress=layers_to_compress,
            device=args.device,
            selection_method=args.selection_method
        )
        compression_time = time.time() - start_time
        print(f"Compression completed in {compression_time:.2f} seconds")
        
        # Calculate compression statistics
        ratio, compressed_params, total_params = calculate_compression_ratio(model, layers_to_compress)
        print(f"\nCompression Statistics:")
        print(f"  Compressed parameters: {compressed_params:,}")
        print(f"  Original parameters: {total_params:,}")
        print(f"  Actual compression ratio: {ratio:.4f}")
        print(f"  Space saved: {(1-ratio)*100:.2f}%")
        
        # Save compressed model
        if args.save_model:
            os.makedirs(args.save_model, exist_ok=True)
            model.save_pretrained(args.save_model)
            tokenizer.save_pretrained(args.save_model)
            print(f"Compressed model saved to {args.save_model}")
    
    # Step 3: Evaluation
    print("\nEvaluating compressed model...")
    ppl = evaluate_perplexity(
        model, tokenizer, args.eval_dataset, args.seqlen, args.batch_size, args.device
    )
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Dataset: {args.eval_dataset}")
    print(f"Perplexity: {ppl:.4f}")
    print("=" * 80)
    
    # Save results
    results = {
        'model': args.model,
        'rank_ratio': args.rank_ratio,
        'selection_method': args.selection_method,
        'layers_compressed': layers_to_compress,
        'dataset': args.eval_dataset,
        'perplexity': ppl,
        'compression_time': compression_time if not args.eval_only else None,
        'timestamp': datetime.now().isoformat()
    }
    
    results_file = 'cur_compression_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()
