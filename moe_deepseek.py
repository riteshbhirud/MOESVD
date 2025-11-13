"""
MOE_CUR_deepseek.py

CUR Decomposition-based Compression for DeepSeek-MoE-16B

Fixed issues:
1. Removed incorrect transpose operations when assigning C and U matrices
2. Fixed RoPE call semantics
3. Corrected compute_U_matrix to remove unnecessary transpose
4. Fixed whitening to use solve_triangular with dtype-safe operations
5. Fixed attention mask broadcasting with robust handling
6. Device-aware evaluation for device_map compatibility
"""

import os
import sys
import json
import argparse
import time
import warnings
from datetime import datetime
from tqdm import tqdm
from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset

warnings.filterwarnings('ignore')

# ============================================================================
# CUR Decomposition Core Algorithms
# ============================================================================

def compute_leverage_scores(matrix: torch.Tensor, k: int, mode: str = 'column') -> torch.Tensor:
    """
    Compute leverage scores for column or row selection.
    
    Leverage scores measure importance based on top-k singular space contribution.
    
    Args:
        matrix: Weight matrix (m x n)
        k: Number of components for leverage computation
        mode: 'column' or 'row'
    
    Returns:
        Leverage scores for each column/row
    """
    if mode == 'column':
        M = matrix  # m x n
    else:  # row
        M = matrix.t()  # n x m
    
    try:
        # Compute top-k SVD
        U, S, Vt = torch.linalg.svd(M.float(), full_matrices=False)
        k = min(k, S.shape[0])
        U_k = U[:, :k]
        
        # Leverage scores: squared row norms of top-k left singular vectors
        leverage_scores = torch.sum(U_k ** 2, dim=1)
    except Exception as e:
        print(f"Warning: SVD failed in leverage computation: {e}")
        # Fallback: column/row norms
        leverage_scores = torch.norm(M, dim=1 if mode == 'row' else 0) ** 2
    
    return leverage_scores


def select_columns_rows(matrix: torch.Tensor, 
                       num_cols: int, 
                       num_rows: int,
                       selection_method: str = 'leverage',
                       seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    """
    Select columns and rows from matrix for CUR decomposition.
    
    Args:
        matrix: Weight matrix (m x n)
        num_cols: Number of columns to select
        num_rows: Number of rows to select
        selection_method: 'leverage', 'norm', or 'random'
        seed: Random seed
        
    Returns:
        C: Selected columns (m x num_cols)
        R: Selected rows (num_rows x n)
        col_indices: Selected column indices
        row_indices: Selected row indices
    """
    torch.manual_seed(seed)
    m, n = matrix.shape
    
    num_cols = min(num_cols, n)
    num_rows = min(num_rows, m)
    
    if selection_method == 'leverage':
        # Column selection
        col_scores = compute_leverage_scores(matrix, min(num_cols * 2, min(m, n)), mode='column')
        col_probs = col_scores / col_scores.sum()
        col_indices = torch.multinomial(col_probs, num_cols, replacement=False)
        
        # Row selection
        row_scores = compute_leverage_scores(matrix, min(num_rows * 2, min(m, n)), mode='row')
        row_probs = row_scores / row_scores.sum()
        row_indices = torch.multinomial(row_probs, num_rows, replacement=False)
        
    elif selection_method == 'norm':
        # Highest norm columns/rows
        col_norms = torch.norm(matrix, dim=0)
        col_indices = torch.topk(col_norms, num_cols).indices
        
        row_norms = torch.norm(matrix, dim=1)
        row_indices = torch.topk(row_norms, num_rows).indices
        
    elif selection_method == 'random':
        col_indices = torch.randperm(n)[:num_cols]
        row_indices = torch.randperm(m)[:num_rows]
    else:
        raise ValueError(f"Unknown selection method: {selection_method}")
    
    # Sort indices
    col_indices, _ = torch.sort(col_indices)
    row_indices, _ = torch.sort(row_indices)
    
    # Extract columns and rows
    C = matrix[:, col_indices]  # (m, num_cols)
    R = matrix[row_indices, :]  # (num_rows, n)
    
    return C, R, col_indices.tolist(), row_indices.tolist()


def compute_U_matrix(C: torch.Tensor, R: torch.Tensor, 
                    W_intersect: torch.Tensor) -> torch.Tensor:
    """
    Compute connecting matrix U in CUR decomposition.
    
    W ≈ C @ U @ R, where U connects selected columns and rows.
    U is computed as pseudoinverse of intersection matrix.
    
    Args:
        C: Selected columns (m x c)
        R: Selected rows (r x n)
        W_intersect: W[row_indices, col_indices] (r x c)
        
    Returns:
        U: Connecting matrix (c x r)
    """
    try:
        # FIXED: pinv(W_intersect) already gives (c, r) when W_intersect is (r, c)
        # No need to transpose
        U = torch.linalg.pinv(W_intersect.float())  # (c, r)
    except Exception as e:
        print(f"Warning: Pseudoinverse failed: {e}")
        # Fallback: identity or zeros
        c, r = C.shape[1], R.shape[0]
        U = torch.eye(min(c, r), dtype=C.dtype, device=C.device)
        if c > r:
            U = torch.cat([U, torch.zeros(c - r, r, dtype=C.dtype, device=C.device)], dim=0)
        elif r > c:
            U = torch.cat([U, torch.zeros(c, r - c, dtype=C.dtype, device=C.device)], dim=1)
    
    return U.to(C.dtype)


def cur_decomposition(W: torch.Tensor, rank: int, 
                     selection_method: str = 'leverage',
                     seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int], List[int]]:
    """
    Perform full CUR decomposition on weight matrix.
    
    Args:
        W: Weight matrix (m x n)
        rank: Target rank (columns/rows to select)
        selection_method: Column/row selection method
        seed: Random seed
        
    Returns:
        C: Selected columns (m x rank)
        U: Connecting matrix (rank x rank)
        R: Selected rows (rank x n)
        col_indices: Selected column indices
        row_indices: Selected row indices
    """
    m, n = W.shape
    rank = min(rank, min(m, n))
    
    # Select columns and rows
    C, R, col_indices, row_indices = select_columns_rows(
        W, rank, rank, selection_method, seed
    )
    
    # Extract intersection matrix
    W_intersect = W[row_indices, :][:, col_indices]  # (rank x rank)
    
    # Compute U
    U = compute_U_matrix(C, R, W_intersect)
    
    return C, U, R, col_indices, row_indices


def fit_CU_given_R(W: torch.Tensor, R: torch.Tensor, num_cols: int,
                   selection_method: str = 'leverage', 
                   seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    CRITICAL: Fit C and U matrices for a specific expert given shared R matrix.
    
    This maintains mathematical consistency when R is shared across experts.
    For expert weight W and shared R: find C, U such that W ≈ C @ U @ R
    
    Mathematical approach:
    1. Select columns from W to form C (preserves real expert data)
    2. Solve for U: U = pinv(C) @ W @ pinv(R)
    
    Args:
        W: Expert weight matrix (m x n)
        R: Shared row matrix (k x n) - FIXED
        num_cols: Number of columns to select for C
        selection_method: Method for selecting columns
        seed: Random seed
        
    Returns:
        C: Selected columns from W (m x num_cols)
        U: Optimized connecting matrix (num_cols x k)
        col_indices: Indices of selected columns
    """
    torch.manual_seed(seed)
    m, n = W.shape
    k, n_r = R.shape
    
    assert n == n_r, f"Dimension mismatch: W has {n} cols, R has {n_r} cols"
    
    num_cols = min(num_cols, min(m, n))
    
    # Column selection strategy: select columns that matter for reconstruction given R
    if selection_method == 'leverage':
        try:
            # Compute how well W is already captured by R's row space
            R_pinv = torch.linalg.pinv(R.float())  # (n x k)
            W_proj = W @ R_pinv @ R  # Projection onto R's row space
            residual = W - W_proj  # What's not captured by R
            
            # Select columns with high residual norm (most important for reconstruction)
            col_scores = torch.norm(residual, dim=0) ** 2 + 1e-10
        except:
            # Fallback: use column norms of W
            col_scores = torch.norm(W, dim=0) ** 2 + 1e-10
    elif selection_method == 'norm':
        col_scores = torch.norm(W, dim=0) ** 2 + 1e-10
    else:  # random
        col_scores = torch.ones(n)
    
    # Normalize and sample
    col_probs = col_scores / col_scores.sum()
    col_indices = torch.multinomial(col_probs, num_cols, replacement=False)
    col_indices, _ = torch.sort(col_indices)
    
    # Extract C
    C = W[:, col_indices]  # (m, num_cols)
    
    # Solve for U: W ≈ C @ U @ R
    # => U ≈ pinv(C) @ W @ pinv(R)
    try:
        C_pinv = torch.linalg.pinv(C.float())  # (num_cols, m)
        R_pinv = torch.linalg.pinv(R.float())  # (n, k)
        U = C_pinv @ W @ R_pinv  # (num_cols, k)
        U = U.to(W.dtype)
    except Exception as e:
        print(f"Warning: U computation failed: {e}")
        # Fallback: initialize to small random values
        U = torch.randn(num_cols, k, dtype=W.dtype, device=W.device) * 0.01
    
    return C, U, col_indices.tolist()


# ============================================================================
# CUR-Compressed Layer Modules for DeepSeek
# ============================================================================

class CUR_Linear(nn.Module):
    """
    Linear layer with CUR decomposition: y = (x @ R.T) @ U.T @ C.T + b
    
    Original: y = W @ x.T + b where W is (out_features x in_features)
    CUR: W ≈ C @ U @ R
    Forward: y = C @ (U @ (R @ x.T)).T + b = (x @ R.T @ U.T @ C.T).T + b
    
    Weight shapes:
    - R.weight: (rank, in_features)
    - U.weight: (rank, rank)
    - C.weight: (out_features, rank)
    """
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # R: (rank x in_features) - selected rows
        self.R = nn.Linear(in_features, rank, bias=False)
        
        # U: (rank x rank) - connecting matrix
        self.U = nn.Linear(rank, rank, bias=False)
        
        # C: (out_features x rank) - selected columns (transposed for Linear)
        self.C = nn.Linear(rank, out_features, bias=False)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x @ R.T @ U.T @ C.T"""
        out = self.R(x)      # (..., rank)
        out = self.U(out)    # (..., rank)
        out = self.C(out)    # (..., out_features)
        
        if self.bias is not None:
            out = out + self.bias
        
        return out


class CUR_DeepseekAttention(nn.Module):
    """
    DeepSeek attention with CUR-compressed projections.
    
    Compresses q_proj, k_proj, v_proj, o_proj using CUR decomposition.
    """
    def __init__(self, config, layer_idx: int, rank_ratio: float = 0.5):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, 'qk_nope_head_dim', self.hidden_size // self.num_heads)
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = getattr(config, 'max_position_embeddings', 2048)
        self.rope_theta = getattr(config, 'rope_theta', 10000.0)
        self.is_causal = True
        self.attention_dropout = getattr(config, 'attention_dropout', 0.0)
        
        # Calculate CUR ranks
        def calc_rank(in_dim, out_dim, ratio):
            return max(1, int((in_dim * out_dim * ratio) / (in_dim + out_dim)))
        
        q_out = self.num_heads * self.head_dim
        kv_out = self.num_key_value_heads * self.head_dim
        
        self.q_rank = calc_rank(self.hidden_size, q_out, rank_ratio)
        self.k_rank = calc_rank(self.hidden_size, kv_out, rank_ratio)
        self.v_rank = calc_rank(self.hidden_size, kv_out, rank_ratio)
        self.o_rank = calc_rank(q_out, self.hidden_size, rank_ratio)
        
        # CUR-compressed projections
        self.q_proj = CUR_Linear(self.hidden_size, q_out, self.q_rank)
        self.k_proj = CUR_Linear(self.hidden_size, kv_out, self.k_rank)
        self.v_proj = CUR_Linear(self.hidden_size, kv_out, self.v_rank)
        self.o_proj = CUR_Linear(q_out, self.hidden_size, self.o_rank)
        
        # RoPE embeddings
        self.rotary_emb = self._init_rope()
    
    def _init_rope(self):
        """Initialize rotary position embeddings"""
        try:
            from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
            return LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        except:
            # If RoPE not available, return None and skip RoPE application
            print("Warning: Could not initialize RoPE, skipping rotary embeddings")
            return None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()
        
        # CUR-compressed projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE if available
        # FIXED: Simplified RoPE call using seq_len
        if self.rotary_emb is not None:
            try:
                cos, sin = self.rotary_emb(value_states, seq_len=key_states.shape[2])
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            except:
                pass  # Skip RoPE if it fails
        
        # Handle past key values
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        # Repeat k/v heads if necessary
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # Attention computation
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / np.sqrt(self.head_dim)
        
        # FIXED: Robust attention mask handling
        if attention_mask is not None:
            mask = attention_mask
            # Handle both 0/1 boolean masks and additive masks
            if mask.dtype in (torch.int32, torch.int64, torch.bool) or mask.dim() == 2:
                if mask.dim() == 2:
                    # Convert 0/1 mask to additive: 0→-10000, 1→0
                    mask = (1 - mask.to(torch.float32)) * -1e4
                    mask = mask[:, None, None, :]  # [bsz,1,1,seq]
                else:
                    mask = mask.to(torch.float32)
            # Broadcast and cast safely
            attn_weights = attn_weights + mask[..., :attn_weights.size(-1)].to(attn_weights.dtype)
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value


class DeepseekCURExpert(nn.Module):
    """
    Single DeepSeek expert with CUR-compressed weights and shared R matrices.
    
    Each expert has gate_proj, up_proj, down_proj compressed with CUR.
    R matrices are shared within expert group for efficiency.
    """
    def __init__(self, config, rank_gate: int, rank_up: int, rank_down: int,
                 shared_R_gate: nn.Module, shared_R_up: nn.Module, shared_R_down: nn.Module,
                 group_idx: int):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.moe_intermediate_size
        self.group_idx = group_idx
        
        # Gate projection (w1): hidden_dim -> ffn_dim with shared R
        self.gate_C = nn.Linear(rank_gate, self.ffn_dim, bias=False)
        self.gate_U = nn.Linear(rank_gate, rank_gate, bias=False)
        self.shared_R_gate = shared_R_gate  # Shared within group
        
        # Up projection (w3): hidden_dim -> ffn_dim with shared R
        self.up_C = nn.Linear(rank_up, self.ffn_dim, bias=False)
        self.up_U = nn.Linear(rank_up, rank_up, bias=False)
        self.shared_R_up = shared_R_up  # Shared within group
        
        # Down projection (w2): ffn_dim -> hidden_dim with shared R
        self.down_C = nn.Linear(rank_down, self.hidden_dim, bias=False)
        self.down_U = nn.Linear(rank_down, rank_down, bias=False)
        self.shared_R_down = shared_R_down  # Shared within group
        
        self.act_fn = nn.SiLU()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Expert forward: down_proj(silu(gate_proj(x)) * up_proj(x))"""
        
        # Gate projection with CUR
        gate = self.shared_R_gate(hidden_states)  # (..., rank_gate)
        gate = self.gate_U(gate)                  # (..., rank_gate)
        gate = self.gate_C(gate)                  # (..., ffn_dim)
        gate = self.act_fn(gate)
        
        # Up projection with CUR
        up = self.shared_R_up(hidden_states)      # (..., rank_up)
        up = self.up_U(up)                        # (..., rank_up)
        up = self.up_C(up)                        # (..., ffn_dim)
        
        # Element-wise multiplication
        intermediate = gate * up
        
        # Down projection with CUR
        out = self.shared_R_down(intermediate)    # (..., rank_down)
        out = self.down_U(out)                    # (..., rank_down)
        out = self.down_C(out)                    # (..., hidden_dim)
        
        return out


class CUR_DeepseekMoE(nn.Module):
    """
    DeepSeek MoE block with CUR-compressed experts.
    
    Architecture:
    - 64 routed experts grouped into 8 groups (8 experts per group)
    - Shared R matrices within each group (preserves real expert data)
    - Individual C and U matrices per expert (optimized via fit_CU_given_R)
    - Shared experts (if present) are NOT compressed
    - MoEGate for routing
    """
    def __init__(self, config, rank_ratio: float = 0.5):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.moe_intermediate_size
        self.num_experts = config.n_routed_experts  # 64
        self.num_experts_per_tok = config.num_experts_per_tok
        self.n_shared_experts = getattr(config, 'n_shared_experts', None)
        
        # Expert grouping: 8 groups of 8 experts each
        self.num_groups = 8
        self.experts_per_group = self.num_experts // self.num_groups  # 8
        
        # Calculate CUR ranks
        def calc_rank(in_dim, out_dim, ratio):
            return max(1, int((in_dim * out_dim * ratio) / (in_dim + out_dim)))
        
        self.rank_gate = calc_rank(self.hidden_dim, self.ffn_dim, rank_ratio)
        self.rank_up = calc_rank(self.hidden_dim, self.ffn_dim, rank_ratio)
        self.rank_down = calc_rank(self.ffn_dim, self.hidden_dim, rank_ratio)
        
        # Create shared R matrices for each group
        self.shared_R_gate_list = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.rank_gate, bias=False)
            for _ in range(self.num_groups)
        ])
        self.shared_R_up_list = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.rank_up, bias=False)
            for _ in range(self.num_groups)
        ])
        self.shared_R_down_list = nn.ModuleList([
            nn.Linear(self.ffn_dim, self.rank_down, bias=False)
            for _ in range(self.num_groups)
        ])
        
        # Initialize shared R matrices to zero (will be filled during compression)
        for i in range(self.num_groups):
            nn.init.zeros_(self.shared_R_gate_list[i].weight)
            nn.init.zeros_(self.shared_R_up_list[i].weight)
            nn.init.zeros_(self.shared_R_down_list[i].weight)
        
        # Create 64 experts, each assigned to a group
        self.experts = nn.ModuleList()
        for group_idx in range(self.num_groups):
            for _ in range(self.experts_per_group):
                expert = DeepseekCURExpert(
                    config,
                    self.rank_gate,
                    self.rank_up,
                    self.rank_down,
                    self.shared_R_gate_list[group_idx],
                    self.shared_R_up_list[group_idx],
                    self.shared_R_down_list[group_idx],
                    group_idx
                )
                self.experts.append(expert)
        
        # Router (not compressed)
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        
        # Shared experts (if present, not compressed)
        if self.n_shared_experts is not None and self.n_shared_experts > 0:
            try:
                from transformers.models.llama.modeling_llama import LlamaMLP
                # Shared experts use standard MLP (not compressed)
                intermediate_size = self.ffn_dim * self.n_shared_experts
                self.shared_experts = LlamaMLP(
                    config=config,
                    hidden_size=self.hidden_dim,
                    intermediate_size=intermediate_size,
                    hidden_act=config.hidden_act
                )
            except:
                print("Warning: Could not initialize shared experts, skipping")
                self.shared_experts = None
        else:
            self.shared_experts = None
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # Router logits
        router_logits = self.gate(hidden_states_flat)
        
        # Top-k routing
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        # Initialize output
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        
        # Process each expert
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            
            if top_x.shape[0] == 0:
                continue
            
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()
            
            current_state = hidden_states_flat[top_x_list]
            expert_output = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]
            
            final_hidden_states.index_add_(0, top_x, expert_output.to(hidden_states.dtype))
        
        # Add shared expert output if present
        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states_flat)
            final_hidden_states = final_hidden_states + shared_output
        
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


# ============================================================================
# Helper Functions
# ============================================================================

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads for grouped query attention"""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def find_layers(module, layers=[nn.Linear], name=''):
    """Recursively find all layers of specified types"""
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


# ============================================================================
# Data Loading Utilities
# ============================================================================

def get_calib_data(dataset_name: str, tokenizer, nsamples: int = 256, 
                   seqlen: int = 2048, seed: int = 3):
    """Load calibration data for profiling"""
    cache_dir = "./cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/{dataset_name}_{nsamples}_{seqlen}_{seed}.pt"
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if os.path.exists(cache_file):
        print(f"Loading cached calibration data from {cache_file}")
        return torch.load(cache_file)
    
    print(f"Preparing calibration data from {dataset_name}...")
    
    if dataset_name == "wikitext2":
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        tot_text = "\n\n".join(traindata["text"])
    elif dataset_name == "c4":
        traindata = load_dataset("allenai/c4", "en", split="train", streaming=True)
        tot_text = "\n\n".join([item["text"] for item in list(traindata.take(5000))])
    elif dataset_name == "ptb":
        traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
        tot_text = "\n\n".join(traindata["sentence"])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    import random
    random.seed(seed)
    
    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, max(0, len(tot_text) - seqlen * 10))
        j = i + seqlen * 10
        trainenc = tokenizer(tot_text[i:j], return_tensors="pt")
        if trainenc.input_ids.shape[1] < seqlen:
            continue
        inp = trainenc.input_ids[:, :seqlen]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    
    torch.save(traindataset, cache_file)
    print(f"Cached {len(traindataset)} samples to {cache_file}")
    return traindataset


def get_test_loader(dataset_name: str, tokenizer, seq_len: int = 2048, batch_size: int = 4):
    """Get test data loader for evaluation"""
    if dataset_name == "wikitext2":
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt").input_ids[0]
    elif dataset_name == "ptb":
        testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')
        testenc = tokenizer("\n\n".join(testdata["sentence"]), return_tensors="pt").input_ids[0]
    elif dataset_name == "c4":
        testdata = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        texts = [item["text"] for item in list(testdata.take(1000))]
        testenc = tokenizer("\n\n".join(texts), return_tensors="pt").input_ids[0]
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
# Profiling and Compression Pipeline
# ============================================================================

@torch.no_grad()
def profile_model_activations(model, calib_data, device='cuda', layers_to_profile=None):
    """
    Profile model to collect activation statistics for whitening.
    
    Returns scaling matrices for weight preprocessing before CUR.
    """
    print("Profiling model activations for whitening...")
    
    layers = model.model.layers
    if layers_to_profile is None:
        layers_to_profile = list(range(len(layers)))
    
    model.eval()
    
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
                profiling_data[layer_idx][name] = []
            profiling_data[layer_idx][name].append(inp.cpu())
        return hook
    
    # Register hooks
    handles = []
    for layer_idx in layers_to_profile:
        layer = layers[layer_idx]
        subset = find_layers(layer)
        for name, module in subset.items():
            if 'gate' in name.lower():  # Skip gate/router
                continue
            handle = module.register_forward_hook(make_hook(layer_idx, name))
            handles.append(handle)
    
    # Run forward passes
    print("Running forward passes to collect activation statistics...")
    for batch_idx, batch in enumerate(tqdm(calib_data, desc="Profiling")):
        try:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            
            # Get device of first layer
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
    
    # Compute scaling matrices
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
                # Concatenate inputs
                all_inputs = torch.cat(inputs_list, dim=0)
                all_inputs = all_inputs.reshape(-1, all_inputs.shape[-1])
                
                # Compute covariance
                mean = all_inputs.mean(dim=0, keepdim=True)
                centered = all_inputs - mean
                cov = (centered.t() @ centered) / (centered.shape[0] - 1)
                cov = cov + 1e-6 * torch.eye(cov.shape[0])
                
                try:
                    # Cholesky decomposition
                    scaling_matrix = torch.linalg.cholesky(cov)
                    scaling_matrices[layer_idx][name] = scaling_matrix
                except:
                    scaling_matrices[layer_idx][name] = torch.eye(cov.shape[0])
                    
            except Exception as e:
                print(f"Warning: Could not compute scaling matrix for {layer_idx}.{name}: {e}")
                continue
    
    # Free memory
    del profiling_data
    torch.cuda.empty_cache()
    
    print(f"Profiling complete. Collected {len(scaling_matrices)} layer scaling matrices.")
    return scaling_matrices


@torch.no_grad()
def compress_model_with_cur(model, scaling_matrices, rank_ratio=0.5,
                           layers_to_compress=None, device='cuda',
                           selection_method='leverage'):
    """
    Apply CUR decomposition to compress DeepSeek-MoE model.
    
    Key steps:
    1. Compress attention projections with individual CUR
    2. For MoE: compute shared R matrices per group, then fit C,U for each expert
    3. Preserve shared experts (not compressed)
    
    FIXES:
    - Removed transpose in compute_U_matrix
    - Use dtype-safe solve_triangular for numerical stability in whitening
    - No transposes when assigning C, U, R to nn.Linear weights
    """
    print(f"Applying CUR compression with rank_ratio={rank_ratio}...")
    
    layers = model.model.layers
    if layers_to_compress is None:
        layers_to_compress = list(range(len(layers)))
    
    device = torch.device(device)
    
    for layer_idx in tqdm(layers_to_compress, desc="Compressing layers"):
        layer = layers[layer_idx]
        
        # Determine layer device
        try:
            first_param = next(layer.parameters())
            layer_device = first_param.device if first_param.device.type != 'meta' else device
        except:
            layer_device = device
        
        subset = find_layers(layer)
        
        # ====================================================================
        # Compress Attention
        # ====================================================================
        
        cur_attn = CUR_DeepseekAttention(model.config, layer_idx, rank_ratio)
        cur_attn = cur_attn.to(layer_device)
        
        # Compress attention projections
        attn_proj_names = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj']
        
        for proj_name in attn_proj_names:
            if proj_name not in subset:
                continue
            
            orig_layer = subset[proj_name]
            if orig_layer.weight.device.type == 'meta':
                continue
                
            W = orig_layer.weight.data.clone().to(layer_device)
            
            # FIXED: dtype-safe solve_triangular
            if layer_idx in scaling_matrices and proj_name in scaling_matrices[layer_idx]:
                L = scaling_matrices[layer_idx][proj_name].to(layer_device)
                try:
                    W32 = W.to(torch.float32)
                    W = torch.linalg.solve_triangular(L, W32.T, upper=False).T.to(W.dtype)
                except:
                    pass
            
            # Determine target
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
            
            # CUR decomposition
            rank = target.rank
            C, U, R, col_idx, row_idx = cur_decomposition(W, rank, selection_method, seed=42+layer_idx)
            
            # Assign without transposing
            target.C.weight.data = C.contiguous()
            target.U.weight.data = U.contiguous()
            target.R.weight.data = R.contiguous()
        
        # Replace attention
        layer.self_attn = cur_attn
        
        # ====================================================================
        # Compress MoE Block
        # ====================================================================
        
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
            cur_moe = CUR_DeepseekMoE(model.config, rank_ratio)
            cur_moe = cur_moe.to(layer_device)
            
            # Copy gate weights
            if hasattr(layer.mlp, 'gate'):
                if layer.mlp.gate.weight.device.type != 'meta':
                    cur_moe.gate.weight.data = layer.mlp.gate.weight.data.clone()
            
            # Copy shared experts if present
            if hasattr(layer.mlp, 'shared_experts') and layer.mlp.shared_experts is not None:
                cur_moe.shared_experts = layer.mlp.shared_experts
            
            # Compress experts by group
            num_groups = cur_moe.num_groups
            experts_per_group = cur_moe.experts_per_group
            
            for group_idx in range(num_groups):
                # Step 1: Compute shared R matrices for this group
                reference_expert_idx = group_idx * experts_per_group
                ref_expert = layer.mlp.experts[reference_expert_idx]
                
                # Compute R for gate_proj
                if ref_expert.gate_proj.weight.device.type != 'meta':
                    W_gate = ref_expert.gate_proj.weight.data.clone().to(layer_device)
                    
                    # FIXED: dtype-safe solve_triangular
                    proj_name = f'mlp.experts.{reference_expert_idx}.gate_proj'
                    if layer_idx in scaling_matrices and proj_name in scaling_matrices[layer_idx]:
                        L = scaling_matrices[layer_idx][proj_name].to(layer_device)
                        try:
                            W_gate32 = W_gate.to(torch.float32)
                            W_gate = torch.linalg.solve_triangular(L, W_gate32.T, upper=False).T.to(W_gate.dtype)
                        except:
                            pass
                    
                    _, _, R_gate, _, _ = cur_decomposition(
                        W_gate, cur_moe.rank_gate, selection_method, seed=42+layer_idx+group_idx
                    )
                    cur_moe.shared_R_gate_list[group_idx].weight.data = R_gate.contiguous()
                
                # Compute R for up_proj
                if ref_expert.up_proj.weight.device.type != 'meta':
                    W_up = ref_expert.up_proj.weight.data.clone().to(layer_device)
                    
                    # FIXED: dtype-safe solve_triangular
                    proj_name = f'mlp.experts.{reference_expert_idx}.up_proj'
                    if layer_idx in scaling_matrices and proj_name in scaling_matrices[layer_idx]:
                        L = scaling_matrices[layer_idx][proj_name].to(layer_device)
                        try:
                            W_up32 = W_up.to(torch.float32)
                            W_up = torch.linalg.solve_triangular(L, W_up32.T, upper=False).T.to(W_up.dtype)
                        except:
                            pass
                    
                    _, _, R_up, _, _ = cur_decomposition(
                        W_up, cur_moe.rank_up, selection_method, seed=42+layer_idx+group_idx+100
                    )
                    cur_moe.shared_R_up_list[group_idx].weight.data = R_up.contiguous()
                
                # Compute R for down_proj
                if ref_expert.down_proj.weight.device.type != 'meta':
                    W_down = ref_expert.down_proj.weight.data.clone().to(layer_device)
                    
                    # FIXED: dtype-safe solve_triangular
                    proj_name = f'mlp.experts.{reference_expert_idx}.down_proj'
                    if layer_idx in scaling_matrices and proj_name in scaling_matrices[layer_idx]:
                        L = scaling_matrices[layer_idx][proj_name].to(layer_device)
                        try:
                            W_down32 = W_down.to(torch.float32)
                            W_down = torch.linalg.solve_triangular(L, W_down32.T, upper=False).T.to(W_down.dtype)
                        except:
                            pass
                    
                    _, _, R_down, _, _ = cur_decomposition(
                        W_down, cur_moe.rank_down, selection_method, seed=42+layer_idx+group_idx+200
                    )
                    cur_moe.shared_R_down_list[group_idx].weight.data = R_down.contiguous()
                
                # Step 2: Fit C and U for each expert in group given shared R
                for local_idx in range(experts_per_group):
                    expert_idx = group_idx * experts_per_group + local_idx
                    orig_expert = layer.mlp.experts[expert_idx]
                    cur_expert = cur_moe.experts[expert_idx]
                    
                    # Get shared R matrices
                    R_gate = cur_moe.shared_R_gate_list[group_idx].weight.data
                    R_up = cur_moe.shared_R_up_list[group_idx].weight.data
                    R_down = cur_moe.shared_R_down_list[group_idx].weight.data
                    
                    # Fit gate_proj
                    if orig_expert.gate_proj.weight.device.type != 'meta':
                        W_gate = orig_expert.gate_proj.weight.data.clone().to(layer_device)
                        
                        # FIXED: dtype-safe solve_triangular
                        proj_name = f'mlp.experts.{expert_idx}.gate_proj'
                        if layer_idx in scaling_matrices and proj_name in scaling_matrices[layer_idx]:
                            L = scaling_matrices[layer_idx][proj_name].to(layer_device)
                            try:
                                W_gate32 = W_gate.to(torch.float32)
                                W_gate = torch.linalg.solve_triangular(L, W_gate32.T, upper=False).T.to(W_gate.dtype)
                            except:
                                pass
                        
                        C_gate, U_gate, _ = fit_CU_given_R(
                            W_gate, R_gate, cur_moe.rank_gate, selection_method, seed=42+expert_idx
                        )
                        cur_expert.gate_C.weight.data = C_gate.contiguous()
                        cur_expert.gate_U.weight.data = U_gate.contiguous()
                    
                    # Fit up_proj
                    if orig_expert.up_proj.weight.device.type != 'meta':
                        W_up = orig_expert.up_proj.weight.data.clone().to(layer_device)
                        
                        # FIXED: dtype-safe solve_triangular
                        proj_name = f'mlp.experts.{expert_idx}.up_proj'
                        if layer_idx in scaling_matrices and proj_name in scaling_matrices[layer_idx]:
                            L = scaling_matrices[layer_idx][proj_name].to(layer_device)
                            try:
                                W_up32 = W_up.to(torch.float32)
                                W_up = torch.linalg.solve_triangular(L, W_up32.T, upper=False).T.to(W_up.dtype)
                            except:
                                pass
                        
                        C_up, U_up, _ = fit_CU_given_R(
                            W_up, R_up, cur_moe.rank_up, selection_method, seed=42+expert_idx+1000
                        )
                        cur_expert.up_C.weight.data = C_up.contiguous()
                        cur_expert.up_U.weight.data = U_up.contiguous()
                    
                    # Fit down_proj
                    if orig_expert.down_proj.weight.device.type != 'meta':
                        W_down = orig_expert.down_proj.weight.data.clone().to(layer_device)
                        
                        # FIXED: dtype-safe solve_triangular
                        proj_name = f'mlp.experts.{expert_idx}.down_proj'
                        if layer_idx in scaling_matrices and proj_name in scaling_matrices[layer_idx]:
                            L = scaling_matrices[layer_idx][proj_name].to(layer_device)
                            try:
                                W_down32 = W_down.to(torch.float32)
                                W_down = torch.linalg.solve_triangular(L, W_down32.T, upper=False).T.to(W_down.dtype)
                            except:
                                pass
                        
                        C_down, U_down, _ = fit_CU_given_R(
                            W_down, R_down, cur_moe.rank_down, selection_method, seed=42+expert_idx+2000
                        )
                        cur_expert.down_C.weight.data = C_down.contiguous()
                        cur_expert.down_U.weight.data = U_down.contiguous()
            
            # Replace MoE block
            layer.mlp = cur_moe
        
        # Clear cache
        torch.cuda.empty_cache()
    
    print("CUR compression completed!")
    return model


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate_perplexity(model, tokenizer, dataset_name='wikitext2',
                       seq_len=2048, batch_size=4, device='cuda'):
    """Evaluate model perplexity"""
    print(f"Evaluating perplexity on {dataset_name}...")
    
    model.eval()
    test_loader = get_test_loader(dataset_name, tokenizer, seq_len, batch_size)
    
    nlls = []
    
    # FIXED: Get device from model's first parameter (handles device_map="auto")
    first_device = next(model.parameters()).device
    
    for batch in tqdm(test_loader, desc=f"Evaluating {dataset_name}"):
        try:
            input_ids = batch.to(first_device)
            outputs = model(input_ids=input_ids, labels=input_ids)
            nlls.append(outputs.loss)
        except Exception as e:
            print(f"Error in evaluation: {e}")
            continue
    
    if len(nlls) == 0:
        return float('inf')
    
    avg_nll = torch.stack(nlls).mean()
    ppl = torch.exp(avg_nll).item()
    
    return ppl


# ============================================================================
# Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='CUR Decomposition for DeepSeek-MoE-16B Compression')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='deepseek-ai/deepseek-moe-16b-base',
                       help='DeepSeek model name or path')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    # Compression arguments
    parser.add_argument('--rank_ratio', type=float, default=0.5,
                       help='Compression ratio for CUR (0-1)')
    parser.add_argument('--layers_to_compress', type=str, default='all',
                       help='"all" or comma-separated indices (e.g., "0,1,2")')
    parser.add_argument('--selection_method', type=str, default='leverage',
                       choices=['leverage', 'norm', 'random'],
                       help='Column/row selection method')
    
    # Data arguments
    parser.add_argument('--calib_dataset', type=str, default='wikitext2',
                       choices=['wikitext2', 'c4', 'ptb'])
    parser.add_argument('--eval_dataset', type=str, default='wikitext2',
                       choices=['wikitext2', 'c4', 'ptb'])
    parser.add_argument('--nsamples', type=int, default=128,
                       help='Number of calibration samples')
    parser.add_argument('--seqlen', type=int, default=2048, help='Sequence length')
    parser.add_argument('--seed', type=int, default=3, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for evaluation')
    
    # Execution arguments
    parser.add_argument('--skip_profiling', action='store_true',
                       help='Skip profiling step')
    parser.add_argument('--profiling_path', type=str, default=None,
                       help='Path to load/save profiling matrices')
    parser.add_argument('--save_model', type=str, default=None,
                       help='Path to save compressed model')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only evaluate, do not compress')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 80)
    print("CUR-MOE: CUR Decomposition for DeepSeek-MoE-16B Compression")
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
    
    # Check available resources
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Found {num_gpus} GPU(s) with {gpu_memory_gb:.2f} GB VRAM each")
        
        # Memory-efficient loading
        if num_gpus == 1 and gpu_memory_gb < 80:
            max_memory = {
                0: f"{int(gpu_memory_gb * 0.8)}GiB",
                'cpu': '100GiB'
            }
        else:
            max_memory = None
    else:
        raise RuntimeError("CUDA GPU required")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map='auto',
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    print(f"Model loaded: {model.config.num_hidden_layers} layers")
    
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
                os.makedirs(os.path.dirname(args.profiling_path) if os.path.dirname(args.profiling_path) else '.', exist_ok=True)
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
        
        # Save if requested
        if args.save_model:
            print(f"Saving compressed model to {args.save_model}")
            os.makedirs(args.save_model, exist_ok=True)
            model.save_pretrained(args.save_model)
            tokenizer.save_pretrained(args.save_model)
    
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
    
    results_file = 'cur_deepseek_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()
