import json
import os
import sys
import argparse
import copy
import torch.jit
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import math
from typing import List, Optional, Tuple, Union
from accelerate import init_empty_weights,infer_auto_device_map, dispatch_model
from utils.data_utils import *
from component.svd_llama import SVD_LlamaAttention, SVD_LlamaMLP
from component.svd_mistral import SVD_MistralAttention, SVD_MistralMLP
from component.svd_opt import SVDOPTDecoderLayer
from component.svd_mixtral_sharing import SVD_MixtralSparseMoeBlock, SVD_MixtralAttention, SVD_MixtralAttention_dict, SVD_MixtralSparseMoeBlock_list
from component.svd_deepseek_sharing import SVD_DeepseekAttention, SVD_DeepseekMoE
from component.deepseek.modeling_deepseek import MoEGate, DeepseekMoE
from component.svd_phimoe_sharing import SVD_PhiMoEAttention, SVD_PhiMoESparseMoeBlock
from utils.model_utils import *
from evaluater import * 
from datetime import datetime
from transformers.models.mixtral.modeling_mixtral import *
import pdb
from accelerate import Accelerator
from functools import partial
import heapq
os.environ['HF_TRUST_REMOTE_CODE'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)
accelerator = Accelerator()

class local_update:
    def __init__(self, layer, scaling_diag_matrix, ratio, name, direct_update=False):
        self.layer = layer
        self.name = name
        self.dev = self.layer.weight.device
        self.W = layer.weight.data.clone()
        self.rows = self.W.shape[0]
        self.columns = self.W.shape[1]
        self.ratio = ratio  # Note: the ratio here is already 1 - the user input ratio
        self.direct_update = direct_update
        self.scaling_diag_matrix = scaling_diag_matrix
        self.data_count = 0
        self.sum_input = None
        self.sum_output = None
        self.initialize_svd()

    def initialize_svd(self):
        if self.direct_update:
            self.U, self.S, self.VT = torch.linalg.svd(self.W.data, full_matrices=False)
        else: 
            try:
                scaling_matrix_inv = torch.linalg.inv(self.scaling_diag_matrix)
            except Exception as e:
                print(f"Warning: scaling_diag_matrix is not full rank for {self.name}!")
                self.scaling_diag_matrix += 1e-6 * torch.eye(self.scaling_diag_matrix.shape[0])
                scaling_matrix_inv = torch.linalg.inv(self.scaling_diag_matrix)
            self.scaling_diag_matrix = self.scaling_diag_matrix.float()
            scaling_matrix_inv = scaling_matrix_inv.float()
            W_scale = torch.matmul(self.W, self.scaling_diag_matrix)
            self.U, self.S, self.VT = torch.linalg.svd(W_scale, full_matrices=False)

        # Calculate the number of singular values to retain
        self.num_s_after_trunc = int(self.W.shape[0] * self.W.shape[1] * self.ratio / (self.W.shape[0] + self.W.shape[1]))
        self.truc_s = self.S[:self.num_s_after_trunc].cuda()
        self.truc_u = self.U[:, :self.num_s_after_trunc].cuda()
        if self.direct_update:
            self.truc_v = self.VT[:self.num_s_after_trunc, :].cuda()
        else:
            self.truc_v = torch.matmul(self.VT[:self.num_s_after_trunc, :].cuda(), scaling_matrix_inv)
        self.truc_sigma = torch.diag(self.truc_s)
        self.new_w = torch.matmul(self.truc_u, torch.matmul(self.truc_sigma, self.truc_v[:self.num_s_after_trunc, :]))
        self.updated_err = self.error = 0

    def add_batch_update_u(self, inp, out):
        if torch.all(inp == 0):
            print(f"Warning: Entirely zero input for {self.name}. Skipping this batch.")
            return

        if inp.dim() == 3:
            inps = inp.view(-1, inp.size(-1))
            outs = out.view(-1, out.size(-1))
        else:
            inps = inp
            outs = out

        non_zero_mask = torch.any(inps != 0, dim=1)
        inps = inps[non_zero_mask]
        outs = outs[non_zero_mask]

        if inps.shape[0] == 0:
            print(f"Warning: No non-zero inputs for {self.name} after filtering. Skipping this batch.")
            return

        self.data_count += inps.shape[0]

        if self.sum_input is None:
            self.sum_input = inps.sum(dim=0)
            self.sum_output = outs.sum(dim=0)
        else:
            self.sum_input += inps.sum(dim=0)
            self.sum_output += outs.sum(dim=0)

        new_w = torch.matmul(self.truc_u, torch.matmul(self.truc_sigma, self.truc_v))
        new_output = inps.matmul(new_w.t())
        self.error = torch.sqrt(torch.sum((outs - new_output)**2)).item() / (torch.norm(outs, p='fro').item() + 1e-9)
        x = torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma)
        
        self.updated_uT = torch.pinverse(x) @ outs

        updated_output = torch.matmul(torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma), self.updated_uT)
        self.updated_error = torch.sqrt(torch.sum((outs - updated_output)**2)).item() / (torch.norm(outs, p='fro').item() + 1e-9)
        
        del inps, outs, new_output, updated_output, x, new_w
        torch.cuda.empty_cache()
    
    def fasterprune(self):
        if self.data_count == 0 or (self.sum_input is not None and torch.all(self.sum_input == 0)):
            print(f"Warning: No valid data passed through {self.name}. Keeping original weights.")
            U, S, VT = torch.linalg.svd(self.W, full_matrices=False)
            sqrtSigma = torch.sqrt(torch.diag(S[:self.num_s_after_trunc]))
            return (U[:, :self.num_s_after_trunc] @ sqrtSigma).to(self.dev), (sqrtSigma @ VT[:self.num_s_after_trunc, :]).to(self.dev)
        else:
            sqrtSigma = torch.sqrt(self.truc_sigma)
            self.appendU = self.updated_uT.t().matmul(sqrtSigma)
            self.appendV = sqrtSigma.matmul(self.truc_v)
            return self.appendU, self.appendV

    def get_compression_ratio(self):
        original_params = self.rows * self.columns
        compressed_params = self.num_s_after_trunc * (self.rows + self.columns + 1)
        return 1 - (compressed_params / original_params)




@torch.no_grad()
def profle_svdllm(model_name, model, calib_loader, dev, selected_layers=None, Attn_or_Experts='both'):

    if "llama" in model_name or "Mistral" in model_name or "vicuna" in model_name or "Mixtral" in model_name:
        layers = model.model.layers
    elif "opt" in model_name:
        layers = model.model.decoder.layers

    if isinstance(selected_layers, str) and selected_layers.lower() == 'all':
        selected_layers = list(range(len(layers)))
    
    model = model.to(dev)
    print("Start obtaining the whitening matrix...")
    def hook(module, input, output):
        inp = input[0].detach().float()
        if inp.dim() == 2:   # for opt
            inp = inp.unsqueeze(0)
        adds = torch.matmul(inp.transpose(1,2), inp)
        adds_sum = torch.sum(adds, dim=0)
        module.raw_scaling_diag_matrix += adds_sum
        del inp, adds, adds_sum
        torch.cuda.empty_cache()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.raw_scaling_diag_matrix = 0
            module.register_forward_hook(hook)
    for batch in tqdm(calib_loader):
        batch = {k: v.to(dev) for k, v in batch.items()}
        
        model(**batch)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
    torch.cuda.empty_cache()
    model = model.cpu()
    for i in selected_layers:
        subset = find_layers(layers[i])
        for name in subset:
            if 'Mixtral' in model_name:
                if Attn_or_Experts == 'Attn' and 'experts' in name:
                    continue
                if Attn_or_Experts == 'Experts' and 'experts' not in name:
                    continue
                if 'gate' in name:  # Skip gate processing
                    continue
            subset[name].raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix.cpu()
    profiling_mat = {}
    print("Start Cholesky Decomposition...")
    for i in tqdm(selected_layers):
        layer_profile = {}
        subset = find_layers(layers[i])
        for name in subset:
            if 'Mixtral' in model_name:
                if Attn_or_Experts == 'Attn' and 'experts' in name:
                    continue
                if Attn_or_Experts == 'Experts' and 'experts' not in name:
                    continue
                if 'gate' in name:  # Skip gate processing
                    continue
            raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix.double().to(dev)
            try:
                scaling_diag_matrix = make_positive_definite(raw_scaling_diag_matrix)
            except Exception as e:
                print(f"Warning: eigen scaling_diag_matrix is not positive for {name}!")
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                eigenvalues = None
                del eigenvalues
            layer_profile[name] = scaling_diag_matrix.cpu()
            scaling_diag_matrix = raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix = None
            del scaling_diag_matrix, raw_scaling_diag_matrix, subset[name].raw_scaling_diag_matrix
            torch.cuda.empty_cache()
        profiling_mat[i] = layer_profile
    return profiling_mat



def nearest_positive_definite(A):
    # Symmetrize the matrix
    B = (A + A.T) / 2
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(B)
    # Clip negative eigenvalues to a small positive number
    min_eig = torch.min(eigenvalues)
    if min_eig < 0:
        eigenvalues = eigenvalues + (-min_eig + 1e-8)
    # Reconstruct the positive definite matrix
    A_pd = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
    # Ensure the matrix is symmetric
    A_pd = (A_pd + A_pd.T) / 2
    return A_pd

def make_positive_definite(matrix, initial_adjustment=1e-6, max_attempts=12, adjustment_factor=6):
    attempts = 0
    
    while attempts < max_attempts:
        try:
            # Cholesky decomposition
            chol_matrix = torch.linalg.cholesky(matrix)
            # Check for NaNs
            if torch.isnan(chol_matrix).any() or torch.isinf(chol_matrix).any():
                print("Warning: NaN or Inf detected in Cholesky decomposition result.")
                # Use Higham's algorithm to adjust the matrix
                matrix_pd = nearest_positive_definite(matrix)
                # Retry Cholesky decomposition
                chol_matrix = torch.linalg.cholesky(matrix_pd)
            if torch.isnan(chol_matrix).any() or torch.isinf(chol_matrix).any():
                print("nan")
            return chol_matrix
        except torch._C._LinAlgError:
            # Fail, try again
            attempts += 1
            eigenvalues = torch.linalg.eigvalsh(matrix)
            adjustment = max(initial_adjustment, -eigenvalues[0] * 1e-3)
            matrix += adjustment * torch.eye(matrix.shape[0]).to(matrix.device)
            initial_adjustment *= adjustment_factor

    raise ValueError("Failed")

@torch.no_grad()
def profle_svdllm_low_resource(model_name, model, calib_loader, dev, selected_layers=None, Attn_or_Experts='both', attention_layers=[],  expert_layers=[]):
    def hook(module, input, output, module_name=None):

        nonlocal scaling_matrix_accumulation_count,expert_selection_counts
        if module_name not in scaling_matrix_accumulation_count:
            scaling_matrix_accumulation_count[module_name] = 0 
        if isinstance(module, MixtralSparseMoeBlock):
            router_logits = output[1]  # Directly get router_logits
            # Calculate routing_weights and selected_experts
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            _, selected_experts = torch.topk(routing_weights, k=module.top_k, dim=-1)

            # Update expert_selection_counts
            for expert_idx in selected_experts.unique():
                expert_selection_counts[i][expert_idx.item()] += (selected_experts == expert_idx).sum().item()
            
            return  # Directly return to avoid affecting `profiling_mat`
    
        if type(module).__name__ == 'MoEGate':
            router_logits = F.linear(input[0], module.weight, None)  # Directly get router_logits
            # Calculate routing_weights and selected_experts
            if module.scoring_func == 'softmax':
                routing_weights = router_logits.softmax(dim=-1)
            _, selected_experts = torch.topk(routing_weights, k=module.top_k, dim=-1, sorted=False)

            # Update expert_selection_counts
            for expert_idx in selected_experts.unique():
                expert_selection_counts[i][expert_idx.item()] += (selected_experts == expert_idx).sum().item()
            
            return  # Directly return to avoid affecting `profiling_mat`
        
        # phi 3.5
        if type(module).__name__ == 'PhiMoESparseMoeBlock':
            router_logits = output[1]  # Directly get router_logits
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            _, selected_experts = torch.topk(routing_weights, k=module.top_k, dim=-1)

            for expert_idx in selected_experts.unique():
                expert_selection_counts[i][expert_idx.item()] += (selected_experts == expert_idx).sum().item()
            
            return
        
        inp = input[0].detach().float()
        if torch.any(torch.isnan(inp)) or torch.any(torch.isinf(inp)):
            with open(log_file_check, 'a') as f:
                f.write(f"NaN or Inf detected in input for {module_name}\n")
        if inp.dim() == 2:  # for opt
            inp = inp.unsqueeze(0)
        adds = torch.matmul(inp.transpose(1, 2), inp)
        adds_sum = torch.sum(adds, dim=0)

        if torch.any(adds_sum < 0):
            with open(log_file_check, 'a') as f:
                f.write(f"Negative value detected in adds_sum for {module_name}\n")
                f.write(f"Min value: {adds_sum.min().item()}\n")
        module.scaling_diag_matrix += adds_sum

        if torch.isnan(module.scaling_diag_matrix).any() or torch.isinf(module.scaling_diag_matrix).any():
            print(f"Warning: NaN or Inf detected in scaling_diag_matrix for {module_name}")
            print(f"scaling_diag_matrix stats: min {module.scaling_diag_matrix.min().item()}, max {module.scaling_diag_matrix.max().item()}, mean {module.scaling_diag_matrix.mean().item()}")

        del inp, adds, adds_sum, output
        torch.cuda.empty_cache()

    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)

 
    layers[0] = layers[0].to(dev)
    # Create log file
    log_dir = "scaling_matrix_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"scaling_matrix_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")  
    log_file_check = os.path.join(log_dir, f'non_positive_check_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    # Count the number of times the scaling matrix is accumulated for each expert
    scaling_matrix_accumulation_count = {}
    if isinstance(selected_layers, str) and selected_layers.lower() == 'all':
        selected_layers = list(range(len(layers)))
        

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(calib_loader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    ) # 1
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.cpu()
            cache['i'] += 1

            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask'].cpu()
                if "opt" not in model_name:
                    cache['position_ids'] = kwargs['position_ids'].cpu()
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask'].cpu()), dim=0)
                if "opt" not in model_name:
                    cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids'].cpu()), dim=0)
 
            raise ValueError

    layers[0] = Catcher(layers[0])
    print("Data input to model completed")
    for batch in tqdm(calib_loader):
        try:
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass
    print("Model run completed")
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "opt" in model_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    else:  
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    if "opt" not in model_name:
        position_ids = cache['position_ids']
    profiling_mat = {}


    expert_selection_counts = {i: [0] * 8 for i in selected_layers}
    if 'deepseek' in model_name:
            expert_selection_counts = {i: [0] * 64 for i in selected_layers}
    if 'phimoe' in model_name:
            expert_selection_counts = {i: [0] * 16 for i in selected_layers}

    for i in tqdm(selected_layers):
        layer_profile = {}
        layer = layers[i].to(dev)
        subset = find_layers(module=layer,layers=[nn.Conv2d, nn.Linear, MixtralSparseMoeBlock, MoEGate,PhiMoESparseMoeBlock], process_moe_block=True)
       
        handles = []
        for name, module in subset.items():
            if 'Mixtral' in model_name or 'deepseek' in model_name or 'phimoe' in model_name:
                if isinstance(module, MixtralSparseMoeBlock) or type(module).__name__ == 'MoEGate' or type(module).__name__ == 'PhiMoESparseMoeBlock':
                    
                    # Handle MixtralSparseMoeBlock separately
                    if Attn_or_Experts in ['Experts', 'both']:
                        full_name = f"layer_{i}_{name}"
                        handles.append(module.register_forward_hook(partial(hook, module_name=full_name)))
                    continue
                
                if Attn_or_Experts == 'Attn' and 'experts' in name:
                    continue
                if Attn_or_Experts == 'Experts' and 'experts' not in name:
                    continue
                if 'experts' in name and i not in expert_layers:
                    continue
                if  'attn' in name and i not in attention_layers:
                    continue
                if  'Mixtral' in model_name and 'gate' in name:  # Skip gate processing
                    continue
                if  'deepseek' in model_name and type(module).__name__ == 'MoEGate':
                    continue
                if  'phimoe' in model_name and 'block_sparse_moe.gate' in name:
                    continue

            if isinstance(module, nn.Linear):
                module.scaling_diag_matrix = 0
                full_name = f"layer_{i}_{name}"
                handles.append(module.register_forward_hook(partial(hook, module_name=full_name)))
        for j in range(inps.shape[0]):
            
            if "opt" not in model_name:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(dev), position_ids=position_ids[j].unsqueeze(0).to(dev))[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(dev))[0]
        for h in handles:
            h.remove()
        layer = layer.cpu()

        # Weighted
        total_selections = sum(expert_selection_counts[i])
        if total_selections > 0:
            for name, module in tqdm(subset.items()):
                if name.startswith('block_sparse_moe.experts.') or name.startswith('mlp.experts.'):
                    if Attn_or_Experts in ['Experts', 'both']:
                        if 'Mixtral' in model_name:
                            parts = name.split('.')
                            if len(parts) == 4 and parts[3] in ['w1', 'w2', 'w3']:
                                expert_idx = int(parts[2])
                                expert_selections = expert_selection_counts[i][expert_idx]
                                usage_factor = expert_selections / total_selections
                                subset[name].scaling_diag_matrix *= (1 + usage_factor)
                        elif 'deepseek' in model_name:
                            parts = name.split('.')
                            if len(parts) == 7 and parts[6] in ['down_proj', 'up_proj', 'gate_proj']:
                                expert_idx = int(parts[5])
                                expert_selections = expert_selection_counts[i][expert_idx]
                                usage_factor = expert_selections / total_selections
                                try:
                                    subset[name].scaling_diag_matrix *= (1 + usage_factor)
                                except:
                                    pdb.set_trace()
                        elif 'phimoe' in model_name:
                            parts = name.split('.')
                            if len(parts) == 7 and parts[6] in ['w1', 'w2', 'w3']:
                                expert_idx = int(parts[5])
                                expert_selections = expert_selection_counts[i][expert_idx]
                                usage_factor = expert_selections / total_selections
                                subset[name].scaling_diag_matrix *= (1 + usage_factor)


        print(f"Log file will be created at: {os.path.abspath(log_file)}")
        print(f"Subset for layer {i}:")
        
        for name in tqdm(subset):
            if 'Mixtral' in model_name  or 'deepseek' in model_name or 'phimoe' in model_name: 
                if Attn_or_Experts == 'Attn' and 'experts' in name:
                    continue
                if Attn_or_Experts == 'Experts' and 'experts' not in name:
                    continue
                if 'experts' in name and i not in expert_layers:
                    continue
                if  'attn' in name and i not in attention_layers:
                    continue
                if  'Mixtral' in model_name and 'gate' in name:  # Skip gate processing
                    continue
                if  'deepseek' in model_name and type(subset[name]).__name__ == 'MoEGate':
                    continue
                if  'phimoe' in model_name and 'block_sparse_moe.gate' in name:
                    continue
                if isinstance(subset[name], MixtralSparseMoeBlock) or type(subset[name]).__name__ == 'MoEGate' or type(subset[name]).__name__ == 'PhiMoESparseMoeBlock':
                    continue 
            if isinstance(subset[name].scaling_diag_matrix, (float,int)):
                raw_scaling_diag_matrix = subset[name].scaling_diag_matrix
            else:
                raw_scaling_diag_matrix = subset[name].scaling_diag_matrix.double().to(dev)

            try:
                if isinstance(raw_scaling_diag_matrix, (float,int)):
                    scaling_diag_matrix = raw_scaling_diag_matrix
                else:
                    scaling_diag_matrix = make_positive_definite(raw_scaling_diag_matrix)
            except Exception as e:
                print(f"Warning: eigen scaling_diag_matrix is not positive for {name}!")
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                eigenvalues = None
                del eigenvalues
            
            if isinstance(scaling_diag_matrix, (float,int)):
                layer_profile[name] = scaling_diag_matrix
            else:
                layer_profile[name] = scaling_diag_matrix.cpu()
            scaling_diag_matrix = raw_scaling_diag_matrix = subset[name].scaling_diag_matrix = None
            del scaling_diag_matrix, raw_scaling_diag_matrix, subset[name].scaling_diag_matrix
            torch.cuda.empty_cache()
        
        layers[i] = layer.cpu()
        profiling_mat[i] = layer_profile
        inps = outs
        torch.cuda.empty_cache()
        
    for i, layer_profile in profiling_mat.items():
        for name, matrix in layer_profile.items():
            if isinstance(matrix, torch.Tensor):
                if torch.isnan(matrix).any() or torch.isinf(matrix).any():
                    print(f"Warning: NaN or Inf detected in final profiling_mat for layer {i}, name {name}")
                    print(f"Matrix stats: min {matrix.min().item()}, max {matrix.max().item()}, mean {matrix.mean().item()}")

    return profiling_mat, expert_selection_counts  # Modify: return expert selection counts

@torch.no_grad()
def whitening(model_name, model, profiling_mat, expert_selection_counts, ratios, dev, selected_layers=None, Attn_or_Experts='both', outlier_or_frequency="frequency", attention_layers=[],  expert_layers=[]):

    model.eval()
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    
    if isinstance(selected_layers, str) and selected_layers.lower() == 'all':
        selected_layers = list(range(len(layers)))
    
    print("Start SVD decomposition after whitening...")

    for i in tqdm(selected_layers, desc="Processing layers", unit="layer"):
        layer = layers[i]

        subset = find_layers(layer)
        

        # This structural change is new: initialize the shared V matrix for the Mixtral model
        if ('Mixtral' in model_name or 'deepseek' in model_name or 'phimoe' in model_name) and i in expert_layers:
            if 'Mixtral' in model_name:
                if outlier_or_frequency == "outlier" and isinstance(ratios, dict):
                    i_outliers={}
                    for exp in range(8):
                        i_outliers[str(exp)] = 0
                    for k,v in ratios:
                        if k.split(".")[3]==i and k.split(".")[-1] in ["w1","w2","w3"]:
                            i_outliers[str(k.split(".")[-2])] += ratios[k]
                    most_frequent_expert = int(max(i_outliers, key=i_outliers.get))
                else:
                    most_frequent_expert = max(range(8), key=lambda x, i=i: expert_selection_counts[i][x])
            elif 'deepseek' in model_name:
                chunk_size = 8
                num_chunks = len(expert_selection_counts[i])//chunk_size
                if outlier_or_frequency == "outlier" and isinstance(ratios, dict):
                    i_outliers={}
                    for exp in range(64):
                        i_outliers[str(exp)] = 0
                    for k,v in ratios:
                        if k.split(".")[3]==i and k.split(".")[-1] in ["gate_proj","up_proj","down_proj"]:
                            i_outliers[str(k.split(".")[-2])] += ratios[k]

                    most_frequent_expert = heapq.nlargest(8, range(64), key=lambda idx: i_outliers.get(str(idx)))
                    most_frequent_expert.sort()
                else:
                    most_frequent_expert = heapq.nlargest(8, range(64), key=lambda x, i=i: expert_selection_counts[i][x])
                    most_frequent_expert.sort()
            elif 'phimoe' in model_name:
                if outlier_or_frequency == "outlier" and isinstance(ratios, dict):
                    i_outliers = {str(exp): 0 for exp in range(16)}  # PhiMoE has 16 experts
                    for k, v in ratios.items():
                        if k.split(".")[2] == str(i) and k.split(".")[-1] in ["w1", "w2", "w3"]:
                            i_outliers[str(k.split(".")[-2])] += ratios[k]
                    most_frequent_expert = int(max(i_outliers, key=i_outliers.get))
                else:
                    most_frequent_expert = max(range(16), key=lambda x: expert_selection_counts[i][x])

            shared_w1_v = shared_w2_v = shared_w3_v = None
            if 'deepseek' in model_name:
                shared_gate_v = {str(i): {} for i in range(8)}
                shared_up_v = {str(i): {} for i in range(8)}
                shared_down_v = {str(i): {} for i in range(8)}

        Moe_List_Avg=[]
        if 'Mixtral' in model_name:
            if isinstance(ratios, dict):
                temp_ratios_dict = copy.deepcopy(ratios)
                w1,w2,w3=[],[],[]
                for k in list(temp_ratios_dict.keys()):
                    v = temp_ratios_dict[k]
                    if k.split(".")[2] != str(i):
                        del temp_ratios_dict[k]
                        continue
                    if k.split(".")[-1] == 'w1':
                        w1.append(v)
                    if k.split(".")[-1] == 'w2':
                        w2.append(v)
                    if k.split(".")[-1] == 'w3':
                        w3.append(v)
                Moe_List_Avg=[sum(w1)/len(w1),sum(w2)/len(w2),sum(w3)/len(w3)]

                svd_attn = SVD_MixtralAttention_dict(config=model.config,layer_idx=i, ratio=temp_ratios_dict)
                svd_moe = SVD_MixtralSparseMoeBlock_list(config=model.config, ratio=[sum(w1)/len(w1),sum(w2)/len(w2),sum(w3)/len(w3)])
            else:
                svd_attn = SVD_MixtralAttention(config=model.config,layer_idx=i, ratio=ratios)
                svd_moe = SVD_MixtralSparseMoeBlock(config=model.config, ratio=ratios)

        if 'phimoe' in model_name:
            if isinstance(ratios, dict):
                temp_ratios_dict = copy.deepcopy(ratios)
                w1, w2, w3 = [], [], []
                for k in list(temp_ratios_dict.keys()):
                    v = temp_ratios_dict[k]
                    if k.split(".")[2] != str(i):
                        del temp_ratios_dict[k]
                        continue
                    if k.split(".")[-1] == 'w1':
                        w1.append(v)
                    if k.split(".")[-1] == 'w2':
                        w2.append(v)
                    if k.split(".")[-1] == 'w3':
                        w3.append(v)
                Moe_List_Avg = [sum(w1)/len(w1), sum(w2)/len(w2), sum(w3)/len(w3)]
                svd_attn = SVD_PhiMoEAttention(config=model.config, layer_idx=i, ratio=temp_ratios_dict)
                svd_moe = SVD_PhiMoESparseMoeBlock(config=model.config, ratio=Moe_List_Avg)
            else:
                svd_attn = SVD_PhiMoEAttention(config=model.config, layer_idx=i, ratio=ratios)
                svd_moe = SVD_PhiMoESparseMoeBlock(config=model.config, ratio=ratios)


        if 'deepseek' in model_name:
            svd_attn = SVD_DeepseekAttention(config=model.config,layer_idx=i, ratio=ratios)
            svd_moe = SVD_DeepseekMoE(config=model.config, ratio=ratios)
        '''if 'phi' in model_name:
            svd_attn = SVD_PhiMoEAttention(config=model.config, layer_idx=i, ratio=ratios)
            svd_moe = SVD_PhiMoESparseMoeBlock(config=model.config, ratio=ratios)'''
        



        for name in subset:
            
            if 'Mixtral' in model_name:
                if Attn_or_Experts == 'Attn' and 'experts' in name:
                    continue
                if Attn_or_Experts == 'Experts' and 'experts' not in name:
                    continue
                if 'experts' in name and i not in expert_layers:
                    continue
                if  'attn' in name and i not in attention_layers:
                    continue
                if 'gate' in name:
                    continue

            if 'deepseek' in model_name:
                if Attn_or_Experts == 'Attn' and 'experts' in name:
                    continue
                if Attn_or_Experts == 'Experts' and 'experts' not in name:
                    continue
                if 'experts' in name and i not in expert_layers:
                    continue
                if  'attn' in name and i not in attention_layers:
                    continue
                if type(subset[name]).__name__ == 'MoEGate' or 'shared_experts' in name:
                    continue

            if 'phi' in model_name:
                if Attn_or_Experts == 'Attn' and 'experts' in name:
                    continue
                if Attn_or_Experts == 'Experts' and 'experts' not in name:
                    continue
                if 'experts' in name and i not in expert_layers:
                    continue
                if  'attn' in name and i not in attention_layers:
                    continue
                if 'block_sparse_moe.gate' in name:
                    continue
            

            if isinstance(ratios, float):
                ratio = ratios
            elif 'moe' in name:
                if 'w1' in name:
                    ratio = Moe_List_Avg[0]
                elif 'w2' in name:
                    ratio = Moe_List_Avg[1]
                elif 'w3' in name:
                    ratio = Moe_List_Avg[2]
            else:
                ratio = ratios['model.layers.'+str(i)+"."+name]
            W = subset[name].weight.data.float().to(dev)
            dtype = W.dtype
            scaling_diag_matrix = profiling_mat[i][name]


            if isinstance(scaling_diag_matrix, (float,int)) and scaling_diag_matrix == 0:
                print(f"Skipping module {name} in layer {i} due to zero scaling matrix.")
                continue
            elif isinstance(scaling_diag_matrix, torch.Tensor) and torch.all(scaling_diag_matrix == 0):
                print(f"Skipping module {name} in layer {i} due to zero scaling matrix.")
                continue
            scaling_diag_matrix=scaling_diag_matrix.to(dev)
            if torch.isnan(scaling_diag_matrix).any() or torch.isinf(scaling_diag_matrix).any():
                print(f"Warning: NaN or Inf detected in scaling_diag_matrix for layer {i}, name {name}")
                print(f"scaling_diag_matrix stats: min {scaling_diag_matrix.min().item()}, max {scaling_diag_matrix.max().item()}, mean {scaling_diag_matrix.mean().item()}")

            epsilon = 1e-6
            max_attempts = 20
            attempts = 0

            while attempts < max_attempts:
                try:
                    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                    break
                except torch._C._LinAlgError:
                    print(f"Warning: scaling_diag_matrix is singular for {name} in layer {i}. Adding small value to diagonal.")
                    scaling_diag_matrix += epsilon * torch.eye(scaling_diag_matrix.shape[0], device=scaling_diag_matrix.device)
                    epsilon *= 10  # Increase epsilon to ensure convergence
                    attempts += 1

            if attempts == max_attempts:
                print(f"Failed to invert scaling_diag_matrix for {name} in layer {i} after {max_attempts} attempts. Using pseudo-inverse instead.")
                scaling_matrix_inv = torch.linalg.pinv(scaling_diag_matrix)
            '''try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                print("Warning: scaling_diag_matrix is not full rank!")
                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(dev)
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)'''

            scaling_diag_matrix = scaling_diag_matrix.float()
            scaling_matrix_inv = scaling_matrix_inv.float()
            W_scale = torch.matmul(W, scaling_diag_matrix)
            U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)

            print(f"For layer {i}, {name}:")
            print(f"W.shape[0]: {W.shape[0]}")
            print(f"W.shape[1]: {W.shape[1]}")
            num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
            print(f"num_s_after_trunc: {num_s_after_trunc}")
            truc_s = S[:num_s_after_trunc]
            truc_u = U[:, :num_s_after_trunc]
            truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv)
            truc_sigma = torch.diag(truc_s)
            #### Replace Attn, MLP ####
            sqrtSigma = torch.sqrt(truc_sigma)
            svd_u = torch.matmul(truc_u, sqrtSigma).cpu().to(dtype)
            svd_v = torch.matmul(sqrtSigma, truc_v).cpu().to(dtype)
            # print(f"svd_u shape: {svd_u.shape}")
            # print(f"svd_v shape: {svd_v.shape}")
            if 'deepseek' in model_name:    
                if "q_proj" in name:
                    svd_attn.q_u_proj.weight.data = svd_u
                    svd_attn.q_v_proj.weight.data = svd_v
                elif "k_proj" in name:
                    svd_attn.k_u_proj.weight.data = svd_u
                    svd_attn.k_v_proj.weight.data = svd_v
                elif "v_proj" in name:
                    svd_attn.v_u_proj.weight.data = svd_u
                    svd_attn.v_v_proj.weight.data = svd_v
                elif "o_proj" in name:
                    svd_attn.o_u_proj.weight.data = svd_u
                    svd_attn.o_v_proj.weight.data = svd_v
                    layer.self_attn = svd_attn
                elif "experts" in name:
                    expert_idx = int(name.split("experts.")[1].split('.')[0])
                    group_idx = expert_idx // 8
                    if "gate" in name:
                        svd_moe.experts[expert_idx].gate_proj_u.weight.data = svd_u
                        # if expert_idx in most_frequent_expert:
                            # shared_gate_v[str(most_frequent_expert.index(expert_idx))]["share_gate"] = svd_v
                        

                        if expert_idx in most_frequent_expert:
                            if "share_gate" not in shared_gate_v[str(group_idx)]:
                                shared_gate_v[str(group_idx)]["share_gate"] = svd_v
                                shared_gate_v[str(group_idx)]["original_expert"] = expert_idx
                            elif expert_selection_counts[i][expert_idx] > expert_selection_counts[i][shared_gate_v[str(group_idx)]["original_expert"]]:
                                # Find an empty group
                                empty_group = next(g for g in range(8) if "share_gate" not in shared_gate_v[str(g)])
                                shared_gate_v[str(empty_group)] = shared_gate_v[str(group_idx)]
                                shared_gate_v[str(group_idx)] = {"share_gate": svd_v, "original_expert": expert_idx}
                    elif "up" in name:
                        svd_moe.experts[expert_idx].up_proj_u.weight.data = svd_u
                        # if expert_idx in most_frequent_expert:
                            # shared_up_v[str(most_frequent_expert.index(expert_idx))]["share_up"] = svd_v
                        
                        if expert_idx in most_frequent_expert:
                            if "share_up" not in shared_up_v[str(group_idx)]:
                                shared_up_v[str(group_idx)]["share_up"] = svd_v
                                shared_up_v[str(group_idx)]["original_expert"] = expert_idx
                            elif expert_selection_counts[i][expert_idx] > expert_selection_counts[i][shared_up_v[str(group_idx)]["original_expert"]]:
                                empty_group = next(g for g in range(8) if "share_up" not in shared_up_v[str(g)])
                                shared_up_v[str(empty_group)] = shared_up_v[str(group_idx)]
                                shared_up_v[str(group_idx)] = {"share_up": svd_v, "original_expert": expert_idx}

                    elif "down" in name:
                        svd_moe.experts[expert_idx].down_proj_u.weight.data = svd_u
                        # if expert_idx in most_frequent_expert:
                            # shared_down_v[str(most_frequent_expert.index(expert_idx))]["share_down"] = svd_v
                        if expert_idx in most_frequent_expert:
                            if "share_down" not in shared_down_v[str(group_idx)]:
                                shared_down_v[str(group_idx)]["share_down"] = svd_v
                                shared_down_v[str(group_idx)]["original_expert"] = expert_idx
                            elif expert_selection_counts[i][expert_idx] > expert_selection_counts[i][shared_down_v[str(group_idx)]["original_expert"]]:
                                empty_group = next(g for g in range(8) if "share_down" not in shared_down_v[str(g)])
                                shared_down_v[str(empty_group)] = shared_down_v[str(group_idx)]
                                shared_down_v[str(group_idx)] = {"share_down": svd_v, "original_expert": expert_idx}                    
                                
            
            if 'Mixtral' in model_name:
                if "q_proj" in name:
                    svd_attn.q_u_proj.weight.data = svd_u
                    svd_attn.q_v_proj.weight.data = svd_v
                    
                elif "k_proj" in name:
                    svd_attn.k_u_proj.weight.data = svd_u
                    svd_attn.k_v_proj.weight.data = svd_v
                elif "v_proj" in name:
                    svd_attn.v_u_proj.weight.data = svd_u
                    svd_attn.v_v_proj.weight.data = svd_v
                elif "o_proj" in name:
                    svd_attn.o_u_proj.weight.data = svd_u
                    svd_attn.o_v_proj.weight.data = svd_v
                    layer.self_attn = svd_attn
                elif "gate_proj" in name:
                    svd_moe.gate.weight.data = subset[name].weight.data  # Keep gate unchanged
                elif "experts" in name:
                    expert_idx = int(name.split("experts.")[1].split('.')[0])
                    if "w1" in name:
                        svd_moe.experts[expert_idx].w1_u.weight.data = svd_u
                        if expert_idx == most_frequent_expert:
                            shared_w1_v = svd_v
                    elif "w2" in name:
                        svd_moe.experts[expert_idx].w2_u.weight.data = svd_u
                        if expert_idx == most_frequent_expert:
                            shared_w2_v = svd_v
                    elif "w3" in name:
                        svd_moe.experts[expert_idx].w3_u.weight.data = svd_u
                        if expert_idx == most_frequent_expert:
                            shared_w3_v = svd_v

            if 'phimoe' in model_name:
                if "q_proj" in name:
                    svd_attn.q_u_proj.weight.data = svd_u
                    svd_attn.q_v_proj.weight.data = svd_v
                    
                elif "k_proj" in name:
                    svd_attn.k_u_proj.weight.data = svd_u
                    svd_attn.k_v_proj.weight.data = svd_v
                elif "v_proj" in name:
                    svd_attn.v_u_proj.weight.data = svd_u
                    svd_attn.v_v_proj.weight.data = svd_v
                elif "o_proj" in name:
                    svd_attn.o_u_proj.weight.data = svd_u
                    svd_attn.o_v_proj.weight.data = svd_v
                    layer.self_attn = svd_attn
                elif "gate_proj" in name:
                    svd_moe.gate.weight.data = subset[name].weight.data  # Keep gate unchanged
                elif "experts" in name:
                    expert_idx = int(name.split("experts.")[1].split('.')[0])
                    if "w1" in name:
                        svd_moe.experts[expert_idx].w1_u.weight.data = svd_u
                        if expert_idx == most_frequent_expert:
                            shared_w1_v = svd_v
                    elif "w2" in name:
                        svd_moe.experts[expert_idx].w2_u.weight.data = svd_u
                        if expert_idx == most_frequent_expert:
                            shared_w2_v = svd_v
                    elif "w3" in name:
                        svd_moe.experts[expert_idx].w3_u.weight.data = svd_u
                        if expert_idx == most_frequent_expert:
                            shared_w3_v = svd_v

            
            W = W_scale = scaling_matrix_inv = scaling_diag_matrix = U = S = VT = truc_s = truc_u = truc_v = sqrtSigma = None
            del W, W_scale, scaling_matrix_inv, scaling_diag_matrix, U, S, VT, truc_s, truc_u, truc_v, sqrtSigma

        if 'Mixtral' in model_name and Attn_or_Experts != 'Attn':
            svd_moe.shared_w1_v.weight.data = shared_w1_v
            svd_moe.shared_w2_v.weight.data = shared_w2_v
            svd_moe.shared_w3_v.weight.data = shared_w3_v
            layer.block_sparse_moe = svd_moe
        if 'phimoe' in model_name and Attn_or_Experts != 'Attn':
            svd_moe.shared_w1_v.weight.data = shared_w1_v
            svd_moe.shared_w2_v.weight.data = shared_w2_v
            svd_moe.shared_w3_v.weight.data = shared_w3_v
            layer.block_sparse_moe = svd_moe
        

        if 'deepseek' in model_name and Attn_or_Experts != 'Attn':
            for expert_idx in range(64):
                group_idx = expert_idx // 8
                if "share_gate" in shared_gate_v[str(group_idx)]:
                    svd_moe.experts[expert_idx].shared_gate_v.weight.data = shared_gate_v[str(group_idx)]["share_gate"]
                if "share_up" in shared_up_v[str(group_idx)]:
                    svd_moe.experts[expert_idx].shared_up_v.weight.data = shared_up_v[str(group_idx)]["share_up"]
                if "share_down" in shared_down_v[str(group_idx)]:
                    svd_moe.experts[expert_idx].shared_down_v.weight.data = shared_down_v[str(group_idx)]["share_down"]
            layer.mlp = svd_moe


        del layer
        torch.cuda.empty_cache()
    
    for i in selected_layers:
        if 'Mixtral' in model_name:        
            if isinstance(layers[i].self_attn, SVD_MixtralAttention):
                print(f"Layer {i}: self_attn successfully updated to SVD version")
            else:
                print(f"Warning: Layer {i}: self_attn was not updated to SVD version")

            if isinstance(layers[i].block_sparse_moe, SVD_MixtralSparseMoeBlock):
                print(f"Layer {i}: block_sparse_moe successfully updated to SVD version")
            else:
                print(f"Warning: Layer {i}: block_sparse_moe was not updated to SVD version")
        elif 'deepseek' in model_name:
            if isinstance(layers[i].self_attn, SVD_DeepseekAttention):
                print(f"Layer {i}: self_attn successfully updated to SVD version")
            else:
                print(f"Warning: Layer {i}: self_attn was not updated to SVD version")
            if isinstance(layers[i].mlp, SVD_DeepseekMoE):
                print(f"Layer {i}: mlp successfully updated to SVD version")
            else:
                print(f"Warning: Layer {i}: mlp was not updated to SVD version")
        elif 'phimoe' in model_name:
            if isinstance(layers[i].self_attn, SVD_PhiMoEAttention):
                print(f"Layer {i}: self_attn successfully updated to SVD version")
            else:
                print(f"Warning: Layer {i}: self_attn was not updated to SVD version")
            if isinstance(layers[i].block_sparse_moe, SVD_PhiMoESparseMoeBlock):
                print(f"Layer {i}: block_sparse_moe successfully updated to SVD version")
            else:
                print(f"Warning: Layer {i}: block_sparse_moe was not updated to SVD version")


@torch.no_grad()
def whitening_local_update(model_name, model, dataloader, profiling_mat, expert_selection_counts, ratios, dev, direct_update=False, selected_layers=None, Attn_or_Experts='both' , outlier_or_frequency="frequency", attention_layers=[],  expert_layers=[]):
    print("Start SVD decomposition then update...")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    if isinstance(selected_layers, str) and selected_layers.lower() == 'all':
        selected_layers = list(range(len(layers)))

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(dataloader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask']
                if "opt" not in model_name:
                    cache['position_ids'] = kwargs['position_ids']
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask']), dim=0)
                if "opt" not in model_name:
                    cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids']), dim=0)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    if "opt" not in model_name:
        position_ids = cache['position_ids']

    for i in tqdm(selected_layers):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        gpts = {}
        Moe_List_Avg = []
        if "llama" in model_name or "vicuna" in model_name:
            svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, ratio=ratio)
        elif "Mistral" in model_name:
            svd_attn = SVD_MistralAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio)
        elif 'opt' in model_name:
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=ratio)
        elif 'Mixtral' in model_name:
            if isinstance(ratios, dict):
                temp_ratios_dict = ratios.deepcopy()
                w1,w2,w3=[],[],[]
                for k in list(temp_ratios_dict.keys()):
                    v = temp_ratios_dict[k]
                    if k.split(".")[2] != str(i):
                        del temp_ratios_dict[k]
                        continue
                    if k.split(".")[-1] == 'w1':
                        w1.append(v)
                    if k.split(".")[-1] == 'w2':
                        w2.append(v)
                    if k.split(".")[-1] == 'w3':
                        w3.append(v)
                Moe_List_Avg = [sum(w1)/len(w1),sum(w2)/len(w2),sum(w3)/len(w3)]
                svd_attn = SVD_MixtralAttention_dict(config=model.config,layer_idx=i, ratio=temp_ratios_dict)
                svd_moe = SVD_MixtralSparseMoeBlock_list(config=model.config, ratio=[sum(w1)/len(w1),sum(w2)/len(w2),sum(w3)/len(w3)])
            else:
                svd_attn = SVD_MixtralAttention(config=model.config,layer_idx=i, ratio=ratios)
                svd_moe = SVD_MixtralSparseMoeBlock(config=model.config, ratio=ratios)
        elif 'deepseek' in model_name:
            svd_attn = SVD_DeepseekAttention(config=model.config,layer_idx=i, ratio=ratios)
            svd_moe = SVD_DeepseekMoE(config=model.config, ratio=ratios)



        if i in expert_layers:
            if 'Mixtral' in model_name:
                if outlier_or_frequency == "outlier" and isinstance(ratios, dict):
                    i_outliers={}
                    for exp in range(8):
                        i_outliers[str(exp)] = 0
                    for k,v in ratios:
                        if k.split(".")[3]==i and k.split(".")[-1] in ["w1","w2","w3"]:
                            i_outliers[str(k.split(".")[-2])] += ratios[k]
                    most_frequent_expert = int(max(i_outliers, key=i_outliers.get))
                else:
                    most_frequent_expert = max(range(8), key=lambda x: expert_selection_counts[i][x])
            elif 'deepseek' in model_name:
                chunk_size = 8
                num_chunks = len(expert_selection_counts[i])//chunk_size
                if outlier_or_frequency == "outlier" and isinstance(ratios, dict):
                    i_outliers={}
                    for exp in range(64):
                        i_outliers[str(exp)] = 0
                    for k,v in ratios:
                        if k.split(".")[3]==i and k.split(".")[-1] in ["gate_proj","up_proj","down_proj"]:
                            i_outliers[str(k.split(".")[-2])] += ratios[k]
                    most_frequent_expert = heapq.nlargest(8, range(64), key=lambda idx: i_outliers.get(str(idx)))
                    most_frequent_expert.sort()
                    '''
                    most_frequent_expert = []
                    for n in range(num_chunks):
                        most_frequent_expert.append(max(range(n*chunk_size, (n+1)*chunk_size), key=lambda idx: i_outliers.get(str(idx))))
                    '''

                else:
                    most_frequent_expert = heapq.nlargest(8, range(64), key=lambda x, i=i: expert_selection_counts[i][x])
                    most_frequent_expert.sort()
                    '''
                    most_frequent_expert = []
                    for n in range(num_chunks):
                        most_frequent_expert.append(max(range(n*chunk_size, (n+1)*chunk_size), key=lambda idx,i=i: expert_selection_counts[i][idx]))
                    '''
            shared_w1_v = shared_w2_v = shared_w3_v = None
            if 'deepseek' in model_name:
                shared_gate_v = {str(i): {} for i in range(8)}
                shared_up_v = {str(i): {} for i in range(8)}
                shared_down_v = {str(i): {} for i in range(8)}
        
        for name in subset:

            if 'Mixtral' in model_name:
                if Attn_or_Experts == 'Attn' and 'experts' in name:
                    continue
                if Attn_or_Experts == 'Experts' and 'experts' not in name:
                    continue
                if 'experts' in name and i not in expert_layers:
                    continue
                if  'attn' in name and i not in attention_layers:
                    continue
                if  'gate' in name:  # Skip gate processing
                    continue

            if 'deepseek' in model_name:
                if Attn_or_Experts == 'Attn' and 'experts' in name:
                    continue
                if Attn_or_Experts == 'Experts' and 'experts' not in name:
                    continue
                if 'experts' in name and i not in expert_layers:
                    continue
                if  'attn' in name and i not in attention_layers:
                    continue
                if type(subset[name]).__name__ == 'MoEGate' or 'shared_experts' in name:
                    continue

            if isinstance(ratios, float):
                ratio = ratios
            elif 'moe' in name:
                if 'w1' in name:
                    ratio = Moe_List_Avg[0]
                elif 'w2' in name:
                    ratio = Moe_List_Avg[1]
                elif 'w3' in name:
                    ratio = Moe_List_Avg[2]
            else:
                ratio = ratios['model.layers.'+str(i)+"."+name]
            
            '''print(f"Processing layer {i}")
            print(f"Available keys in profiling_mat[{i}]: {list(profiling_mat[i].keys())}")
            print(f"Current layer name: {name}")'''
            
            if profiling_mat is not None:
                scaling_diag_matrix = profiling_mat[i][name].to(dev)
            else: 
                scaling_diag_matrix = None
            
            
            gpts[name] = local_update(subset[name], scaling_diag_matrix = scaling_diag_matrix, ratio=ratio, name=name, direct_update=direct_update)
        
        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch_update_u(inp[0].data, out.data)
            return tmp
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        if "opt" not in model_name:
            outs = layer(inps, attention_mask=attention_masks, position_ids=position_ids)[0]
        else:
            outs = layer(inps, attention_mask=attention_masks)[0]
        for h in handles:
            h.remove()
        for name in gpts:
            svd_u, svd_v = gpts[name].fasterprune()
            svd_u, svd_v = svd_u.to(dtype), svd_v.to(dtype)
            if 'Mixtral' in model_name:
                if "q_proj" in name:
                    svd_attn.q_u_proj.weight.data = svd_u
                    svd_attn.q_v_proj.weight.data = svd_v
                elif "k_proj" in name:
                    svd_attn.k_u_proj.weight.data = svd_u
                    svd_attn.k_v_proj.weight.data = svd_v
                elif "v_proj" in name:
                    svd_attn.v_u_proj.weight.data = svd_u
                    svd_attn.v_v_proj.weight.data = svd_v
                elif "o_proj" in name:
                    svd_attn.o_u_proj.weight.data = svd_u
                    svd_attn.o_v_proj.weight.data = svd_v
                    layer.self_attn = svd_attn
                elif "gate_proj" in name:
                    svd_moe.gate.weight.data = subset[name].weight.data  # Keep gate unchanged
                elif "experts" in name:
                    # print(f"sname: {name}")
                    # print(f"svd_u weight shape: {svd_u.shape}")
                    # print(f"svd_v weight shape: {svd_v.shape}")
                    expert_idx = int(name.split("experts.")[1].split('.')[0])
                    if "w1" in name:
                        svd_moe.experts[expert_idx].w1_u.weight.data = svd_u
                        if expert_idx == most_frequent_expert:
                            shared_w1_v = svd_v
                    elif "w2" in name:
                        svd_moe.experts[expert_idx].w2_u.weight.data = svd_u
                        if expert_idx == most_frequent_expert:
                            shared_w2_v = svd_v
                    elif "w3" in name:
                        svd_moe.experts[expert_idx].w3_u.weight.data = svd_u
                        if expert_idx == most_frequent_expert:
                            shared_w3_v = svd_v
                # if Attn_or_Experts != 'Attn':
                    # layer.block_sparse_moe = svd_moe
            elif 'deepseek' in model_name:
                if "q_proj" in name:
                    svd_attn.q_u_proj.weight.data = svd_u
                    svd_attn.q_v_proj.weight.data = svd_v
                elif "k_proj" in name:
                    svd_attn.k_u_proj.weight.data = svd_u
                    svd_attn.k_v_proj.weight.data = svd_v
                elif "v_proj" in name:
                    svd_attn.v_u_proj.weight.data = svd_u
                    svd_attn.v_v_proj.weight.data = svd_v
                elif "o_proj" in name:
                    svd_attn.o_u_proj.weight.data = svd_u
                    svd_attn.o_v_proj.weight.data = svd_v
                    layer.self_attn = svd_attn
                elif "experts" in name:
                    expert_idx = int(name.split("experts.")[1].split('.')[0])
                    if "gate" in name:
                        svd_moe.experts[expert_idx].gate_proj_u.weight.data = svd_u
                        if expert_idx in most_frequent_expert:
                            shared_gate_v[str(most_frequent_expert.index(expert_idx))]["share_gate"] = svd_v
                    elif "up" in name:
                        svd_moe.experts[expert_idx].up_proj_u.weight.data = svd_u
                        if expert_idx in most_frequent_expert:
                            shared_up_v[str(most_frequent_expert.index(expert_idx))]["share_up"] = svd_v
                    elif "down" in name:
                        svd_moe.experts[expert_idx].down_proj_u.weight.data = svd_u
                        if expert_idx in most_frequent_expert:
                            shared_down_v[str(most_frequent_expert.index(expert_idx))]["share_down"] = svd_v
                    '''
                    if "gate" in name:
                        svd_moe.experts[expert_idx].gate_proj_u.weight.data = svd_u
                        if expert_idx in most_frequent_expert:
                            shared_gate_v[str(expert_idx//8)]["share_gate"] = svd_v
                    elif "up" in name:
                        svd_moe.experts[expert_idx].up_proj_u.weight.data = svd_u
                        if expert_idx in most_frequent_expert:
                            shared_up_v[str(expert_idx//8)]["share_up"] = svd_v
                    elif "down" in name:
                        svd_moe.experts[expert_idx].down_proj_u.weight.data = svd_u
                        if expert_idx in most_frequent_expert:
                            shared_down_v[str(expert_idx//8)]["share_down"] = svd_v
                    '''
            
            elif 'opt' in model_name:
                if "q_proj" in name:
                    svd_decoder.self_attn.q_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.q_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.q_u_proj.bias.data = layer.self_attn.q_proj.bias.data
                elif "k_proj" in name:
                    svd_decoder.self_attn.k_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.k_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.k_u_proj.bias.data = layer.self_attn.k_proj.bias.data
                elif "v_proj" in name:
                    svd_decoder.self_attn.v_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.v_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.v_u_proj.bias.data = layer.self_attn.v_proj.bias.data
                elif "out_proj" in name:
                    svd_decoder.self_attn.out_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.out_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.out_u_proj.bias.data = layer.self_attn.out_proj.bias.data
                elif "fc1" in name:
                    svd_decoder.fc1_u_proj.weight.data = svd_u
                    svd_decoder.fc1_v_proj.weight.data = svd_v
                    svd_decoder.fc1_u_proj.bias.data = layer.fc1.bias.data
                elif "fc2" in name:
                    svd_decoder.fc2_u_proj.weight.data = svd_u
                    svd_decoder.fc2_v_proj.weight.data = svd_v
                    svd_decoder.fc2_u_proj.bias.data = layer.fc2.bias.data
                    svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
                    svd_decoder.final_layer_norm = layer.final_layer_norm
                    layers[i] = svd_decoder
            else:
                if "q_proj" in name:
                    svd_attn.q_u_proj.weight.data = svd_u
                    svd_attn.q_v_proj.weight.data = svd_v
                elif "k_proj" in name:
                    svd_attn.k_u_proj.weight.data = svd_u
                    svd_attn.k_v_proj.weight.data = svd_v
                elif "v_proj" in name:
                    svd_attn.v_u_proj.weight.data = svd_u
                    svd_attn.v_v_proj.weight.data = svd_v
                elif "o_proj" in name:
                    svd_attn.o_u_proj.weight.data = svd_u
                    svd_attn.o_v_proj.weight.data = svd_v
                    layer.self_attn = svd_attn
                elif "gate_proj" in name:
                    svd_mlp.gate_u_proj.weight.data = svd_u
                    svd_mlp.gate_v_proj.weight.data = svd_v
                elif "down_proj" in name:
                    svd_mlp.down_u_proj.weight.data = svd_u
                    svd_mlp.down_v_proj.weight.data = svd_v
                elif "up_proj" in name:
                    svd_mlp.up_u_proj.weight.data = svd_u
                    svd_mlp.up_v_proj.weight.data = svd_v
                    layer.mlp = svd_mlp


        if 'Mixtral' in model_name and Attn_or_Experts != 'Attn':
            svd_moe.shared_w1_v.weight.data = shared_w1_v
            svd_moe.shared_w2_v.weight.data = shared_w2_v
            svd_moe.shared_w3_v.weight.data = shared_w3_v
            layer.block_sparse_moe = svd_moe

        if 'deepseek' in model_name and Attn_or_Experts != 'Attn':
            for expert_idx in range(64):
                if expert_idx in most_frequent_expert:
                    continue
                svd_moe.experts[expert_idx].shared_gate_v.weight.data = shared_gate_v[str(expert_idx//8)]["share_gate"]
                svd_moe.experts[expert_idx].shared_up_v.weight.data = shared_up_v[str(expert_idx//8)]["share_up"]
                svd_moe.experts[expert_idx].shared_down_v.weight.data = shared_down_v[str(expert_idx//8)]["share_down"]
            layer.mlp = svd_moe

        layer = layer.to(dev)
        if "opt" not in model_name:
            outs = layer(inps, attention_mask=attention_masks, position_ids=position_ids)[0]
        else:
            outs = layer(inps, attention_mask=attention_masks)[0]
        layers[i] = layer.cpu()
        del gpts
        torch.cuda.empty_cache()
        inps = outs
        outs = None
        del outs
        torch.cuda.empty_cache()
    
    for i in selected_layers:
        if 'Mixtral' in model_name:        
            if isinstance(layers[i].self_attn, SVD_MixtralAttention):
                print(f"Layer {i}: self_attn successfully updated to SVD version")
            else:
                print(f"Warning: Layer {i}: self_attn was not updated to SVD version")

            if isinstance(layers[i].block_sparse_moe, SVD_MixtralSparseMoeBlock):
                print(f"Layer {i}: block_sparse_moe successfully updated to SVD version")
            else:
                print(f"Warning: Layer {i}: block_sparse_moe was not updated to SVD version")
        else:
            if isinstance(layers[i].self_attn, SVD_DeepseekAttention):
                print(f"Layer {i}: self_attn successfully updated to SVD version")
            else:
                print(f"Warning: Layer {i}: self_attn was not updated to SVD version")
            if isinstance(layers[i].mlp, SVD_DeepseekMoE):
                print(f"Layer {i}: self_attn successfully updated to SVD version")
            else:
                print(f"Warning: Layer {i}: self_attn was not updated to SVD version")

    model.config.use_cache = use_cache

def generate_experiment_name(args):
    layers = args.selected_layers if args.selected_layers is not None else "All"
    experiment_name = f"Layers: {layers} | "


    if args.layer_ratios_path:

        keep_rate = float(args.layer_ratios_path.split('keep_rate=')[1].split('%')[0])
        compression_ratio = 1 - keep_rate / 100
        experiment_name += f"Ratio: {compression_ratio:.4f} | "
    elif args.ratio:
        experiment_name += f"Ratio: {args.ratio} | "
    else:
        experiment_name += "Ratio: None | "
    
    experiment_name += f"Attn_or_Experts: {args.Attn_or_Experts} | "
    experiment_name += f"Model: {args.model} | "
    if args.step == 1:
        experiment_name += "Method: Whitening Only"
    elif args.step == 2:
        experiment_name += "Method: Whitening Then Update"
    elif args.step == 3:
        experiment_name += "Method: Update Only"
    elif args.step == 4:
        experiment_name += "Method: Evaluation"
    else:
        experiment_name += "Method: Unknown"
    return experiment_name


def create_boxed_string(*args):
    # Convert each input argument to a string and split by lines
    lines = [str(arg).split('\n') for arg in args]
    # Flatten all lines into a single list
    lines = [line for sublist in lines for line in sublist]
    # Calculate the maximum width of all lines
    width = max(len(line) for line in lines)
    box_top = "┌" + "─" * (width + 2) + "┐"
    box_bottom = "└" + "─" * (width + 2) + "┘"
    # Format each line, ensuring consistent width
    boxed_lines = [f"│ {line:<{width}} │" for line in lines]
    return "\n".join([box_top] + boxed_lines + [box_bottom])

def save_result_to_summary(result, experiment_name, results_summary_path):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    boxed_result = create_boxed_string(
        f"Timestamp: {timestamp}",
        f"Experiment: {experiment_name}",
        result
    )
    
    os.makedirs(os.path.dirname(results_summary_path), exist_ok=True)
    
    with open(results_summary_path, 'a') as f:
        f.write(f"{boxed_result}\n\n")
    print(f"Result appended to {results_summary_path}")

def generate_keep_configs(expert_selection_counts, methods=['adaptive', 'cumulative', 'entropy', 'top_n-1', 'top_n-2', 'top_n-3', 'top_n-4', 'top_n-5', 'top_n-6', 'top_n-7','top_n-8','top_n-9','top_n-10','top_n-11','top_n-12','top_n-24','top_n-48'], importance_threshold=0.05, cumulative_threshold=0.9, entropy_threshold=0.7):
    keep_configs = {}

    for method in methods:
        keep_config = {"layer_experts_idx": {}}

        for layer, expert_counts in expert_selection_counts.items():
            total_selections = sum(expert_counts)
            
            if method.startswith('top_n-'):
                k = int(method.split('-')[1])
                num_experts = len(expert_counts)
                sorted_experts = sorted(enumerate(expert_counts), key=lambda x: x[1], reverse=True)
                experts_to_keep = [expert for expert, _ in sorted_experts[:num_experts - k]]
            
            elif method == 'adaptive':
                experts_to_keep = [expert for expert, count in enumerate(expert_counts) 
                                   if (count / total_selections) >= importance_threshold]
            
            elif method == 'cumulative':
                sorted_experts = sorted(enumerate(expert_counts), key=lambda x: x[1], reverse=True)
                cumulative_importance = 0
                experts_to_keep = []
                for expert, count in sorted_experts:
                    importance = count / total_selections
                    if cumulative_importance + importance > cumulative_threshold:
                        break
                    cumulative_importance += importance
                    experts_to_keep.append(expert)
            
            elif method == 'entropy':
                probabilities = np.array(expert_counts) / total_selections
                entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
                max_entropy = np.log(len(expert_counts))
                normalized_entropy = entropy / max_entropy
                
                if normalized_entropy > entropy_threshold:
                    num_to_keep = int((1 - normalized_entropy) * len(expert_counts))
                else:
                    num_to_keep = len(expert_counts)
                
                sorted_experts = sorted(enumerate(expert_counts), key=lambda x: x[1], reverse=True)
                experts_to_keep = [expert for expert, _ in sorted_experts[:num_to_keep]]
            
            else:
                raise ValueError(f"Unknown method: {method}")

            keep_config["layer_experts_idx"][layer] = experts_to_keep

        keep_configs[method] = keep_config

    return keep_configs


def save_keep_configs(keep_configs, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for method, keep_config in keep_configs.items():
        save_path = os.path.join(save_dir, f'keep_config_{method}.json')
        with open(save_path, 'w') as f:
            json.dump(keep_config, f, indent=4)
    print(f"Keep configs saved to {save_dir}")

def move_model_to_gpu(model, no_split_module_classes=None):
    if no_split_module_classes is None:
        no_split_module_classes = ['MixtralDecoderLayer']

    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    usage_ratio = 0.5
    # Calculate available memory for each GPU
    gpu_memory = [torch.cuda.get_device_properties(i).total_memory for i in range(num_gpus)]
    
    # Initialize model with empty weights to get accurate memory requirements
    with init_empty_weights():
        empty_model = type(model)(model.config)
    
    # Infer device mapping considering all available GPUs
    device_map = infer_auto_device_map(
        empty_model,
        no_split_module_classes=no_split_module_classes,
        dtype=next(model.parameters()).dtype,
        max_memory={i: str(mem / 1024**3 * usage_ratio) + "GB" for i, mem in enumerate(gpu_memory)},
    )
    
    # Dispatch model using the inferred device mapping
    model = dispatch_model(model, device_map=device_map)
    
    '''print("Model distribution across GPUs:")
    for module, device in device_map.items():
        print(f"{module}: {device}")'''
    
    return model
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='jeffwan/llama-7b-hf', help='LLaMA model to load, pass `jeffwan/llama-7b-hf`')
    parser.add_argument('--model_path', type=str, default=None, help='local compressed model path or whitening information path')
    parser.add_argument('--layer_ratios_path', type=str, default=None, help='Target compression ratio,(0,1), default=0.2, means only keeping about 20% of the params.')
    parser.add_argument('--ratio', type=float, default=None, help='Target compression ratio,(0,1), default=0.2, means only keeping about 80% of the params.')
    parser.add_argument('--run_low_resource', action='store_true', help='whether to run whitening in low resource, exp, compress LLaMA-7B below 15G gpu')
    parser.add_argument('--dataset', type=str, default='wikitext2',help='Where to extract calibration data from [wikitext2, ptb, c4]')
    parser.add_argument('--whitening_nsamples', type=int, default=256, help='Number of calibration data samples for whitening.')
    parser.add_argument('--outlier_or_frequency', type=str, default='frequency')
    parser.add_argument('--updating_nsamples', type=int, default=16, help='Number of calibration data samples for udpating.')
    parser.add_argument('--save_path', type=str, default=None, help='the path to save the compressed model checkpoints.`')
    parser.add_argument('--profiling_mat_path', type=str, default=None, help='Local path to load the profiling matrices`')
    parser.add_argument('--seed',type=int, default=0, help='Seed for sampling the calibration data')
    parser.add_argument('--DEV', type=str, default="cuda", help='device')
    parser.add_argument('--model_seq_len', type=int, default=2048, help='the default sequence length of the LLM')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='inference bactch size')
    parser.add_argument('--gen_seq_len', type=int, default=1024, help='generated sequence len for efficiency evaluation')
    parser.add_argument('--step', type=int, default=4, help='the step to run the compression')
    parser.add_argument('--lora', type=str, default=None, help='the lora updated weight path to run the accuracy evaluation')
    parser.add_argument('--selected_layers', type=str, default=[], help='Comma-separated list of layer indices to compress')
    parser.add_argument('--Attn_or_Experts', type=str, default='both', choices=['Attn', 'Experts', 'both'], help='Which parts to compress: Attention, Experts, or both')
    parser.add_argument('--test_mode', type=str, default='custom', choices=['custom', 'huggingface'], help='Which parts to compress: Attention, Experts, or both')
    parser.add_argument('--evaluate_after_compression', action='store_true', help='Evaluate the model immediately after compression')
    parser.add_argument('--params_only', action='store_true', help='Only calculate and display parameter statistics without evaluating perplexity')
    parser.add_argument('--drop_methods', nargs='+', default=['adaptive', 'cumulative', 'entropy'], 
                        help='Methods to determine experts to drop')
    parser.add_argument('--importance_threshold', type=float, default=0.1, 
                        help='Importance threshold for adaptive method')
    parser.add_argument('--cumulative_threshold', type=float, default=0.8, 
                        help='Cumulative importance threshold')
    parser.add_argument('--entropy_threshold', type=float, default=0.5, 
                        help='Entropy threshold')
    parser.add_argument('--attention_layers', type=str, default=[], help='Comma-separated list of attention layers')
    parser.add_argument('--expert_layers', type=str, default=[], help='Comma-separated list of expert layers')
    parser.add_argument('--group', type=str, default=None, help='Experiment group identifier (e.g., "group_A")')

    args = parser.parse_args()
    # Handle selected_layers

    assert not (args.layer_ratios_path and args.ratio), "layer_ratios_path and ratio can't be input at the same time"
    
    if args.layer_ratios_path is not None:
        with open(args.layer_ratios_path, 'r', encoding='utf-8') as file:
            layer_ratios = json.load(file)
        # Perform 1 - value operation on all values in layer_ratios
        layer_ratios = {k: 1 - v if isinstance(v, (int, float)) else 
                        {inner_k: 1 - inner_v for inner_k, inner_v in v.items()} if isinstance(v, dict) else v
                        for k, v in layer_ratios.items()}
    # Convert expert_layers to a list of integers
    if args.selected_layers and isinstance(args.selected_layers, str):
        args.selected_layers = [int(layer) for layer in args.selected_layers.split(',')]
    # Convert expert_layers to a list of integers
    if args.expert_layers and isinstance(args.expert_layers, str):
        args.expert_layers = [int(layer) for layer in args.expert_layers.split(',')]
    
    # Convert attention_layers to a list of integers
    if args.attention_layers and isinstance(args.attention_layers, str):
        args.attention_layers = [int(layer) for layer in args.attention_layers.split(',')]

    experiment_name = generate_experiment_name(args)
    if args.ratio:
        args.ratio = 1- args.ratio

    if args.step == 4:
        results_summary_path = os.path.join(args.save_path, 'baseline_results.txt') if args.save_path else 'baseline_results.txt'
    elif args.ratio is None and args.layer_ratios_path is not None:
        if args.outlier_or_frequency == 'outlier':
            results_summary_path = os.path.join(args.save_path, 'results_summary_owl_outlier.txt') if args.save_path else 'results_summary_owl_outlier.txt'
        else:
            results_summary_path = os.path.join(args.save_path, 'results_summary_owl_frequency.txt') if args.save_path else 'results_summary_owl_frequency.txt'
    else:
        if args.params_only:
            results_summary_path = os.path.join(args.save_path, 'results_summary_params_only.txt') if args.save_path else 'results_summary_params_only.txt'
        else:
            results_summary_path = os.path.join(args.save_path, 'results_summary_uniform_test.txt') if args.save_path else 'results_summary_uniform_test.txt'

    print(f"selected layers: {args.selected_layers}")

    if args.step == 1:
        model, tokenizer = get_model_from_huggingface(model_id=args.model)
        model = model.eval()
        if args.profiling_mat_path is None:
            cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
            
            start_time = time.time()
            profiling_mat, expert_selection_counts = profle_svdllm_low_resource(args.model, model, cali_white_data, args.DEV, args.selected_layers, args.Attn_or_Experts, attention_layers=args.attention_layers,  expert_layers=args.expert_layers)
            end_time = time.time()
            profiling_time = end_time - start_time
            print(f"Profiling took {profiling_time:.2f} seconds")
            if args.save_path is not None:
                torch.save((profiling_mat, expert_selection_counts), args.save_path + "/" + '_'.join(map(str, args.selected_layers)) + args.Attn_or_Experts + args.model.replace("/", "_").replace("-", "_") + '_profiling_' + args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt')
                profiling_file = args.save_path + "/" + '_'.join(map(str, args.selected_layers)) + args.Attn_or_Experts + args.model.replace("/", "_").replace("-", "_") + '_profiling_' + args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt'
        else:
            profiling_mat, expert_selection_counts = torch.load(args.profiling_mat_path)
        if args.ratio:
            start_time = time.time()
            whitening(args.model, model, profiling_mat, expert_selection_counts, args.ratio, args.DEV, args.selected_layers, args.Attn_or_Experts, args.outlier_or_frequency, attention_layers=args.attention_layers,  expert_layers=args.expert_layers)
            end_time = time.time()
            whitening_time = end_time - start_time
            print(f"Whitening took {whitening_time:.2f} seconds")
            
            model_path_save=args.save_path + "/"+'attn' +'_'.join(map(str, args.attention_layers))+'expert'+ '_'.join(map(str, args.expert_layers)) + args.model.replace("/", "_").replace("-", "_") +'_whitening_only_'+str(args.ratio) + '_' + args.group + '.pt'
            print(f"Saved model: {model_path_save}", flush=True)
        else:
            whitening(args.model, model, profiling_mat, expert_selection_counts, layer_ratios, args.DEV, args.selected_layers, args.Attn_or_Experts, args.outlier_or_frequency, attention_layers=args.attention_layers,  expert_layers=args.expert_layers)
            model_path_save=args.save_path + "/"+'attn' +'_'.join(map(str, args.attention_layers))+'expert'+ '_'.join(map(str, args.expert_layers)) + args.model.replace("/", "_").replace("-", "_") +'_whitening_only' +'_' + args.group +'.pt'
            print(f"Saved model: {model_path_save}", flush=True)
        # Generate and save drop configs
        num_layers = len(model.model.layers)

        
        keep_configs = generate_keep_configs(
            expert_selection_counts,
            importance_threshold=args.importance_threshold,
            cumulative_threshold=args.cumulative_threshold,
            entropy_threshold=args.entropy_threshold
        )
        # Determine save path
        if args.ratio:
            keep_config_dir = os.path.join(args.save_path,'results','expert_drop', 'attn' +'_'.join(map(str, args.attention_layers))+'expert'+ '_'.join(map(str, args.expert_layers)) + args.model.replace("/", "_").replace("-", "_") +'_whitening_only_'+str(args.ratio)+'_' + args.group,'checkpoint')
        else:
            keep_config_dir = os.path.join(args.save_path,'results','expert_drop', 'attn' +'_'.join(map(str, args.attention_layers))+'expert'+ '_'.join(map(str, args.expert_layers)) + args.model.replace("/", "_").replace("-", "_") +'_whitening_only'+'_' + args.group,'checkpoint')
        # Create a directory with the same name as the model file (without .pt extension)
        os.makedirs(keep_config_dir, exist_ok=True)
        # Save expert_selection_counts
        if 'Mixtral' in args.model:
            expert_counts_path = os.path.join(keep_config_dir, 'expert_selection_counts.json')
            with open(expert_counts_path, 'w') as f:
                json.dump(expert_selection_counts, f, indent=4)
            print(f"Expert selection counts saved to {expert_counts_path}")
    
        # Save keep configs
        save_keep_configs(keep_configs, keep_config_dir)
    
        if args.save_path is not None:
            print("Model architecture:")
            print(model)
            torch.save({'model': model, 'tokenizer': tokenizer}, model_path_save)
        
        if args.evaluate_after_compression:
            model, tokenizer = get_model_from_local_gpu(model_path_save, args.model, args.test_mode)
            result = ppl_eval_sharing(model, tokenizer, experiment_name, datasets=['wikitext2'], model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, params_only=args.params_only)
            print(result)
            save_result_to_summary(result, experiment_name, results_summary_path)
        
        if os.path.exists(profiling_file):
            os.remove(profiling_file)
            print(f"Deleted profiling file: {profiling_file}")
        
        '''if os.path.exists(model_path_save):
            os.remove(model_path_save)
            print(f"Deleted model file: {model_path_save}")'''


    elif args.step == 2:
        model, tokenizer = get_model_from_huggingface(model_id=args.model)
        dataloader, _ = get_loaders(args.dataset, nsamples=args.updating_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
        model = model.eval()
        model = model.float()
        if args.profiling_mat_path is None:
            cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
            profiling_mat, expert_selection_counts = profle_svdllm_low_resource(args.model, model, cali_white_data, args.DEV, args.selected_layers, args.Attn_or_Experts, attention_layers=args.attention_layers,  expert_layers=args.expert_layers)
            if args.save_path is not None:
                torch.save((profiling_mat, expert_selection_counts), args.save_path + "/" + '_'.join(map(str, args.selected_layers)) + args.Attn_or_Experts+args.model.replace("/", "_").replace("-", "_") + '_profiling_'+ args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt')
                profiling_file = args.save_path + "/" + '_'.join(map(str, args.selected_layers)) + args.Attn_or_Experts+args.model.replace("/", "_").replace("-", "_") + '_profiling_'+ args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt'
        else:
            profiling_mat, expert_selection_counts = torch.load(args.profiling_mat_path)
        if args.ratio:
            whitening_local_update(args.model, model, dataloader, profiling_mat, expert_selection_counts, args.ratio, args.DEV, selected_layers=args.selected_layers, Attn_or_Experts=args.Attn_or_Experts, attention_layers=args.attention_layers,  expert_layers=args.expert_layers)
            model_path_save=args.save_path + "/" +'attn'+'_'.join(map(str, args.attention_layers))+'expert'+ '_'.join(map(str, args.expert_layers))+ args.model.replace("/", "_").replace("-", "_") +'_whitening_then_update_' + str(args.ratio) +'_' + args.group+ '.pt'
            print(f"Saved model: {model_path_save}", flush=True)
        else:
            whitening_local_update(args.model, model, dataloader, profiling_mat, expert_selection_counts, layer_ratios, args.DEV, selected_layers=args.selected_layers, Attn_or_Experts=args.Attn_or_Experts,attention_layers=args.attention_layers,  expert_layers=args.expert_layers)
            model_path_save=args.save_path + "/" +'attn'+'_'.join(map(str, args.attention_layers))+'expert'+ '_'.join(map(str, args.expert_layers))+ args.model.replace("/", "_").replace("-", "_") +'_whitening_then_update' +'_' + args.group+ '.pt'
            print(f"Saved model: {model_path_save}", flush=True)
        if args.save_path is not None:
            torch.save({'model': model, 'tokenizer': tokenizer}, model_path_save)

        # Generate and save drop configs
        num_layers = len(model.model.layers)
        keep_configs = generate_keep_configs(
            expert_selection_counts, 
            importance_threshold=args.importance_threshold,
            cumulative_threshold=args.cumulative_threshold,
            entropy_threshold=args.entropy_threshold
        )
        # Determine save path
        if args.ratio:
            keep_config_dir = os.path.join(args.save_path,'results','expert_drop', 'attn' +'_'.join(map(str, args.attention_layers))+'expert'+ '_'.join(map(str, args.expert_layers)) + args.model.replace("/", "_").replace("-", "_") +'_whitening_only_'+str(args.ratio)+'_' + args.group,'checkpoint')
        else:
            keep_config_dir = os.path.join(args.save_path,'results','expert_drop', 'attn' +'_'.join(map(str, args.attention_layers))+'expert'+ '_'.join(map(str, args.expert_layers)) + args.model.replace("/", "_").replace("-", "_") +'_whitening_only'+'_' + args.group,'checkpoint')
        # Create a directory with the same name as the model file (without .pt extension)
        os.makedirs(keep_config_dir, exist_ok=True)
        # Save expert_selection_counts
        if 'Mixtral' in args.model:
            expert_counts_path = os.path.join(keep_config_dir, 'expert_selection_counts.json')
            with open(expert_counts_path, 'w') as f:
                json.dump(expert_selection_counts, f, indent=4)
            print(f"Expert selection counts saved to {expert_counts_path}")        
        
        
        
        # Save keep configs
        save_keep_configs(keep_configs, keep_config_dir)

        if args.evaluate_after_compression:
            model, tokenizer = get_model_from_local_gpu(model_path_save, args.model, args.test_mode)
            result = ppl_eval_sharing(model, tokenizer, experiment_name, datasets=['wikitext2'], model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, params_only=args.params_only)
            print(result)
            save_result_to_summary(result, experiment_name, results_summary_path)
        
        if os.path.exists(profiling_file):
            os.remove(profiling_file)
            print(f"Deleted profiling file: {profiling_file}")
        
        '''if os.path.exists(model_path_save):
            os.remove(model_path_save)
            print(f"Deleted model file: {model_path_save}")'''

    elif args.step == 3:
        model, tokenizer = get_model_from_huggingface(args.model)
        model = model.eval()
        model = model.float()
        dataloader, _ = get_loaders(args.dataset, nsamples=args.updating_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
        # Add this line to get expert_selection_counts
        _, expert_selection_counts = profle_svdllm_low_resource(args.model, model, dataloader, args.DEV, args.selected_layers, args.Attn_or_Experts)
        whitening_local_update(model_name=args.model, model=model, dataloader=dataloader, profiling_mat=None, expert_selection_counts=expert_selection_counts, ratio=args.ratio, dev=args.DEV, direct_update=True, selected_layers=args.selected_layers, Attn_or_Experts=args.Attn_or_Experts)
        model_path_save=args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_update_only_' + str(args.ratio) + '.pt'
        if args.save_path is not None:
            torch.save({'model': model, 'tokenizer': tokenizer, 'expert_selection_counts': expert_selection_counts},model_path_save )
        
        if args.evaluate_after_compression:
            model, tokenizer, expert_selection_counts = get_model_from_local_gpu(model_path_save, args.model, args.test_mode)
            result = ppl_eval_sharing(model, tokenizer, experiment_name, datasets=['wikitext2'], model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, expert_selection_counts=expert_selection_counts)
            print(result)
            save_result_to_summary(result, experiment_name, results_summary_path)

    elif args.step >= 4:
        print(f"evaluating {args.model_path}...")
        if args.model_path == "original":
            model, tokenizer = get_model_from_huggingface_gpu(args.model)
            expert_selection_counts = None
        else:
            model, tokenizer, expert_selection_counts = get_model_from_local_gpu(args.model_path, args.model, args.test_mode)
            if args.lora is not None:
                from utils.peft import PeftModel
                model = PeftModel.from_pretrained(
                    model,
                    args.lora,
                    torch_dtype=torch.float16,
                )
                model = model.merge_and_unload()
        model = model.float()
        if args.step == 4:

            result = ppl_eval_sharing(model, tokenizer, experiment_name, datasets=['wikitext2'], model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size)
            print(result)
            save_result_to_summary(result, experiment_name, results_summary_path)
            
        elif args.step == 5:
            eff_eval(model, tokenizer, generated_len=args.gen_seq_len, batch_size=args.eval_batch_size, device=args.DEV)