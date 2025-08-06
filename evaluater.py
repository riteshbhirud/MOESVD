import torch
import numpy as np
from tqdm import tqdm
import time
import itertools
from utils.data_utils import get_test_data,get_test_dataset,get_test_loader
import os
import sys
import pdb
import inspect
from datasets import load_dataset
from component.svd_mixtral_sharing import SVD_MixtralSparseMoeBlock, SVD_MixtralAttention
# from component.svd_mixtral_share_old import SVD_MixtralSparseMoeBlock, SVD_MixtralAttention
from torch.utils.data import DataLoader


from accelerate import Accelerator

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)


def print_memory_usage():
    total_gpus = torch.cuda.device_count()
    total_allocated = 0
    total_reserved = 0
    
    for i in range(total_gpus):
        allocated = torch.cuda.memory_allocated(device=i) / 1024 / 1024
        reserved = torch.cuda.memory_reserved(device=i) / 1024 / 1024
        total_allocated += allocated
        total_reserved += reserved
        # print(f"GPU {i} - Allocated: {allocated:.2f} MiB, Reserved: {reserved:.2f} MiB")
    
    # print(f"Total - Allocated: {total_allocated:.2f} MiB, Reserved: {total_reserved:.2f} MiB")
    
    return total_allocated, total_reserved
    
def print_in_box(*args):
    # Find the longest string to determine box width
    max_length = max(len(str(arg)) for arg in args)
    box_width = max_length + 4  # Adding padding for the box

    print("┌" + "─" * box_width + "┐")
    for arg in args:
        print(f"│ {str(arg):<{max_length}} │")
    print("└" + "─" * box_width + "┘")    
    
@torch.no_grad()
def ppl_eval(model, tokenizer, experiment_name, datasets=['wikitext2', 'ptb', 'c4'], model_seq_len=2048, batch_size=32):
    model.eval()
    ppls = {}
    total_allocated_list = []
    total_reserved_list = []

    for dataset in datasets:
        test_loader = get_test_data(dataset, tokenizer, seq_len=model_seq_len, batch_size=batch_size)
        nlls = []

        for batch in tqdm(test_loader):
            try:
                # Move input batch to the first device of the model
                input_ids = batch.to(next(model.parameters()).device)
                allocated, reserved = print_memory_usage()
                total_allocated_list.append(allocated)
                total_reserved_list.append(reserved)
                output = model(input_ids)
                lm_logits = output.logits if hasattr(output, "logits") else output[0]

                if torch.isfinite(lm_logits).all():
                    shift_logits = lm_logits[:, :-1, :].contiguous()
                    shift_labels = input_ids[:, 1:].contiguous().to(shift_logits.device)

                    assert shift_logits.device == shift_labels.device, "shift_logits and shift_labels are on different devices"

                    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    nlls.append(loss.to("cpu"))
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"Out of memory error for batch size {batch.size(0)}. Skipping batch.")
                    torch.cuda.empty_cache()
                else:
                    print(f"RuntimeError: {str(e)}")
                    raise e

        if nlls:
            ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
            ppls[dataset] = ppl
    avg_allocated = sum(total_allocated_list) / len(total_allocated_list)
    avg_reserved = sum(total_reserved_list) / len(total_reserved_list)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params_B = trainable_params / 1e9
    result_str = f"Experiment: {experiment_name}\n"
    result_str += f"PPL after pruning: {ppls}\n"
    result_str += f"Average Allocated Memory: {avg_allocated:.2f} MiB\n"
    result_str += f"Average Reserved Memory: {avg_reserved:.2f} MiB\n"
    result_str += f"Total number of trainable parameters: {trainable_params_B:.2f}B\n"
    
    print_in_box(
        f"Experiment: {experiment_name}",
        f"PPL after pruning: {ppls}",
        f"Average Allocated Memory: {avg_allocated:.2f} MiB",
        f"Average Reserved Memory: {avg_reserved:.2f} MiB",
        f"Total number of trainable parameters: {trainable_params_B:.2f}B"
    )
    
    return result_str


@torch.no_grad()
def ppl_eval_share_old(model, tokenizer, experiment_name, datasets=['wikitext2', 'ptb', 'c4'], model_seq_len=2048, batch_size=32, expert_selection_counts=None):
    model.eval()
    ppls = {}
    total_allocated_list = []
    total_reserved_list = []

    for dataset in datasets:
        test_loader = get_test_data(dataset, tokenizer, seq_len=model_seq_len, batch_size=batch_size)
        nlls = []

        for batch in tqdm(test_loader):
            try:
                # Move input batch to the first device of the model
                input_ids = batch.to(next(model.parameters()).device)
                allocated, reserved = print_memory_usage()
                total_allocated_list.append(allocated)
                total_reserved_list.append(reserved)

                # Check if the model has SVD compressed Mixtral features
                if hasattr(model, 'model') and hasattr(model.model, 'layers') and expert_selection_counts is not None:
                    for layer_idx, layer in enumerate(model.model.layers):
                        if hasattr(layer, 'block_sparse_moe') and isinstance(layer.block_sparse_moe, SVD_MixtralSparseMoeBlock):
                            moe_block = layer.block_sparse_moe
                            if expert_selection_counts.get(layer_idx):
                                most_frequent_expert = max(range(len(moe_block.experts)), key=lambda x: expert_selection_counts[layer_idx][x])
                                shared_vt = moe_block.experts[most_frequent_expert].w2_v.weight.data
                                for expert in moe_block.experts:
                                    expert.w2_v.weight.data = shared_vt

                output = model(input_ids)
                lm_logits = output.logits if hasattr(output, "logits") else output[0]

                if torch.isfinite(lm_logits).all():
                    shift_logits = lm_logits[:, :-1, :].contiguous()
                    shift_labels = input_ids[:, 1:].contiguous().to(shift_logits.device)

                    assert shift_logits.device == shift_labels.device, "shift_logits and shift_labels are on different devices"

                    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    nlls.append(loss.to("cpu"))
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"Out of memory error for batch size {batch.size(0)}. Skipping batch.")
                    torch.cuda.empty_cache()
                else:
                    print(f"RuntimeError: {str(e)}")
                    raise e

        if nlls:
            ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
            ppls[dataset] = ppl

    avg_allocated = sum(total_allocated_list) / len(total_allocated_list)
    avg_reserved = sum(total_reserved_list) / len(total_reserved_list)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params_B = trainable_params / 1e9

    result_str = f"Experiment: {experiment_name}\n"
    result_str += f"PPL after pruning: {ppls}\n"
    result_str += f"Average Allocated Memory: {avg_allocated:.2f} MiB\n"
    result_str += f"Average Reserved Memory: {avg_reserved:.2f} MiB\n"
    result_str += f"Total number of trainable parameters: {trainable_params_B:.2f}B\n"
    
    return result_str

def count_parameters(model):
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
    return total_params, trainable_params


@torch.no_grad()
def ppl_eval_sharing(model, tokenizer, experiment_name, datasets=['wikitext2', 'ptb', 'c4'], model_seq_len=2048, batch_size=16, params_only=False):
    def _perplexity(nlls, n_samples, seqlen):
        return torch.exp(torch.stack(nlls).sum() / (n_samples * seqlen))

    model.eval()
    ppls = {}
    total_allocated_list = []
    total_reserved_list = []

    # Get the main device of the model
    main_device = next(model.parameters()).device
    if not params_only:
        for dataset in datasets:
            '''if dataset == 'wikitext2':
                # Use the same data loading method as the new code
                data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
                data = tokenizer("\n\n".join(data["text"]), return_tensors="pt")
                data = data.input_ids.to(main_device)'''
            # For other datasets, use the original loading method
            data = get_test_data(dataset, tokenizer, seq_len=model_seq_len, batch_size=batch_size)
            # data = next(iter(data)).to(main_device)  # Assuming get_test_data returns a DataLoader

            seqlen = model_seq_len
            n_samples = len(data)
            nlls = []

            with tqdm(range(n_samples), desc=f"Evaluating {dataset} - Perplexity") as progress_bar:
                for i in progress_bar:
                    batch = next(iter(data)).to(main_device)

                    allocated, reserved = print_memory_usage()
                    total_allocated_list.append(allocated)
                    total_reserved_list.append(reserved)

                    with torch.no_grad():
                        output = model(batch)
                        logits = output.logits if hasattr(output, "logits") else output[0]

                    # Ensure logits are on the correct device
                    logits = logits.to(main_device)
                    shift_logits = logits[:, :-1, :].contiguous().float()
                    shift_labels = batch[:, 1:].contiguous()

                    loss_fct = torch.nn.CrossEntropyLoss()
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    neg_log_likelihood = loss.float() * seqlen
                    nlls.append(neg_log_likelihood)

                    curr_ppl = _perplexity(nlls, i + 1, seqlen)
                    progress_bar.set_description(f"Evaluating {dataset} - Perplexity {curr_ppl:.3f}")

            ppl = _perplexity(nlls, n_samples, seqlen)
            ppls[dataset] = ppl.item()

    # Calculate parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    # Check for SVD compressed Mixtral features
    svd_layers = sum(1 for m in model.modules() if isinstance(m, SVD_MixtralSparseMoeBlock))
    result_str = f"Experiment: {experiment_name}\n"
    if not params_only:
        avg_allocated = sum(total_allocated_list) / len(total_allocated_list)
        avg_reserved = sum(total_reserved_list) / len(total_reserved_list)
        result_str += f"PPL after evaluation: {ppls}\n"
        result_str += f"Average Allocated Memory: {avg_allocated:.2f} MiB\n"
        result_str += f"Average Reserved Memory: {avg_reserved:.2f} MiB\n"
    
    result_str += f"Total number of parameters: {total_params / 1e9:.2f}B\n"
    result_str += f"Number of trainable parameters: {trainable_params / 1e9:.2f}B\n"
    result_str += f"Number of non-trainable parameters: {non_trainable_params / 1e9:.2f}B\n"
    result_str += f"Number of SVD compressed Mixtral layers: {svd_layers}\n"

    print(result_str)
    return result_str





@torch.no_grad()
def ppl_eval_gpu(model, tokenizer, datasets=['wikitext2'], model_seq_len=2048, batch_size=8):
    accelerator = Accelerator()
    
    ppls = {}
    for dataset_name in datasets:
        test_dataset = get_test_dataset(dataset_name, tokenizer, seq_len=model_seq_len)
        test_loader = get_test_loader(test_dataset, batch_size, accelerator)
        
        nlls = []
        for batch in test_loader:
            # Ensure input data is on the correct device
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            nlls.append(accelerator.gather(loss))
        
        ppl = torch.exp(torch.cat(nlls).mean()).item()
        ppls[dataset_name] = ppl
        
        del test_dataset, test_loader, nlls
        torch.cuda.empty_cache()
    
    if accelerator.is_main_process:
        print("PPL after evaluation: {}".format(ppls))
        print("Max GPU Memory Usage: {} MiB".format(torch.cuda.max_memory_allocated() / 1024 / 1024))
    
    return ppls


@torch.no_grad()
def eff_eval(model, tokenizer, dataset='wikitext2', original_len=4, generated_len=128, batch_size=1, device="cuda"):
    model.eval()
    throughput = 0
    token_num = 0
    end_memory = 0
    num_batches_to_fetch = 10
    test_loader = get_test_data(dataset, tokenizer, seq_len=original_len, batch_size = batch_size)
    weight_memory = torch.cuda.memory_allocated()
    for batch_idx, batch_data in enumerate(itertools.islice(test_loader, num_batches_to_fetch)):
        batch = batch_data.to(device)
        token_num += batch.shape[0] * generated_len
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize()
        start_time = time.time()
        generation_output = model.generate(
                input_ids=batch,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                use_cache=True,
                top_k=50,
                max_length=original_len+generated_len,
                top_p=0.95,
                temperature=1,
        )
        torch.cuda.synchronize()
        end_time = time.time()
        end_memory = max(torch.cuda.max_memory_allocated(0), end_memory)
        if torch.isfinite(generation_output[0]).all():  # check if the generation is successful since fp16 may cause nan
            throughput += end_time - start_time
            print("time: {}".format(end_time - start_time))
    print("Total Memory: {} GB".format(end_memory/(1024 ** 3)))
    print("Weight Memory: {} GB".format(weight_memory/(1024 ** 3)))
    print("Activation Memory: {} GB".format((end_memory - start_memory)/(1024 ** 3)))
    print("Throughput: {} tokens/sec".format(token_num / throughput))

