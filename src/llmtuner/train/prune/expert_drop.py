import logging
import os
import sys
import io
import warnings
from argparse import Namespace
import types
import torch
import time
from accelerate import hooks
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
sys.path.append('/workspace/MOE-SVD')
from evaluater import ppl_eval_sharing
import pickle
import dill
import cloudpickle
from llmtuner.model.mixtral.modeling_mixtral import MixtralSparseMoeBlock, MixtralPreTrainedModel
from transformers import MixtralForCausalLM
from .io import create_dir, save_json
from .utils import print_gpu_memory, prepare_calibration_input, find_modules
from .wrapper import MixtralExpertDropWrapper, DeepseekExpertDropWrapper
from ...model.deepseek.modeling_deepseek import DeepseekPreTrainedModel, MoEGate
import pdb
import sys
sys.path.append('/root/.cache/huggingface/modules/')
from transformers_modules.phimoe.modeling_phimoe import PhiMoEForCausalLM
from transformers_modules.deepseek.modeling_deepseek import DeepseekForCausalLM
logger = logging.getLogger(__name__)


def fill_missing_values_for_non_moe_layers(values: list, moe_layer_indices: list, num_layers: int):
    # Fill missing values for non-MoE layers
    filled_values = []

    for i in range(num_layers):
        if i not in moe_layer_indices:
            filled_values.append(None)
        else:
            filled_values.append(values[moe_layer_indices.index(i)])

    return filled_values


@torch.no_grad()
def layerwise_pruning(args: Namespace, model, dataloader: DataLoader, accelerator: Accelerator, num_samples: int):
    # Perform layer-wise pruning
    # num_samples: samples on each device, calculated as "num_samples = n_compression_samples // num_processes"
    device = accelerator.device
    unwrapped_model = accelerator.unwrap_model(model)  # Unwrap model first
    use_cache = unwrapped_model.config.use_cache
    unwrapped_model.config.use_cache = False
    layers = unwrapped_model.model.layers

    if args.score_save_file is not None:
        routing_scores = []

    # Get MoE layer ids
    if isinstance(unwrapped_model, MixtralPreTrainedModel):
        num_experts = unwrapped_model.config.num_local_experts
        num_layers = unwrapped_model.config.num_hidden_layers
        moe_layer_indices = list(range(num_layers))
    elif isinstance(unwrapped_model, DeepseekPreTrainedModel):
        num_experts = unwrapped_model.config.n_routed_experts
        num_layers = unwrapped_model.config.num_hidden_layers
        moe_layer_indices = [layer_idx for layer_idx in range(num_layers) if (unwrapped_model.config.n_routed_experts is not None and layer_idx >= unwrapped_model.config.first_k_dense_replace and layer_idx % unwrapped_model.config.moe_layer_freq == 0)]
    else:
        raise NotImplementedError

    accelerator.print("moe_layer_indices", moe_layer_indices)

    # Get valid MoE layer ids
    if isinstance(num_experts, list):
        valid_moe_layer_indices = [i for i in moe_layer_indices if num_experts[i] >= 0]
    else:
        valid_moe_layer_indices = moe_layer_indices

    accelerator.print("Getting features...")
    inputs, outputs, attention_mask, position_ids = prepare_calibration_input(unwrapped_model, dataloader, num_samples)

    accelerator.print('Starting ...')
    update_num_experts_list = []
    update_experts_idx = []

    for i in tqdm(range(num_layers), desc="Dropping layers...", disable=not accelerator.is_main_process):
        sys.stderr.flush()
        torch.cuda.empty_cache()
        print_gpu_memory(accelerator)
        layer = layers[i]

        if i in valid_moe_layer_indices:
            this_layer_num_experts = num_experts[i] if isinstance(num_experts, list) else num_experts

            if this_layer_num_experts > args.r:
                if args.r > 0:
                    # Find modules
                    if isinstance(unwrapped_model, MixtralPreTrainedModel):
                        subset = find_modules(layer, [MixtralSparseMoeBlock])
                    elif isinstance(unwrapped_model, DeepseekPreTrainedModel):
                        subset = find_modules(layer, [MoEGate])
                    else:
                        raise NotImplementedError

                    # Wrap layers
                    wrapped_layers = {}
                    for name in subset:
                        if isinstance(unwrapped_model, MixtralPreTrainedModel):
                            wrapped_layers[name] = MixtralExpertDropWrapper(subset[name])
                        elif isinstance(unwrapped_model, DeepseekPreTrainedModel):
                            wrapped_layers[name] = DeepseekExpertDropWrapper(subset[name])
                        else:
                            raise NotImplementedError

                    # Forward hook for recording metrics
                    def add_batch(name):
                        def mixtral_hook(_, input, output):
                            wrapped_layers[name].add_batch(input[0].data, output[1].data)  # output[1] is router_logits (before softmax)

                        def deepseek_hook(_, input, output):
                            wrapped_layers[name].add_batch(input[0].data, output[0].data, output[1].data)  # output[0] is topk ids, output[1] is topk scores (after softmax)

                        if isinstance(unwrapped_model, MixtralPreTrainedModel):
                            return mixtral_hook
                        elif isinstance(unwrapped_model, DeepseekPreTrainedModel):
                            return deepseek_hook
                        else:
                            raise NotImplementedError

                    # Get importance
                    handles = []
                    for name in wrapped_layers:
                        handles.append(subset[name].register_forward_hook(add_batch(name)))
                    for j in range(num_samples):
                        outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]
                    for h in handles:
                        h.remove()

                    # Expert Drop
                    for name in subset:  # should be only one element in subset
                        module_state_dict_name = f"model.layers.{i}.{name}"
                        accelerator.print(f"Dropping for {module_state_dict_name}")

                        # Sort total scores
                        # [IMPORTANT] all reduce across devices
                        scores = wrapped_layers[name].scores
                        scores = accelerator.reduce(scores, reduction="sum")  # Here we use "sum" as the number of tokens processed by each device may be different.
                        accelerator.print(f"layer {i} scores: {scores}")

                        _, experts_to_drop = torch.topk(scores, this_layer_num_experts - args.r, largest=args.reverse_drop)
                        accelerator.print("largest:", args.reverse_drop, bool(args.reverse_drop))
                        experts_to_drop = experts_to_drop.tolist()
                        accelerator.print(f"layer {i} experts_to_drop: {experts_to_drop}")
                        experts_to_preserve = sorted(list(set(range(this_layer_num_experts)) - set(experts_to_drop)))
                        update_num_experts_list.append(len(experts_to_preserve))
                        update_experts_idx.append(experts_to_preserve)

                        if args.score_save_file is not None:
                            routing_scores.append(scores.float().cpu())

                else:  # No expert left
                    update_num_experts_list.append(0)  # This denotes that this layer has no MoE, but has Norm
                    update_experts_idx.append([])

            else:  # Do not drop as the remaining experts have already satisfied the requirement
                update_num_experts_list.append(this_layer_num_experts)
                update_experts_idx.append(list(range(this_layer_num_experts)))

        else:
            update_num_experts_list.append(-1)  # This denotes that this layer has no MoE & Norm
            update_experts_idx.append(None)
            for j in range(num_samples):
                outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]

        # Update inputs & outputs
        inputs, outputs = outputs, inputs

    # Fill in the missing values for non-MoE layers
    update_num_experts_list = fill_missing_values_for_non_moe_layers(update_num_experts_list, moe_layer_indices, num_layers)
    update_experts_idx = fill_missing_values_for_non_moe_layers(update_experts_idx, moe_layer_indices, num_layers)
    accelerator.print("update_num_experts_list", update_num_experts_list)
    accelerator.print("update_experts_idx", update_experts_idx)

    # Update the config
    accelerator.print("Expert dropping done!")
    unwrapped_model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    if isinstance(unwrapped_model, MixtralPreTrainedModel):
        accelerator.print("Updating model config...")
        setattr(unwrapped_model.config, "num_local_experts", update_num_experts_list)
        setattr(unwrapped_model.config, "layer_experts_idx", update_experts_idx)
    elif isinstance(unwrapped_model, DeepseekPreTrainedModel):
        setattr(unwrapped_model.config, "n_routed_experts", update_num_experts_list)
        setattr(unwrapped_model.config, "layer_experts_idx", update_experts_idx)
    else:
        raise NotImplementedError

    # Save routing scores
    if args.score_save_file is not None:
        if isinstance(num_experts, list):
            warnings.warn("Recording routing scores with list type \"num_experts\" is not supported!")
        else:
            if accelerator.is_main_process:
                routing_scores = torch.stack(routing_scores, dim=0)
                create_dir(os.path.dirname(args.score_save_file))
                torch.save(routing_scores, args.score_save_file)
            accelerator.wait_for_everyone()


@torch.no_grad()
def global_pruning(args: Namespace, model, dataloader: DataLoader, accelerator: Accelerator, num_samples: int):
    # Perform global pruning
    unwrapped_model = accelerator.unwrap_model(model)  # Unwrap model first
    use_cache = unwrapped_model.config.use_cache
    unwrapped_model.config.use_cache = False
    layers = unwrapped_model.model.layers

    if args.score_save_file is not None:
        warnings.warn("Recording routing scores is only supported under \"layer-wise\" mode!")

    # Get MoE layer ids
    if isinstance(unwrapped_model, MixtralPreTrainedModel):
        num_experts = unwrapped_model.config.num_local_experts
        num_layers = unwrapped_model.config.num_hidden_layers
        moe_layer_indices = list(range(num_layers))
    elif isinstance(unwrapped_model, DeepseekPreTrainedModel):
        num_experts = unwrapped_model.config.n_routed_experts
        num_layers = unwrapped_model.config.num_hidden_layers
        moe_layer_indices = [layer_idx for layer_idx in range(num_layers) if (unwrapped_model.config.n_routed_experts is not None and layer_idx >= unwrapped_model.config.first_k_dense_replace and layer_idx % unwrapped_model.config.moe_layer_freq == 0)]
    else:
        raise NotImplementedError

    accelerator.print("moe_layer_indices", moe_layer_indices)

    # Get valid MoE layer ids
    if isinstance(num_experts, list):
        valid_moe_layer_indices = [i for i in moe_layer_indices if num_experts[i] >= 0]
    else:
        valid_moe_layer_indices = moe_layer_indices

    accelerator.print("Getting features...")
    inputs, outputs, attention_mask, position_ids = prepare_calibration_input(unwrapped_model, dataloader, num_samples)

    accelerator.print('Starting ...')
    global_scores = []

    for i in tqdm(range(num_layers), desc="Gathering scores...", disable=not accelerator.is_main_process):
        sys.stderr.flush()
        torch.cuda.empty_cache()
        print_gpu_memory(accelerator)
        layer = layers[i]

        if i in valid_moe_layer_indices:
            # Find modules
            if isinstance(unwrapped_model, MixtralPreTrainedModel):
                subset = find_modules(layer, [MixtralSparseMoeBlock])
            elif isinstance(unwrapped_model, DeepseekPreTrainedModel):
                subset = find_modules(layer, [MoEGate])
            else:
                raise NotImplementedError

            # Wrap layers
            wrapped_layers = {}
            for name in subset:
                if isinstance(unwrapped_model, MixtralPreTrainedModel):
                    wrapped_layers[name] = MixtralExpertDropWrapper(subset[name])
                elif isinstance(unwrapped_model, DeepseekPreTrainedModel):
                    wrapped_layers[name] = DeepseekExpertDropWrapper(subset[name])
                else:
                    raise NotImplementedError

            # Forward hook for recording metrics
            def add_batch(name):
                def mixtral_hook(_, input, output):
                    wrapped_layers[name].add_batch(input[0].data, output[1].data)  # output[1] is router_logits (before softmax)

                def deepseek_hook(_, input, output):
                    wrapped_layers[name].add_batch(input[0].data, output[0].data, output[1].data)  # output[0] is topk ids, output[1] is topk scores (after softmax)

                if isinstance(unwrapped_model, MixtralPreTrainedModel):
                    return mixtral_hook
                elif isinstance(unwrapped_model, DeepseekPreTrainedModel):
                    return deepseek_hook
                else:
                    raise NotImplementedError

            # Get importance
            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(num_samples):
                outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]
            for h in handles:
                h.remove()

            # Expert Drop
            for name in subset:
                module_state_dict_name = f"model.layers.{i}.{name}"
                accelerator.print(f"Dropping for {module_state_dict_name}")

                # Sort total scores
                # [IMPORTANT] all reduce across devices
                scores = wrapped_layers[name].scores
                scores = accelerator.reduce(scores, reduction="sum")  # Here we use "sum" as the number of tokens processed by each device may be different.
                global_scores.append(scores)
                accelerator.print(f"layer {i} scores: {scores}")

        else:
            for j in range(num_samples):
                outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]

        # Update inputs & outputs
        inputs, outputs = outputs, inputs

    print_gpu_memory(accelerator)
    torch.cuda.empty_cache()

    # Get number of experts to drop
    if isinstance(num_experts, list):
        total_num_experts = sum([n for n in num_experts if n is not None])
    else:
        total_num_experts = num_experts * len(moe_layer_indices)

    avg_experts_per_moe_layer = total_num_experts / len(valid_moe_layer_indices)
    num_experts_to_drop = round((avg_experts_per_moe_layer - args.r) * len(valid_moe_layer_indices))

    if num_experts_to_drop > 0:
        if num_experts_to_drop < total_num_experts:  # Not all experts are dropped
            # Cat scores
            global_scores = torch.cat(global_scores, dim=0)  # Gather the scores.
            accelerator.print(f"global_scores: {global_scores}")

            _, experts_to_drop = torch.topk(global_scores, num_experts_to_drop, largest=args.reverse_drop)
            accelerator.print("largest:", args.reverse_drop, bool(args.reverse_drop))
            experts_to_drop = sorted(experts_to_drop.tolist())
            accelerator.print(f"experts_to_drop: {experts_to_drop}")

            # Expert Drop
            update_num_experts_list = []
            update_experts_idx = []

            if isinstance(num_experts, list):
                for layer_id in tqdm(moe_layer_indices, desc='Dropping Experts...'):
                    if layer_id in valid_moe_layer_indices:
                        position_begin_id = sum([n for n in num_experts[:layer_id] if n is not None])
                        position_end_id = sum([n for n in num_experts[:(layer_id + 1)] if n is not None])
                        experts_to_preserve = sorted(list(set(range(position_begin_id, position_end_id)) - set(experts_to_drop)))
                        experts_to_preserve = [i - position_begin_id for i in experts_to_preserve]
                        accelerator.print(f"layer {layer_id} experts_to_preserve: {experts_to_preserve}")

                        update_num_experts_list.append(len(experts_to_preserve))
                        update_experts_idx.append(experts_to_preserve)
                    else:
                        update_num_experts_list.append(-1)  # This denotes that this layer has no MoE & Norm
                        update_experts_idx.append(None)

            else:
                for position_id, layer_id in tqdm(enumerate(moe_layer_indices), desc='Dropping Experts...'):
                    # position_id: position of the element in the list
                    position_begin_id = num_experts * position_id
                    position_end_id = num_experts * (position_id + 1)
                    experts_to_preserve = sorted(list(set(range(position_begin_id, position_end_id)) - set(experts_to_drop)))
                    experts_to_preserve = [i - position_begin_id for i in experts_to_preserve]
                    accelerator.print(f"layer {layer_id} experts_to_preserve: {experts_to_preserve}")

                    update_num_experts_list.append(len(experts_to_preserve))
                    update_experts_idx.append(experts_to_preserve)

        else:  # No expert left
            if isinstance(num_experts, list):
                update_num_experts_list = []
                update_experts_idx = []

                for layer_id in tqdm(moe_layer_indices, desc='Dropping Experts...'):
                    if layer_id in valid_moe_layer_indices:
                        update_num_experts_list.append(0)  # This denotes that this layer has no MoE, but has Norm
                        update_experts_idx.append([])
                    else:
                        update_num_experts_list.append(-1)  # This denotes that this layer has no MoE & Norm
                        update_experts_idx.append(None)
            else:
                update_num_experts_list = [0 for i in range(len(moe_layer_indices))]  # All layers has no MoE, but has Norm
                update_experts_idx = [[] for i in range(len(moe_layer_indices))]

    else:  # Do not drop as the remaining number of experts has already satisfied the requirement
        update_num_experts_list = num_experts
        update_experts_idx = [list(range(num_experts[i] if isinstance(num_experts, list) else num_experts)) for i in range(len(moe_layer_indices))]

    # Fill in the missing values for non-MoE layers
    update_num_experts_list = fill_missing_values_for_non_moe_layers(update_num_experts_list, moe_layer_indices, num_layers)
    update_experts_idx = fill_missing_values_for_non_moe_layers(update_experts_idx, moe_layer_indices, num_layers)
    accelerator.print("update_num_experts_list", update_num_experts_list)
    accelerator.print("update_experts_idx", update_experts_idx)

    # Update the config
    accelerator.print("Expert dropping done!")
    unwrapped_model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    if isinstance(unwrapped_model, MixtralPreTrainedModel):
        accelerator.print("Updating model config...")
        setattr(unwrapped_model.config, "num_local_experts", update_num_experts_list)
        setattr(unwrapped_model.config, "layer_experts_idx", update_experts_idx)
    elif isinstance(unwrapped_model, DeepseekPreTrainedModel):
        setattr(unwrapped_model.config, "n_routed_experts", update_num_experts_list)
        setattr(unwrapped_model.config, "layer_experts_idx", update_experts_idx)
    else:
        raise NotImplementedError


@torch.no_grad()
def post_experts_drop(compressed_model_save_path,config_path, model, tokenizer, config, accelerator: Accelerator, preserve_gate=False):
    # Post-processing after expert dropping
    unwrapped_model = accelerator.unwrap_model(model)  # Unwrap model first
    layers = unwrapped_model.model.layers
    layer_experts_idx = config["layer_experts_idx"]
    gate_num_experts = []

    if accelerator.is_main_process:
        # Modify weights for experts & gates
        start_time = time.time()

        for layer_id, layer in tqdm(list(enumerate(layers)), desc='Dropping Experts...'):
            if str(layer_id) not in layer_experts_idx:
                continue  # If the layer is not in the configuration, skip it
            experts_to_preserve = layer_experts_idx[str(layer_id)]
            accelerator.print(f"layer {layer_id} experts_to_preserve: {experts_to_preserve}")

            if experts_to_preserve is not None:  # This layer is MoE
                r = len(experts_to_preserve)

                if isinstance(unwrapped_model, MixtralForCausalLM) or isinstance(unwrapped_model,PhiMoEForCausalLM ):
                    if r > 0:
                        # Drop experts
                        layer.block_sparse_moe.experts = nn.ModuleList([layer.block_sparse_moe.experts[i] for i in experts_to_preserve])

                        # Rewrite gate
                        this_layer_num_experts = layer.block_sparse_moe.gate.out_features
                        experts_to_drop = sorted(list(set(range(this_layer_num_experts)) - set(experts_to_preserve)))
                        if r==1:
                            layer.block_sparse_moe.top_k=1

                        if not preserve_gate:  # Remove gate weights for dropped experts
                            new_gate_weight = layer.block_sparse_moe.gate.weight.data[experts_to_preserve]
                            layer.block_sparse_moe.gate = nn.Linear(in_features=layer.block_sparse_moe.gate.in_features, out_features=r, bias=False, device=layer.block_sparse_moe.gate.weight.device, dtype=layer.block_sparse_moe.gate.weight.dtype)
                            layer.block_sparse_moe.gate.weight.data = new_gate_weight
                            layer.block_sparse_moe.num_experts = r
                        else:  # Re-order gate weights for all experts, the dropped weights are preserved
                            new_gate_weight = layer.block_sparse_moe.gate.weight.data[experts_to_preserve + experts_to_drop]
                            layer.block_sparse_moe.gate.weight.data = new_gate_weight
                            layer.block_sparse_moe.num_experts = r

                        gate_num_experts.append(this_layer_num_experts)

                    else:
                        layer.block_sparse_moe = None
                        gate_num_experts.append(None)

                elif isinstance(unwrapped_model, DeepseekForCausalLM):
                    if r > 0:
                        # Drop experts
                        layer.mlp.experts = nn.ModuleList([layer.mlp.experts[i] for i in experts_to_preserve])

                        # Rewrite gate
                        this_layer_num_experts = layer.mlp.gate.weight.data.shape[0]
                        experts_to_drop = sorted(list(set(range(this_layer_num_experts)) - set(experts_to_preserve)))

                        if not preserve_gate:  # Remove gate weights for dropped experts
                            new_gate_weight = layer.mlp.gate.weight.data[experts_to_preserve]
                            layer.mlp.gate.weight.data = new_gate_weight
                        else:  # Re-order gate weights for all experts, the dropped weights are preserved
                            new_gate_weight = layer.mlp.gate.weight.data[experts_to_preserve + experts_to_drop]
                            layer.mlp.gate.weight.data = new_gate_weight

                        gate_num_experts.append(this_layer_num_experts)

                    else:
                        layer.mlp = None
                        gate_num_experts.append(None)

                else:
                    raise NotImplementedError

            else:  # This layer is not MoE
                gate_num_experts.append(None)

        end_time = time.time()
        execution_time = end_time - start_time
        accelerator.print(f"Execution time for dropping experts: {execution_time:.2f} seconds")


        # Set gate configs
        if preserve_gate:
            accelerator.print(f"Preserve dropped gate weights the model.")
            accelerator.print("gate_num_experts", gate_num_experts)
            config["gate_num_experts"] = gate_num_experts
        else:
            accelerator.print(f"Do not preserve dropped gate weights for the model.")

        # Save
        # unwrapped_model.save_pretrained(compressed_model_save_path)
        # tokenizer.save_pretrained(compressed_model_save_path)

        hooks.remove_hook_from_module(unwrapped_model, recurse=True)
        # unwrapped_model = unwrapped_model.cpu()
        unwrapped_model.to('cpu')
        accelerator.print("Preparing to serialize model...")
        
        # Use cloudpickle to serialize the model
        model_bytes = cloudpickle.dumps(unwrapped_model)
        
        accelerator.print(f"Model serialization completed")
        accelerator.print(f"Serialized model size: {len(model_bytes) / (1024 * 1024):.2f} MB")
        
        # Replace the unwrapped_model with its serialized version
        unwrapped_model = model_bytes

        # Save the entire model (including backup of dynamic methods and tokenizer)
        torch.save({
            'model': unwrapped_model,  # Save the entire model object
            'tokenizer': tokenizer  # Save tokenizer
        }, os.path.join(compressed_model_save_path, compressed_model_save_path.split('/')[-2] + config_path.replace('.json', '')+ ".pt"), pickle_protocol=pickle.HIGHEST_PROTOCOL)

        save_json(config, os.path.join(compressed_model_save_path, "config.json"), indent=2)


    accelerator.wait_for_everyone()
    accelerator.print(f"model: {model}")
