#coding:utf8
import os
import sys
import torch
import torch.nn as nn
import gc
from accelerate import Accelerator, infer_auto_device_map
from accelerate.utils import set_module_tensor_to_device
from transformers.models.mixtral.modeling_mixtral import *

from component.deepseek.modeling_deepseek import MoEGate, DeepseekMoE
from component.phimoe.modeling_phimoe import PhiMoESparseMoeBlock
import pdb

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

# bandaid fix
dev = torch.device("cuda")

'''def get_model_from_huggingface(model_id):
    from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer, LlamaForCausalLM
    if "opt" in model_id or "mistral" in model_id or "Mixtral" in model_id:
        tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="cpu", trust_remote_code=True)
    elif 'deepseek' in model_id:
        tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="cpu")
    else:
        tokenizer = LlamaTokenizer.from_pretrained(model_id, device_map="cpu", trust_remote_code=True)
    if 'deepseek' in model_id:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float32, local_files_only=True, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float32, trust_remote_code=True, cache_dir=None)
    model.seqlen = 2048
    return model, tokenizer'''
def get_model_from_huggingface(model_id):
    from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer
    if "opt" in model_id or "mistral" in model_id or "Mixtral" in model_id:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    elif 'deepseek' in model_id:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Enable quantization for 'mistralai/Mixtral-8x22B-v0.1'
    if model_id == 'mistralai/Mixtral-8x22B-v0.1':
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cpu",
            load_in_8bit=True,  # Enable 8-bit quantization
            trust_remote_code=True
        )
    elif 'deepseek' in model_id:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cpu",
            torch_dtype=torch.float32,
            local_files_only=True,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cpu",
            torch_dtype=torch.float32,
            trust_remote_code=True
        )

    model.seqlen = 2048
    return model, tokenizer
def get_model_from_huggingface_gpu(model_id):
    from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer
    
    if "opt" in model_id or "mistral" in model_id or "Mixtral" in model_id or 'deepseek' in model_id:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto",
        torch_dtype=torch.float32, 
        trust_remote_code=True, 
        cache_dir=None
    )
    model.seqlen = 2048
    return model, tokenizer


'''def get_model_from_huggingface_gpu(model_id):
    from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer, LlamaForCausalLM
    if "opt" in model_id or "mistral" in model_id or "Mixtral" in model_id:
        tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto", trust_remote_code=True)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(model_id, device_map="auto", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float32, trust_remote_code=True, cache_dir=None)
    model.seqlen = 2048
    return model, tokenizer'''

def get_model_from_local(model_id):
    pruned_dict = torch.load(model_id, map_location='cpu')
    tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
    return model, tokenizer

def load_custom_model(model_path):
    accelerator = Accelerator()
    
    # 加载整个字典
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # 获取模型和tokenizer
    model = checkpoint['model']
    tokenizer = checkpoint['tokenizer']
    
    # 推断设备映射
    device_map = infer_auto_device_map(model)
    
    # 使用 to 方法将模型分布到多个 GPU 上
    model = model.to(device_map)
    
    return accelerator, model, tokenizer


def get_model_from_local_gpu(model_id, model_name, mode='custom'):
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if mode == 'custom':
        # 加载自定义模型
        pruned_dict = torch.load(model_id, map_location='cpu')
        tokenizer = pruned_dict['tokenizer']
        model = pruned_dict['model']
        # 使用 accelerate 的 load_checkpoint_and_dispatch 加载和分配模型
        model = load_checkpoint_and_dispatch(
            model=model,
            checkpoint=model_id,
            device_map='auto',
            no_split_module_classes=['MixtralDecoderLayer','DeepseekDecoderLayer','PhiMoEDecoderLayer']
        )
        # del pruned_dict
        # gc.collect()
        return model, tokenizer

    elif mode == 'huggingface':
        # 加载 Huggingface 模型和 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 初始化空模型
        # with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print(model_id)
        # 使用 accelerate 的 load_checkpoint_and_dispatch 加载和分配模型
        model = load_checkpoint_and_dispatch(
            model=model,
            checkpoint=model_id,
            device_map='auto',
            no_split_module_classes=['MixtralDecoderLayer','DeepseekDecoderLayer','PhiMoEDecoderLayer']
        )

        return model, tokenizer

    else:
        raise ValueError("Invalid mode. Choose either 'custom' or 'huggingface'.")


'''def get_model_from_local_gpu(model_id):
    from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer, LlamaForCausalLM
    from accelerate import infer_auto_device_map, dispatch_model
    # Load the pruned model dictionary from the local file
    pruned_dict = torch.load(model_id, map_location='cpu')
    
    # Extract the tokenizer and model state_dict from the dictionary
    tokenizer_state_dict, model_state_dict = pruned_dict['tokenizer'], pruned_dict['model']
    
    # Initialize the tokenizer and model
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_state_dict)
    # model = AutoModelForCausalLM.from_pretrained(model_state_dict)
    
    # Generate device map automatically
    device_map = infer_auto_device_map(model_state_dict)
    
    # Dispatch the model to the appropriate devices based on the device map
    model = dispatch_model(model_state_dict, device_map)
    model=model.half()
    tokenizer=tokenizer_state_dict
    
    return model, tokenizer'''

'''def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res'''

'''def find_layers(module, layers=[nn.Conv2d, nn.Linear, MixtralSparseMoeBlock], name=''):
    res = {}
    if type(module) in layers:
        res[name] = module
        # 如果是 MixtralSparseMoeBlock，继续递归进入其内部
        if isinstance(module, MixtralSparseMoeBlock):
            for name1, child in module.named_children():
                res.update(find_layers(
                    child, layers=[nn.Conv2d, nn.Linear], name=name + '.' + name1 if name != '' else name1
                ))
    else:
        for name1, child in module.named_children():
            res.update(find_layers(
                child, layers=layers, name=name + '.' + name1 if name != '' else name1
            ))
    return res'''

def find_layers(module, layers=[nn.Conv2d, nn.Linear, MixtralSparseMoeBlock, MoEGate,PhiMoESparseMoeBlock], name='', process_moe_block=False):
    res = {}

    # 1. 特别处理 MixtralSparseMoeBlock 模块
    if isinstance(module, MixtralSparseMoeBlock) or type(module).__name__ == 'MoEGate' or type(module).__name__ == 'PhiMoESparseMoeBlock':
        # pdb.set_trace()
        if process_moe_block:
            # 如果要处理 MixtralSparseMoeBlock，将其自身加入结果
            res[name] = module
            # 并且递归处理其子模块
            for name1, child in module.named_children():
                res.update(find_layers(
                    child, layers=layers, name=name + '.' + name1 if name != '' else name1, process_moe_block=process_moe_block
                ))
        else:
            # 如果不处理 MixtralSparseMoeBlock，只递归处理其子模块
            for name1, child in module.named_children():
                res.update(find_layers(
                    child, layers=layers, name=name + '.' + name1 if name != '' else name1, process_moe_block=False
                ))
        return res  # 处理完 MixtralSparseMoeBlock 后直接返回，避免重复处理

    # 2. 判断当前模块是否属于其他指定的层类型 (排除 MixtralSparseMoeBlock)
    elif type(module) in layers or 'gate' in name:
        res[name] = module

    # 3. 递归处理其他非指定类型的模块
    else:
        for name1, child in module.named_children():
            res.update(find_layers(
                child, layers=layers, name=name + '.' + name1 if name != '' else name1, process_moe_block=process_moe_block
            ))

    return res



def find_linear_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res