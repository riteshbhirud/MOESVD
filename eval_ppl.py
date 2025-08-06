# Import necessary libraries
import torch
import cloudpickle
import argparse
import os
import sys
import datetime
from accelerate import load_checkpoint_and_dispatch
sys.path.append("/workspace/MOE-SVD")
sys.path.append("/workspace/MOE-SVD/component/deepseek")
sys.path.append('/root/.cache/huggingface/modules/')
from evaluater import ppl_eval_sharing

def get_model_from_local_gpu(model_id, model_name, mode='custom'):
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if mode == 'custom':
        # Load custom model
        
        pruned_dict = torch.load(model_id, map_location='cpu')
        
        tokenizer = pruned_dict['tokenizer']
        model = cloudpickle.loads(pruned_dict['model'])
        # Use accelerate's load_checkpoint_and_dispatch to load and distribute the model
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
        # Load Huggingface model and tokenizer
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        from transformers import LlamaTokenizer, AutoTokenizer
        if "opt" in model_id or "mistral" in model_id or "Mixtral" in model_id:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        elif 'deepseek' in model_id:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        else:
            tokenizer = LlamaTokenizer.from_pretrained(model_id, trust_remote_code=True)
        # Initialize empty model
        # with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(model_id,trust_remote_code=True)
        print(model_id)
        # Use accelerate's load_checkpoint_and_dispatch to load and distribute the model
        model = load_checkpoint_and_dispatch(
            model=model,
            checkpoint=model_id,
            device_map='auto',
            no_split_module_classes=['MixtralDecoderLayer','SVD_DeepseekMoE','DeepseekMoE','PhiMoEDecoderLayer']
        )

        return model, tokenizer

    else:
        raise ValueError("Invalid mode. Choose either 'custom' or 'huggingface'.")
    


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str, required=True, help='Path to save evaluation results')
parser.add_argument('--model_path', type=str, required=True, help='Path to load the model')
parser.add_argument('--load_mode', type=str, required=True, help='custom or huggingface')
args = parser.parse_args()


# Load the model
model, tokenizer = get_model_from_local_gpu(args.model_path,args.model_path,mode=args.load_mode)

# Get model name
model_name = os.path.basename(args.model_path).replace('.pt', '')
experiment_name = f"dense_model_{model_name}"

# Evaluate the model
evaluation_result = ppl_eval_sharing(model, tokenizer, experiment_name)

# Save evaluation results
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(os.path.join(args.output_path, "evaluation_result.txt"), "a") as f:
    f.write(f"\n\n--- New Evaluation: {timestamp} ---\n")
    f.write(f"Experiment: {experiment_name}\n\n")
    f.write(evaluation_result)

# Now you can use model and tokenizer