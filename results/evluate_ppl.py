# Import necessary libraries
import torch
import cloudpickle
import argparse
import os
import sys
import datetime
from accelerate import load_checkpoint_and_dispatch

# Add necessary paths to sys.path
sys.path.append("/workspace/MOE-SVD")
# sys.path.append("/workspace/MOE-SVD/component/deepseek")
sys.path.append('/root/.cache/huggingface/modules/')

# Import ppl_eval_sharing function from evaluater
from evaluater import ppl_eval_sharing

def load_model_and_tokenizer(load_path,test_mode='custom'):
    if test_mode=='custom':
        # Load the saved dictionary
        saved_dict = torch.load(load_path)
        # Deserialize the model from bytes
        model = cloudpickle.loads(saved_dict['model'])
        # Get the tokenizer
        tokenizer = saved_dict['tokenizer']
        # Load and dispatch the model using accelerate
        model = load_checkpoint_and_dispatch(
            model=model,
            checkpoint=load_path,
            device_map='auto',
            no_split_module_classes=['MixtralDecoderLayer','DeepseekDecoderLayer','PhiMoEDecoderLayer']
        )
        return model, tokenizer
    else:
        # Import necessary classes from transformers
        from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer
        # Choose the appropriate tokenizer based on the model
        if "opt" in load_path or "mistral" in load_path or "Mixtral" in load_path:
            tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
        elif 'deepseek' in load_path:
            tokenizer = AutoTokenizer.from_pretrained(load_path)
        else:
            tokenizer = LlamaTokenizer.from_pretrained(load_path, trust_remote_code=True)

        # Enable quantization for 'mistralai/Mixtral-8x22B-v0.1'
        if load_path == 'mistralai/Mixtral-8x22B-v0.1':
            model = AutoModelForCausalLM.from_pretrained(
                load_path,
                device_map="cpu",
                load_in_8bit=True,  # Enable 8-bit quantization
                trust_remote_code=True
            )
        elif 'deepseek' in load_path:
            model = AutoModelForCausalLM.from_pretrained(
                load_path,
                device_map="cpu",
                torch_dtype=torch.float32,
                local_files_only=True,
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                load_path,
                device_map="cpu",
                torch_dtype=torch.float32,
                trust_remote_code=True
            )

        model.seqlen = 2048
        return model, tokenizer

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--compressed_model_save_path', type=str, required=True)
parser.add_argument('--config_path', type=str, required=True)
parser.add_argument('--model_name',type=str,required=True)
args = parser.parse_args()

# Example usage
load_path = os.path.join(args.compressed_model_save_path, args.model_name + args.config_path.replace('.json', '')+ ".pt")

# Load the model and tokenizer
model, tokenizer = load_model_and_tokenizer(load_path)
experiment_name = f"drop_{os.path.basename(args.config_path)}"
evaluation_result = ppl_eval_sharing(model, tokenizer, experiment_name)

# Open file in append mode and write evaluation results
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(os.path.join(args.compressed_model_save_path, "evaluation_result.txt"), "a") as f:
    f.write(f"\n\n--- New Evaluation: {timestamp} ---\n")
    f.write(f"Experiment: {experiment_name}\n\n")
    f.write(evaluation_result)

# Delete .pt model file (commented out)
'''if os.path.exists(load_path):
    os.remove(load_path)
    print(f"Deleted model file: {load_path}")
else:
    print(f"Model file not found: {load_path}")'''

# Now you can use model and tokenizer