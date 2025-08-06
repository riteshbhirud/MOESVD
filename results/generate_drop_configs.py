import json
import argparse
import os
import numpy as np

def generate_keep_configs(expert_selection_counts, methods=['adaptive', 'cumulative', 'entropy', 'top_n-1', 'top_n-2', 'top_n-3', 'top_n-4', 'top_n-5', 'top_n-6', 'top_n-7'], importance_threshold=0.05, cumulative_threshold=0.9, entropy_threshold=0.7):
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
    for method, keep_config in keep_configs.items():
        save_path = os.path.join(save_dir, f'keep_config_{method}.json')
        with open(save_path, 'w') as f:
            json.dump(keep_config, f, indent=4)
    print(f"Keep configs saved to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate keep configs from expert selection counts")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the expert_selection_counts.json file")
    parser.add_argument("--methods", nargs='+', default=['adaptive', 'cumulative', 'entropy', 'top_n-1', 'top_n-2', 'top_n-3', 'top_n-4', 'top_n-5', 'top_n-6', 'top_n-7'], help="Methods to use for generating keep configs")
    parser.add_argument("--importance_threshold", type=float, default=0.1, help="Importance threshold for adaptive method")
    parser.add_argument("--cumulative_threshold", type=float, default=0.8, help="Cumulative importance threshold")
    parser.add_argument("--entropy_threshold", type=float, default=0.5, help="Entropy threshold")

    args = parser.parse_args()

    # Load expert selection counts
    expert_counts_path = os.path.join(args.model_dir, 'expert_selection_counts.json')
    with open(expert_counts_path, 'r') as f:
        expert_selection_counts = json.load(f)

    # Generate keep configs
    keep_configs = generate_keep_configs(
        expert_selection_counts, 
        methods=args.methods,
        importance_threshold=args.importance_threshold,
        cumulative_threshold=args.cumulative_threshold,
        entropy_threshold=args.entropy_threshold
    )
    print(keep_configs)
    # Save keep configs
    save_keep_configs(keep_configs, args.model_dir)