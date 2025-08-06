#!/bin/bash

# Set the base directory for the compressed models
BASE_DIR="/workspace/MOE-SVD/results/expert_drop"

# Loop through all subdirectories in the base directory
for dir in "$BASE_DIR"/*; do
    if [ -d "$dir" ]; then
        # Get the directory name
        dir_name=$(basename "$dir")
        
        # Check if the directory contains a checkpoint subdirectory
        if [ -d "$dir/checkpoint" ]; then
            # Loop through all .pt files in the checkpoint directory
            for pt_file in "$dir/checkpoint"/*.pt; do
                if [ -f "$pt_file" ]; then
                    # Extract the config name from the .pt file name
                    config_name=$(basename "$pt_file" .pt)
                    config_name="${config_name#"$dir_name"}"
                    config_name="${config_name}.json"
                    
                    # Run the evaluation script
                    python results/evluate_ppl.py \
                        --compressed_model_save_path "$dir/checkpoint" \
                        --config_path "$config_name"
                    
                    echo "Evaluated $pt_file"
                fi
            done
        else
            echo "No checkpoint directory found in $dir"
        fi
    fi
done

echo "Evaluation complete for all models."
