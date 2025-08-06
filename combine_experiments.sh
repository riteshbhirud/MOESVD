#!/bin/bash
export PYTHONPATH='./lm-evaluation-harness':'/workspace/MOE-SVD'
# Global variables
SVD_MODEL_PATH=""

# Common parameters


MODEL_PATH="/workspace/MOE-SVD/component/phimoe"
# MODEL_PATH="mistralai/Mixtral-8x7B-v0.1"
WHITENING_NSAMPLES=512
UPDATING_NSAMPLES=16
SEED=42
MODEL_SEQ_LEN=2048



SAVE_PATH="/workspace/MOE-SVD"
DATASETS=("wikitext2")


NUM_NODES=1
NUM_PROCESSES=1
DATASET="c4_train"
DATA_TYPE="pt"
N_COMPRESSION_SAMPLES=128
COMPRESS_METHOD="expert_drop"

OUTPUT_DIR="/workspace/MOE-SVD/results/expert_drop"
REVERSE_DROP="False"
PRESERVE_GATE="False"

COMBO_LOG_DIR="combo_log"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${COMBO_LOG_DIR}/combined_processing_log_${TIMESTAMP}.txt"

mkdir -p "$COMBO_LOG_DIR"
CONFIG_PATHS=(
  'keep_config_top_n-12.json'
)


log_message() {
    echo "$(date +"%Y-%m-%d %H:%M:%S") - $1" | tee -a "$LOG_FILE"
}


read -r -d '' EXPERIMENT_GROUPS << EOM
group_A|3,5,7,9,12,15,20,23,26|NONE|3,5,7,9,12,15,20,23,26|0.2
EOM


run_svd() {
    local SELECTED_LAYERS=$1
    local ATTENTION_LAYERS=$2
    local EXPERT_LAYERS=$3
    local RATIO=$4
    local EXP_GROUP=$5
    local STEP=$6

    # Log message for running SVD with specific parameters
    log_message "Running SVD for group $EXP_GROUP with Selected Layers: $SELECTED_LAYERS, Attention Layers: $ATTENTION_LAYERS, Expert Layers: $EXPERT_LAYERS, Ratio: $RATIO, Step: $STEP"

    SVD_MODEL_PATH=""

    for DATASET in "${DATASETS[@]}"; do
        # Prepare Python command
        PYTHON_CMD="python MOE_SVD_share.py \
            --model \"$MODEL_PATH\" \
            --step \"$STEP\" \
            --ratio \"$RATIO\" \
            --whitening_nsamples \"$WHITENING_NSAMPLES\" \
            --updating_nsamples \"$UPDATING_NSAMPLES\" \
            --dataset \"$DATASET\" \
            --seed \"$SEED\" \
            --model_seq_len \"$MODEL_SEQ_LEN\" \
            --save_path \"$SAVE_PATH\" \
            --DEV \"cuda\" \
            --eval_batch_size 16 \
            --group \"$EXP_GROUP\""

        # Add selected_layers to command if not NONE
        if [ "$SELECTED_LAYERS" != "NONE" ] && [ -n "$SELECTED_LAYERS" ]; then
            PYTHON_CMD+=" --selected_layers \"$SELECTED_LAYERS\""
        fi

        # Add expert_layers to command if not NONE
        if [ "$EXPERT_LAYERS" != "NONE" ] && [ -n "$EXPERT_LAYERS" ]; then
            PYTHON_CMD+=" --expert_layers \"$EXPERT_LAYERS\""
        fi

        # Add attention_layers to command if not NONE
        if [ "$ATTENTION_LAYERS" != "NONE" ] && [ -n "$ATTENTION_LAYERS" ]; then
            PYTHON_CMD+=" --attention_layers \"$ATTENTION_LAYERS\""
        fi

        # Run SVD compression and evaluation, display output in real-time and log
        log_message "Executing command: $PYTHON_CMD"
        eval $PYTHON_CMD 2>&1 | tee -a "$LOG_FILE"

        # Extract SVD model path from log file
        TEMP_PATH=$(grep -i "Saved model:" "$LOG_FILE" | tail -n 1 | awk '{print $NF}')
        
        if [ -n "$TEMP_PATH" ]; then
            SVD_MODEL_PATH=$TEMP_PATH
            log_message "Found SVD model path: $SVD_MODEL_PATH"
        fi
    done

    if [ -n "$SVD_MODEL_PATH" ]; then
        if [ -f "$SVD_MODEL_PATH" ]; then
            log_message "SVD model file confirmed at: $SVD_MODEL_PATH"
        else
            log_message "Error: SVD model file not found at path: $SVD_MODEL_PATH"
            SVD_MODEL_PATH=""
        fi
    else
        log_message "Error: Could not find SVD model path in output."
    fi
}

run_expert_drop() {
    local MODEL_PATH=$1
    local MODEL_NAME=$(basename "${MODEL_PATH}" .pt)

    # Log message for running Expert Drop
    log_message "Running Expert Drop for model: $MODEL_NAME"

    for CONFIG_PATH in "${CONFIG_PATHS[@]}"; do
        log_message "Processing config: $(basename "$CONFIG_PATH")"

        local OUTPUT_DIR="${OUTPUT_DIR}/${MODEL_NAME}"
        local COMPRESSED_MODEL_SAVE_PATH="${OUTPUT_DIR}/checkpoint"

        # Build Expert Drop command, ensuring all parameters are correctly quoted
        EXPERT_DROP_CMD="accelerate launch \
            --config_file \"config/accelerate/mixtral_normal.yaml\" \
            --num_processes ${NUM_PROCESSES} \
            --num_machines ${NUM_NODES} \
            src/run_prune.py \
            --stage prune \
            --model_name_or_path \"${MODEL_PATH}\" \
            --dataset \"${DATASET}\" \
            --split \"train\" \
            --data_type \"${DATA_TYPE}\" \
            --cutoff_len ${MODEL_SEQ_LEN} \
            --output_dir \"${OUTPUT_DIR}\" \
            --logging_steps 10 \
            --bf16 \
            --n_compression_samples ${N_COMPRESSION_SAMPLES} \
            --compress_method ${COMPRESS_METHOD} \
            --expert_drop_method \"post_dropping\" \
            --reverse_drop ${REVERSE_DROP} \
            --preserve_gate ${PRESERVE_GATE} \
            --compressed_model_save_path \"${COMPRESSED_MODEL_SAVE_PATH}\" \
            --config_path \"${CONFIG_PATH}\""

        log_message "Executing Expert Drop command: $EXPERT_DROP_CMD"
        eval $EXPERT_DROP_CMD 2>&1 | tee -a "$LOG_FILE"

        if [ $? -ne 0 ]; then
            log_message "Error: Expert Drop failed for model $MODEL_NAME"
            return 1
        fis

        log_message "Expert Drop processing completed for config: $(basename "$CONFIG_PATH")"

        # Run the evaluation script
        log_message "Executing evaluation command..."
        # EVAL_CMD="python results/evluate_ppl.py \
        #     --compressed_model_save_path \"${COMPRESSED_MODEL_SAVE_PATH}\" \
        #     --config_path \"${CONFIG_PATH}\" \
        #     --model_name \"${MODEL_NAME}\""

        eval $EVAL_CMD 2>&1 | tee -a "$LOG_FILE"

        if [ $? -ne 0 ]; then
            log_message "Error: Evaluation failed for model $MODEL_NAME"
            log_message "Evaluation command was: $EVAL_CMD"
            return 1
        fi

        log_message "PPL Evaluation completed for model: $MODEL_NAME"

        # Run the lm-evaluation-harness
        local TUNE_ID="${COMPRESSED_MODEL_SAVE_PATH}/${MODEL_NAME}${CONFIG_PATH/.json/}.pt"
        log_message "Running lm-evaluation-harness for tune_id: $TUNE_ID"

        # LM_EVAL_CMD="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python lm-evaluation-harness/lm_eval --model ${TUNE_ID} \
        #     --model_args parallelize=True \
        #     --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,mathqa \
        #     --output_path \"${COMPRESSED_MODEL_SAVE_PATH}\" \
        #     --batch_size 2"

        eval $LM_EVAL_CMD 2>&1 | tee -a "$LOG_FILE"

        if [ $? -ne 0 ]; then
            log_message "Error: lm-evaluation-harness failed for model $MODEL_NAME"
            log_message "lm-evaluation-harness command was: $LM_EVAL_CMD"
            return 1
        fi

        log_message "lm-evaluation-harness completed for model: $MODEL_NAME"

        # Delete TUNE_ID file
        if [ "$DELETE_TUNE_ID" == "True" ]; then
            if [ -f "$TUNE_ID" ]; then
                log_message "Attempting to delete TUNE_ID file: $TUNE_ID"
                if rm "$TUNE_ID"; then
                    log_message "Successfully deleted TUNE_ID file: $TUNE_ID"
                else
                    log_message "Error: Failed to delete TUNE_ID file: $TUNE_ID"
                fi
            else
                log_message "Warning: TUNE_ID file not found: $TUNE_ID"
            fi
        else
            log_message "Skipping deletion of TUNE_ID file: $TUNE_ID"
        fi
    done

    return 0
}


main() {
    local STEP=1
    DELETE_TUNE_ID="False"  # Set switch to control whether to delete $TUNE_ID
    
    while IFS='|' read -r GROUP SELECTED_LAYERS ATTENTION_LAYERS EXPERT_LAYERS RATIO; do
        log_message "Processing Experiment Group: $GROUP"
        
        log_message "Parsed values:"
        log_message "SELECTED_LAYERS: $SELECTED_LAYERS"
        log_message "ATTENTION_LAYERS: $ATTENTION_LAYERS"
        log_message "EXPERT_LAYERS: $EXPERT_LAYERS"
        log_message "RATIO: $RATIO"
        
        # Run SVD processing
        run_svd "$SELECTED_LAYERS" "$ATTENTION_LAYERS" "$EXPERT_LAYERS" "$RATIO" "$GROUP" "$STEP"
        log_message "After run_svd, SVD_MODEL_PATH is: $SVD_MODEL_PATH"
        
        if [ -n "$SVD_MODEL_PATH" ] && [ -f "$SVD_MODEL_PATH" ]; then
            log_message "SVD model file found: $SVD_MODEL_PATH"
            if run_expert_drop "$SVD_MODEL_PATH"; then
                log_message "Expert Drop completed successfully"
                
                # Delete SVD model file
                if rm "$SVD_MODEL_PATH"; then
                    log_message "SVD model file deleted successfully: $SVD_MODEL_PATH"
                else
                    log_message "Error: Failed to delete SVD model file: $SVD_MODEL_PATH"
                fi

            else
                log_message "Expert Drop failed, skipping evaluation"
            fi
        else
            log_message "Error: SVD model file not found or path is empty. Skipping Expert Drop for this configuration."
        fi
        
    done <<< "$EXPERIMENT_GROUPS"
}


main