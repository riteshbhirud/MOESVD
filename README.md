# MoE-SVD: Singular Value Decomposition for Mixture of Experts

This project implements SVD-based compression techniques for Mixture of Experts (MoE) models, with a focus on the Mixtral-8x7B model.

## Main Components

1. `MOE_SVD_share.py`: Core implementation of SVD compression and evaluation methods.
2. `combine_experiments.sh`: Main script for running experiments.
3. `svd_mixtral_sharing.py`: Implementation of SVD compression for Mixtral model.
4. `svd_deepseek_sharing.py`: Implementation of SVD compression for DeepSeek model.

## Setup



1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run an experiment, use the `combine_experiments.sh` script:
bash
bash combine_experiments.sh

This script sets up the experiment parameters and calls `MOE_SVD_share.py` with the appropriate arguments.

### Key Parameters in `combine_experiments.sh`

The `combine_experiments.sh` script contains several important parameters that control the compression and evaluation process:

- `MODEL_PATH`: Specifies the path or name of the model to be compressed (e.g., "mistralai/Mixtral-8x7B-v0.1").
- `WHITENING_NSAMPLES` and `UPDATING_NSAMPLES`: Control the number of samples used in the whitening and updating processes.
- `SEED`: Sets the random seed for reproducibility.
- `MODEL_SEQ_LEN`: Defines the sequence length for the model.
- `SAVE_PATH`: Specifies where to save the compressed model.
- `DATASETS`: Lists the datasets used for evaluation (e.g., "wikitext2").
- `NUM_NODES` and `NUM_PROCESSES`: Control parallelization settings.
- `DATASET` and `DATA_TYPE`: Specify the dataset and its format for compression.
- `N_COMPRESSION_SAMPLES`: Sets the number of samples used for compression.
- `COMPRESS_METHOD`: Defines the compression method (e.g., "expert_drop").
- `OUTPUT_DIR`: Specifies where to save the results.
- `REVERSE_DROP` and `PRESERVE_GATE`: Control aspects of the expert dropping process.
- `COMBO_LOG_DIR` and `LOG_FILE`: Define logging settings for the experiments.

These parameters allow for flexible configuration of the compression process, enabling experimentation with different models, datasets, and compression settings.


### Running SVD Compression and Evaluation

The main loop in `combine_experiments.sh` runs SVD compression and evaluation for different configurations, including various compression ratios and layer selections.

## Core Functionality in `MOE_SVD_share.py`

`MOE_SVD_share.py` contains the implementation of SVD compression and evaluation methods. Key functions include:

1. `profle_svdllm_low_resource`: Profiles the model for SVD compression.
2. `whitening`: Applies whitening transformation to the model.
3. `whitening_local_update`: Performs local updates after whitening.
4. `ppl_eval_sharing`: Evaluates the perplexity of the compressed model.


## Evaluation

The script automatically evaluates the compressed model using perplexity on specified datasets. Results are saved in a log file.

## Customization

You can modify the parameters in `combine_experiments.sh` to experiment with different compression ratios, layer selections, and model configurations.

## Contributing

Contributions to improve the MoE-SVD project are welcome. Please submit pull requests or open issues for any bugs or feature requests.

