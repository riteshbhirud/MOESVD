python generate_drop_configs.py \
  --model_dir /workspace/MOE-SVD/results/expert_drop/attnexpert3_5_7_9_10_12_23_24_26mistralai_Mixtral_8x7B_v0.1_whitening_only_0.8_group_A/checkpoint \
  --importance_threshold 0.1 \
  --cumulative_threshold 0.8 \
  --entropy_threshold 0.6