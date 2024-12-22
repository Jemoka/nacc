#!/bin/bash

for lr in 5e-4 5e-5 5e-6; do
    CUDA_VISIBLE_DEVICES=5 uv run python main.py "e3_no_transformer_${lr}" --featureset neuralpsych  --no_transformer --lr $lr --validation_interval 64 --checkpoint_interval 64 --report_interval 16 --wandb
done

