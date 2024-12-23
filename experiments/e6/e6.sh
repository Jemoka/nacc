#!/bin/bash

for lr in 5e-4 5e-5 5e-6; do
    CUDA_VISIBLE_DEVICES=1 uv run python main.py "e6_no_longitudinal_${lr}" --hidden_dim 2048 --n_layers 3 --batch_size 64 --lr ${lr} --featureset combined --validation_interval 64 --checkpoint_interval 64 --report_interval 16 --wandb --no_transformer --no_longitudinal 
done

CUDA_VISIBLE_DEVICES=1 uv run python main.py "e6_ours" --hidden_dim 2048 --n_layers 3 --batch_size 64 --lr 5e-6 --featureset combined --validation_interval 64 --checkpoint_interval 64 --report_interval 16 --wandb --no_transformer

