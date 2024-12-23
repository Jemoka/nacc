#!/bin/bash

# sweep for some basic parameters
FEATURESET=neuralpsych # the featurest to learn

for fold in 5 6 7 8 9; do
    CUDA_VISIBLE_DEVICES=5 uv run python main.py "e5_kfold_${FEATURESET}_${fold}" --hidden_dim 2048 --n_layers 3 --batch_size 64 --lr 5e-6 --featureset ${FEATURESET} --validation_interval 64 --checkpoint_interval 64 --report_interval 16 --wandb --no_transformer --fold ${fold}
done

