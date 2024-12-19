#!/bin/bash

# sweep for some basic parameters
FEATURESET=neuralpsych # the featurest to learn

for fold in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 uv run python main.py "e2_kfold_${FEATURESET}_${fold}" --fold $fold --featureset $FEATURESET   --validation_interval 64 --checkpoint_interval 64 --report_interval 16 --wandb
done

