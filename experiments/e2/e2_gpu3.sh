#!/bin/bash

# sweep for some basic parameters
FEATURESET=combined # the featurest to learn

for fold in 5 6 7 8 9; do
    CUDA_VISIBLE_DEVICES=3 uv run python main.py "e2_kfold_${FEATURESET}_${fold}" --fold $fold --featureset $FEATURESET   --validation_interval 64 --checkpoint_interval 64 --report_interval 16 --wandb
done

