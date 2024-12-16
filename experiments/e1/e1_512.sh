#!/bin/bash

# sweep for some basic parameters
BATCH_SIZE=64 # we will fix batch size
NLAYERS=3 # we will fix the model at 3 layers
HIDDEN=512 # we will sweep through various hidden dimensions


for lr in 5e-3 5e-4 5e-5 5e-6; do
    for featureset in neuralpsych combined; do
        CUDA_VISIBLE_DEVICES=0 uv run python main.py "e1_${featureset}_${HIDDEN}_${lr}" --hidden_dim $HIDDEN --n_layers $NLAYERS --batch_size $BATCH_SIZE --lr $lr --featureset $featureset --wandb 
    done
done

