# common standard library utilities
import os
import sys
import glob
import time
import json
import math
import random
from random import Random

from pathlib import Path
from argparse import Namespace

# machine learning and data utilities
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR

from torch.utils.data import DataLoader, Dataset

# huggingface
from transformers import AutoConfig, AutoModel, AutoTokenizer

# MLOps
import wandb
from accelerate import Accelerator

# logging
from loguru import logger

# data utilities
import datasets
from datasets import Dataset, DatasetDict
from tqdm import tqdm

tqdm.pandas()

def process_sample(sample):
    datas = sample["normalized"]
    masks = (datas < 0) # True means masked, and dataloader already sets negatives as the misisng samples

    # check if all elements of the tensor are all equal to each other across time
    # this is the "time invariant" samples
    time_invariance = torch.any(datas.T.roll(shifts=(0,1), dims=(0,1)) == datas.T, dim=1)

    # get the time invariant samples as a seperate set (data 1)
    data_inv = datas[0].clone()
    data_inv_mask = masks[0].clone()
    data_inv[~time_invariance] = 0.00
    data_inv_mask[~time_invariance] = True

    # and mask out the time invariant data from timeseries
    data_var = datas.clone()
    data_var[:, time_invariance] = 0.00
    data_var_mask = masks.clone() 
    data_var_mask[:, time_invariance] = True

    # filter out any data which is all zero
    var_mask = ~data_var_mask.all(dim=1)
    data_var = data_var[var_mask]
    data_var_mask = data_var_mask[var_mask]

    # seed the one-hot vector
    one_hot_target = [0 for _ in range(3)]
    # and set it
    one_hot_target[sample["target"]] = 1

    return (data_inv, data_inv_mask,
            data_var, data_var,
            one_hot_target)

def collate_fn(batch, no_longitudinal=False):
    di, dim, dv, dvm, target = zip(*[process_sample(i) for i in batch])

    # crop the data such that all of them are not longitudinal
    if no_longitudinal:
        dv = [i[-1:] for i in dv]
        dvm = [i[-1:] for i in dvm]

    # invariant data can just be stacked
    inv_data = torch.stack(di)
    inv_mask = torch.stack(dim)
    out = torch.stack([torch.tensor(i).float() for i in target])

    # get the batch's maximum length
    time_max = max(len(i) for i in dv)
    to_pad = [time_max-i.shape[0] for i in dv]

    # pad the data and mask tensors
    var_data = torch.stack([F.pad(i, (0,0,0,j), "constant", 0) for i,j in zip(dv, to_pad)])
    var_mask = torch.stack([F.pad(i, (0,0,0,j), "constant", True) for i,j in zip(dvm, to_pad)])

    # calculate which of the samples is padding only
    is_pad = var_mask.all(dim=2)

    return inv_data, inv_mask.bool(), var_data, var_mask.bool(), is_pad, out

def get_dataloaders(featureset="combined", fold=0, batch_size=16, no_longitudinal=False):
    dataset = datasets.load_dataset("./data/nacc",
                                    f"{featureset}_longitudinal_{str(fold)}",
                                    trust_remote_code=True)

    dataset_val = dataset["validation"]
    dataset_val.set_format("torch", ["normalized", "target"])

    dataset_train = dataset["train"]
    dataset_train.set_format("torch", ["normalized", "target"])

    train_dl = DataLoader(dataset_train, batch_size=batch_size,
                          collate_fn=lambda x: collate_fn(x, no_longitudinal),
                          shuffle=True)
    val_dl = DataLoader(dataset_val, batch_size=batch_size,
                        collate_fn=lambda x: collate_fn(x, no_longitudinal),
                        shuffle=True)

    return (train_dl, val_dl), len(dataset_val[0]["normalized"][0])

