# common standard library utilities
import os
import sys
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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# huggingface
from transformers import AutoConfig, AutoModel, AutoTokenizer

# MLOps
import wandb
from accelerate import Accelerator

# logging
from loguru import logger

class NACCTemporalLSTM(nn.Module):
    def __init__(self, nlayers=3, hidden=128):
        super().__init__()

        # the encoder network
        self.lstm = nn.LSTM(hidden, hidden, nlayers)

        # precompute position embeddings
        # we assume that no samples span more than 50 years
        self._nlayers = nlayers

    def forward(self, xs, temporal_mask):
        res = torch.zeros((xs.size(0), xs.size(-1)), device=xs.device)

        # compute sequence lengths of input
        seq_lens = (~temporal_mask).sum(1)

        # if we have no temporal data, we skip all the LSTM
        if (sum(seq_lens) == 0).all():
            return res

        # packem!
        packed = pack_padded_sequence(xs[seq_lens > 0], seq_lens[seq_lens > 0].cpu().tolist(),
                                      batch_first=True, enforce_sorted=False)
        # brrrr
        _, (out, __) = self.lstm(packed)
        # squash down hidden dims by averaging it
        out = out.sum(dim=0)
        # create a backplate for non-temporal data (i.e. those with seq_lens < 0)
        # insert these outputs from above as well as raw x for non temporal data
        res[seq_lens > 0] = out

        # and return everything we got
        return res
        
class NACCFeatureExtraction(nn.Module):

    def __init__(self, nhead=4, nlayers=3, hidden=128):
        super(NACCFeatureExtraction, self).__init__()

        # the entry network ("linear value embedding")
        # bigger than 80 means that its going to be out of bounds and therefore
        # be masked out; so hard code 81
        self.linear0 = nn.Linear(1, hidden)
        
        # the encoder network
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers, enable_nested_tensor=False)
        self.__hidden = hidden

    def forward(self, x, mask):
        # don't forward pass on the padding; otherwise we'll nan
        is_padding = mask.all(dim=1)

        # the backplate to insert back
        backplate = torch.zeros((x.shape[0], self.__hidden)).float().to(x.device)

        # forward pass only on the non-paddng
        net = self.linear0(torch.unsqueeze(x[~is_padding], dim=2))
        # recall transformers are seq first
        net = self.encoder(net.transpose(0,1), src_key_padding_mask=mask[~is_padding]).transpose(0,1)

        # put the results back
        backplate[~is_padding] = net.mean(dim=1)

        # average the output sequence information
        return backplate

# the transformer network
class NACCFuseModel(nn.Module):

    def __init__(self, num_classes, num_features, nhead=4, nlayers=3,
                 hidden=128, no_transformer=False):
        # call early initializers
        super().__init__()

        # initial feature extraction system
        self.extraction = NACCFeatureExtraction(nhead, nlayers, hidden)
        self.temporal = NACCTemporalLSTM(nlayers, num_features)
        self.hidden = hidden

        # if requested, skip the transformer and just project
        self.no_transformer = no_transformer
        if self.no_transformer:
            # if we skip the transformer encoder, our feature extractor
            # will also just be a projection
            self.extraction = nn.Linear(num_features, hidden, bias=True)

        # create a mapping between feature and hidden space
        # so temporal can be fused with hidden
        # we don't have bias to ensure zeros stay zeros
        self.proj = nn.Linear(num_features, hidden, bias=True)

        # mix attention projection
        self.feature_offset = nn.Parameter(torch.rand(hidden), requires_grad=True)
        self.mix_projection = nn.Linear(hidden, 1, bias=False)

        # prediction network
        self.ffnn = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            # nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes),
            nn.Softmax(1)
        )

        # loss
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self,
                feats_invariant, mask_invariant,
                feats_temporal, mask_temporal,
                padding_mask, # True if its padding
                labels=None):

        # and encode the temporal features
        temporal_encoding = self.temporal(feats_temporal, padding_mask)

        if self.no_transformer:
            invariant_encoding = self.extraction(feats_invariant)
        else:
            # encnode the invariant featrues first
            invariant_encoding = self.extraction(feats_invariant, mask_invariant)

        # late fuse and predict
        # apply a learned offset shift to the temporal data
        # we do this instead of bias to ensure that each slot
        # recieves the same offset value if no temporal
        temporal_encoding = self.proj(temporal_encoding)

        offsets = self.feature_offset.sigmoid()
        mix = self.mix_projection(temporal_encoding).squeeze(-1).sigmoid()

        mixed_invariants = torch.einsum("b,bh -> bh", 1-mix, invariant_encoding*offsets)
        mixed_temporal = torch.einsum("b,bh -> bh", mix, temporal_encoding)
        fused = mixed_invariants + mixed_temporal

        net = self.ffnn(fused)

        loss = None
        if labels is not None:
            # TODO put weight on MCI
            # loss = (torch.log(net)*labels)*torch.tensor([1,1.3,1,1])
            loss = self.cross_entropy(net, labels)

        return { "logits": net, "loss": loss }




