import os
import sys
import argparse

import torch
import random
import numpy as np
import inspect
import logging
from loguru import logger
from dotenv import load_dotenv

import parameters
from commands import execute

load_dotenv()

logger.remove()

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

if __name__ == "__main__":
    args = parameters.parser.parse_args()

    logger.add(
        sys.stderr,
        format="<cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> |"
        "<level>{level: ^8}</level>| "
        "<magenta>({name}:{line})</magenta> <level>{message}</level>",
        level=("DEBUG" if args.verbose > 0 else "INFO"),
        colorize=True,
        enqueue=True
    )

    execute(args)

