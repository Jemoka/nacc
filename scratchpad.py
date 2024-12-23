"""
scratchpad.py
A place to test code snippets and experiment with new ideas.
"""

import sys
from loguru import logger

import numpy as np

logger.remove()
logger.add(
    sys.stderr,
    format="<cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> |"
    "<level>{level: ^8}</level>| "
    "<magenta>({name}:{line})</magenta> <level>{message}</level>",
    level="DEBUG",
    colorize=True,
    enqueue=True
)

from trainer import Trainer
from commands import configure
from datasets import load_dataset
from experiments.scripts import validate

dataset = load_dataset("./data/nacc", "combined_longitudinal_0")["validation"]
trainer = Trainer.from_pretrained("./output/e1_neuralpsych_512_5e-5/best")



