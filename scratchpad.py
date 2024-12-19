"""
scratchpad.py
A place to test code snippets and experiment with new ideas.
"""

import sys
from loguru import logger

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

# ds = load_dataset("./data/nacc", "neuralpsych_longitudinal_0")
# trainer = Trainer.from_pretrained("./output/e1_neuralpsych_512_5e-5/best")

