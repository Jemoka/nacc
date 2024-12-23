"""
logits.py
read out the validation logits from a model
"""

import torch
from trainer import Trainer
from commands import configure
from datasets import load_dataset

def logits(in_model):
    trainer = Trainer.from_pretrained(in_model)

    logits = []
    labels = []

    from tqdm import tqdm
    for batch in tqdm(iter(trainer.val_dl), total=len(trainer.val_dl)):
        results = trainer.model(*batch)
        logits += results["logits"].detach().cpu().tolist()
        labels += batch[-1].detach().cpu().tolist()

    import json
    with open("./data.json", 'w') as df:
        json.dump({"logits": logits, "labels": labels}, df, indent=4)

if __name__ == "__main__":
    logits("./output/e4_combined_2048_5e-6/best")

