"""
a0.py
Experiment 2, analysis 0

This script will evaluate the model on the validation set and save the results
in a json file. The results will include:

- Overall accuracy
- One versus rest accuracy
- Precision and recall by class
- Normalized confusion matrix
- Normalized confusion matrix, per number of longitudinal samples included
"""

import json
from trainer import Trainer
from commands import configure
from datasets import load_dataset

in_model = "./output/e1_neuralpsych_512_5e-5/best"
out_path = "./results/e1_neuralpsych_512_5e-5.json"

trainer = Trainer.from_pretrained(in_model)

# outer index: true
# inner index: predicted
confusion = [[[0 for _ in range(3)]
                      for _ in range(3)]
                      for _ in range(20)]

from tqdm import tqdm
for batch in tqdm(iter(trainer.val_dl), total=len(trainer.val_dl)):
    results = trainer.model(*batch)
    results = results["logits"].argmax(-1).cpu()
    targets = batch[-1].argmax(-1).cpu()

    for x,i,j in zip(batch[2], targets, results):
        confusion[sum(x.sum(dim=1)>0)][i][j] += 1

# sum the confusion matriarchies
confusion = np.array(confusion)

# overall normalized confusion
overall = confusion.sum(axis=0)
overall_normalized = overall/overall.sum(axis=1)

# normalized confusion by year
temporal_normalized = np.einsum("tap,ta->tap",
                                confusion,
                                (1/(confusion.sum(axis=2)+1e-12)))

overall_accuracy = sum(overall.diagonal())/overall.sum()
recall_by_class = overall.diagonal()/overall.sum(axis=1)
precision_by_class = overall.diagonal()/overall.sum(axis=0)

one_v_rest_control = ((overall[1:, 1:]).sum() + overall[0][0])/overall.sum()
one_v_rest_mci = (overall[0][0] + overall[0][2] +
                  overall[2][0] + overall[2][2] +
                  overall[1][1])/overall.sum()
one_v_rest_dementia = ((overall[:2, :2]).sum() + overall[2][2])/overall.sum()

results = {
    "accuracy": {
        "overall": overall_accuracy,
        "one_versus_rest": [one_v_rest_control, one_v_rest_mci, one_v_rest_dementia]
    },
    "pr": {
        "recall_by_class": recall_by_class.tolist(),
        "precision_by_class": precision_by_class.tolist()
    },
    "confusion": {
        "overall": overall_normalized.tolist(),
        "temporal": temporal_normalized.tolist()
    }
    
}

with open(out_path, "w") as df:
    json.dump(results, df, indent=4)



