import json
from glob import glob
from pathlib import Path
from collections import defaultdict
from experiments.scripts import validate, vfields_

import numpy as np
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m.tolist(), h.tolist()

##################################################

# first, validate each experiment
experiments = "./output/e5_kfold_*/best"
for i in glob(experiments):
    run_name = Path(i).parent.stem
    validate(i, str(Path("./results")/f"{run_name}.json"))

def get_via_list(data, list):
    partial = data
    for i in list:
        try:
            partial = partial[i]
        except KeyError:
            print(partial, list)
    return partial

##################################################

def average(data_list):
    """return the average and the confidence interval of the data_list

    Arguments
    ----------
        data_list : list
                list of paths to json files

    Returns
    ----------
        dict
            dictionary with the average and confidence interval
    """

    values = defaultdict(list)

    for i in data_list:
        with open(i) as f:
            data = json.load(f)
            for indx, i in enumerate(vfields_):
                values[indx].append(get_via_list(data, i))
    values = dict(values)

    # covert each list to err bounds
    values_bounds = {i:mean_confidence_interval(j) for i,j in values.items()}

    # cast back to normal values
    converted_values = {}
    for i,j in values_bounds.items():
        feature = vfields_[i]

        partial = converted_values
        # first, create enogh of the dictionary to
        # write the value
        for f in feature[:-1]:
            res = partial.get(f, {})
            partial[f] = res
            partial = res
        # then, write the value
        partial[feature[-1]] = j

    return converted_values

# this is the set with the transformer encoder 
## ...combined
e2_combined = glob("./results/e2_kfold_combined_*")
e2_combined = average(e2_combined)

with open("./results/FINAL_RESULTS_transformer_combined.json", 'w') as df:
    json.dump(e2_combined, df, indent=4)

## ...neuralpsych
e2_nps = glob("./results/e2_kfold_neuralpsych_*")
e2_nps = average(e2_nps)

with open("./results/FINAL_RESULTS_transformer_neuralpsych.json", 'w') as df:
    json.dump(e2_nps, df, indent=4)

# this is the set without transformer encoder
## ...combined
e5_combined = glob("./results/e5_kfold_combined_*")
e5_combined = average(e5_combined)

with open("./results/FINAL_RESULTS_no_transformer_combined.json", 'w') as df:
        json.dump(e5_combined, df, indent=4)

## ...neuralpsych
e5_nps = glob("./results/e5_kfold_neuralpsych_*")
e5_nps = average(e5_nps)

with open("./results/FINAL_RESULTS_no_transformer_neuralpsych.json", 'w') as df:
        json.dump(e5_nps, df, indent=4)
        



