import re
import os
import glob
import random
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict

from tqdm import tqdm
tqdm.pandas()

_CITATION = """\
@article{weintraub2009alzheimer,
  title={The Alzheimer's disease centers' Uniform Data Set (UDS): The neuropsychologic test battery},
  author={Weintraub, Sandra and Salmon, David and Mercaldo, Nathaniel and Ferris, Steven and Graff-Radford, Neill R and Chui, Helena and Cummings, Jeffrey and DeCarli, Charles and Foster, Norman L and Galasko, Douglas and others},
  journal={Alzheimer Disease \& Associated Disorders},
  volume={23},
  number={2},
  pages={91--101},
  year={2009},
  publisher={LWW}
}
"""

_DESCRIPTION = """\
The Alzheimer’s Disease Centers’ Uniform Data Set (UDS) from the National Alzheimer's Coordinating Center (NACC)
"""

R = random.Random(7)

NFOLDS = 10

class NACC(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        j 
        for i in range(NFOLDS)
        for j in 
        [datasets.BuilderConfig(name=f"neuralpsych_1shot_{i}",
                                description=f"NACC neuralpsych battery, 1 shot diagosis, kfold {i}"),
         datasets.BuilderConfig(name=f"combined_1shot_{i}",
                                description=f"NACC combined dataset, 1 shot diagosis, kfold {i}"),
         datasets.BuilderConfig(name=f"combined_longitudinal_{i}",
                                description=f"NACC combined dataset, longitudinal prediction, kfold {i}"),
         datasets.BuilderConfig(name=f"neuralpsych_longitudinal_{i}",
                                description=f"NACC neuralpsych battery, longitudinal prediction, kfold {i}")]
    ]

    DEFAULT_CONFIG_NAME = "combined_1shot_0"

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # features=features,  
            # Homepage of the dataset for documentation
            # homepage=_HOMEPAGE,
            # License for the dataset if available
            # license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        self.means_ = None
        self.stds_ = None

        data = dl_manager.download("investigator_nacc57.csv")
        name, analysis, foldid = self.config.name.split("_")
        feature = dl_manager.download(os.path.join("features", name))
        df = pd.read_csv(data)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data": df,
                    "data_path": data,
                    "feature_path": feature,
                    "analysis": analysis,
                    "split": "train",
                    "fold": int(foldid),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data": df,
                    "data_path": data,
                    "feature_path": feature,
                    "analysis": analysis,
                    "split": "dev",
                    "fold": int(foldid),
                },
            ),
        ]

    def _generate_examples(self, data, data_path, feature_path, analysis, split, fold):
        featureset = feature_path

        df = data

        #### drop NA data ####
        df_numeric = df.apply(pd.to_numeric, errors='coerce')
        df_res = df_numeric.apply((lambda x: x.fillna(-4) if not (x!=x).all() else df[x.name]))
        data = df_res.dropna(axis=1, how="all")

        #### calculate current prediction targets ####
        # construct the artificial target 
        # everything is to be ignored by default
        # this new target has: 0 - Control; 1 - MCI; 2 - Dementia
        data.loc[:, "current_target"] = -1

        # NACCETPR == 88; DEMENTED == 0 means Control
        data.loc[(data.NACCETPR == 88)&
                (data.DEMENTED == 0), "current_target"] = 0
        # NACCETPR == 1; DEMENTED == 1; means AD Dementia
        data.loc[(data.NACCETPR == 1)&
                (data.DEMENTED == 1), "current_target"] = 2
        # NACCETPR == 1; DEMENTED == 0; NACCTMCI = 1 or 2 means amnestic MCI
        data.loc[((data.NACCETPR == 1)&
                (data.DEMENTED == 0)&
                ((data.NACCTMCI == 1) |
                (data.NACCTMCI == 2))), "current_target"] = 1

        # drop the columns that are irrelavent to us (i.e. not the labels above)
        data = data[data.current_target != -1]
        data = data[data.NACCAGE > 65]

        # create kfolds
        folds = []
        ids = set(data.NACCID)
        fold_size = (len(ids)//NFOLDS)

        while len(ids) >= fold_size:
            subset = R.sample(list(ids), fold_size)
            folds.append(subset)
            ids = ids - set(subset)

        # create hf datasets
        with open(featureset, 'r') as f:
            features = f.read().split("\n\n")
            features = [[j for j in i.split("\n") if j.strip() != "" and j.strip()[0] != "#"]
                        for i in features]
            grouped_features = [i for i in features if len(i) != 0]
        features_ = list(set([j for i in grouped_features 
                for j in i if j.strip() != ""]))
        features = features_ + ["NACCID", "NACCAGE", "current_target"]
        features = list(set(features))

        folds = [data[data.NACCID.isin(i)][features]
                 for indx, i in enumerate(folds)
                 if ((split == "dev" and (indx == fold))
                     or (split == "train" and (indx != fold)))]
        res = pd.concat(folds)

        if analysis == "1shot":
            # yield the dataset, one shot per row
            for indx, row in res.reset_index(drop=True).iterrows():
                yield indx, row.to_dict()
        else:
            # yield the dataset, grouped by prediction + augmentation
            res_data_converted = res.groupby(res.NACCID).apply(lambda x:self.process_participant(x, True))
            res_data_not_converted = res.groupby(res.NACCID).apply(lambda x:self.process_participant(x, False))
            # filter out for blanks
            res_data_converted = [j for i in res_data_converted if i for j in i]
            res_data_not_converted = [j for i in res_data_not_converted if i for j in i]

            # balance for counts of each
            res_data = res_data_converted+res_data_not_converted

            # compute sample of each type
            control_samples = []
            mci_samples = []
            dementia_samples = []

            for elem in res_data:
                i,j,k = elem
                if j == 0:
                    control_samples.append((i,j, k))
                elif j == 1:
                    mci_samples.append((i,j, k))
                elif j == 2:
                    dementia_samples.append((i,j,k))

            # min elements to sample from each class
            num_samples = min(len(mci_samples),
                            len(dementia_samples),
                            len(control_samples))
            res_data = (R.sample(mci_samples, num_samples) +
                                R.sample(dementia_samples, num_samples) +
                                R.sample(control_samples, num_samples))

            # generate the normalized features
            data = [i[0][features_] for i in res_data]
            targets = [i[1] for i in res_data]
            concat = pd.concat([i for indx, i in enumerate(data)
                                if targets[indx] == 0])[features_]

            if split == "train":
                means = concat.mean()
                stds = concat.std()

                # write down for dev split
                self.means_ = means
                self.stds_ = stds
            elif split == "dev":
                means = self.means_
                stds = self.stds_

            # to normalize the data + compute "variance of variances"
            def norm(x):
                partial = ((x-means)/stds)
                partial[(x < 0) | (x > 80)] = -1
                return partial
            def compute_variances(x):
                group = [x[j].std(axis=1).to_numpy() for j in grouped_features]
                return group

            # yield them together
            for indx,(x,y,m) in enumerate(res_data):
                normalized = norm(x[features_])

                results = {
                    "predictors": x[features_].to_numpy(),
                    "normalized": normalized.to_numpy(),
                    "variations": np.stack(compute_variances(normalized)).transpose(),
                    "time": m,
                    "target": y,
                }

                yield indx, results

    @staticmethod
    def process_participant(part, converted=False):
        if len(part) <= 1:
            return None
        sorted = part.sort_values(by=["NACCAGE"])
        crops = list(range(len(sorted)))[1:]
        # crops = R.sample(possible_crops, R.randint(1, len(possible_crops)))

        if converted:
            res = [(sorted.iloc[:j], sorted.iloc[j].current_target,
                    # sorted.iloc[:j+1].NACCAGE-sorted.iloc[0].NACCAGE) for j in crops
                    sorted.iloc[:j+1].NACCAGE-65) for j in crops
                if sorted.iloc[j-1].current_target > sorted.iloc[j].current_target]
        else:
            res = [(sorted.iloc[:j], sorted.iloc[j].current_target,
                    # sorted.iloc[:j+1].NACCAGE-sorted.iloc[0].NACCAGE) for j in crops
                    sorted.iloc[:j+1].NACCAGE-65) for j in crops
                if sorted.iloc[j-1].current_target <= sorted.iloc[j].current_target]


        if len(res) == 0:
            return None

        return res

