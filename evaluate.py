from glob import glob
from pathlib import Path
from experiments.scripts import validate

# first, validate each experiment
experiments = "./output/e5_kfold_*/best"
for i in glob(experiments):
    run_name = Path(i).parent.stem
    validate(i, str(Path("./results")/f"{run_name}.json"))

