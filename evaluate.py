from glob import glob
from pathlib import Path
from experiments.scripts import validate

# first, validate each experiment
experiments = "./output/e2_kfold_*/best"
for i in in experiments:
    run_name = Path(i).parent.stem
    validate(i, str(Path("./resources")/f"{run_name}.json"))

