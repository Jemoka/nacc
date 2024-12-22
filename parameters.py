import argparse

parser = argparse.ArgumentParser(prog='nacc')

# logistics
parser.add_argument("experiment", help="name for the experiment", type=str)
parser.add_argument('-v', '--verbose', action='count', default=0, help="log level")
parser.add_argument("--wandb", default=False, action="store_true", help="whether to use wandb")
parser.add_argument("--warm_start", default=None, type=str, help="recover trainer from this path")

# intervals
parser.add_argument("--report_interval", default=32, type=int, help="save to wandb every this many steps")
parser.add_argument("--checkpoint_interval", default=256, type=int, help="checkpoint every this many steps")
parser.add_argument("--validation_interval", default=256, type=int, help="validate every this many steps")

# dataset
parser.add_argument("--out_dir", help="directory to save checkpoints and outputs", type=str, default="output")
parser.add_argument("--featureset", help="which feature set to run", type=str, default="combined")
parser.add_argument("--fold", help="kfold fold", type=int, default=0)
parser.add_argument("--no_longitudinal", default=False, action="store_true", help="even if longitudinal data is available, use only one past sample")


# training hyperparameters
parser.add_argument("--lr", help="learning rate", type=float, default=5e-5)
parser.add_argument("--epochs", help="number of epochs to train", type=int, default=55)
parser.add_argument("--batch_size", help="batch size", type=int, default=64)

# model hyperparemeters
parser.add_argument("--hidden_dim", help="hidden dimension", type=int, default=512)
parser.add_argument("--n_layers", help="number of layers for the model", type=int, default=3)
parser.add_argument("--no_transformer", default=False, action="store_true", help="use a simple, linear encoding scheme instead of Transformer encoding")



