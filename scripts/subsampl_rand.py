import pandas
from pathlib import Path

# TODO: set data path specific to the node user
# read from ~/.config/something or at least hard-code by host-name
path_data = Path("/mnt/storage/Data/NLP/datasets/importance_sampling/")
df = pandas.read_csv(path_data / "reference/MNLI/train.tsv",
    delimiter="\t", header=0, quoting=3)
