import pandas
from pathlib import Path

# TODO: set data path specific to the node user
# read from ~/.config/something or at least hard-code by host-name
path_data = Path("/mnt/storage/Data/NLP/datasets/importance_sampling/")
df = pandas.read_csv(path_data / "reference/MNLI/train.tsv",
                     sep="\t",
                     header=0,
                     quoting=3)

# TODO: iterate over seeds and sampe sizes
seed = 42
cnt_samples = 256
subsample = df.sample(n=cnt_samples, random_state=seed)
print(subsample)
path_out = path_data / "subsample_rand" / f"seed_{seed}" / f"{cnt_samples}" / "MNLI/"
path_out.mkdir(parents=True, exist_ok=True)
df.to_csv(path_out / "train.tsv", sep="\t")
# TODO: copy dev set
