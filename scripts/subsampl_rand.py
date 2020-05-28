import pandas
import shutil
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
# TODO: maybe N sample from each class?
subsample = df.sample(n=cnt_samples, random_state=seed)
path_out = path_data / "subsample_rand" / f"seed_{seed}" / f"{cnt_samples}"
path_out.mkdir(parents=True, exist_ok=True)
df.to_csv(path_out / "train.tsv", sep="\t")
shutil.copy(path_data / "reference/MNLI/dev_matched.tsv", path_out / "dev_matched.tsv")
shutil.copy(path_data / "reference/MNLI/dev_mismatched.tsv", path_out / "dev_mismatched.tsv")
