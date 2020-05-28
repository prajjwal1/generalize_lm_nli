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
cnt_samples_per_class = 1
cnt_samples = cnt_samples_per_class * 3
# random sampling
# subsample = df.sample(n=cnt_samples, random_state=seed)
# sample from each label equally
subsample = df.sample(frac=1, random_state=seed).groupby('gold_label').head(cnt_samples_per_class)
path_out = path_data / "subsample_rand" / f"seed_{seed}" / f"{cnt_samples}"
path_out.mkdir(parents=True, exist_ok=True)
print("saving to", path_out)
subsample.to_csv(path_out / "train.tsv", sep="\t")
shutil.copy(path_data / "reference/MNLI/dev_matched.tsv", path_out / "dev_matched.tsv")
shutil.copy(path_data / "reference/MNLI/dev_mismatched.tsv", path_out / "dev_mismatched.tsv")
