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

print("loaded", df.shape)
for seed in range(42, 52):
    cnt_samples_per_class = 1
    while True:
        cnt_samples = cnt_samples_per_class * 3
        if cnt_samples > len(df):
            cnt_samples = len(df)
        shuffled = df.sample(frac=1, random_state=seed)
        subsample = shuffled.groupby('gold_label').head(cnt_samples_per_class)
        path_out = path_data / "subsample_rand" / f"seed_{seed}" / f"{cnt_samples}"
        path_out.mkdir(parents=True, exist_ok=True)
        print("saving to", path_out)
        subsample.to_csv(path_out / "train.tsv", sep="\t")
        shutil.copy(path_data / "reference/MNLI/dev_matched.tsv", path_out / "dev_matched.tsv")
        shutil.copy(path_data / "reference/MNLI/dev_mismatched.tsv",
                    path_out / "dev_mismatched.tsv")
        if cnt_samples == len(df):
            break
        cnt_samples_per_class *= 2
