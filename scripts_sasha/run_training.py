import subprocess
from pathlib import Path

path_data = Path("/mnt/storage/Data/NLP/datasets/importance_sampling/")
model_name = "albert-base-v2"
subsample_type = "subsample_rand"
for dir_seed in (path_data / subsample_type).iterdir():
    seed = int(str(dir_seed).split("_")[-1])
    print(seed)
    for cnt_samples in dir_seed.iterdir():
        # TODO: skip if there are results in the destination folder
        command = ["python3", "run_glue.py",
                   "--model_name_or_path", model_name,
                   "--task_name", "MNLI",
                   "--do_train",
                   "--do_eval",
                   "--data_dir", cnt_samples,
                   "--max_seq_length", "128",
                   "--per_gpu_train_batch_size", "256",
                   "--learning_rate", "2e-5",
                   "--num_train_epochs", "3.0",
                   "--output_dir",
                   path_data / "results" / model_name / subsample_type / f"seed_{seed}" / cnt_samples.name,
                   "--fp16",
                   "--cbt_samples", cnt_samples.name]

        # TODO: instead of running locally, submit to a job scheduler
        # if the platform supports
        proc = subprocess.Popen(command, shell=False)
        proc.communicate()
