import subprocess
from pathlib import Path

path_data = Path("/mnt/storage/Data/NLP/datasets/importance_sampling/")
# TODO: iterate all subfolders
seed = 42
cnt_samples = 256
subsample_type = "subsample_rand"
model_name = "albert-base-v2"
command = ["python3", "run_glue.py",
           "--model_name_or_path", model_name,
           "--task_name", "MNLI",
           "--do_train",
           "--do_eval",
           "--data_dir", path_data / subsample_type / f"seed_{seed}" / str(cnt_samples),
           "--max_seq_length", "128",
           "--per_gpu_train_batch_size", "256",
           "--learning_rate", "2e-5",
           "--num_train_epochs", "3.0",
           "--output_dir",
           path_data / "results" / model_name / subsample_type / f"seed_{seed}" / str(cnt_samples),
           "--fp16",
           "--data_pct 0.02"]


proc = subprocess.Popen(command, shell=False)
proc.communicate()
