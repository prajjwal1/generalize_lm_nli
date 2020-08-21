import dataclasses
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler
from tqdm.auto import tqdm, trange
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    GlueDataset,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)
from transformers.trainer import SequentialDistributedSampler

from core.meta_fs import MetaTrainer
from dataset_utils import get_dataset_dict, processor_dict
from datasets.meta_dataset import MetaDataset
from hans.utils_hans import HansDataset, InputFeatures

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    data_dir: str = field(default=None, metadata={"help": "GLUE directory"})
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization."
                " Sequences longer than this will be truncated, sequences"
                " shorter will be padded."
            )
        },
    )
    # compatibility with Hf
    task_name: str = field(default=None)
    overwrite_cache: bool = field(default=False)


@dataclass
class MetaTrainingArguments(TrainingArguments):
    output_dir: str = field(metadata={"help": "Output directory to save models"})
    eval_steps: int = field(
        default=100, metadata={"help": "Steps after which evaluation will be run"},
    )
    step_size: float = field(default=None)
    max_sample_limit: int = field(default=None)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": (
                "Path to pretrained model or model identifier from"
                " huggingface.co/models"
            )
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Where do you want to store the pretrained models downloaded from s3"
            )
        },
    )


def hans_data_collator(features: List[InputFeatures]) -> Dict[str, torch.Tensor]:
    """
    Data collator that removes the "pairID" key if present.
    """
    batch = default_data_collator(features)
    _ = batch.pop("pairID", None)
    return batch


def main():

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, MetaTrainingArguments)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (model_args, data_args, training_args,) = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and"
            " is not empty. Use --overwrite_output_dir to overcome."
        )
    if not os.path.exists(training_args.output_dir):
        os.mkdir(training_args.output_dir)

    logging.info("Training dataset %s", data_args.task_name)
    # Set seed
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    def build_compute_metrics_fn(task_name: str,) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction) -> Dict:
            if output_mode == "classification":
                preds = np.argmax(p.predictions, axis=1)
            elif output_mode == "regression":
                preds = np.squeeze(p.predictions)
            return glue_compute_metrics(data_args.task_name, preds, p.label_ids)

        return compute_metrics_fn

    train_dataset = MetaDataset(GlueDataset(data_args, tokenizer))
    eval_dataset = GlueDataset(data_args, tokenizer, mode="dev")

    try:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found")

    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, config=config, cache_dir=model_args.cache_dir,
    )

    # Compute indices of dataset for subsampling
    cur_len = len(train_dataset)
    indices = np.arange(cur_len)
    np.random.shuffle(indices)
    train_sampler = SubsetRandomSampler(indices[: training_args.max_sample_limit])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        sampler=train_sampler,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=default_data_collator,
    )

    trainer = MetaTrainer(
        model,
        training_args,
        train_dataloader,
        eval_dataloader,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
    )

    trainer.train()


if __name__ == "__main__":
    main()
