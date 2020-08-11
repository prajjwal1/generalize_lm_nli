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
    task_list: str = field(default=None)
    eval_task_list: str = field(default=None)
    total_task_list: str = field(default=None)
    task_shared: bool = field(default=True)
    num_update_steps: int = field(default=5)
    num_sample_tasks: int = field(default=5)
    eval_steps: int = field(
        default=100, metadata={"help": "Steps after which evaluation will be run"},
    )
    output_file_name: str = field(default=None)
    # num_samples: int = field(default=None)
    max_sample_limit: int = field(default=None)
    num_tasks: int = field(default=None)


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

    # py_parser = argparse.ArgumentParser()
    # py_parser.add_argument("--task_list", nargs="*", type=str)
    # py_parser, _ = py_parser.parse_known_args()
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

    # Parsing string arguments to list
    training_args.task_list = list(map(str, training_args.task_list.split(",")))
    training_args.eval_task_list = list(
        map(str, training_args.eval_task_list.split(","))
    )

    training_args.total_task_list = (
        training_args.task_list + training_args.eval_task_list
    )
    if "mnli" in training_args.total_task_list:
        training_args.total_task_list.append("mnli-mm")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s,"
        " 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logging.info("Training datasets %s", training_args.task_list)
    logging.info("Evaluation datasets %s (no training)", training_args.eval_task_list)
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

    processors = [processor_dict[task]() for task in training_args.task_list]

    dataset_dict = get_dataset_dict(data_args)

    data_dirs = [dataset_dict[task] for task in training_args.task_list]

    label_lists = [processor.get_labels() for processor in processors]

    num_labels = [len(label_list) for label_list in label_lists]

    train_dataset_list, eval_dataset_list = [], []
    for task in training_args.task_list:
        data_args.task_name = task
        data_args.data_dir = dataset_dict[task]
        train_dataset_list.append(MetaDataset(GlueDataset(data_args, tokenizer)))

    # This to to avoid repetition for additional_dataset_list
    if "hans" in training_args.total_task_list:
        data_args.task_name = "hans"
        data_args.data_dir = dataset_dict["hans"]
        hans_dataset = HansDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task="hans",
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            evaluate=True,
        )

    for task in training_args.total_task_list:
        data_args.task_name = task
        data_args.data_dir = dataset_dict[task]
        if task != "hans":
            eval_dataset_list.append(GlueDataset(data_args, tokenizer, mode="dev"))
        if task == "hans":
            eval_dataset_list.append(hans_dataset)
    try:
        assert len(eval_dataset_list) == len(training_args.total_task_list)
    except AssertionError:
        print(len(eval_dataset_list), len(training_args.total_task_list))

    # TODO: Make it work on variable number of classes
    try:
        num_labels = glue_tasks_num_labels[training_args.task_list[0]]
        output_mode = glue_output_modes[training_args.task_list[0]]
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
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Compute indices of dataset for subsampling
    indices_train_dataset = []
    for dataset in train_dataset_list:
        cur_len = len(dataset)
        indices = np.arange(cur_len)
        np.random.shuffle(indices)
        indices_train_dataset.append(indices[: training_args.max_sample_limit])

    # Samplers for each train and eval datasets
    train_sampler_list, eval_sampler_list = [], []
    for indices, dataset in zip(indices_train_dataset, train_dataset_list):
        train_sampler_list.append(SubsetRandomSampler(indices))

    for dataset in eval_dataset_list:
        if training_args.local_rank != -1:
            eval_sampler_list.append(SequentialDistributedSampler(dataset))
        else:
            eval_sampler_list.append(SequentialSampler(dataset))

    # Dataloader for each train and eval datasets
    train_dataloader_list, eval_dataloader_list = [], []
    data_collator = default_data_collator

    for train_dataset, train_sampler in tqdm(
        zip(train_dataset_list, train_sampler_list)
    ):
        train_dataloader_list.append(
            DataLoader(
                train_dataset,
                batch_size=training_args.per_device_train_batch_size,
                sampler=train_sampler,
                #         collate_fn=data_collator,
                drop_last=True,
            )
        )

    for task, eval_dataset, eval_sampler in tqdm(
        zip(training_args.total_task_list, eval_dataset_list, eval_sampler_list)
    ):
        data_collator = hans_data_collator if task == "hans" else default_data_collator
        eval_dataloader_list.append(
            DataLoader(
                eval_dataset,
                batch_size=training_args.per_device_eval_batch_size,
                sampler=eval_sampler,
                collate_fn=data_collator,
                drop_last=True,
            )
        )

    train_examples = [
        processor.get_train_examples(data_dir)
        for processor, data_dir in tqdm(zip(processors, data_dirs))
    ]

    train_steps_per_task = [
        math.floor(
            (len(train_example) / training_args.per_device_train_batch_size)
            / (training_args.num_update_steps + 1)
        )
        for train_example in train_examples
    ]

    additional_dataset_list = {"hans": hans_dataset}

    total_steps = sum(train_steps_per_task) * training_args.num_train_epochs
    logging.info("***** Total steps: {} *****".format(total_steps))

    trainer = MetaTrainer(
        model,
        training_args,
        train_dataloader_list,
        eval_dataloader_list,
        compute_metrics=build_compute_metrics_fn,
        train_steps_per_task=train_steps_per_task,
        additional_dataset_list=additional_dataset_list,
    )

    trainer.train()


if __name__ == "__main__":
    main()
