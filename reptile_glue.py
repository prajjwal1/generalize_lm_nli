import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler
from tqdm.auto import tqdm, trange
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DefaultDataCollator,
    EvalPrediction,
    GlueDataset,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)
from transformers.data.processors.glue import (
    ColaProcessor,
    MnliProcessor,
    MrpcProcessor,
    QnliProcessor,
    QqpProcessor,
    RteProcessor,
    Sst2Processor,
    StsbProcessor,
    WnliProcessor,
)

from core.meta import MetaTrainer

sys.path.append("..")


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
    target_task: str = field(default="mrpc", metadata={"help": "Target Task"})
    task_shared: bool = field(default=True)
    num_update_steps: int = field(default=5)
    num_sample_tasks: int = field(default=5)
    eval_steps: int = field(
        default=100, metadata={"help": "Steps after which evaluation will be run"},
    )


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
    training_args.task_list = list(map(str, training_args.task_list.split(",")))
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

    processor_dict = {
        "mrpc": MrpcProcessor,
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "sst-2": Sst2Processor,
        "rte": RteProcessor,
        "wnli": WnliProcessor,
        "qqp": QqpProcessor,
        "qnli": QnliProcessor,
        "sts-b": StsbProcessor,
    }
    processors = [processor_dict[task]() for task in training_args.task_list]

    dataset_dict = {
        "mrpc": data_args.data_dir + "/MRPC",
        "cola": data_args.data_dir + "/CoLA",
        "mnli": data_args.data_dir + "/MNLI",
        "sst-2": data_args.data_dir + "/SST-2",
        "rte": data_args.data_dir + "/RTE",
        "wnli": data_args.data_dir + "/WNLI",
        "qqp": data_args.data_dir + "/QQP",
        "qnli": data_args.data_dir + "/QNLI",
        "sts-b": data_args.data_dir + "/STS-B",
    }
    data_dirs = [dataset_dict[task] for task in training_args.task_list]

    for i, task in enumerate(training_args.task_list):
        if task == training_args.target_task:
            target_task_id = i
            break

    task_cluster_dict = {
        "mrpc": 0,
        "cola": 1,
        "mnli": 0,
        "sst-2": 1,
        "rte": 0,
        "wnli": 0,
        "qqp": 0,
        "qnli": 2,
        "sts-b": 3,
    }
    task_clusters = (
        [task_cluster_dict[task] for task in training_args.task_list]
        if training_args.task_shared
        else None
    )

    label_lists = [processor.get_labels() for processor in processors]
    if not training_args.task_shared:
        num_labels = [len(label_list) for label_list in label_lists]
    else:
        cluster_num_labels = {0: 3, 1: 2, 2: 2, 3: 1}
    num_labels = [cluster_num_labels[task_cluster] for task_cluster in task_clusters]

    train_dataset_list, eval_dataset_list = [], []
    for task, data_dir in zip(training_args.task_list, data_dirs):
        data_args.task_name = task
        data_args.data_dir = dataset_dict[task]
        train_dataset_list.append(GlueDataset(data_args, tokenizer))
        eval_dataset_list.append(GlueDataset(data_args, tokenizer, mode="dev"))

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
    train_sampler_list = []
    for dataset in train_dataset_list:
        train_sampler_list.append(RandomSampler(dataset))

    train_dataloader_list, eval_dataloader_list = [], []
    data_collator = DefaultDataCollator()

    for train_dataset, eval_dataset, sampler in tqdm(
        zip(train_dataset_list, eval_dataset_list, train_sampler_list)
    ):

        train_dataloader_list.append(
            DataLoader(
                train_dataset,
                batch_size=training_args.train_batch_size,
                sampler=sampler,
                collate_fn=data_collator.collate_batch,
                drop_last=True,
            )
        )

        eval_dataloader_list.append(
            DataLoader(
                eval_dataset,
                batch_size=training_args.train_batch_size,
                sampler=sampler,
                collate_fn=data_collator.collate_batch,
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
    total_steps = sum(train_steps_per_task) * training_args.num_train_epochs
    print(f"Total steps: {total_steps}")

    t_total = int(
        len(train_dataloader_list)
        // training_args.gradient_accumulation_steps
        * training_args.num_train_epochs
    )
    num_train_epochs = training_args.num_train_epochs

    trainer = MetaTrainer(
        model,
        training_args,
        train_dataloader_list,
        eval_dataloader_list,
        build_compute_metrics_fn,
        train_steps_per_task,
    )

    trainer.train()


if __name__ == "__main__":
    main()
