# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on HANS."""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from core.siamese_trainer import SiameseTrainer
from models.siamese_model import SiameseTransformer
from utils_siamese_hans import (
    InputFeatures,
    SiameseHansDataset,
    hans_processors,
    siamese_hans_data_collator,
)

sys.path.append("..")

logger = logging.getLogger(__name__)


@dataclass
class SiameseModelArguments:
    """
    Arguments pertaining to SiameseTransformer
    """

    model_name: str = field(
        metadata={
            "help": (
                "Path to pretrained model or model identifier from"
                " huggingface.co/models"
            )
        }
    )
    load_model_path: str = field(
        default=None, metadata={"help": "Path from where weights will be loaded"}
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
    freeze_a: bool = field(default=False, metadata={"help": "freeze model a"})
    freeze_b: bool = field(default=False, metadata={"help": "freeze model b"})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(
        metadata={
            "help": "The name of the task to train selected in the list: "
            + ", ".join(hans_processors.keys())
        }
    )
    data_dir: str = field(
        metadata={
            "help": (
                "The input data dir. Should contain the .tsv files (or other data"
                " files) for the task."
            )
        }
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences"
                " longer than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (SiameseModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not"
            " empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits"
        " training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        processor = hans_processors[data_args.task_name]()
        label_list = processor.get_labels()
        num_labels = len(label_list)
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.load_model_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.load_model_path,
        cache_dir=model_args.cache_dir,
    )
    model = SiameseTransformer(model_args, config)

    if model_args.load_model_path:
        model_path = os.path.join(model_args.load_model_path, "pytorch_model.bin")
        if os.path.isfile(model_path):
            ckpt = torch.load(model_path)
            logger.info(
                "*** Loading model weights from %s***", model_args.load_model_path
            )
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            raise ValueError("Model --load_model_path is not valid")

    # Get datasets
    train_dataset = (
        SiameseHansDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        SiameseHansDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            evaluate=True,
        )
        if training_args.do_eval
        else None
    )

    # Initialize our Trainer
    trainer = SiameseTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=siamese_hans_data_collator,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path
            if os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        output = trainer.predict(eval_dataset)
        preds = output.predictions
        preds = np.argmax(preds, axis=1)

        pair_ids = [ex["a"].pairID for ex in eval_dataset]
        output_eval_file = os.path.join(
            training_args.output_dir, "hans_predictions.txt"
        )
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                writer.write("pairID,gold_label\n")
                for pid, pred in zip(pair_ids, preds):
                    writer.write("ex" + str(pid) + "," + label_list[int(pred)] + "\n")

        trainer._log(output.metrics)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
