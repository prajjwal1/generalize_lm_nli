import dataclasses
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    EvalPrediction,
    GlueDataset,
)
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)

from core.siamese_trainer import SiameseTrainer
from datasets.siamese_dataset import SiameseGlueDataset, siamese_data_collator
from models.siamese_model import SiameseTransformer, SiameseTransformer2

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


def main():
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
            f"Output directory ({training_args.output_dir}) already exists and"
            " is not empty. Use --overwrite_output_dir to overcome."
        )

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

    set_seed(training_args.seed)

    try:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.load_model_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

    model = SiameseTransformer2(model_args, config)

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

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.load_model_path,
        cache_dir=model_args.cache_dir,
    )
    train_dataset = (
        SiameseGlueDataset(data_args, tokenizer) if training_args.do_train else None
    )
    eval_dataset = (
        SiameseGlueDataset(data_args, tokenizer, mode="dev")
        if training_args.do_eval
        else None
    )

    def build_compute_metrics_fn(task_name: str,) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction) -> Dict:
            if output_mode == "classification":
                preds = np.argmax(p.predictions, axis=1)
            elif output_mode == "regression":
                preds = np.squeeze(p.predictions)
            return glue_compute_metrics(data_args.task_name, preds, p.label_ids)

        return compute_metrics_fn

    trainer = SiameseTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=siamese_data_collator,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
    )
    trainer.evaluate()
    if training_args.do_train:
        if model_args.load_model_path:
            trainer.train(model_args.load_model_path)
        else:
            trainer.train()
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    eval_results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            eval_datasets.append(
                SiameseGlueDataset(
                    mnli_mm_data_args,
                    tokenizer=tokenizer,
                    mode="dev",
                    cache_dir=model_args.cache_dir,
                )
            )

        for eval_dataset in eval_datasets:
            result = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(
                training_args.output_dir,
                f"eval_results_{eval_dataset.args.task_name}.txt",
            )
            with open(output_eval_file, "w") as writer:
                logger.info(
                    "***** Eval results {} *****".format(eval_dataset.args.task_name)
                )
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

            eval_results.update(result)


if __name__ == "__main__":
    main()
