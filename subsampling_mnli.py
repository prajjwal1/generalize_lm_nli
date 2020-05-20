import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, EvalPrediction, GlueDataset)
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (HfArgumentParser, Trainer, TrainingArguments,
                          glue_compute_metrics, glue_output_modes,
                          glue_tasks_num_labels, set_seed)

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class Arguments:
    data_pct: float


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, Arguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
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
        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    train_dataset = GlueDataset(data_args, tokenizer=tokenizer) if training_args.do_train else None
    eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None

    train_dataset_lengths = {
        0.1: int(len(train_dataset) * 0.1),
        0.2: int(len(train_dataset) * 0.2),
        0.3: int(len(train_dataset) * 0.3),
        0.4: int(len(train_dataset) * 0.4),
        0.5: len(train_dataset) - int(len(train_dataset) * 0.5),
        0.6: len(train_dataset) - int(len(train_dataset) * 0.4),
        0.7: len(train_dataset) - int(len(train_dataset) * 0.3),
        0.8: len(train_dataset) - int(len(train_dataset) * 0.2),
        0.9: len(train_dataset) - int(len(train_dataset) * 0.1),
    }

    train_dataset_10_pct, train_dataset_90_pct = torch.utils.data.random_split(
        dataset=train_dataset, lengths=[train_dataset_lengths[0.1], train_dataset_lengths[0.9]]
    )
    train_dataset_20_pct, train_dataset_80_pct = torch.utils.data.random_split(
        dataset=train_dataset, lengths=[train_dataset_lengths[0.2], train_dataset_lengths[0.8]]
    )
    train_dataset_30_pct, train_dataset_70_pct = torch.utils.data.random_split(
        dataset=train_dataset, lengths=[train_dataset_lengths[0.3], train_dataset_lengths[0.7]]
    )
    train_dataset_40_pct, train_dataset_60_pct = torch.utils.data.random_split(
        dataset=train_dataset, lengths=[train_dataset_lengths[0.4], train_dataset_lengths[0.6]]
    )
    train_dataset_50_pct, _ = torch.utils.data.random_split(
        dataset=train_dataset, lengths=[train_dataset_lengths[0.5], train_dataset_lengths[0.5]]
    )

    train_dataset_pct_dict = {
        0.1: train_dataset_10_pct,
        0.2: train_dataset_20_pct,
        0.3: train_dataset_30_pct,
        0.4: train_dataset_40_pct,
        0.5: train_dataset_50_pct,
        0.6: train_dataset_60_pct,
        0.7: train_dataset_70_pct,
        0.8: train_dataset_80_pct,
        0.9: train_dataset_90_pct,
    }
    # Specify the percentage
    train_dataset = train_dataset_pct_dict[args.data_pct]
    log_data_pct = str(args.data_pct * 100)
    logger.info("*** Using %f of the dataset ***", log_data_pct)

    def compute_metrics(p: EvalPrediction) -> Dict:
        if output_mode == "classification":
            preds = np.argmax(p.predictions, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(p.predictions)
        return glue_compute_metrics(data_args.task_name, preds, p.label_ids)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            eval_datasets.append(GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, evaluate=True))

        for eval_dataset in eval_datasets:
            result = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            )
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

            results.update(result)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
