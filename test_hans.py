import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
from tqdm.auto import tqdm, trange
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, EvalPrediction)
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import HfArgumentParser, TrainingArguments, set_seed

from hans.dataset import HansDataset
from hans.processors import (glue_output_modes, glue_tasks_num_labels,
                             load_and_cache_examples)

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


def evaluate(model_args, data_args, training_args, eval_dataset, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if data_args.task_name == "mnli" else (data_args.task_name,)
    eval_outputs_dirs = (
        (training_args.output_dir, training_args.output_dir + "-MM")
        if data_args.task_name == "mnli"
        else (training_args.output_dir,)
    )

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset, label_list = load_and_cache_examples(
            model_args, data_args, training_args, eval_task, tokenizer, evaluate=True
        )

        if not os.path.exists(eval_output_dir) and training_args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=training_args.per_gpu_eval_batch_size
        )

        # multi-gpu eval
        if training_args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", training_args.per_gpu_eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval().to(training_args.device)
            batch = tuple(t.to(training_args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                # Model_type is not an argument of model_args
                # if args.model_type != "distilbert":
                #    inputs["token_type_ids"] = (
                #        batch[2] if args.model_type in ["bert", "xlnet"] else None
                #    )  # XLM, DistilBERT and RoBERTa don't use segment_ids
                inputs["token_type_ids"] = None
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
                pair_ids = batch[4].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
                pair_ids = np.append(pair_ids, batch[4].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if data_args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif data_args.output_mode == "regression":
            preds = np.squeeze(preds)

        output_eval_file = os.path.join(eval_output_dir, "hans_predictions.txt")
        with open(output_eval_file, "w") as writer:
            writer.write("pairID,gld_label\n")
            for pid, pred in zip(pair_ids, preds):
                writer.write("ex" + str(pid) + "," + label_list[int(pred)] + "\n")

    return results


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use"
            " --overwrite_output_dir to overcome."
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

    num_labels = glue_tasks_num_labels[data_args.task_name]
    output_mode = glue_output_modes[data_args.task_name]

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="MNLI",
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

    eval_dataset = HansDataset(data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None

    # Training
    if training_args.do_train:
        print("Please turn training_args.do_train off. Only evaluation supported on HANS.")

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        data_args.output_mode = output_mode
        result = evaluate(model_args, data_args, training_args, eval_dataset, model, tokenizer)

        output_eval_file = os.path.join(training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt")
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
