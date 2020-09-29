import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from types import MethodType
from typing import Callable, Dict, Optional

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
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

from models.cbow import CBOW
from models.orthogonal_transformer import OrthogonalTransformer

logger = logging.getLogger(__name__)


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
    model_weights_path: Optional[str] = field(default=None)

@dataclass
class SpecificTrainingArguments(TrainingArguments):
    hyperparam_search: Optional[bool] = field(default=False)
    lamb: Optional[float] = field(default=None)

def _save(self, output_dir: Optional[str] = None):
    output_dir = output_dir if output_dir is not None else self.args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Saving model checkpoint to %s", output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(
        {"model_state_dict": self.model.state_dict()},
        os.path.join(output_dir, "pytorch_model.bin"),
    )
    torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def my_hp_space(trial):
    return {
        #  "learning_rate": trial.suggest_float("learning_rate", 5e-5, 2e-5, log=True),
        #  "seed": trial.suggest_int("seed", 1, 40),
        "lamb": trial.suggest_float("lamb", 0.00001, 0.0001, log=True),
    }

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, SpecificTrainingArguments)
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
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    tfmr = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    train_dataset = GlueDataset(data_args, tokenizer=tokenizer)

    eval_dataset = GlueDataset(
            data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir,
        )
    test_dataset = (
        GlueDataset(
            data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir,
        )
        if training_args.do_predict
        else None
    )

    #  lstm = LSTM(lstm_args)
    cbow = CBOW(config)

    def model_init():
        return OrthogonalTransformer(tfmr, cbow, config, training_args)

    if not training_args.hyperparam_search:
        model = OrthogonalTransformer(tfmr, cbow, config, training_args)

    if model_args.model_weights_path:
        logging.info("**** Loading nn.Module() weights ****")
        if training_args.hyperparam_search:
            raise ValueError("Cannot load weights when hyperparam_search=True")
        ckpt = torch.load(
            os.path.join(model_args.model_weights_path, "pytorch_model.bin")
        )
        model.load_state_dict(ckpt["model_state_dict"])

    def build_compute_metrics_fn(task_name: str,) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction) -> Dict:
            if output_mode == "classification":
                preds = np.argmax(p.predictions, axis=1)
            elif output_mode == "regression":
                preds = np.squeeze(p.predictions)
            return glue_compute_metrics(data_args.task_name, preds, p.label_ids)

        return compute_metrics_fn

    # Initialize our Trainer
    if training_args.hyperparam_search:
        trainer = Trainer(
            model_init=model_init,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=build_compute_metrics_fn(data_args.task_name),
        )
        trainer._save = MethodType(_save, trainer)
        trainer.hyperparameter_search(direction="maximize", hp_space=my_hp_space)
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            compute_metrics = build_compute_metrics_fn(data_args.task_name)
        )
        trainer._save = MethodType(_save, trainer)

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
    eval_results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            eval_datasets.append(
                GlueDataset(
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

    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            test_datasets.append(
                GlueDataset(
                    mnli_mm_data_args,
                    tokenizer=tokenizer,
                    mode="test",
                    cache_dir=model_args.cache_dir,
                )
            )

        for test_dataset in test_datasets:
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            if output_mode == "classification":
                predictions = np.argmax(predictions, axis=1)

            output_test_file = os.path.join(
                training_args.output_dir,
                f"test_results_{test_dataset.args.task_name}.txt",
            )
            if trainer.is_world_master():
                with open(output_test_file, "w") as writer:
                    logger.info(
                        "***** Test results {} *****".format(
                            test_dataset.args.task_name
                        )
                    )
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if output_mode == "regression":
                            writer.write("%d\t%3.3f\n" % (index, item))
                        else:
                            item = test_dataset.get_labels()[item]
                            writer.write("%d\t%s\n" % (index, item))
    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
