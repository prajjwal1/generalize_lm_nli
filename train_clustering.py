import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.cluster import DBSCAN
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, EvalPrediction, GlueDataset)
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (HfArgumentParser, Trainer, TrainingArguments,
                          glue_compute_metrics, glue_output_modes,
                          glue_tasks_num_labels, set_seed)

logger = logging.getLogger(__name__)


@dataclass
class Clustering_Arguments:
    eps: float = field(
        metadata={"help": "Threshold value to form clusters"}
    )
    min_samples: int = field(
        metadata={"help": "Minimum samples for clustering"}
    )
    embedding_path: str = field(
        metadata={"help": "Path from where embeddings will be loaded"}
    )
    data_pct: float = field(
        metadata={"help": "specifies how much data will be used"}
    )
    cluster_output_path: str = field(
        default=None, metadata={"help": "Path where embedding will be stored"}
    )
    cluster_labels_path: Optional[str] = field(
        default = None, metadata={"help": "Path from there clustering labels will be loaded"}
    )

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


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, Clustering_Arguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, clustering_args = parser.parse_args_into_dataclasses()

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

    logger.info("Loading embeddings")
    try:
        os.path.isfile(clustering_args.embedding_path)
        if clustering_args.cluster_labels_path:
            os.path.isfile(clustering_args.cluster_labels_path)
        #else:
        #    raise ValueError(f"Cluster labels not found at ({clustering_args.cluster_labels_path}")
    except FileNotFoundError:
        raise ValueError(f"Embeddings not found at ({clustering.embedding_path})")

    embeddings = torch.load(clustering_args.embedding_path)
    embeddings = np.concatenate(embeddings)
    logging.info("*** Loaded ", len(embeddings), " samples ***")
    # Load pretrained model and tokenizer

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

    logging.info("Forming clusters")
    if clustering_args.cluster_labels_path is None:
        clustering = DBSCAN(eps=clustering_args.eps, min_samples=clustering_args.min_samples).fit(embeddings)
        with open(clustering_args.cluster_output_path + "/" + "cluster_labels.npy", 'wb') as f:
            np.save(f, clustering.labels_)
            logging.info("*** INFO: Clustering labels saved ***")
    else:
        cluster_labels = np.load(clustering_args.cluster_labels_path)
        logging.info("INFO: Clustering labels loaded")
        class cluster_labels_patch:
            def __init__(self, cluster_labels):
                self.labels_ = cluster_labels

        clustering = cluster_labels_patch(cluster_labels)

    def get_cluster_indices(cluster_num, labels_array):
        return np.where(labels_array == cluster_num)[0]

    def get_concat_cluster_indices(labels, data_pct, original_len):
        current_len, cluster_indices = 0, []
        for i in set(labels):
            curr_cluster_indices = get_cluster_indices(i, labels)
            current_len += len(curr_cluster_indices)
            if current_len < int(original_len*data_pct):
                cluster_indices.extend(curr_cluster_indices)
            else:
                return cluster_indices
    
    if training_args.do_train:
        train_dataset = GlueDataset(data_args, tokenizer)
        cluster_indices = get_concat_cluster_indices(clustering.labels_, clustering_args.data_pct, len(train_dataset))
        train_dataset = torch.utils.data.Subset(train_dataset, cluster_indices)
        if len(train_dataset) < 100:
            sys.exit(0)
    eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None

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
