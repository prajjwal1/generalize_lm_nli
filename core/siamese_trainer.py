import logging
import math
import os
import random
import re
import shutil
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from tqdm.auto import tqdm, trange
from transformers import set_seed, torch_distributed_zero_first
from transformers.data.data_collator import DataCollator, default_data_collator
from transformers.file_utils import is_apex_available, is_torch_tpu_available
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.trainer import SequentialDistributedSampler, Trainer, get_tpu_sampler
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    EvalPrediction,
    PredictionOutput,
    TrainOutput,
    is_wandb_available,
)
from transformers.training_args import TrainingArguments

if is_apex_available():
    from apex import amp


if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

try:
    from torch.utils.tensorboard import SummaryWriter

    _has_tensorboard = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter

        _has_tensorboard = True
    except ImportError:
        _has_tensorboard = False


def is_tensorboard_available():
    return _has_tensorboard


if is_wandb_available():
    import wandb


logger = logging.getLogger(__name__)


class SiameseTrainer(Trainer):
    def _training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        optimizer: torch.optim.Optimizer,
    ) -> float:
        model.train()
        # input_a, input_b = inputs["a"], inputs["b"]
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)
        labels = inputs["a"]["labels"]
        outputs = model(inputs["a"], inputs["b"])
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.item()

    def _prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.
        Works both with or without labels.
        """

        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else self.prediction_loss_only
        )

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        eval_losses: List[float] = []
        preds: torch.Tensor = None
        label_ids: torch.Tensor = None
        model.eval()

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(
                dataloader, [self.args.device]
            ).per_device_loader(self.args.device)

        for inputs in tqdm(dataloader, desc=description):
            has_labels = any(
                inputs["a"].get(k) is not None
                for k in ["labels", "lm_labels", "masked_lm_labels"]
            )
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.args.device)
            labels = inputs["a"]["labels"]
            with torch.no_grad():
                outputs = model(inputs["a"], inputs["b"])
                if has_labels:
                    step_eval_loss, logits = outputs[:2]
                    eval_losses += [step_eval_loss.mean().item()]
                else:
                    logits = outputs[0]

            if not prediction_loss_only:
                if preds is None:
                    preds = logits.detach()
                else:
                    preds = torch.cat((preds, logits.detach()), dim=0)
                if inputs["a"].get("labels") is not None:
                    if label_ids is None:
                        label_ids = inputs["a"]["labels"].detach()
                    else:
                        label_ids = torch.cat(
                            (label_ids, inputs["a"]["labels"].detach()), dim=0
                        )

        if self.args.local_rank != -1:
            # In distributed mode, concatenate all results from all nodes:
            if preds is not None:
                preds = self.distributed_concat(
                    preds, num_total_examples=self.num_examples(dataloader)
                )
            if label_ids is not None:
                label_ids = self.distributed_concat(
                    label_ids, num_total_examples=self.num_examples(dataloader)
                )
        elif is_torch_tpu_available():
            # tpu-comment: Get all predictions and labels from all worker shards of eval dataset
            if preds is not None:
                preds = xm.mesh_reduce("eval_preds", preds, torch.cat)
            if label_ids is not None:
                label_ids = xm.mesh_reduce("eval_label_ids", label_ids, torch.cat)

        # Finally, turn the aggregated tensors into numpy arrays.
        if preds is not None:
            preds = preds.cpu().numpy()
        if label_ids is not None:
            label_ids = label_ids.cpu().numpy()

        if (
            self.compute_metrics is not None
            and preds is not None
            and label_ids is not None
        ):
            metrics = self.compute_metrics(
                EvalPrediction(predictions=preds, label_ids=label_ids)
            )
        else:
            metrics = {}
        if len(eval_losses) > 0:
            metrics["eval_loss"] = np.mean(eval_losses)

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)
        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

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
