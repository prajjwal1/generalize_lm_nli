# coding=utf-8
# Author: Prajjwal Bhargava

import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from tqdm.auto import tqdm, trange
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    GlueDataset,
    Trainer,
    default_data_collator,
    get_linear_schedule_with_warmup,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.training_args import TrainingArguments

try:
    from apex import amp

    _has_apex = True
except ImportError:
    _has_apex = False


def is_apex_available():
    return _has_apex


logger = logging.getLogger(__name__)


@dataclass
class MetaTrainer(Trainer):
    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        train_dataloader_list: [DataLoader],
        eval_dataloader_list: [DataLoader],
        train_steps_per_task: List,
        data_collator: Optional[DataCollator] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        prediction_loss_only=False,
        optimizers: Tuple[
            torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR
        ] = None,
    ):

        self.model = model.to(args.device)
        self.args = args
        self.compute_metrics_list = [
            compute_metrics(task) for task in self.args.total_task_list
        ]
        self.train_dataloader_list = train_dataloader_list
        self.eval_dataloader_list = eval_dataloader_list
        self.data_collator = (
            data_collator if data_collator is not None else default_data_collator
        )
        self.prediction_loss_only = prediction_loss_only
        self.optimizers = optimizers
        self.eval_results = {}
        self.train_steps_per_task = train_steps_per_task
        self.num_tasks = len(self.eval_dataloader_list)
        self._setup_wandb()
        if self.args.fp16:
            self.scaler = torch.cuda.amp.GradScaler()

    def update_model_params(self, model, fast_params):
        for idx, param in enumerate(model.parameters()):
            param.data = fast_params[idx]
        return model

    def compute_grad(self, loss, params, optimizer):
        if self.args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_grad_params = torch.autograd.grad(
                    self.scaler.scale(scaled_loss), params, allow_unused=True,
                )
            inv_scale = 1.0 / self.scaler.get_scale()
            grad_params = [p * inv_scale for p in scaled_grad_params]
            return grad_params

        grad_params = torch.autograd.grad(loss, params, allow_unused=True)
        return grad_params

    def get_loss_mean(self, loss):
        return loss.mean() if self.args.n_gpu > 1 else loss

    def train(self):
        # Plotting data support
        columns = self.args.total_task_list
        metrics = ["eval_loss", "eval_acc", "eval_f1", "eval_acc_and_f1"]
        df = pd.DataFrame(columns=columns, index=metrics)
        for i in range(len(df.columns)):
            for j in range(len(metrics)):
                df[columns[i]][metrics[j]] = []

        model = self.model
        # TODO: Make scheduler dataset agnostic
        optimizer, scheduler = self.get_optimizers(
            int(
                len(self.train_dataloader_list[0])
                // self.args.gradient_accumulation_steps
                * self.args.num_train_epochs
            )
        )
        # FP-16, Multi-GPU, Distributed Training
        # TODO: Make FP-16 work, scaled_loss, OOM issue
        if self.args.fp16:
            if is_apex_available:
                model, optimizer = amp.initialize(
                    model, optimizer, opt_level=self.args.fp16_opt_level
                )
            else:
                raise ValueError("Apex is not installed")

        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )
        # TODO: Make calculation of num_epochs with HF
        num_train_epochs = self.args.num_train_epochs
        total_train_batch_size = (
            self.args.train_batch_size
            * self.args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
        )

        logger.info("***** Running training *****")
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info(
            "  Instantaneous batch size per device = %d",
            self.args.per_device_train_batch_size,
        )
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            total_train_batch_size,
        )
        logger.info(
            "  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps
        )

        model.zero_grad()
        self.global_step = 0

        # main loop
        for epoch in trange(int(self.args.num_train_epochs), desc="Epoch"):
            model.train()
            tr_loss = 0

            # Set epoch if distributed training is enabled
            for train_dataloader in self.train_dataloader_list:
                if isinstance(train_dataloader, DataLoader) and isinstance(
                    train_dataloader, DistributedSampler
                ):
                    train_dataloader.sampler.set_epoch(epoch)

            train_dataloaders_iters = [
                iter(train_dataloader)
                for train_dataloader in self.train_dataloader_list
            ]

            train_dl_ids = []
            for t_id in range(len(self.args.task_list)):
                train_dl_ids += [t_id] * self.train_steps_per_task[
                    t_id
                ]  # math.ceil(len(train_examples[t_id]))
            train_dl_ids = np.random.choice(
                train_dl_ids, len(train_dl_ids), replace=False
            )

            for step, task_id in enumerate(tqdm(train_dl_ids, desc="Task IDs")):
                for update_step in range(self.args.num_update_steps + 1):

                    try:
                        inputs = next(train_dataloaders_iters[task_id])
                    except StopIteration:
                        break

                    for k, v in inputs.items():
                        inputs[k] = v.to(self.args.device)
                        if not isinstance(inputs["labels"], torch.cuda.LongTensor):
                            inputs["labels"] = inputs["labels"].long()

                    """
                    # TODO: Make Gradient clipping work because loss.backward() is not being used
                    if (
                        (step + 1) % self.args.gradient_accumulation_steps == 0
                        and update_step > 1
                    ):
                        if self.args.fp16:
                            torch.nn.utils.clip_grad_norm_(
                                amp.master_params(optimizer), self.args.max_grad_norm
                            )
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), self.args.max_grad_norm
                            )
                        model.zero_grad()
                        scheduler.step()
                    """
                    # MAML specific
                    if update_step == 0:
                        tr_loss = model(**inputs)[0]
                        tr_loss = self.get_loss_mean(tr_loss)
                        grad_params = self.compute_grad(
                            tr_loss, model.parameters(), optimizer
                        )
                        fast_params = list(
                            map(
                                lambda p: p[1] - self.args.learning_rate * p[0]
                                if p[0] is not None
                                else p[1],
                                zip(grad_params, model.parameters()),
                            )
                        )

                    elif update_step == self.args.num_update_steps:
                        if update_step == 0:
                            raise ValueError("update_step cannot be 0!")

                        for param, f_param in zip(model.parameters(), fast_params):
                            if not param.requires_grad:
                                continue
                            cur_grad = (
                                (param - f_param)
                                / update_step
                                / self.args.learning_rate
                            )
                            if param.grad is None:
                                param.grad = torch.zeros(cur_grad.size()).cuda()
                                param.grad.add_(cur_grad / inputs["labels"].size(0))

                    elif update_step < self.args.num_update_steps:
                        model = self.update_model_params(model, fast_params)
                        tr_loss = model(**inputs)[0]
                        tr_loss = self.get_loss_mean(tr_loss)
                        grad_params = self.compute_grad(tr_loss, fast_params, optimizer)
                        fast_params = list(
                            map(
                                lambda p: p[1] - self.args.learning_rate * p[0]
                                if p[0] is not None
                                else p[1],
                                zip(grad_params, fast_params),
                            )
                        )

                if step % self.num_tasks == (self.args.num_sample_tasks - 1):
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    model.zero_grad()

                self.global_step += 1

                # Run evaluation on task list
                if self.global_step % self.args.eval_steps == 0:
                    for idx, eval_dataloader in enumerate(self.eval_dataloader_list):
                        self.compute_metrics = self.compute_metrics_list[idx]
                        result = self.evaluate(eval_dataloader.dataset)

                        for key, value in result.items():
                            logger.info(
                                "%s  %s = %s",
                                self.args.total_task_list[idx],
                                key,
                                value,
                            )
                            df[self.args.total_task_list[idx]][key].append(value)
                    # Save model
                if (
                    self.args.save_steps > 0
                    and self.global_step % self.args.save_steps == 0
                ):
                    if hasattr(model, "module"):
                        assert model.module is self.model
                    else:
                        assert model is self.model

                    output_dir = os.path.join(
                        self.args.output_dir,
                        f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}",
                    )

                    self.save_model(output_dir)

        logging.info("*** Results have been saved at %s ***", self.args.output_dir)
        df.to_csv(self.args.output_dir + self.args.output_file_name + ".csv")
