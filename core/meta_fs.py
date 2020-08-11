# coding=utf-8
# Author: Prajjwal Bhargava

import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import higher
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import autocast
from tqdm.auto import tqdm
from transformers import EvalPrediction, Trainer, default_data_collator, set_seed
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.training_args import TrainingArguments


logger = logging.getLogger(__name__)


def hans_data_collator(features) -> Dict[str, torch.Tensor]:
    """
    Data collator that removes the "pairID" key if present.
    """
    batch = default_data_collator(features)
    _ = batch.pop("pairID", None)
    return batch


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
        ] = (None, None),
        additional_dataset_list: Optional[Dict] = None,
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
        self.optimizer, self.lr_scheduler = optimizers
        self.eval_results = {}
        self.train_steps_per_task = train_steps_per_task
        self.additional_dataset_list = additional_dataset_list
        if self.args.fp16:
            self.scaler = torch.cuda.amp.GradScaler()
        self.epoch = None
        self.tb_writer = None
        set_seed(self.args.seed)

    #    def update_model_params(self, model, fast_params):
    #        for idx, param in enumerate(model.parameters()):
    #            param.data = fast_params[idx]
    #        return model

    #   def compute_grad(self, loss, params, optimizer):
    #       if self.args.fp16:
    #           with amp.scale_loss(loss, optimizer) as scaled_loss:
    #               scaled_grad_params = torch.autograd.grad(
    #                   self.scaler.scale(scaled_loss), params, allow_unused=True,
    #               )
    #           inv_scale = 1.0 / self.scaler.get_scale()
    #           grad_params = [p * inv_scale for p in scaled_grad_params]
    #           return grad_params

    #       grad_params = torch.autograd.grad(loss, params, allow_unused=True)
    #       return grad_params

    def get_loss_mean(self, loss):
        return loss.mean() if self.args.n_gpu > 1 else loss

    def train(self):
        # Plotting data support
        columns = self.args.total_task_list
        metrics = [
            "eval_loss",
            "eval_acc",
            "eval_f1",
            "eval_acc_and_f1",
            "eval_mnli-mm/acc",
            "epoch"
        ]
        df = pd.DataFrame(columns=columns, index=metrics)
        print(columns, metrics)
        for i in range(len(df.columns)):
            for j in range(len(metrics)):
                df[columns[i]][metrics[j]] = []

        model = self.model
        # TODO: Make scheduler dataset agnostic
        self.create_optimizer_and_scheduler(
            int(
                len(self.train_dataloader_list[0])
                // self.args.gradient_accumulation_steps
                * self.args.num_train_epochs
            )
        )
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
        self.epoch = 0

        eval_step = [2 ** i for i in range(1, 20)]
        inner_optimizer = torch.optim.SGD(
            model.parameters(), lr=self.args.learning_rate
        )
        model.train()

        tqdm_iterator = tqdm(self.train_dataloader_list[0], desc="Batch Index")

        for batch_idx, meta_batch in enumerate(tqdm_iterator):
            model.zero_grad()
            target_batch = next(iter(self.train_dataloader_list[0]))
            outer_loss = 0.0
            for inputs, target_inputs in zip(meta_batch, target_batch):
                for k, v in inputs.items():
                    inputs[k] = v.to(self.args.device)
                    target_inputs[k] = v.to(self.args.device)
                with higher.innerloop_ctx(
                    model, inner_optimizer, copy_initial_weights=False
                ) as (fmodel, diffopt):

                    with autocast():
                        inner_loss = model(**inputs)[0]
                        inner_loss = self.get_loss_mean(inner_loss)
                        diffopt.step(inner_loss)
                        outer_loss += model(**target_inputs)[0]

            self.global_step += 1
            outer_loss = self.get_loss_mean(outer_loss)
            with autocast():
                outer_loss.backward()

            if self.args.fp16:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.args.max_grad_norm
                )

            # Run evaluation on task list
            if self.global_step in eval_step:
                for idx, eval_dataloader in enumerate(self.eval_dataloader_list):
                    task = self.args.total_task_list[idx]
                    self.compute_metrics = self.compute_metrics_list[idx]
                    if task != "hans":
                        result = self.evaluate(eval_dataloader.dataset)

                        for key, value in result.items():
                            logger.info(
                                "%s  %s = %s",
                                self.args.total_task_list[idx],
                                key,
                                value,
                            )
                            df[self.args.total_task_list[idx]][key].append(value)
                    if task == "hans":
                        label_list = ["contradiction", "entailment", "neutral"]
                        dataset = self.additional_dataset_list[task]
                        self.data_collator = hans_data_collator
                        output = self.predict(dataset)  # , description="Prediction")
                        self.data_collator = default_data_collator
                        self.log(output.metrics)
                        preds = output.predictions
                        preds = np.argmax(preds, axis=1)
                        pair_ids = [ex.pairID for ex in dataset]
                        output_eval_file = os.path.join(
                            self.args.output_dir,
                            "hans_predictions_" + str(batch_idx+1) + ".txt",
                        )
                        if self.is_world_master():
                            with open(output_eval_file, "w") as writer:
                                writer.write("pairID,gold_label\n")
                                for pid, pred in zip(pair_ids, preds):
                                    writer.write(
                                        "ex"
                                        + str(pid)
                                        + ","
                                        + label_list[int(pred)]
                                        + "\n"
                                    )

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
                    self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}",
                )

                self.save_model(output_dir)
                if self.is_world_master():
                    self._rotate_checkpoints()

                logging.info(
                    "*** Results have been saved at %s ***", self.args.output_dir
                )
                df.to_csv(self.args.output_dir + self.args.output_file_name + ".csv")
