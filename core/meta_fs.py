# coding=utf-8
# Author: Prajjwal Bhargava

import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
from copy import deepcopy
import gc

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import trange
from transformers import EvalPrediction, Trainer, set_seed
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.training_args import TrainingArguments

logger = logging.getLogger(__name__)


@dataclass
class MetaTrainer(Trainer):
    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        prediction_loss_only=False,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
    ):

        self.model = model.to(args.device)
        self.args = args
        self.compute_metrics = compute_metrics
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.prediction_loss_only = prediction_loss_only
        self.optimizer, self.lr_scheduler = optimizers
        self.epoch = None
        self.tb_writer = None
        set_seed(self.args.seed)

    def run_maml(self):
        self.create_optimizer_and_scheduler(self.args.max_sample_limit)
        dataloader = iter(self.train_dataloader)
        self.model.train()
        self.global_step = 0
        eval_step = [2 ** i for i in range(1, 20)]

        for task_id in trange(2*self.args.max_sample_limit):

            fast_model = deepcopy(self.model)
            fast_model.to(self.args.device)
            inner_optimizer = torch.optim.AdamW(fast_model.parameters(), lr=self.args.step_size)
            fast_model.train()
            support_set = next(dataloader)
            sum_gradients = []
            #  query_set = next(dataloader)

            # Support set [classes]
            for task in support_set:
                task = self._prepare_inputs(task)
                loss = fast_model(**task)[0]
                loss.backward()
                inner_optimizer.step()
                inner_optimizer.zero_grad()

            self.global_step += self.args.train_batch_size
            # Query Set [classes]
            #  for task in query_set:
                #  task = self._prepare_inputs(task)
                #  query_loss = fast_model(**task)[0]
            #      query_loss.backward()

            meta_weights = list(self.model.parameters())
            fast_weights = list(fast_model.parameters())

            for i, (meta_params, fast_params) in enumerate(zip(meta_weights, fast_weights)):
                gradient = meta_params - fast_params
                sum_gradients.append(gradient)

            del fast_model, inner_optimizer
            torch.cuda.empty_cache()

            # Outer loop is here since the task is 1
            #  for i in range(0, len(sum_gradients)):
            #  sum_gradients[i] = sum_gradients[i] / self.args.max_sample_limit
            for i, params in enumerate(self.model.parameters()):
                params.grad = sum_gradients[i]

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.lr_scheduler.step()

            del sum_gradients
            gc.collect()

            if self.global_step in eval_step:
                output = self.prediction_loop(self.eval_dataloader, description = "Evaluation")
                self.log(output.metrics)

                output_dir = os.path.join(
                    self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}",
                )
                self.save_model(output_dir)

    def train(self):

        self.create_optimizer_and_scheduler(self.args.max_sample_limit)

        logger.info("***** Running training *****")

        self.global_step = 0
        self.epoch = 0

        eval_step = [2 ** i for i in range(1, 20)]
        inner_optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.args.step_size
        )
        self.model.train()

        tqdm_iterator = tqdm(self.train_dataloader, desc="Batch Index")

        #  self.model.zero_grad()
        self.optimizer.zero_grad()
        query_dataloader = iter(self.train_dataloader)

        for batch_idx, meta_batch in enumerate(tqdm_iterator):
            target_batch = next(query_dataloader)
            outer_loss = 0.0
            # Loop through all classes
            for inputs, target_inputs in zip(meta_batch, target_batch):

                for k, v in inputs.items():
                    inputs[k] = v.to(self.args.device)
                    target_inputs[k] = v.to(self.args.device)

                with higher.innerloop_ctx(
                    self.model, inner_optimizer, copy_initial_weights=False
                ) as (fmodel, diffopt):

                    inner_loss = fmodel(**inputs)[0]
                    diffopt.step(inner_loss)
                    outer_loss += fmodel(**target_inputs)[0]

            self.global_step += 1
            self.optimizer.step()
            self.lr_scheduler.step()
            outer_loss.backward()
            #  self.model.zero_grad()

            if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.max_grad_norm
                )

            # Run evaluation on task list
            if self.global_step in eval_step:
                output = self.prediction_loop(self.eval_dataloader, description = "Evaluation")
                self.log(output.metrics)

                output_dir = os.path.join(
                    self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}",
                )
                self.save_model(output_dir)
