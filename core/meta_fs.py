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
from tqdm.auto import tqdm
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
        self.model.train()
        self.global_step = 0
        eval_step = [2 ** i for i in range(1, 20)]

        for task_id, support_set in enumerate(tqdm(self.train_dataloader)):

            fast_model = deepcopy(self.model)
            fast_model.to(self.args.device)
            inner_optimizer = torch.optim.AdamW(fast_model.parameters(), lr=self.args.step_size)
            fast_model.train()
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

