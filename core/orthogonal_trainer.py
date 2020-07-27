import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import EvalPrediction, Trainer
from transformers.file_utils import is_apex_available, is_torch_tpu_available
from transformers.trainer_utils import PredictionOutput

if is_apex_available():
    from apex import amp

logger = logging.getLogger(__name__)


class OrthogonalTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
