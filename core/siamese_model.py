import logging

import torch
from torch import nn
from transformers import AutoModel

logger = logging.getLogger(__name__)


class SiameseTransformer(nn.Module):
    def __init__(self, args, config):
        super(SiameseTransformer, self).__init__()
        self.args = args
        self.model_a = AutoModel.from_pretrained(
            self.args.model_name, config=config, cache_dir=self.args.cache_dir
        )
        self.model_b = AutoModel.from_pretrained(
            self.args.model_name, config=config, cache_dir=self.args.cache_dir
        )

        if self.args.freeze_a:
            logger.info("**** Freezing Model A ****")
            for param in self.model_a.encoder.parameters():
                param.requires_grad = False

        if self.args.freeze_b:
            logger.info("**** Freezing Model B ****")
            for param in self.model_b.encoder.parameters():
                param.requires_grad = False

        self.linear = nn.Sequential(
            nn.Linear(self.args.input_dim, self.args.linear_dim),
            nn.Linear(self.args.linear_dim, self.args.num_labels),
        )

    def forward(self, input_a, input_b):
        loss_fct = nn.CrossEntropyLoss()
        labels = input_a["labels"]
        input_a.pop("labels")
        input_b.pop("labels")
        output_a = self.model_a(**input_a)[0][:, 0, :]
        output_b = self.model_b(**input_b)[0][:, 0, :]
        concat_output = torch.cat([output_a, output_b])
        concat_output = concat_output.view(labels.size(0), -1)
        logits = self.linear(concat_output)
        loss = loss_fct(logits, labels)
        return loss, logits
