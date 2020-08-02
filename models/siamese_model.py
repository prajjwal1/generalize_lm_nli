import logging

import torch
from torch import nn
from transformers import AutoModel, AutoModelForSequenceClassification, PreTrainedModel

logger = logging.getLogger(__name__)


class PredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((8, 128))
        self.dense = nn.Linear(4096, len(config.id2label))

    def forward(self, features):
        features = self.pool(features)
        features = features.view(features.shape[0] // 4, -1)
        features = self.dense(features)
        return features


class SiameseTransformer(nn.Module):
    def __init__(self, args, config):
        super(SiameseTransformer, self).__init__()
        self.args = args
        self.model_a = AutoModelForSequenceClassification.from_pretrained(
            self.args.model_name, config=config, cache_dir=self.args.cache_dir
        )
        self.model_b = AutoModelForSequenceClassification.from_pretrained(
            self.args.model_name, config=config, cache_dir=self.args.cache_dir
        )

        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, a, b):
        output_a = self.model_a(**a)  # [bs, seq_len, 768]
        output_b = self.model_b(**b)
        outputs = []
        for i in range(len(output_a)):
            outputs.append(output_a[i] + output_b[i])
        return outputs

class SiameseTransformer2(nn.Module):
    def __init__(self, args, config):
        super(SiameseTransformer2, self).__init__()
        self.args = args
        self.model_a = AutoModel.from_pretrained(
            self.args.model_name, config=config, cache_dir=self.args.cache_dir
        )
        self.model_b = AutoModel.from_pretrained(
            self.args.model_name, config=config, cache_dir=self.args.cache_dir
        )
        self.linear_1 = nn.Linear(config.hidden_size*3, config.hidden_size)
        self.linear_2 = nn.Linear(config.hidden_size, config.num_labels)

        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, a, b):
        labels = a['labels']
        a.pop('labels')
        b.pop('labels')
        output_a = self.model_a(**a)[1]  # [bs, seq_len, 768]
        output_b = self.model_b(**b)[1]
        output = torch.cat([output_a, output_b, output_a-output_b], dim=1)
        output = self.linear_1(output)
        logits = self.linear_2(output)
        loss = self.loss_fct(logits, labels)
        return loss, logits
