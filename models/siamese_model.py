import logging

import torch
from torch import nn
from transformers import AutoModel


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class SiameseTransformer(nn.Module):
    def __init__(self, args, config):
        super(SiameseTransformer, self).__init__()
        self.model_a = AutoModel.from_pretrained(args.model_name, config=config, cache_dir=args.cache_dir)
        self.model_b = AutoModel.from_pretrained(args.model_name, config=config, cache_dir=args.cache_dir)

        logging.info("**** Encoder will not be trained ****")
        for param in self.model_a.parameters():
            param.requires_grad = False
        for param in self.model_b.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(config.hidden_size*3, config.num_labels)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, a, b):
        labels = a.pop('labels')
        b.pop('labels')
        output_a = self.model_a(**a)
        output_b = self.model_b(**b)
        embeddings_a = mean_pooling(output_a,  a['attention_mask'])
        embeddings_b = mean_pooling(output_b, b['attention_mask'])
        output = torch.cat([embeddings_a, embeddings_b, embeddings_a-embeddings_b], dim=1)
        logits = self.classifier(output)
        loss = self.criterion(logits, labels)
        return loss, logits
