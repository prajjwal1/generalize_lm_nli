import torch
from torch import nn


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class SBERT(nn.Module):
    def __init__(self, model, config):
        super(SBERT, self).__init__()
        self.encoder = model
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, **inputs):
        labels = inputs.pop('labels')
        model_output = self.encoder(**inputs)
        sentence_embeddings = mean_pooling(model_output,  inputs['attention_mask'])
        logits = self.classifier(sentence_embeddings)
        loss = self.criterion(logits, labels)
        return loss, logits
