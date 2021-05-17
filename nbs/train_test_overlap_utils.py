import math
from collections import Counter
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertEmbeddings, BertSelfAttention
from transformers import glue_compute_metrics
from sentence_transformers import util

class CBOW(nn.Module):
    def __init__(self, config):
        super(CBOW, self).__init__()

        self.embeddings = BertEmbeddings(config)
        self.attention = BertSelfAttention(config)
        self.act_fn = nn.ReLU()
        self.linear_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_3 = nn.Linear(config.hidden_size, 3)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids, token_type_ids, labels, **kwargs):
        embeds = self.embeddings(input_ids, token_type_ids)
        out = self.attention(embeds)[0].sum(1)
        out = self.linear_1(out)
        out = self.linear_2(F.relu(out))
        out = self.linear_3(F.relu(out))
        loss = self.loss_fct(out, labels)
        return loss, out
    
def build_compute_metrics_fn(task_name):
    def compute_metrics_fn(p):
        preds = np.argmax(p.predictions, axis=1)
        return glue_compute_metrics('mnli', preds, p.label_ids)
    return compute_metrics_fn

# def cosine_similarity(sentence_a, sentence_b):
#     embeddings1 = model.encode(sentence_a, device = device)
#     embeddings2 = model.encode(sentence_b, device = device)
#     cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2).item()
#     return cosine_scores

def cosine_similarity(embeddings_1, sentence_b, model):
    if model is not None:
        #  embeddings1 = model.encode(sentence_a, device = device)
        embeddings_2 = model.encode(sentence_b, device='cuda')
        cosine_scores = util.pytorch_cos_sim(embeddings_1, embeddings_2)
        return cosine_scores

    vec1 = Counter(embeddings_1)
    vec2 = Counter(sentence_b)
    
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    return float(numerator) / denominator
