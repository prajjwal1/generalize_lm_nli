from torch import nn
from transformers.modeling_bert import BertEmbeddings, BertSelfAttention


class CBOW(nn.Module):
    def __init__(self, config):
        super(CBOW, self).__init__()

        self.embeddings = BertEmbeddings(config)
        self.attention = BertSelfAttention(config)
        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.act_fn = nn.ReLU()
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, input_ids, token_type_ids):
        embeds = self.embeddings(input_ids, token_type_ids)
        out = self.attention(embeds)[0].sum(1)
        out = self.act_fn(out)
        out = self.linear2(out)
        return out
