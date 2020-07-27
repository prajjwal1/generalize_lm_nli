import torch
import torch.nn as nn


class Bottle(nn.Module):
    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0] * size[1], -1))
        return out.view(size[0], size[1], -1)


class Linear(Bottle, nn.Linear):
    pass


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        input_size = config.d_proj if config.projection else config.d_embed
        dropout = 0 if config.n_layers == 1 else config.dp_ratio
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=config.d_hidden,
            num_layers=config.n_layers,
            dropout=dropout,
            bidirectional=config.birnn,
        )

    def forward(self, inputs):
        batch_size = inputs.size()[1]
        state_shape = self.config.n_cells, batch_size, self.config.d_hidden
        h0 = c0 = inputs.new_zeros(state_shape)
        self.rnn.flatten_parameters()
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        return (
            ht[-1]
            if not self.config.birnn
            else ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)
        )


class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.d_embed)
        self.projection = Linear(config.d_embed, config.d_proj)
        self.encoder = Encoder(config)

    def forward(self, batch):
        embed = self.embed(batch)
        if self.config.projection:
            embed = self.relu(self.projection(embed))
        embed = self.encoder(embed)
        return embed
