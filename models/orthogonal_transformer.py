import torch
import torch.nn.functional as F
from torch import nn


class HEXProjection(nn.Module):
    def __init__(self, config):
        super(HEXProjection, self).__init__()
        self.standardize_dim = nn.Linear(config.hidden_size, config.batch_size)
        self.inverse_param = nn.Parameter(torch.rand(1))

    def forward(self, x, y):
        x = self.standardize_dim(x)
        y = self.standardize_dim(y)

        x = x / x.max(0, keepdim=True)[0]
        y = y / y.max(0, keepdim=True)[0]
        F_a = torch.cat([x, y], dim=1)
        F_p = torch.cat([torch.zeros_like(x), x], dim=1)
        F_g = torch.cat([y, torch.zeros_like(y)], dim=1)
        #  print(F_g)
        internal_prod = torch.matmul(F_g.t(), F_g)
        print(internal_prod)
        inverse_inside = torch.inverse(
            internal_prod + self.inverse_param * torch.eye(*internal_prod.shape)
        )

        F_l = torch.eye(*F_a.shape) - torch.matmul(
            torch.matmul(torch.matmul(F_g, inverse_inside), F_g.t()), F_a,
        )
        return F_l


class OrthogonalTransformer(nn.Module):
    def __init__(self, network_a, network_b, config):
        super(OrthogonalTransformer, self).__init__()
        self.network_a = network_a
        self.network_b = network_b
        self.hex = HEXProjection(config)
        self.out = nn.Linear(config.batch_size, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_tuple=None,
    ):

        output_a = self.network_a(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )[1]
        # loss_a, output_a_logits = output_a[0], output_a[1]

        output_b = self.network_b(input_ids, token_type_ids)

        if self.train():
            projected_logits = self.hex(output_a, output_b)
        elif self.eval():
            return output_a

        output = self.out(projected_logits.t())
        loss = self.loss_fct(output, labels)
        return loss, output
