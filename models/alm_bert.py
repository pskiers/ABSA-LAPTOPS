import torch
import torch.nn as nn
from transformers import BertModel


class ALMBert(nn.Module):

  def __init__(self, n_classes, dropout):
    super().__init__()
    self.bert = BertModel.from_pretrained("bert-base-cased")
    self.bert.requires_grad_(False)
    self.drop = nn.Dropout(p=dropout)

    self.ctx_att = nn.MultiheadAttention(embed_dim=self.bert.config.hidden_size, num_heads=4, dropout=dropout, batch_first=True)
    self.ctx_bn1 = nn.BatchNorm1d(768)
    self.ctx_lin = nn.Linear(768, 64)
    self.ctx_drp = nn.Dropout(dropout)
    self.ctx_bn2 = nn.BatchNorm1d(64)
    self.ctx_avg = nn.AvgPool1d(64)

    self.asp_att = nn.MultiheadAttention(embed_dim=self.bert.config.hidden_size, num_heads=4, dropout=dropout, batch_first=True)
    self.asp_bn1 = nn.BatchNorm1d(768)
    self.asp_lin = nn.Linear(768, 64)
    self.asp_drp = nn.Dropout(dropout)
    self.asp_bn2 = nn.BatchNorm1d(64)
    self.asp_avg = nn.AvgPool1d(64)

    self.linear = nn.Linear(int(75 + 75), n_classes)

  def forward(self, text_ids, text_attention, aspect_ids, aspect_attention):
    context, _ = self.bert(
      input_ids=text_ids,
      attention_mask=text_attention,
      return_dict=False
    )
    aspect, _ = self.bert(
      input_ids=aspect_ids,
      attention_mask=aspect_attention,
      return_dict=False
    )
    asp_out, _ = self.asp_att(context, aspect, aspect)
    asp_out = self.asp_bn1(torch.swapaxes(asp_out, 1, 2))
    asp_out = self.asp_lin(torch.swapaxes(asp_out, 1, 2))
    asp_out = nn.functional.relu(asp_out)
    asp_out = self.asp_drp(asp_out)
    asp_out = self.asp_bn2(torch.swapaxes(asp_out, 1, 2))
    asp_out = self.asp_avg(torch.swapaxes(asp_out, 1, 2)).squeeze(dim=2)

    ctx_out, _ = self.ctx_att(context, context, context)
    ctx_out = self.ctx_bn1(torch.swapaxes(ctx_out, 1, 2))
    ctx_out = self.ctx_lin(torch.swapaxes(ctx_out, 1, 2))
    ctx_out = nn.functional.relu(ctx_out)
    ctx_out = self.ctx_drp(ctx_out)
    ctx_out = self.ctx_bn2(torch.swapaxes(ctx_out, 1, 2))
    ctx_out = self.ctx_avg(torch.swapaxes(ctx_out, 1, 2)).squeeze(dim=2)

    out = torch.concat([asp_out, ctx_out], dim=1)

    return self.linear(out)