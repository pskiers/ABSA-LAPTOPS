import torch
import torch.nn as nn
from transformers import BertModel


class BasicBert(nn.Module):

  def __init__(self, n_classes):
    super().__init__()
    self.bert = BertModel.from_pretrained("bert-base-cased")
    self.bert.requires_grad_(False)
    self.drop = nn.Dropout(p=0.6)

    self.lin1 = nn.Linear(768, 256)
    self.drp1 = nn.Dropout(0.6)
    self.bn1 = nn.BatchNorm1d(256)

    self.lin2 = nn.Linear(256, 1)
    self.drp2 = nn.Dropout(0.6)
    self.bn2 = nn.BatchNorm1d(1)

    self.output = nn.Linear(75 + 4, n_classes)

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

    out = self.lin1(torch.concat([aspect, context], dim=1))
    out = nn.functional.relu(out)
    out = self.drp1(out)
    out = self.bn1(torch.swapaxes(out, 1, 2))

    out = self.lin2(torch.swapaxes(out, 1, 2))
    out = nn.functional.relu(out)
    out = self.drp2(out)
    out = self.bn2(torch.swapaxes(out, 1, 2))

    return self.output(out)