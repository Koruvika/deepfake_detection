import torch
from torch import nn


class UnifiedContrastive(nn.Module):
    def __init__(self, reduction='mean'):
        super(UnifiedContrastive, self).__init__()
        self.reduction = reduction

    def forward(self, logits, target):
        sum_neg = ((1 - target) * torch.exp(logits)).sum(1)
        sum_pos = (target * torch.exp(-logits)).sum(1)
        loss = torch.log(1 + sum_neg * sum_pos)
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss