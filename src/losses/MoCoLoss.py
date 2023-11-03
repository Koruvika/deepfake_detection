import torch
from torch import nn


class UnifiedContrastive(nn.Module):
    def __init__(self, reduction='mean'):
        super(UnifiedContrastive, self).__init__()
        self.reduction = reduction

    def forward(self, logits, target):
        """

        Args:
            logits: similarity between query and samples in the queue
            target: Check if the query has the same label as the samples in the queue --> one-hot encoding

        Returns:

        """
        sum_neg = ((1 - target) * torch.exp(logits)).sum(1)
        sum_pos = (target * torch.exp(-logits)).sum(1)
        loss = torch.log(1 + sum_neg * sum_pos)
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss