import torch
from torch import nn
import torch.nn.functional as F

from .ResNet import model_dict

class SupConResNet(nn.Module):
    """backbone + projection head"""

    def __init__(self, name='resnet34', head='linear', feat_dim=128, device="gpu"):
        super(SupConResNet, self).__init__()
        model, dim_in = model_dict[name]
        self.encoder = model()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

        self.encoder.to(device)
        self.head.to(device)

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat