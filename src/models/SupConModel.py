import torch
from torch import nn
import torch.nn.functional as F
import timm
# from .ResNet import model_dict

model_dict = {
    'resnet18': [timm.create_model('resnet18', num_classes=512), 512],
    'resnet34': [timm.create_model('resnet34', num_classes=1024), 1024],
    'resnet50': [timm.create_model('resnet50', num_classes=2048), 2048],
    'resnet101': [timm.create_model('resnet101', num_classes=2048), 2048],
}

class SupConResNet(nn.Module):
    """backbone + projection head"""

    def __init__(self, name='resnet34', head='linear', feat_dim=128, device="gpu"):
        super(SupConResNet, self).__init__()
        model, dim_in = model_dict[name]
        self.encoder = model
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


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc1 = nn.Linear(feat_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, features):
        out = self.fc1(features)
        out = self.fc2(out)
        return out