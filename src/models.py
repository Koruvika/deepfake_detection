import torch
from efficientnet_pytorch import EfficientNet
from torch import nn

from src import SAM


class SBIDetector(nn.Module):
    def __init__(self, base_optimizer=torch.optim.SGD, lr=0.001, momentum=0.9):
        super(SBIDetector, self).__init__()
        self.net = EfficientNet.from_pretrained("efficientnet-b4", advprop=True, num_classes=2)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = self.optimizer = SAM(self.parameters(), base_optimizer, lr=lr, momentum=momentum)

    def forward(self, inputs):
        x = self.net(inputs)
        return x

    def train_step(self, inputs, target):

        # first step
        pred = self(inputs)
        pred_first = pred.clone()
        loss = self.loss_fn(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.first_step(zero_grad=True)

        # second step
        pred = self(inputs)
        loss = self.loss_fn(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.second_step(zero_grad=True)

        return pred_first