from functools import partial

import torch
from torch import nn
from torchvision.models import resnet


# SplitBatchNorm: simulate multi-gpu behavior of BatchNorm in one gpu by splitting alone the batch dimension
# implementation adapted from https://github.com/davidcpage/cifar10-fast/blob/master/torch_backend.py
class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits

    def forward(self, inputs):
        N, C, H, W = inputs.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = nn.functional.batch_norm(
                inputs.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return nn.functional.batch_norm(
                inputs, self.running_mean, self.running_var,
                self.weight, self.bias, False, self.momentum, self.eps)


class ModelBase(nn.Module):

    def __init__(self, feature_dim=128, arch=None, bn_splits=16):
        super(ModelBase, self).__init__()

        # use split batchnorm
        norm_layer = partial(SplitBatchNorm, num_splits=bn_splits) if bn_splits > 1 else nn.BatchNorm2d
        resnet_arch = getattr(resnet, arch)
        net = resnet_arch(num_classes=feature_dim, norm_layer=norm_layer)

        self.net = []
        for name, module in net.named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if isinstance(module, nn.MaxPool2d):
                continue
            if isinstance(module, nn.Linear):
                self.net.append(nn.Flatten(1))
            self.net.append(module)

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        x = self.net(x)
        # note: not normalized here
        return x

class SingleGPUMoCo(nn.Module):
    def __init__(self, dim=128, K=4096, m=0.99, T=0.1, arch='resnet18', bn_splits=8, symmetric=True):
        super(SingleGPUMoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.symmetric = symmetric

        # create the encoders
        self.encoder_q = ModelBase(feature_dim=dim, arch=arch, bn_splits=bn_splits)
        self.encoder_k = ModelBase(feature_dim=dim, arch=arch, bn_splits=bn_splits)

        print(self.encoder_q)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("label_queue", torch.zeros(self.K).long() - 1)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        self.label_queue[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x, labels):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], labels[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, labels, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle], labels[idx_unshuffle]

    def forward(self, im_q, im_k, labels):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            labels: a batch of label for images (-1 for unsupervised images)
        Output:
            logits: with shape Nx(1+K)
            targets: with shape Nx(1+K)
            fake_targets: to report the top-1, top-5 accuracy as MoCo,
                          it returns the index of the data augmented image.
        """

        batch_size = labels.shape[0]

        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()
            im_k_, labels_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k, labels)

            k = self.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized

            # undo shuffle
            k, labels = self._batch_unshuffle_single_gpu(k, labels_, idx_unshuffle)

        logit_aug = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        logit_queue = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([logit_aug, logit_queue], dim=1) / self.T

        positive_target = torch.ones((batch_size, 1)).cuda()
        target_queue = labels[:, None] == self.label_queue[None, :]
        target_queue &= (labels[:, None] != -1)
        target_queue = target_queue.float().cuda()
        targets = torch.cat([positive_target, target_queue], dim=1)
        # targets = ((labels[:, None] == self.label_queue[None, :]) & (labels[:, None] != -1)).float().cuda()
        # targets = torch.cat([positive_target, targets], dim=1)

        self._dequeue_and_enqueue(k, labels)

        return logits, targets, torch.zeros(logits.shape[0], dtype=torch.long).cuda()  # to report the top-1, top-5


if __name__ == "__main__":
    pass
    # create model
    # model = SingleGPUMoCo(
    #         dim=128,
    #         K=4096,
    #         m=0.99,
    #         T=0.07,
    #         arch="resnet50",
    #         bn_splits=8,
    #         symmetric=False,
    #     ).cuda()
    # print(model.encoder_q)
    #
    # pseudo_inputs_1 = torch.rand((16, 3, 160, 160)).cuda()
    # pseudo_inputs_2 = torch.rand((16, 3, 160, 160)).cuda()
    # labels = torch.zeros((32, ))
    #
    # model(pseudo_inputs_1, pseudo_inputs_2, labels)