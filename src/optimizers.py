import math
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.5, **kwargs):
        assert rho >= 0.0

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups


    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(torch.stack(
            [p.grad.norm(p=2).to(shared_device) for group in self.param_groups for p in group["params"] if
             p.grad is not None]), p=2)
        return norm

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()


class LinearDecayLR(_LRScheduler):
    def __init__(self, optimizer, n_epoch, start_decay, last_epoch=-1):
        self.start_decay=start_decay
        self.n_epoch=n_epoch
        super(LinearDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        last_epoch = self.last_epoch
        n_epoch=self.n_epoch
        b_lr=self.base_lrs[0]
        start_decay=self.start_decay
        if last_epoch>start_decay:
            lr=b_lr-b_lr/(n_epoch-start_decay)*(last_epoch-start_decay)
        else:
            lr=b_lr
        return [lr]

# def adjust_learning_rate(configs, optimizer, epoch):
#     lr = configs.learning_rate
#     if configs.cosine:
#         eta_min = lr * (configs.lr_decay_rate ** 3)
#         lr = eta_min + (lr - eta_min) * (
#                 1 + math.cos(math.pi * epoch / configs.epochs)) / 2
#     else:
#         steps = np.sum(epoch > np.asarray(configs.lr_decay_epochs))
#         if steps > 0:
#             lr = lr * (configs.lr_decay_rate ** steps)
#
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def adjust_learning_rate(optimizer, epoch, learning_rate, cosine, lr_decay_rate, n_epochs, lr_decay_epochs):
    if cosine:
        eta_min = learning_rate * (lr_decay_rate ** 3)
        learning_rate = eta_min + (learning_rate - eta_min) * (1 + math.cos(math.pi * epoch / n_epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(lr_decay_epochs))
        if steps > 0:
            learning_rate = learning_rate ** (lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def warmup_learning_rate(optimizer, epoch, batch_id, total_batches, warm, warm_epochs, warmup_from, warmup_to):
    if warm and epoch <= warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / (warm_epochs * total_batches)
        lr = warmup_from + p * (warmup_to - warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


