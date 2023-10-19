import datetime

import math
from yacs.config import CfgNode

configs = CfgNode()

# basic
configs.device = "cuda:0"
configs.seed = 42

# model
configs.model = CfgNode()
configs.model.base_model = "resnet18"
configs.model.proj_head = "linear"
configs.model.feat_dim = 128

# loss
configs.loss = CfgNode()
configs.loss.temperature = 0.07

# dataset
configs.dataset = CfgNode()
configs.dataset.batch_size = 16
configs.dataset.num_workers = 6

# opt
configs.opt = CfgNode()
configs.opt.epochs = 100
configs.opt.learning_rate = 0.05
configs.opt.momentum = 0.9
configs.opt.weight_decay = 1e-4
configs.opt.lr_decay_rate = 0.1
configs.opt.cosine = True

if configs.dataset.batch_size > 256:
    configs.opt.warm = True
else:
    configs.opt.warm = False

if configs.opt.warm:
    configs.opt.warm_epochs = 10
    configs.opt.warmup_from = 0.01
    if configs.opt.cosine:
        eta_min = configs.opt.learning_rate * (configs.opt.lr_decay_rate ** 3)
        configs.opt.warmup_to = eta_min + (configs.opt.learning_rate - eta_min) * (
                1 + math.cos(math.pi * configs.opt.warm_epochs / configs.opt.epochs)) / 2
    else:
        configs.opt.warmup_to = configs.opt.learning_rate

# log
t = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
configs.logs = CfgNode()
configs.logs.log_interval = 10
configs.logs.epoch_interval = 1
configs.logs.checkpoints_dir = f'/mnt/data/duongdhk/checkpoints/contrast/contrast_{t}'
configs.logs.log_folder = f'/mnt/data/duongdhk/logs/contrast/contrast_{t}'
configs.logs.log_file = f'{t}.log'
configs.logs.config_file = f'config.json'
configs.logs.artifact = f"contrastive_learning"
configs.logs.name = f"contrastive_learning"