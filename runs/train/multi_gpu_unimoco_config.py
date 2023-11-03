import math
from yacs.config import CfgNode

import datetime

## BASIC
configs = CfgNode()
configs.seed = 42

configs.num_workers = 8
configs.world_size = 1
configs.rank = 0
configs.dist_url = "tcp://localhost:10001"
configs.dist_backend = "nccl"
configs.device = [0, 1]
configs.multiprocessing_distributed = True

configs.knn_k = 200
configs.knn_t = 0.1


## DATASET
configs.dataset = CfgNode()
configs.dataset.batch_size = 16
configs.dataset.train_root = "/mnt/data/duongdhk/datasets/processed_deepfake_detection_dataset/processed_FFPP"
configs.dataset.test_root = "/mnt/data/duongdhk/datasets/processed_deepfake_detection_dataset/Celeb-DF-v2/images"

## MODEL
configs.model = CfgNode()
configs.model.dim = 128  # change it
configs.model.K = 4096  # change it
configs.model.m = 0.999
configs.model.T = 0.07
configs.model.arch = "resnet18"  # change it
configs.model.bn_splits = 4  # change it

## OPTIMIZER
configs.optimizer = CfgNode()
configs.optimizer.lr = 0.03  * configs.dataset.batch_size / 256  # 0.1
configs.optimizer.momentum = 0.9
configs.optimizer.weight_decay = 1e-4
configs.optimizer.n_epochs = 200
configs.optimizer.start_epoch = 0

# Scheduler
configs.optimizer.cosine = True
configs.optimizer.lr_decay_rate = 0.1
configs.optimizer.lr_decay_schedule = [120, 160]

if configs.dataset.batch_size > 256:
    configs.optimizer.warm = True
    configs.optimizer.warm_epochs = 10
    configs.optimizer.warmup_from = 0.01
    if configs.optimizer.cosine:
        eta_min = configs.optimizer.lr * (configs.optimizer.lr_decay_rate ** 3)
        configs.optimizer.warmup_to = eta_min + (configs.optimizer.lr - eta_min) * (
                1 + math.cos(math.pi * configs.optimizer.warm_epochs / configs.optimizer.n_epochs)) / 2
    else:
        configs.optimizer.warmup_to = configs.optimizer.lr
else:
    configs.optimizer.warm = False



# LOG
configs.logs = CfgNode()
t = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
configs.logs = CfgNode()
configs.logs.time = t
configs.logs.log_interval = 10
configs.logs.epoch_interval = 1
configs.logs.checkpoints_dir = f'/mnt/data/duongdhk/checkpoints/contrast/unimoco_{t}'
configs.logs.log_folder = f'/mnt/data/duongdhk/logs/contrast/unimoco_{t}'
configs.logs.log_file = f'{t}.log'
configs.logs.config_file = f'config.json'
configs.logs.artifact = f"UniMoCo with Single GPU"
configs.logs.name = f"UniMoCo with Single GPU"

# RESUME
configs.resume = None

