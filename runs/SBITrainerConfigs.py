import datetime

from yacs.config import CfgNode

configs = CfgNode()

# basic
configs.device = "cuda:0"
configs.image_size = 256
configs.batch_size = 32
configs.dataset_path = "/mnt/data/duongdhk/datasets/FFPP"
configs.n_frames = 8
configs.num_workers = 6
configs.n_epochs = 100

# log
t = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
configs.logs = CfgNode()
configs.logs.log_interval = 10
configs.logs.epoch_interval = 1
configs.logs.checkpoints_dir = f'/mnt/data/duongdhk/checkpoints/sbi/sbi_{t}'
configs.logs.log_folder = f'/mnt/data1/duongdhk/logs/sbi/sbi_{t}'
configs.logs.log_file = f'{t}.log'
configs.logs.config_file = f'config.json'
configs.logs.artifact = f"sbi"
configs.logs.name = f"sbi"