import json
import logging
import os
import random
import sys

import numpy as np
import torch
import wandb
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

sys.path.insert(0, "")
from runs.train.unimoco_config import configs as UniMoCoConfig
from src.models import SingleGPUMoCo
from src.losses import UnifiedContrastive
from src.datasets import SupConDataset
from src.transforms import GaussianBlur
from src.optimizers import adjust_learning_rate, warmup_learning_rate


# TODO 1: Implement training code for UniMoCo
    # TODO 1.1: Understand UniMoCo
    # TODO 1.2: Understand Split Batch Normalization
# TODO 2: Deal with problem about CelebDF dataset


class UniMoCoTrainer:
    def __init__(self, configs=UniMoCoConfig):
        self.configs = configs
        self.device = torch.device(self.configs.device)

        self.best_val_loss = 1e6

        self.init_seed()
        self.init_model()
        self.init_data()
        self.init_log()

    def init_seed(self):
        self.seed = self.configs.seed
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def init_model(self):
        self.model = SingleGPUMoCo(
            dim=self.configs.model.dim,
            K=self.configs.model.K,
            m=self.configs.model.m,
            T=self.configs.model.T,
            arch=self.configs.model.arch,
            bn_splits=self.configs.model.bn_splits,
        ).to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=self.configs.optimizer.lr,
                                   momentum=self.configs.optimizer.momentum,
                                   weight_decay=self.configs.optimizer.weight_decay)

        self.criterion = UnifiedContrastive().to(self.device)

    def init_data(self):
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(160, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = SupConDataset(self.configs.dataset, "train", train_transform)
        self.train_dataloader = DataLoader(
            dataset,
            batch_size=self.configs.dataset.batch_size,
            num_workers=self.configs.num_workers,
            shuffle=True,
            drop_last=True
        )

    def init_log(self):
        os.makedirs(self.configs.logs.log_folder, exist_ok=True)
        os.makedirs(self.configs.logs.checkpoints_dir, exist_ok=True)

        self.log_file = os.path.join(self.configs.logs.log_folder, self.configs.logs.log_file)
        self.tensorboard = os.path.join(self.configs.logs.log_folder, "tensorboard")
        self.config_file = os.path.join(self.configs.logs.log_folder, self.configs.logs.config_file)

        logging.basicConfig(
            format='[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
            level=logging.INFO,
            handlers=[
                logging.FileHandler(self.log_file, mode='w'),
                logging.StreamHandler()
            ]
        )

        wandb.init(
            project=f"Deepfake Detection with UniMoCo on Single GPU",
            config=dict(self.configs),
            id = self.configs.logs.time,
            entity="duongnpc239",
            dir="/mnt/data/duongdhk/tmp/wandb"
        )

        config_json = json.dumps(self.configs, indent=4)
        with open(self.config_file, "w") as f:
            f.write(config_json)

        self.writer = SummaryWriter(self.tensorboard)

    def train(self, epoch):
        self.model.train()

        losses = []
        top1 = []
        top5 = []

        for i, (images, labels, _) in enumerate(self.train_dataloader):
            images[0] = images[0].to(self.device)
            images[1] = images[1].to(self.device)
            labels = labels.to(self.device)
            output, target, fake_targets = self.model(im_q=images[0], im_k=images[1], labels=labels)

            loss = self.criterion(output, target)

            losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % self.configs.logs.log_interval == 0:
                logging.info(
                    f"Epoch {epoch} - {i}/{len(self.train_dataloader)}: Loss: {loss.item()}"
                )

        return np.mean(losses)


    def validate(self, validate_set):
        pass

    def log(self, losses, epoch):
        if epoch % self.configs.logs.epoch_interval == 0:
            filename = os.path.join(str(self.configs.logs.checkpoints_dir), "checkpoint_epoch_" + str(epoch) + '.pth')
            save_checkpoint({
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'losses': losses,
            }, filename)

        if losses["train_loss"] < self.best_val_loss:
            filename = os.path.join(str(self.configs.logs.checkpoints_dir), "checkpoint_best_loss.pth")
            save_checkpoint({
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'losses': losses,
            }, filename)
            self.best_val_loss = losses["train_loss"]

        # log
        self.writer.add_scalar("Train/Loss", losses["train_loss"], epoch)

        wandb.log({
            "Train/Loss": losses["train_loss"],
        })

        logging.info(
            f"Epoch: {epoch}, Train/Loss: {losses['train_loss']}"
        )

    def run(self):

        for epoch in range(1, self.configs.optimizer.n_epochs + 1):
            adjust_learning_rate(
                self.optimizer,
                epoch,
                self.configs.optimizer.lr,
                self.configs.optimizer.cosine,
                self.configs.optimizer.lr_decay_rate,
                self.configs.optimizer.n_epochs,
                self.configs.optimizer.lr_decay_schedule,
            )
            train_loss = self.train(epoch)
            losses = {
                "train_loss": train_loss,
            }
            self.log(losses, epoch)
        self.writer.close()

def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)
    logging.info('Save checkpoint to {0:}'.format(filename))


def main():
    trainer = UniMoCoTrainer()
    trainer.run()


if __name__ == "__main__":
    main()