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

sys.path.insert(0, "")
from runs.supcon_config import configs as SupConTrainerConfig
from src.models import SupConResNet
from src.losses import SupConLoss
from src.optimizers import adjust_learning_rate, warmup_learning_rate
from src.datasets import SupConDataset

class SupConTrainer:
    def __init__(self, configs=SupConTrainerConfig):
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
        self.model = SupConResNet(self.configs.model.base_model, self.configs.model.proj_head, self.configs.model.feat_dim, device=self.device)
        self.criterion = SupConLoss(temperature=self.configs.loss.temperature)
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=self.configs.opt.learning_rate,
                                   momentum=self.configs.opt.momentum,
                                   weight_decay=self.configs.opt.weight_decay)
        self.scheduler = [adjust_learning_rate, warmup_learning_rate]

    def init_data(self):
        # TODO: add contrast transform here
        dataset = SupConDataset(self.configs.dataset, "train", None)
        n = len(dataset)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, (n - n//5, n//5))
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.configs.dataset.batch_size,
            num_workers=self.configs.dataset.num_workers,
            shuffle=True,
            drop_last=False
        )
        self.val_dataloader = DataLoader(
            train_dataset,
            batch_size=self.configs.dataset.batch_size,
            num_workers=self.configs.dataset.num_workers,
            shuffle=False,
            drop_last=False
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
            project=f"Deepfake Detection with Contrastive Learning",
            config=dict(self.configs),
            id=self.configs.logs.name,
            entity="duongnpc239"
        )

        config_json = json.dumps(self.configs, indent=4)
        with open(self.config_file, "w") as f:
            f.write(config_json)

        self.writer = SummaryWriter(self.tensorboard)

    def train(self, epoch):
        self.model.train()

        for i, (images, labels, _) in enumerate(self.train_dataloader):
            images = torch.cat([images[0], images[1]], dim=0)
            images = images.to(self.device)
            labels = labels.to(self.device)

            warmup_learning_rate(self.configs.opt, epoch, i, len(self.train_dataloader), self.optimizer)

            features = self.model(images)
            f1, f2 = torch.split(features, [self.configs.dataset.batch_size, self.configs.dataset.batch_size], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = self.criterion(features, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


            if i % self.configs.logs.log_interval == 0:
                logging.info(
                    f"Epoch {epoch} - {i}/{len(self.train_dataloader)}: Loss: {loss.item()}"
                )


    def validate(self, validation_set):
        self.model.eval()
        val_losses = []
        with torch.no_grad():
            for i, (images, labels, _) in enumerate(validation_set):
                images = torch.cat([images[0], images[1]], dim=0)

                images = images.to(self.device)
                labels = labels.to(self.device)

                features = self.model(images)
                f1, f2 = torch.split(features, [self.configs.dataset.batch_size, self.configs.dataset.batch_size], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                loss = self.criterion(features, labels)
                val_losses.append(loss.item())

        val_loss = np.mean(val_losses)
        return val_loss


    def log(self, epoch, losses):
        if epoch % self.configs.logs.epoch_interval == 0:
            filename = os.path.join(str(self.configs.logs.checkpoints_dir), "checkpoint_epoch_" + str(epoch) + '.pth')
            save_checkpoint({
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'losses': losses,
            }, filename)

        if losses["val_loss"] < self.best_val_loss:
            filename = os.path.join(str(self.configs.logs.checkpoints_dir), "checkpoint_best_val_loss.pth")
            save_checkpoint({
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'losses': losses,
            }, filename)

            art = wandb.Artifact(f"{self.configs.logs.artifact}-{wandb.run.id}-best_model", type="model")
            art.add_file(os.path.join(str(self.configs.logs.checkpoints_dir), "checkpoint_best_val_loss.pth"))
            wandb.log_artifact(art)
            self.best_val_loss = losses["val_loss"]

        # log
        self.writer.add_scalar("Train/Loss", losses["train_loss"], epoch)
        self.writer.add_scalar("Validate/Loss", losses["val_loss"], epoch)

        wandb.log({
            "Train/Loss": losses["train_loss"],
            "Validate/Loss": losses["val_loss"],
        })

        logging.info(
            f"Epoch: {epoch}, Train/Loss: {losses['train_loss']}, Validate/Loss: {losses['val_loss']}"
        )

    def run(self):
        for epoch in range(self.configs.opt.epochs):
            adjust_learning_rate(self.configs.opt, self.optimizer, epoch)
            self.train(epoch)
            train_loss = self.validate(self.train_dataloader)
            val_loss = self.validate(self.val_dataloader)
            losses = {
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            self.log(epoch, losses)
        self.writer.close()


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)
    logging.info('Save checkpoint to {0:}'.format(filename))


def main():
    trainer = SupConTrainer()
    trainer.run()


if __name__ == "__main__":
    main()