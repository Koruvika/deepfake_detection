import json
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, "")
from src import SBIFaceForencisDataset, SBIDetector, LinearDecayLR, compute_accuray
from runs.SBITrainerConfigs import configs as SBITrainerConfig


class SBITrainer:
    def __init__(self, configs=SBITrainerConfig):
        self.configs = configs
        self.device = torch.device(configs.device)

        self.best_val_auc = 0.

        seed = 5
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.init_model()
        self.init_optimizer()
        self.init_data()
        self.init_log()

    def init_model(self):
        self.model = SBIDetector().to(self.device)

    def init_optimizer(self):
        self.scheduler = LinearDecayLR(self.model.optimizer, self.configs.n_epochs, int(self.configs.n_epochs / 4 * 3))
        self.criterion = nn.CrossEntropyLoss()

    def init_data(self):
        train_dataset = SBIFaceForencisDataset(phase='train', dataset_path=self.configs.dataset_path,
                                               image_size=self.configs.image_size, n_frames=self.configs.n_frames)

        val_dataset = SBIFaceForencisDataset(phase='val', dataset_path=self.configs.dataset_path,
                                             image_size=self.configs.image_size, n_frames=self.configs.n_frames)

        self.train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.configs.batch_size // 2,  # with each batch load 2 images: one fake & one real
            shuffle=True,
            collate_fn=train_dataset.collate_fn,
            num_workers=self.configs.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=train_dataset.worker_init_fn
        )
        self.val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=self.configs.batch_size,
            shuffle=False,
            collate_fn=val_dataset.collate_fn,
            num_workers=self.configs.num_workers,
            pin_memory=True,
            worker_init_fn=train_dataset.worker_init_fn
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
            project=f"Self-blending Images Trainer",
            config=dict(self.configs),
            id=self.configs.logs.name,
            entity="duongnpc239"
        )

        config_json = json.dumps(self.configs, indent=4)
        with open(self.config_file, "w") as f:
            f.write(config_json)

        self.writer = SummaryWriter(self.tensorboard)

    def train(self, epoch):
        """
        Train for a epoch
        """

        self.model.train()
        for i, data in enumerate(self.train_dataloader):
            image = data["img"].to(self.device, non_blocking=True)
            target = data["label"].to(self.device, non_blocking=True)
            output = self.model.training_step(image, target)
            loss = self.criterion(output, target)
            acc = compute_accuray(F.log_softmax(output, dim=1), target)

            if i % self.configs.logs.log_interval == 0:
                logging.info(
                    f"Epoch {epoch} - {i}/{len(self.train_dataloader)}: Loss: {loss.item()}, Accuracy: {acc}"
                )

        self.scheduler.step()

    def validate(self, dataloader):
        """
        Validate (train/val) dataloader
        Args:
            dataloader: train or val dataloader
        Returns:

        """
        self.model.eval()
        val_losses, val_accs = [], []
        output_dict, target_dict = [], []
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                image = data["img"].to(self.device, non_blocking=True)
                target = data["label"].to(self.device, non_blocking=True)
                output = self.model(image)
                loss = self.criterion(output, target)
                acc = compute_accuray(F.log_softmax(output, dim=1), target)

                val_losses.append(loss.item())
                val_accs.append(acc)
                output_dict += output.softmax(1)[:, 1].cpu().data.numpy().tolist()
                target_dict += target.cpu().data.numpy().tolist()

        val_loss = np.mean(val_losses)
        val_acc = np.mean(val_accs)
        val_auc = roc_auc_score(target_dict, output_dict)
        return val_loss, val_acc, val_auc

    def log(self, epoch, losses):
        if epoch % self.configs.logs.epoch_interval == 0:
            filename = os.path.join(str(self.configs.logs.checkpoints_dir), "checkpoint_epoch_" + str(epoch) + '.pth')
            save_checkpoint({
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optimizer': self.model.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'losses': losses,
            }, filename)

        if losses["val_auc"] < self.best_val_auc:
            filename = os.path.join(str(self.configs.logs.checkpoints_dir), "checkpoint_best_val_auc.pth")
            save_checkpoint({
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optimizer': self.model.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'losses': losses,
            }, filename)
            art = wandb.Artifact(f"{self.configs.logs.artifact}-{wandb.run.id}-best_model", type="model")
            art.add_file(os.path.join(str(self.configs.logs.checkpoints_dir), "checkpoint_best_val_auc.pth"))
            wandb.log_artifact(art)
            self.best_val_auc = losses["val_auc"]

        # log
        self.writer.add_scalar("Train/Loss", losses["train_loss"], epoch)
        self.writer.add_scalar("Train/Accuracy", losses["train_acc"], epoch)
        self.writer.add_scalar("Train/Area Under Curve", losses["train_auc"], epoch)
        self.writer.add_scalar("Validate/Loss", losses["val_loss"], epoch)
        self.writer.add_scalar("Validate/Accuracy", losses["val_acc"], epoch)
        self.writer.add_scalar("Validate/Area Under Curve", losses["val_auc"], epoch)

        wandb.log({
            "Train/Loss": losses["train_loss"],
            "Train/Accuracy": losses["train_acc"],
            "Train/Area Under Curve": losses["train_auc"],
            "Validate/Loss": losses["val_loss"],
            "Validate/Accuracy": losses["val_acc"],
            "Validate/Area Under Curve": losses["val_auc"],
        })

        logging.info(
            f"Epoch: {epoch}, "
            f"Train/Loss: {losses['train_loss']}, Train/Accuracy: {losses['train_acc']}, Train/Area Under Curve: {losses['train_auc']}, "
            f"Validate/Loss: {losses['val_loss']}, Validate/Accuracy: {losses['val_acc']}, Validate/Area Under Curve: {losses['val_auc']}"
        )

    def run(self):
        for epoch in range(self.configs.n_epochs):
            self.train(epoch)
            train_loss, train_acc, train_auc = self.validate(self.train_dataloader)
            val_loss, val_acc, val_auc = self.validate(self.val_dataloader)
            losses = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_auc": train_auc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_auc": val_auc,
            }
            self.log(epoch, losses)
            # self.scheduler.step()
        self.writer.close()


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)
    logging.info('Save checkpoint to {0:}'.format(filename))


def main():
    trainer = SBITrainer()
    trainer.run()


if __name__ == "__main__":
    main()
