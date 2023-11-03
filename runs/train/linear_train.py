import json
import logging
import os

import torch
import random
import numpy as np
import wandb
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from linear_config import configs as LinearConfigs
import sys


sys.path.insert(0, "")
from src.models import SupConResNet, LinearClassifier
from src.datasets import LinearDataset, CelebValidateDataset
from src.metrics import compute_accuracy
from src.optimizers import adjust_learning_rate, warmup_learning_rate


class LinearTrainer:
    def __init__(self, configs=LinearConfigs):
        self.configs = configs
        self.device = torch.device(self.configs.device)

        self.best_test_accuracy = 0.

        self.init_seed()
        self.init_model()
        self.init_dataset()
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
        # init model
        self.model = SupConResNet(self.configs.model.base_model, self.configs.model.proj_head, self.configs.model.feat_dim, device=self.device)
        self.classifier = LinearClassifier(self.configs.model.base_model, 2).cuda(self.configs.device)

        # load state dict
        checkpoint = torch.load(self.configs.model.ckpt, map_location=self.configs.device)
        self.model.load_state_dict(checkpoint["model"])

        # cross-entropy loss
        self.criterion = nn.CrossEntropyLoss().cuda(self.configs.device)

        # init optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=self.configs.opt.learning_rate,
                                         momentum=self.configs.opt.momentum,
                                         weight_decay=self.configs.opt.weight_decay)


    def init_dataset(self):
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        val_transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_dataset = LinearDataset(self.configs.dataset, "train", train_transform)
        test_dataset = CelebValidateDataset(self.configs.dataset, "test", val_transform)
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.configs.dataset.batch_size,
            num_workers=self.configs.dataset.num_workers,
            shuffle=True,
            drop_last=False
        )

        self.test_dataloader = DataLoader(
            test_dataset,
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
            project=f"Deepfake Detection with Linear Classifier",
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
        self.classifier.train()

        # for p in self.model.parameters():
        #     p.requires_grad = False

        losses = []
        accuracies = []
        for i, (images, labels) in enumerate(self.train_dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            warmup_learning_rate(self.configs.opt, epoch, i, len(self.train_dataloader), self.optimizer)

            # with torch.no_grad():
            features = self.model.encoder(images)
            output = self.classifier(features)
            loss = self.criterion(output, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            acc = compute_accuracy(output, labels)

            losses.append(loss.item())
            accuracies.append(acc)

            if i % self.configs.logs.log_interval == 0:
                logging.info(
                    f"Epoch {epoch} - {i}/{len(self.train_dataloader)}: Loss: {loss.item()}, Acc: {acc}"
                )

        return np.mean(losses), np.mean(accuracies)

    def validate(self, validation_set):
        self.model.eval()
        self.classifier.eval()

        accuracies = []
        losses = []

        with torch.no_grad():
            for i, (images, labels) in enumerate(validation_set):
                images = images.to(self.device)
                labels = labels.to(self.device)

                features = self.model.encoder(images)
                output = self.classifier(features.detach())
                loss = self.criterion(output, labels)

                acc = compute_accuracy(output, labels)

                losses.append(loss.item())
                accuracies.append(acc)

        return np.mean(losses), np.mean(accuracies)


    def log(self, losses, epoch):
        if epoch % self.configs.logs.epoch_interval == 0:
            filename = os.path.join(str(self.configs.logs.checkpoints_dir), "checkpoint_epoch_" + str(epoch) + '.pth')
            save_checkpoint({
                'epoch': epoch,
                'model': self.model.state_dict(),
                'classifier': self.classifier.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'losses': losses,
            }, filename)

        if losses["val_acc"] < self.best_test_accuracy:
            filename = os.path.join(str(self.configs.logs.checkpoints_dir), "checkpoint_best_test_acc.pth")
            save_checkpoint({
                'epoch': epoch,
                'model': self.model.state_dict(),
                'classifier': self.classifier.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'losses': losses,
            }, filename)
            self.best_test_accuracy = losses["val_acc"]

        # log
        self.writer.add_scalar("Train/Loss", losses["train_loss"], epoch)
        self.writer.add_scalar("Train/Accuracy", losses["train_acc"], epoch)
        self.writer.add_scalar("Test/Loss", losses["val_loss"], epoch)
        self.writer.add_scalar("Test/Accuracy", losses["val_acc"], epoch)

        wandb.log({
            "Train/Loss": losses["train_loss"],
            "Train/Accuracy": losses["train_acc"],
            "Test/Loss": losses["val_loss"],
            "Test/Accuracy": losses["val_acc"],
        })

        logging.info(
            f"Epoch: {epoch}, Train/Loss: {losses['train_loss']}, Train/Accuracy: {losses['train_acc']}, "
            f"Test/Loss: {losses['val_loss']}, Test/Accuracy: {losses['val_acc']}"
        )

    def run(self):
        # self.validate(self.test_dataloader)
        for epoch in range(1, self.configs.opt.epochs + 1):
            adjust_learning_rate(self.configs.opt, self.optimizer, epoch)
            train_loss, train_acc = self.train(epoch)
            val_loss, val_acc = self.validate(self.test_dataloader)
            losses = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            self.log(losses, epoch)
        self.writer.close()


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)
    logging.info('Save checkpoint to {0:}'.format(filename))


def main():
    trainer = LinearTrainer()
    trainer.run()


if __name__ == "__main__":
    main()