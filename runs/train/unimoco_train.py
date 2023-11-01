import argparse
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
from tqdm import tqdm
from torch.nn import functional as F
sys.path.insert(0, "")
from runs.train.unimoco_config import configs as UniMoCoConfig
from src.models import SingleGPUMoCo
from src.losses import UnifiedContrastive
from src.datasets import SupConDataset, CelebValidateDataset
from src.transforms import GaussianBlur
from src.optimizers import adjust_learning_rate, warmup_learning_rate
from src.metrics import knn_predict


# TODO 1: Implement training code for UniMoCo
    # TODO 1.1: Understand UniMoCo
    # TODO 1.2: Understand Split Batch Normalization
# TODO 2: Deal with problem about CelebDF dataset


class UniMoCoTrainer:
    def __init__(self, configs=UniMoCoConfig):
        self.configs = configs
        self.device = torch.device(self.configs.device)

        self.best_val_loss = 1e6
        self.best_val_acc = 0.
        self.prev_val_acc = 0.

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
        test_transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])

        train_dataset = SupConDataset(self.configs.dataset, "train", train_transform)
        memory_dataset = SupConDataset(self.configs.dataset, "train", test_transform, False)
        test_dataset = CelebValidateDataset(self.configs.dataset, "test", test_transform)

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.configs.dataset.batch_size,
            num_workers=self.configs.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True
        )
        self.memory_dataloader = DataLoader(
            memory_dataset,
            batch_size=self.configs.dataset.batch_size,
            num_workers=self.configs.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=True
        )
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.configs.dataset.batch_size,
            num_workers=self.configs.num_workers,
            pin_memory=True,
            shuffle=False,
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
        train_bar = tqdm(self.train_dataloader)
        for images, labels, _ in train_bar:
            images[0] = images[0].to(self.device)
            images[1] = images[1].to(self.device)
            labels = labels.to(self.device)
            output, target, fake_targets = self.model(im_q=images[0], im_k=images[1], labels=labels)

            loss = self.criterion(output, target)

            losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_bar.set_description(
                'Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch, self.configs.optimizer.n_epochs,
                                                                        self.optimizer.param_groups[0]['lr'],
                                                                        loss.item()))

            # if i % self.configs.logs.log_interval == 0:
            #     logging.info(
            #         f"Epoch {epoch} - {i}/{len(self.train_dataloader)}: Loss: {loss.item()}"
            #     )

        return losses[-1]


    def validate(self):
        self.model.eval()
        total_top1, total_num, feature_bank = 0.0, 0.0, []


        with torch.no_grad():
            for data, target, _ in tqdm(self.memory_dataloader, desc="Feature Extraction"):
                feature = self.model.encoder_q(data.cuda(non_blocking=True))
                feature = F.normalize(feature, dim=1)
                feature_bank.append(feature)

            feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            feature_labels = torch.tensor(self.memory_dataloader.dataset.label_list.astype(np.float32)).cuda()

            test_bar = tqdm(self.test_dataloader, desc="KNN Evaluation")
            for data, target in test_bar:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                feature = self.model.encoder_q(data)
                feature = F.normalize(feature, dim=1)

                pred_labels = knn_predict(feature, feature_bank, feature_labels, 2, self.configs.knn_k,
                                          self.configs.knn_t)

                total_num += data.size(0)
                total_top1 += (pred_labels[:, 0] == target).float().sum().item()
                test_bar.set_description(
                    'Test Epoch: Acc@1:{:.2f}%'.format(total_top1 / total_num * 100))

        return total_top1 / total_num * 100

    def log(self, losses, epoch):
        if epoch % self.configs.logs.epoch_interval == 0:
            filename = os.path.join(str(self.configs.logs.checkpoints_dir), "checkpoint_epoch_" + str(epoch) + '.pth')
            save_checkpoint({
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'losses': losses,
            }, filename)

        if losses["validate_accuracy"] < self.best_val_acc:
            filename = os.path.join(str(self.configs.logs.checkpoints_dir), "checkpoint_best_acc.pth")
            save_checkpoint({
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'losses': losses,
            }, filename)
            self.best_val_acc = losses["validate_accuracy"]

        # log
        self.writer.add_scalar("Train/Loss", losses["train_loss"], epoch)
        self.writer.add_scalar("Validate/Accuracy", losses["validate_accuracy"], epoch)

        wandb.log({
            "Train/Loss": losses["train_loss"],
            "Validate/Accuracy": losses["validate_accuracy"],
        })

        logging.info(
            f"Epoch: {epoch}, Train/Loss: {losses['train_loss']}, Validate/Accuracy: {losses['validate_accuracy']}"
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
            if epoch % 5 == 0:
                validate_acc = self.validate()
                self.prev_val_acc = validate_acc
            else:
                validate_acc = self.prev_val_acc
            losses = {
                "train_loss": train_loss,
                "validate_accuracy": validate_acc,
            }
            self.log(losses, epoch)
        self.writer.close()

def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)
    logging.info('Save checkpoint to {0:}'.format(filename))


def parse_args():
    parser = argparse.ArgumentParser(description='Train MoCo on CIFAR-10')
    parser.add_argument('-a', '--arch', default='resnet18')
    parser.add_argument('--lr', '--learning-rate', default=0.06, type=float, metavar='LR', help='initial learning rate',
                        dest='lr')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch-size', default=512, type=int, metavar='N', help='mini-batch size')


def main():
    trainer = UniMoCoTrainer()
    trainer.run()


if __name__ == "__main__":
    main()