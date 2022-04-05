from argparse import Namespace
import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
from torch.utils.data import DataLoader, Subset
from torch.multiprocessing import cpu_count
from torch.optim import RMSprop

from torchvision import transforms
from torchvision.datasets import ImageFolder

from models import FinetunedModel
from utils import get_train_val_split
from ds_augmentations import AugDatasetWrapper


# training loop components
class FinetunedClassifierModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        hparams = Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.hparam = hparams
        self.model = FinetunedModel(hparams.n_classes, hparams.freeze_base,
                                    self.hparam.hidden_size)
        self.loss = nn.BCEWithLogitsLoss()

    def total_steps(self):
        return len(self.train_dataloader()) // self.hparam.epochs

    def get_dataloader(self, split):
        ds = ImageFolder(self.hparam.ds_name)  # change this for the datasets
        if split == 'train':
            split_ds = Subset(ds, self.hparam.train_ids)
        elif split == 'test':
            split_ds = Subset(ds, self.hparam.validation_ids)
        return DataLoader(
            AugDatasetWrapper(split_ds, target_size=self.hparam.img_size),
            batch_size=self.hparam.batch_size,
            shuffle=split == "train",
            num_workers=cpu_count(),
            drop_last=False)

    def train_dataloader(self):
        return self.get_dataloader("train")

    def val_dataloader(self):
        return self.get_dataloader("test")

    def forward(self, x):
        return self.model(x)

    def step(self, batch, step_name="train"):
        x, y = batch
        y_out = self.forward(x)
        loss = self.loss(y_out, y)
        loss_key = f"{step_name}_loss"
        tensorboard_logs = {loss_key: loss}
        self.log('training_metrics', tensorboard_logs)  # this is needed to get the logs

        return {("loss" if step_name == "train" else loss_key): loss, 'log': tensorboard_logs,
                "progress_bar": {loss_key: loss}}

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def validation_end(self, outputs):
        if len(outputs) == 0:
            return {"val_loss": torch.tensor(0)}
        else:
            loss = torch.stack([x["val_loss"] for x in outputs]).mean()
            return {"val_loss": loss, "log": {"val_loss": loss}}

    def configure_optimizers(self):
        optimizer = RMSprop(self.model.parameters(), lr=self.hparam.lr)
        schedulers = [
            CosineAnnealingLR(optimizer, self.hparam.epochs)
        ] if self.hparam.epochs > 1 else []
        return [optimizer], schedulers


def train(args, device):
    train_idx, val_idx, n_classes = get_train_val_split(args.ds_name, get_n_classes = True)

    # using the suggested lr
    hparams_cls = Namespace(
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_classes=n_classes,
        train_ids=train_idx,
        validation_ids=val_idx,
        hidden_size=args.hidden_size,
        freeze_base=args.freeze_base,
        img_size=(args.image_size, args.image_size)
    )

    module = FinetunedClassifierModule(hparams_cls)

    logger = WandbLogger(project='finetuning-classifier-on-paintings',
                         name='uncropped',
                         config={
                             'learning_rate': args.lr,
                             'architecture': 'CNN',
                             'dataset': 'Paintings',
                             'epochs': args.epochs,
                         })
    logger.watch(module, log='all', log_freq=args.log_interval)

    trainer = pl.Trainer(gpus=1, max_epochs=hparams_cls.epochs, logger=logger,
                         log_every_n_steps=args.log_interval)  # need the last arg to log the training iterations per step

    trainer.fit(module)


if __name__ == 'main':

    #add an option to do hyperparameter search (the lr finetuning bit)
    # Training settings
    parser = argparse.ArgumentParser(description='Paintings Classifier')

    parser.add_argument('--ds-name', type=str, default='datasets/paintings', metavar='S',
                        help='Name of the dataset to train and validate on (default: datasets/paintings)')
    parser.add_argument('--image-size', type=int, default=512, metavar='N',
                        help='input image size for model training and inference (default: 512)')

    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 16)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--hidden-size', type=int, default=512, metavar='N',
                        help='hidden size of the fc layer of the model (default: 512)')
    parser.add_argument('--freeze-base', action='store_true', default=True,
                        help='Freeze the pretrained model before training? (default: True)')

    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='enables/disables CUDA training')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='Number of workers (default: 4)')

    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status (default: 50)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    args = parser.parse_args()
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')

    train(args, device)