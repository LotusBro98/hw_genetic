import sys
import logging

import torch
import torch.utils.data
import torchvision

from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torchvision import transforms
import torch.nn.functional as F
import pytorch_lightning as pl

from hyps import Hyps
from model import Classifier


class MyProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar


class TrainModel(pl.LightningModule):
    def __init__(self, hyps: Hyps, n_classes):
        super().__init__()
        self.hyps = hyps
        self.model = Classifier(hyps, n_classes)

    def training_step(self, batch, batch_idx):
        images, labels = batch

        pred = self.model(images)
        loss = F.cross_entropy(pred, labels, reduction="mean")
        acc = (pred.argmax(-1) == labels).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        pred = self.model(images)
        loss = F.cross_entropy(pred, labels, reduction="mean")
        acc = (pred.argmax(-1) == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hyps.lr, weight_decay=self.hyps.weight_decay)
        return optimizer


def train(hyps: Hyps, epochs=10, val_part=0.2):
    print(hyps)
    dataset = torchvision.datasets.CIFAR100("./cifar-100", download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))
    val_size = int(val_part * len(dataset))
    gen = torch.Generator()
    gen.manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        lengths=[len(dataset) - val_size, val_size],
        generator=gen
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128)

    model = TrainModel(hyps, n_classes=100)
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[
            MyProgressBar(),
            EarlyStopping(
                monitor="val_loss",
                min_delta=0.01,
                patience=3,
            )
        ],
    )

    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    val_accuracy = trainer.logged_metrics["val_acc"].item()

    return val_accuracy


if __name__ == '__main__':
    fitness = train(Hyps())
    print(fitness)
