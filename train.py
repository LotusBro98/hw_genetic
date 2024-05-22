from typing import Any

import torch
import torchvision
import tqdm
from pytorch_lightning.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torchvision import transforms
import torch.nn.functional as F
import pytorch_lightning as pl

from genome import Hyps
from model import Classifier


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

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hyps.lr)
        return optimizer



def train(hyps: Hyps, epochs=10):
    dataset = torchvision.datasets.CIFAR100("./cifar-100", download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))
    loader = torch.utils.data.DataLoader(dataset, batch_size=16)

    model = TrainModel(hyps, n_classes=100)
    trainer = pl.Trainer(
        max_epochs=epochs,
        # devices=1,
        # accelerator="cpu"
    )

    trainer.fit(model, loader)

    return loss


if __name__ == '__main__':
    loss = train(Hyps())
    print(loss)
