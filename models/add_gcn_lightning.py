import logging
from typing import Optional

import pytorch_lightning as pl
import torch
from torch import nn, optim
from torchmetrics import F1, ConfusionMatrix


class AddGcnModel(pl.LightningModule):
    def __init__(
            self,
            architecture: nn.Module,
            optimizer: Optional[optim.Optimizer] = None,
            lr_scheduler: Optional[object] = None
    ):
        super(AddGcnModel, self).__init__()
        self.model = architecture
        self.loss_function = torch.nn.BCELoss()
        self.optimizer = optimizer
        self.schedulers = lr_scheduler
        self.f1_score = F1(self.model.num_classes, threshold=0.5, average='macro')
        self.confmat = ConfusionMatrix(num_classes=self.model.num_classes, multilabel=True)
        # Necessary for logging hyperparams with tensorboard
        # TODO Understand why this is super slow
        # self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self) -> dict:
        # get default optimizer
        if not self.optimizer:
            logging.info("Optimizer not defined. Using default Adam optimizer")
            self.optimizer = optim.Adam(self.feature_extractor.parameters(), lr=3e-4)

        # get default scheduler
        if not self.schedulers:
            logging.info("Scheduler not defined. Using default StepLR")
            self.schedulers = optim.lr_scheduler.StepLR(self.optimizer, step_size=50)

        configuration_dict = {
            "optimizer": self.optimizer,
            "lr_scheduler": self.schedulers,
            "monitor": "val_loss",
        }
        return configuration_dict

    def training_step(self, batch, batch_idx):
        x, y = batch
        s_m, s_r = self.forward(x)

        s_output = (s_m + s_r) / 2

        out_dict = {"predictions": s_output, "y": y}

        return out_dict

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        s_m, s_r = self.forward(x)

        s_output = (s_m + s_r) / 2
        return s_output, y

    def training_step_end(self, outputs):
        loss = self.loss_function(outputs['predictions'], outputs['y'].float())
        f1_score = self.f1_score(outputs['predictions'], outputs['y'])

        out_dict = {
            "loss": loss,
            "train_acc": f1_score
        }

        return out_dict

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train_loss", avg_loss.item(), prog_bar=True, logger=True)
        avg_acc = torch.stack([x["train_acc"] for x in outputs]).mean()
        self.log("train_acc", avg_acc.item(), prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        s_m, s_r = self.forward(x)

        s_output = (s_m + s_r) / 2
        out_dict = {"predictions": s_output, "y": y}

        return out_dict

    def validation_step_end(self, outputs):
        loss = self.loss_function(outputs['predictions'], outputs['y'].float())
        f1_score = self.f1_score(outputs['predictions'], outputs['y'])

        out_dict = {
            "val_loss": loss,
            "val_acc": f1_score
        }

        return out_dict

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss.item(), prog_bar=True, logger=True)

        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        self.log("val_acc", avg_acc.item(), prog_bar=True, logger=True)
