import os

import hydra
import torch.optim
import torchvision
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
)
from pytorch_lightning.loggers import MLFlowLogger
from data.coco import COCO2014
import albumentations as alb
from albumentations.pytorch import ToTensorV2

from models import ADD_GCN
from models.add_gcn_lightning import AddGcnModel
from data.coco_datamodule import CocoDataModule
from data import get_transform


def get_config_optim(model, lr, lrp):
    params = list(model.named_parameters())

    return [
        {'params': [p for n, p in params if 'features' in n], 'lr': lr * lrp},
        {'params': [p for n, p in params if 'features' not in n], 'lr': lr},
    ]


@hydra.main(config_path="configs", config_name="train.yaml")
def run_training(params):
    # Dataset definition
    train_dataset = hydra.utils.instantiate(params.train_dset,
                                            transform=hydra.utils.instantiate(params.aug_train))

    val_dataset = hydra.utils.instantiate(params.test_dset,
                                          transform=hydra.utils.instantiate(params.aug_test))

    data_module = CocoDataModule(
        train_dataset,
        val_dataset,
        batch_size=16,
        num_workers=8,
    )

    # Model definition
    architecture = ADD_GCN(
        torchvision.models.resnet101(pretrained=True),
        train_dataset.num_classes
    )

    # optimizer = hydra.utils.instantiate(
    #     params.optimizer, params=architecture.parameters()
    # )

    # TODO Understand why get config optim is not working directly through hydra instantiate
    optimizer = torch.optim.SGD(
        params=get_config_optim(architecture,
                                params.optimizer.lr,
                                0.1),
        lr=params.optimizer.lr,
        weight_decay=params.optimizer.weight_decay,
        nesterov=params.optimizer.nesterov,
        momentum=params.optimizer.momentum
    )
    # optimizer = hydra.utils.instantiate(
    #     params.optimizer,
    #     params=get_config_optim(architecture,
    #                             params.optimizer.lr,
    #                             0.1)
    # )

    scheduler = hydra.utils.instantiate(params.scheduler, optimizer=optimizer)

    model = AddGcnModel(
        architecture,
        optimizer=optimizer,
        lr_scheduler=scheduler,
    )

    if params.logger.mlflow:
        logger = MLFlowLogger(
            experiment_name=params.logger.experiment_name,
            tracking_uri=params.logger.server_address,
        )
        logger.log_hyperparams({"experiments": os.getcwd()})
    else:
        logger = True

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    checkpoint = hydra.utils.instantiate(params.checkpoint)
    early_stopping = hydra.utils.instantiate(params.early_stopping)
    trainer = Trainer(
        **params.trainer,
        callbacks=[checkpoint, lr_monitor, early_stopping],
        logger=logger
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    run_training()
