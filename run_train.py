import os

import cv2
import hydra
import torch.optim
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
)
from pytorch_lightning.loggers import MLFlowLogger
import numpy as np
from models import ADD_GCN
from models.add_gcn_lightning import AddGcnModel
from data.coco_datamodule import CocoDataModule
import glob

from models.mobilevit import MobileViT


def get_config_optim(model, lr, lrp):
    params = list(model.named_parameters())

    return [
        {"params": [p for n, p in params if "features" in n], "lr": lr * lrp},
        {"params": [p for n, p in params if "features" not in n], "lr": lr},
    ]


@hydra.main(config_path="configs", config_name="train_uva_2.yaml")
def run_training(params):
    cv2.setNumThreads(1)

    if not params.model.pretrained:
        params.aug_train.transforms[-2].mean = [0, 0, 0]
        params.aug_test.transforms[-2].mean = [0, 0, 0]
        params.aug_train.transforms[-2].std = [1, 1, 1]
        params.aug_test.transforms[-2].std = [1, 1, 1]

    if params.dataset.use_patches:
        img_names = glob.glob(os.path.join(params.train_dset.data_dir, "patches", "*"))
        img_names = [x.split("/")[-1] for x in img_names]
        img_names_unique = np.array(sorted(list(set(["_".join(x.split("_")[0:-1]) for x in img_names]))))
        rng = np.random.default_rng(42)

        indices = np.array(range(len(img_names_unique)), dtype=int)
        indices = rng.permutation(indices)

        img_names_unique = img_names_unique[indices].tolist()
        split_index = int(np.floor(len(img_names_unique) * params.dataset.test_size))
        img_names_train = img_names_unique[0:-split_index]
        img_names_val = img_names_unique[-split_index:]
        img_names_train = list(filter(lambda x: any(y in x for y in img_names_train), img_names))
        img_names_val = list(filter(lambda x: any(y in x for y in img_names_val), img_names))

        # Dataset definition
        train_dataset = hydra.utils.instantiate(
            params.train_dset, transform=hydra.utils.instantiate(params.aug_train), valid_files=img_names_train
        )

        val_dataset = hydra.utils.instantiate(
            params.test_dset, transform=hydra.utils.instantiate(params.aug_test), valid_files=img_names_val
        )
    else:
        # Dataset definition
        train_dataset = hydra.utils.instantiate(params.train_dset, transform=hydra.utils.instantiate(params.aug_train))

        val_dataset = hydra.utils.instantiate(params.test_dset, transform=hydra.utils.instantiate(params.aug_test))

    data_module = CocoDataModule(
        train_dataset,
        val_dataset,
        batch_size=16,
        num_workers=8,
    )

    # Model definition
    architecture = ADD_GCN(
        hydra.utils.instantiate(params.architecture),
        train_dataset.num_classes,
        skip_gcn=params.model.skip_gcn,
    )

    if params.model.pretrained:
        optimizer_params = get_config_optim(architecture, params.optimizer.lr, 0.1)
    else:
        optimizer_params = architecture.parameters()

    # TODO Understand why get config optim is not working directly through hydra instantiate when using get_config_optim
    #  Apparently list of dicts are automatically converted to Hydra dicts
    optimizer = torch.optim.SGD(
        params=optimizer_params,
        lr=params.optimizer.lr,
        weight_decay=params.optimizer.weight_decay,
        nesterov=params.optimizer.nesterov,
        momentum=params.optimizer.momentum,
    )

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
    trainer = Trainer(**params.trainer, callbacks=[checkpoint, lr_monitor, early_stopping], logger=logger)
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    run_training()
