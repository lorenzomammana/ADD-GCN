from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class CocoDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_dataset: Dataset,
            val_dataset: Dataset,
            batch_size: Optional[int] = 256,
            num_workers: int = 8,
    ):
        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str = None):
        pass

    def train_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

        return dataloader

    def val_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )

        return dataloader
