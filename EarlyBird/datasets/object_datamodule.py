import lightning as pl
import os
from torch.utils.data import DataLoader
from typing import Optional
from datasets.sampler import RandomPairSampler
from datasets.messytable_dataset import Messytable
from datasets.object_dataset import ObjectDataset


class ObjectDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str = "../data/MultiviewX",
            batch_size: int = 1,
            num_workers: int = 4,
            resolution=None,
            bounds=None,
            load_depth=False,
            train_ratio=.5026,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resolution = resolution
        self.bounds = bounds
        self.load_depth = load_depth
        self.dataset = os.path.basename(self.data_dir)
        self.train_ratio = train_ratio

        self.data_predict = None
        self.data_test = None
        self.data_val = None
        self.data_train = None

    def setup(self, stage: Optional[str] = None):
        if 'messytable' in self.dataset.lower():
            base = Messytable(self.data_dir)
        else:
            raise ValueError(f'Unknown dataset name {self.dataset}')

        if stage == 'fit':
            self.data_train = ObjectDataset(
                base,
                is_train=True,
                resolution=self.resolution,
                bounds=self.bounds,
                train_ratio=self.train_ratio,
            )
        if stage == 'fit' or stage == 'validate':
            self.data_val = ObjectDataset(
                base,
                is_train=False,
                resolution=self.resolution,
                bounds=self.bounds,
                train_ratio=self.train_ratio,
            )
        if stage == 'test':
            self.data_test = ObjectDataset(
                base,
                is_train=False,
                resolution=self.resolution,
                bounds=self.bounds,
                train_ratio=self.train_ratio,
            )
        if stage == 'predict':
            self.data_predict = ObjectDataset(
                base,
                is_train=False,
                resolution=self.resolution,
                bounds=self.bounds,
                train_ratio=self.train_ratio,
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=RandomPairSampler(self.data_train)
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.data_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
