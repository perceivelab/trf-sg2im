import os
from pathlib import Path

from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from utils.visualize import *

from data_modules.loader import *


class BaseDataModule:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)

    def train_dataloader(self, batch_size, device='cpu', num_workers=os.cpu_count(),
                         persistent_workers=False):
        sampler = RandomSampler(self.train_ds)
        dl = DataLoader(self.train_ds, batch_size=batch_size, num_workers=num_workers, sampler=sampler,
                        persistent_workers=persistent_workers, drop_last=True, collate_fn=self.collate_fn, pin_memory=True)
        return DeviceDataLoader(dl, device)

    def val_dataloader(self, batch_size, device='cpu', num_workers=os.cpu_count(), persistent_workers=False):
        dl = DataLoader(self.val_ds, batch_size=batch_size, num_workers=num_workers, persistent_workers=persistent_workers,
                        drop_last=True, collate_fn=self.collate_fn, pin_memory=True)
        return DeviceDataLoader(dl, device)

    def test_dataloader(self, batch_size, device='cpu', num_workers=os.cpu_count(), persistent_workers=False):
        dl = DataLoader(self.test_ds, batch_size=batch_size, num_workers=1, persistent_workers=persistent_workers,
                        drop_last=True, collate_fn=self.collate_fn, pin_memory=True)
        return DeviceDataLoader(dl, device)
