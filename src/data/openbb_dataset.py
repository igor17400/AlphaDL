import os
import sys

import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import torch
from openbb import obb
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.namespaces.data_config import DataConfig

class OpenBBDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DataConfig):
        super().__init__()
        self.ticker = cfg.ticker
        self.start_date = cfg.start_date
        self.end_date = cfg.end_date
        self.sequence_length = cfg.sequence_length
        self.batch_size = cfg.batch_size
        self.market = cfg.market
        self.interval = cfg.interval
        self.cache_dir = cfg.cache_dir
        self.use_cache = cfg.use_cache
        self.provider = cfg.provider 

    def prepare_data(self):
        cache_file = os.path.join(self.cache_dir, f"{self.ticker}_{self.market}_{self.interval}.csv")

        if self.use_cache and os.path.exists(cache_file):
            self.data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        else:
            # Fetch data using OpenBB
            self.data = self._fetch_data()
            
            # Cache the data
            os.makedirs(self.cache_dir, exist_ok=True)
            self.data.to_csv(cache_file)

    def _fetch_data(self):
        return obb.equity.price.historical(
            symbol=self.ticker,
            start_date=self.start_date,
            end_date=self.end_date,
            interval=self.interval,
            provider=self.provider  
        )

    def setup(self, stage=None):
        # Prepare sequences
        sequences = []
        targets = []

        for i in range(len(self.data) - self.sequence_length):
            seq = self.data['Close'].values[i:i+self.sequence_length]
            target = self.data['Close'].values[i+self.sequence_length]
            sequences.append(seq)
            targets.append(target)

        # Convert to tensors
        self.sequences = torch.FloatTensor(sequences).unsqueeze(2)
        self.targets = torch.FloatTensor(targets).unsqueeze(1)

        # Split into train and validation sets
        train_size = int(0.8 * len(self.sequences))
        self.train_data = TensorDataset(self.sequences[:train_size], self.targets[:train_size])
        self.val_data = TensorDataset(self.sequences[train_size:], self.targets[train_size:])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

# Usage
from src.namespaces.data_config import DataConfig

config = DataConfig(
    ticker='AAPL',
    start_date='2022-01-01',
    end_date='2023-01-01',
    sequence_length=10,
    batch_size=32,
    market='US',
    interval='1d',
    cache_dir='./data_cache',
    use_cache=True,
    provider='yfinance'
)

data_module = OpenBBDataModule(config)
data_module.prepare_data()
data_module.setup()
