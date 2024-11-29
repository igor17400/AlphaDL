import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv

import lightning as L
from torch.utils.data import DataLoader, TensorDataset
import torch
from openbb import obb
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Access the API key
openbb_api_key = os.getenv("OPENBB_API_KEY")

# Set the API key for OpenBB
obb.user.preferences.api_key = openbb_api_key


class OpenBBDataModule(L.LightningDataModule):
    def __init__(
        self,
        ticker: str,
        train_date: str,
        val_date: str,
        test_date: str,
        interval: str,
        market: str,
        valid_time_split: str,
        sequence_length: int,
        batch_size: int,
        cache_dir: str,
        use_cache: bool,
        provider: str,
        openbb_output_type: str,
        **kwargs,
    ):
        """
        Initializes the OpenBBDataModule.

        Parameters:
        - ticker (str): The stock ticker symbol to fetch data for.
        - train_date (str): The start date for the training data in 'YYYY-MM-DD' format.
        - val_date (str): The start date for the validation data in 'YYYY-MM-DD' format.
        - test_date (str): The start date for the test data in 'YYYY-MM-DD' format.
        - interval (str): The data interval (e.g., '1m' for 1 minute).
        - market (str): The market identifier (e.g., 'US' for the United States).
        - valid_time_split (str): The datetime to split validation data.
        - sequence_length (int): The length of the sequence window for model input.
        - batch_size (int): The number of samples per batch for the DataLoader.
        - cache_dir (str): The directory to cache fetched data.
        - use_cache (bool): Whether to use cached data if available.
        - provider (str): The data provider to use with OpenBB.
        - openbb_output_type (str): The output type setting for OpenBB (e.g., 'dataframe').
        """
        super().__init__()
        self.ticker = ticker
        self.train_date = train_date
        self.val_date = val_date
        self.test_date = test_date
        self.interval = interval
        self.market = market
        self.valid_time_split = valid_time_split
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.provider = provider
        self.openbb_output_type = openbb_output_type

        # Apply OpenBB settings to ensure output is a dataframe
        obb.user.preferences.output_type = self.openbb_output_type

    def prepare_data(self):
        """
        Prepares the data for the model by either loading it from a cache file or fetching it from OpenBB.
        """
        # Define the path to the cache file
        cache_file = os.path.join(
            self.cache_dir, f"{self.ticker}_{self.market}_{self.interval}.csv"
        )
        logger.info(
            f"Preparing data for {self.ticker} from {self.train_date} to {self.test_date}"
        )

        # Check if the cache file exists and use it if available
        if self.use_cache and os.path.exists(cache_file):
            logger.info(f"Loading data from cache file: {cache_file}")
            self.data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        else:
            # If cache file does not exist, fetch the data
            logger.info("Fetching data using OpenBB")
            self.data = self._fetch_data()

            # Cache the fetched data
            logger.info("Caching the data")
            os.makedirs(
                self.cache_dir, exist_ok=True
            )  # Ensure the cache directory exists
            self.data.to_csv(cache_file)  # Save the data to the cache file

    def _fetch_data(self):
        """
        Fetches historical data for the specified ticker symbol using OpenBB.
        """
        logger.info(
            f"Fetching data for {self.ticker} from {self.train_date} to {self.test_date}"
        )
        return obb.equity.price.historical(
            symbol=self.ticker,
            start_date=self.train_date,
            end_date=self.test_date,
            interval=self.interval,
            provider=self.provider,
        )

    def setup(self, stage=None):
        logger.info("Setting up data sequences")

        # Convert date strings to datetime objects
        val_date = datetime.strptime(self.val_date, "%Y-%m-%d")
        test_date = datetime.strptime(self.test_date, "%Y-%m-%d")

        train_data = self.data[self.data.index < val_date]
        logger.info(f"Train data shape: {train_data.shape}")
        val_data = self.data[
            (self.data.index >= val_date) & (self.data.index < test_date)
        ]
        logger.info(f"Val data shape: {val_data.shape}")
        test_data = self.data[self.data.index >= test_date]
        logger.info(f"Test data shape: {test_data.shape}")

        self.train_sequences, self.train_targets = self._prepare_sequences(train_data)
        self.val_sequences, self.val_targets = self._prepare_sequences(val_data)
        self.test_sequences, self.test_targets = self._prepare_sequences(test_data)

        self.train_data = TensorDataset(self.train_sequences, self.train_targets)
        self.val_data = TensorDataset(self.val_sequences, self.val_targets)
        self.test_data = TensorDataset(self.test_sequences, self.test_targets)

    def _prepare_sequences(self, data):
        """
        Prepares sequences and targets for training from the given data.

        This method iterates over the 'close' prices in the data, creating sequences of a specified length and their corresponding targets.
        The sequence is a window of 'close' prices, and the target is the 1-day return ratio calculated from the next 'close' price.
        The sequences and targets are then converted to tensors for use in training.

        Parameters:
        - data (pd.DataFrame): The data from which to prepare sequences and targets.

        Returns:
        - sequences_tensor (torch.FloatTensor): A tensor of sequences, each sequence being a window of 'close' prices.
        - targets_tensor (torch.FloatTensor): A tensor of targets, each target being the 1-day return ratio calculated from the next 'close' price.
        """
        logger.info("Preparing sequences")
        sequences = []
        targets = []
        close_prices = data["close"].values

        for i in range(len(close_prices) - self.sequence_length - 1):
            seq = close_prices[
                i : i + self.sequence_length
            ]  # A sequence is a window of 'close' prices
            # Calculate the 1-day return ratio as the target
            target = (
                close_prices[i + self.sequence_length]
                - close_prices[i + self.sequence_length - 1]
            ) / close_prices[i + self.sequence_length - 1]
            sequences.append(seq)
            targets.append(target)

        # Convert lists to numpy arrays
        sequences = np.array(sequences)
        targets = np.array(targets)

        # Convert numpy arrays to tensors
        sequences_tensor = torch.FloatTensor(sequences).unsqueeze(
            -1
        )  # Add a new dimension at the end
        targets_tensor = torch.FloatTensor(targets).unsqueeze(
            -1
        )  # Add a new dimension at the end

        return sequences_tensor, targets_tensor

    def train_dataloader(self):
        logger.info("Creating train dataloader")
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        logger.info("Creating validation dataloader")
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        logger.info("Creating test dataloader")
        return DataLoader(self.test_data, batch_size=self.batch_size)
