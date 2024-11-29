import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv
import time

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

# Login into OpenBB
obb.account.login(pat=openbb_api_key)


class OpenBBDataModule(L.LightningDataModule):
    def __init__(
        self,
        ticker: str,
        train_date: list,
        val_date: list,
        test_date: list,
        interval: str,
        market: str,
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
        - train_date (list): The start and end dates for the training data in 'YYYY-MM-DD' format.
        - val_date (list): The start and end dates for the validation data in 'YYYY-MM-DD' format.
        - test_date (list): The start and end dates for the test data in 'YYYY-MM-DD' format.
        - interval (str): The data interval (e.g., '1m' for 1 minute).
        - market (str): The market identifier (e.g., 'US' for the United States).
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
        # Define the path to the cache directory
        cache_dir = os.path.join(
            self.cache_dir, self.market, self.interval, self.ticker
        )
        os.makedirs(cache_dir, exist_ok=True)  # Ensure the cache directory exists

        # Define paths for the train, val, and test CSV files
        train_file = os.path.join(cache_dir, "train.csv")
        val_file = os.path.join(cache_dir, "val.csv")
        test_file = os.path.join(cache_dir, "test.csv")

        # Check if the cache files exist and use them if available
        if self.use_cache and all(
            os.path.exists(f) for f in [train_file, val_file, test_file]
        ):
            logger.info(f"Loading data from cache files in: {cache_dir}")
            train_data = pd.read_csv(train_file, index_col=0, parse_dates=True)
            val_data = pd.read_csv(val_file, index_col=0, parse_dates=True)
            test_data = pd.read_csv(test_file, index_col=0, parse_dates=True)
        else:
            # If cache files do not exist, fetch the data
            logger.info("Fetching data using OpenBB")
            full_data = self._fetch_data()

            # Log the range of the fetched data
            logger.info(
                f"Fetched data from {full_data.index.min()} to {full_data.index.max()}"
            )

            # Convert date strings to datetime objects
            train_start_date, train_end_date = map(
                lambda x: datetime.strptime(x, "%Y-%m-%d"), self.train_date
            )
            val_start_date, val_end_date = map(
                lambda x: datetime.strptime(x, "%Y-%m-%d"), self.val_date
            )
            test_start_date, test_end_date = map(
                lambda x: datetime.strptime(x, "%Y-%m-%d"), self.test_date
            )

            # Split the data
            train_data = full_data[
                (full_data.index >= train_start_date)
                & (full_data.index < train_end_date)
            ]
            val_data = full_data[
                (full_data.index >= val_start_date) & (full_data.index < val_end_date)
            ]
            test_data = full_data[
                (full_data.index >= test_start_date) & (full_data.index < test_end_date)
            ]

            # Log the size of each dataset
            logger.info(f"Train data size: {train_data.shape}")
            logger.info(f"Validation data size: {val_data.shape}")
            logger.info(f"Test data size: {test_data.shape}")

            # Cache the split data
            logger.info("Caching the split data")
            train_data.to_csv(train_file)
            val_data.to_csv(val_file)
            test_data.to_csv(test_file)

        # Assign the data to the class attributes
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def _fetch_data(self):
        """
        Fetches historical data for the specified ticker symbol using OpenBB.
        """
        # Determine the overall start and end dates for fetching data
        overall_start_date = datetime.strptime(self.train_date[0], "%Y-%m-%d")
        overall_end_date = datetime.strptime("2024-11-07", "%Y-%m-%d")

        # Initialize an empty DataFrame to store the fetched data
        all_data = pd.DataFrame()

        # Set the initial start date for the first API call
        current_start_date = overall_start_date

        # Track the number of API calls
        api_call_count = 0
        api_call_limit = 300

        while current_start_date < overall_end_date:
            # Calculate the end date for the current window
            current_end_date = current_start_date + pd.Timedelta(days=3)
            if current_end_date > overall_end_date:
                current_end_date = overall_end_date

            # Convert dates to string format for the API call
            start_str = current_start_date.strftime("%Y-%m-%d")
            end_str = current_end_date.strftime("%Y-%m-%d")

            logger.info(
                f"Fetching data for {self.ticker} from {start_str} to {end_str}"
            )

            # Fetch data for the current window
            data_chunk = obb.equity.price.historical(
                symbol=self.ticker,
                start_date=start_str,
                end_date=end_str,
                interval=self.interval,
                provider=self.provider,
            )

            # Append the fetched data to the all_data DataFrame
            all_data = pd.concat([all_data, data_chunk])

            # Update the start date for the next window
            current_start_date = current_end_date

            # Increment the API call count
            api_call_count += 1

            # Check if the API call limit is reached
            if api_call_count >= api_call_limit:
                logger.info("API call limit reached, waiting for 60 seconds...")
                time.sleep(60)  # Wait for 60 seconds
                api_call_count = 0  # Reset the API call count

        # Log the fetched data to check its content
        logger.info(f"Fetched data head:\n{all_data.head()}")
        logger.info(f"Fetched data tail:\n{all_data.tail()}")

        return all_data

    def setup(self, stage=None):
        logger.info("Setting up data sequences")

        # Convert date strings to datetime objects
        val_start_date, val_end_date = map(
            lambda x: datetime.strptime(x, "%Y-%m-%d"), self.val_date
        )
        test_start_date, test_end_date = map(
            lambda x: datetime.strptime(x, "%Y-%m-%d"), self.test_date
        )

        train_data = self.train_data[self.train_data.index < val_start_date]
        logger.info(f"Train data shape: {train_data.shape}")
        val_data = self.val_data[
            (self.val_data.index >= val_start_date)
            & (self.val_data.index < val_end_date)
        ]
        logger.info(f"Val data shape: {val_data.shape}")
        test_data = self.test_data[
            (self.test_data.index >= test_start_date)
            & (self.test_data.index < test_end_date)
        ]
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
            seq = close_prices[i : i + self.sequence_length]
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

        # Apply z-score normalization to sequences
        seq_mean = sequences.mean(axis=0)
        seq_std = sequences.std(axis=0)
        sequences = (sequences - seq_mean) / seq_std

        # Apply z-score normalization to targets
        target_mean = targets.mean()
        target_std = targets.std()
        targets = (targets - target_mean) / target_std

        # Convert numpy arrays to tensors
        sequences_tensor = torch.FloatTensor(sequences).unsqueeze(-1)
        targets_tensor = torch.FloatTensor(targets).unsqueeze(-1)

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
