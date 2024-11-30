import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv
import time

import lightning as L
from torch.utils.data import DataLoader
import torch
from openbb import obb
import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

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
        index: str,
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
        max_prediction_length: int,
        selected_tickers: list = None,
        **kwargs,
    ):
        super().__init__()
        self.index = index
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
        self.max_prediction_length = max_prediction_length
        self.selected_tickers = selected_tickers

        # Apply OpenBB settings to ensure output is a dataframe
        obb.user.preferences.output_type = self.openbb_output_type

    def prepare_data(self):
        """
        Prepares the data for the model by either loading it from a cache file or fetching it from OpenBB.
        """
        # Define the path to the cache directory
        cache_dir = os.path.join(
            self.cache_dir, self.market, self.interval, "SP500"
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

            # Add a unified time index
            full_data = self._add_time_idx(full_data)

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

            # Ensure the index is unique for each dataset
            train_data = train_data[~train_data.index.duplicated(keep='first')]
            val_data = val_data[~val_data.index.duplicated(keep='first')]
            test_data = test_data[~test_data.index.duplicated(keep='first')]

            # Save the data to cache
            train_data.to_csv(train_file)
            val_data.to_csv(val_file)
            test_data.to_csv(test_file)

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def _fetch_data(self):
        # Fetch the list of constituents for the specified index
        constituents = obb.index.constituents(symbol=self.index, provider=self.provider)
        
        tickers = constituents['symbol'].tolist()

        # If selected_tickers is provided, filter the tickers list
        if self.selected_tickers:
            tickers = [ticker for ticker in tickers if ticker in self.selected_tickers]

        # Initialize an empty DataFrame to store the fetched data
        all_data = pd.DataFrame()

        # Set the overall start and end dates
        overall_start_date = datetime.strptime(self.train_date[0], "%Y-%m-%d")
        overall_end_date = datetime.strptime(self.test_date[1], "%Y-%m-%d")

        # Track the number of API calls
        api_call_count = 0
        api_call_limit = 300

        for ticker in tickers:
            current_start_date = overall_start_date
            ticker_data = pd.DataFrame()  # DataFrame to store data for the current ticker
            while current_start_date < overall_end_date:
                # Calculate the end date for the current window
                current_end_date = current_start_date + pd.Timedelta(days=3)
                if current_end_date > overall_end_date:
                    current_end_date = overall_end_date

                # Convert dates to string format for the API call
                start_str = current_start_date.strftime("%Y-%m-%d")
                end_str = current_end_date.strftime("%Y-%m-%d")

                logger.info(
                    f"Fetching data for {ticker} from {start_str} to {end_str}"
                )

                try:
                    # Fetch data for the current window
                    data_chunk = obb.equity.price.historical(
                        symbol=ticker,
                        start_date=start_str,
                        end_date=end_str,
                        interval=self.interval,
                        provider=self.provider,
                    )

                    # Add a ticker column to the data_chunk
                    data_chunk['ticker'] = ticker

                    # Append the fetched data to the ticker_data DataFrame
                    ticker_data = pd.concat([ticker_data, data_chunk])

                except Exception as e:
                    logger.error(f"Failed to fetch data for {ticker}: {e}")
                    break  # Exit the while loop and continue with the next ticker

                # Update the start date for the next window
                current_start_date = current_end_date

                # Increment the API call count
                api_call_count += 1

                # Check if the API call limit is reached
                if api_call_count >= api_call_limit:
                    logger.info("API call limit reached, waiting for 60 seconds...")
                    time.sleep(60)  # Wait for 60 seconds
                    api_call_count = 0  # Reset the API call count

            # Ensure the index is unique for the ticker data
            ticker_data = ticker_data[~ticker_data.index.duplicated(keep='first')]

            # Save the data for the current ticker
            ticker_file = os.path.join(self.cache_dir, self.market, self.interval, "SP500", f"{ticker}.csv")
            ticker_data.to_csv(ticker_file)

            # Append the ticker data to the all_data DataFrame
            all_data = pd.concat([all_data, ticker_data])

        # Ensure the index is unique for the combined data
        all_data = all_data[~all_data.index.duplicated(keep='first')]

        # Save all data to ALL.csv
        all_file = os.path.join(self.cache_dir, self.market, self.interval, "SP500", "ALL.csv")
        all_data.to_csv(all_file)

        # Log the fetched data to check its content
        logger.info(f"Fetched data head:\n{all_data.head()}")
        logger.info(f"Fetched data tail:\n{all_data.tail()}")

        return all_data

    def _add_time_idx(self, df):
        """
        Adds a time index to the DataFrame for each group (ticker).
        """
        # Ensure the index is a datetime index
        df.index = pd.to_datetime(df.index)

        # Extract month and day from the date
        df["month"] = df.index.month
        df["day"] = df.index.day

        def create_time_idx(group):
            # Use pd.factorize to create a continuous index for each symbol's time series
            group['time_idx'] = pd.factorize(group.index)[0]
            return group

        # Apply the time index creation function to each group
        df_index = df.index
        df = df.groupby('ticker').apply(create_time_idx).reset_index(drop=True).set_index(df_index)

        return df

    def setup(self, stage=None):
        logger.info("Setting up data sequences")

        # Create the training TimeSeriesDataSet
        self.training = TimeSeriesDataSet(
            self.train_data,
            time_idx="time_idx",
            target="close",
            group_ids=["ticker"],
            max_encoder_length=self.sequence_length,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=["ticker"],
            time_varying_known_reals=["open", "high", "low", "volume"],
            time_varying_unknown_reals=["close"],
            target_normalizer=GroupNormalizer(
                groups=["ticker"], transformation="softplus"
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        # Create the validation TimeSeriesDataSet
        self.validation = TimeSeriesDataSet.from_dataset(
            self.training, self.val_data, predict=True, stop_randomization=True
        )

    def train_dataloader(self):
        logger.info("Creating train dataloader")
        return self.training.to_dataloader(
            train=True, batch_size=self.batch_size, num_workers=0
        )

    def val_dataloader(self):
        logger.info("Creating validation dataloader")
        return self.validation.to_dataloader(
            train=False, batch_size=self.batch_size * 10, num_workers=0
        )

    def test_dataloader(self):
        logger.info("Creating test dataloader")
        test_dataset = TimeSeriesDataSet.from_dataset(
            self.training, self.test_data, predict=True, stop_randomization=True
        )
        return test_dataset.to_dataloader(
            train=False, batch_size=self.batch_size * 10, num_workers=0
        )
