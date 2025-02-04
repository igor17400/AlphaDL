import os
import logging
from datetime import datetime
from dotenv import load_dotenv
import time
from typing import Dict, List, Optional
from tqdm import tqdm
import pickle

import torch
from torch.utils.data import Dataset
from openbb import obb
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Access the API key
openbb_api_key = os.getenv("OPENBB_API_KEY")

# Login into OpenBB
obb.account.login(pat=openbb_api_key)


class StocksDataFrame(Dataset):
    """Creates a dataframe for stock market data using OpenBB.

    Attributes:
        index: Stock market index to fetch data for
        scraping_date: Date range for data collection
        train_date: Training date range
        val_date: Validation date range
        test_date: Test date range
        interval: Data interval (e.g. '1d' for daily)
        market: Market name
        sequence_length: Length of input sequences
        cache_dir: Directory to cache downloaded data
        use_cache: Whether to use cached data
        provider: Data provider name
        openbb_output_type: OpenBB output format
        selected_tickers: Optional list of specific tickers to use
        batch_ticker_processing: Whether to process all tickers at once
    """

    def __init__(
        self,
        index: str,
        scraping_date: list,
        train_date: list,
        val_date: list,
        test_date: list,
        interval: str,
        market: str,
        sequence_length: int,
        cache_dir: str,
        use_cache: bool,
        provider: str,
        openbb_output_type: str,
        train: bool,
        validation: bool,
        selected_tickers: Optional[List[str]] = None,
        batch_ticker_processing: bool = True,
    ) -> None:
        super().__init__()

        self.index = index
        self.scraping_date = scraping_date
        self.train_date = train_date
        self.val_date = val_date
        self.test_date = test_date
        self.interval = interval
        self.market = market
        self.sequence_length = sequence_length
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.provider = provider
        self.openbb_output_type = openbb_output_type
        self.selected_tickers = selected_tickers
        self.batch_ticker_processing = batch_ticker_processing
        self.train = train
        self.validation = validation

        # Apply OpenBB settings
        obb.user.preferences.output_type = self.openbb_output_type

        # Load and process data
        full_data = self.download_data()
        logger.info("Processing dataset...")
        self.data = self.process_data(full_data)
        logger.info(f"Dataset processed!")

        if self.batch_ticker_processing:
            logger.info(
                f"Batch processing activated. "
                f"Sequence length: {self.sequence_length}"
            )
        else:
            logger.info(
                f"Single-ticker processing mode. "
                f"Sequence length: {self.sequence_length}"
            )

    def __len__(self):
        """Returns the number of valid prediction points in the dataset."""
        return sum(self.data["pred_idx"])

    def __getitem__(self, idx):
        """Returns a batch of data in StockMixer format.

        Returns:
            tuple: (data_batch, mask_batch, price_batch, gt_batch, dates_timestamp) where:
                - data_batch: shape [n_stocks, sequence_length, features]
                - mask_batch: shape [n_stocks, 1]
                - price_batch: shape [n_stocks, 1]
                - gt_batch: shape [n_stocks, 1]
                - dates_timestamp: timestamps for the sequence
        """
        if not self.batch_ticker_processing:
            raise NotImplementedError("Only batch_ticker_processing=True is supported")

        # print(f"########## idx:{idx} ##########")
        # print("self.data")
        # print(self.data)

        # Find indices where pred_idx is True
        pred_indices = np.where(self.data["pred_idx"])[0]
        # print("pred_indices")
        # print(pred_indices)
        current_pred_idx = pred_indices[idx]
        # print("current_pred_idx")
        # print(current_pred_idx)

        # Get sequence window for all stocks
        sequence_start = current_pred_idx - self.sequence_length + 1
        sequence_end = current_pred_idx + 1

        # Extract features (excluding returns) and returns separately
        features = self.data["features"][
            :, sequence_start:sequence_end, :-1
        ]  # All features except returns
        # print("features")
        # print(features)

        returns = self.data["features"][
            :, sequence_start:sequence_end, -1
        ]  # Returns only
        # print("returns")
        # print(returns)

        dates = self.data["dates"][sequence_start:sequence_end]
        # print("dates")
        # print(dates)

        # Get current prices (last timestep of sequence)
        prices = features[:, -1, -1]  # Last timestep, close price
        # print("prices")
        # print(prices)

        # Get ground truth (return at the prediction point)
        ground_truth = returns[:, -1]  # Last timestep's return

        # Convert dates to timestamps
        dates_timestamp = dates.astype(np.int64) // 10**9

        return (
            torch.tensor(features, dtype=torch.float32),  # [n_stocks, seq_len, features]
            torch.tensor(prices, dtype=torch.float32).unsqueeze(1),  # [n_stocks, 1]
            torch.tensor(ground_truth, dtype=torch.float32).unsqueeze(1),  # [n_stocks, 1]
            dates_timestamp,
        )

    def download_data(self):
        """Downloads raw data from OpenBB or loads it from cache if available."""
        # Define cache paths
        cache_dir = os.path.join(self.cache_dir, self.market, self.interval, self.index)
        all_file = os.path.join(cache_dir, "ALL.csv")

        if self.use_cache and os.path.exists(all_file):
            logger.info(f"Loading raw data from cache: {all_file}")
            full_data = pd.read_csv(all_file, index_col=0, parse_dates=True)
        else:
            logger.info("Fetching data using OpenBB")
            full_data = self._fetch_data()

        return full_data

    def process_data(self, full_data):
        """Processes the raw data into a single processed dataset."""
        processed_dir = os.path.join(
            self.cache_dir, self.market, self.interval, self.index
        )
        os.makedirs(processed_dir, exist_ok=True)
        processed_file = os.path.join(processed_dir, "ALL_processed.pkl")

        if self.use_cache and os.path.exists(processed_file):
            logger.info(f"Loading processed data from cache: {processed_file}")
            with open(processed_file, "rb") as f:
                all_data_dict = pickle.load(f)
        else:
            logger.info("Processing dataset...")

            # List to store feature arrays for each valid ticker
            # Each element will be a 2D array of shape (n_timesteps, n_features)
            features_list = []

            # List to store dates for each ticker
            # Each element will be a 1D array of datetime values
            dates_list = []

            # List to store tickers that have sufficient data
            # Only includes tickers with >= sequence_length data points
            valid_tickers = []

            for ticker in tqdm(full_data["ticker"].unique(), desc="Processing tickers"):
                ticker_data = full_data[full_data["ticker"] == ticker].copy()
                if len(ticker_data) >= self.sequence_length:
                    normalized_data = self._check_and_convert_format(
                        ticker_data, ticker
                    )
                    features = normalized_data[
                        ["open", "high", "low", "volume", "close"]
                    ].values
                    features_list.append(features)
                    dates_list.append(normalized_data["date"].values)
                    valid_tickers.append(ticker)

            features_array = np.stack(features_list)
            dates_array = dates_list[0]  # All dates should be the same for all tickers

            # Calculate returns
            close_prices = features_array[..., -1]
            returns = np.zeros_like(close_prices)
            returns[..., 1:] = (
                close_prices[..., 1:] - close_prices[..., :-1]
            ) / close_prices[..., :-1]

            # Add returns as a new feature
            features_array = np.dstack([features_array, returns[..., np.newaxis]])

            all_data_dict = {
                "features": features_array,
                "tickers": valid_tickers,
                "dates": dates_array,
            }

            with open(processed_file, "wb") as f:
                pickle.dump(all_data_dict, f)

        # Calculate split indices with sequence_length adjustment
        dates = pd.to_datetime(all_data_dict["dates"])

        # Create initial pred_idx array (all False)
        pred_idx = np.zeros_like(dates, dtype=bool)

        # Set True only for valid prediction points
        if self.train:
            if self.validation:
                start_date = pd.to_datetime(self.val_date[0])
                end_date = pd.to_datetime(self.val_date[1])
            else:
                start_date = pd.to_datetime(self.train_date[0])
                end_date = pd.to_datetime(self.train_date[1])
        else:
            start_date = pd.to_datetime(self.test_date[0])
            end_date = pd.to_datetime(self.test_date[1])

        # Find the index where our prediction start_date begins
        start_idx = np.where(dates >= start_date)[0][0]

        # Calculate relevant_start_idx by going back sequence_length steps
        relevant_start_idx = start_idx - self.sequence_length
        relevant_start = dates[relevant_start_idx]

        # Create relevant_idx that includes lookback window
        relevant_idx = (dates >= relevant_start) & (dates <= end_date)

        # Set prediction points (only after the lookback window)
        pred_idx[(dates >= start_date) & (dates <= end_date)] = True

        # Log the counts for verification
        logger.info(f"Sequence length: {self.sequence_length}")
        logger.info(f"Number of elements in relevant_idx: {sum(relevant_idx)}")
        logger.info(
            f"Number of prediction points (pred_idx): {sum(pred_idx[relevant_idx])}"
        )
        logger.info(
            f"Expected total: prediction points + sequence_length = {sum(pred_idx[relevant_idx]) + self.sequence_length}"
        )

        # Filter the data
        all_data_dict["features"] = all_data_dict["features"][:, relevant_idx]
        all_data_dict["dates"] = all_data_dict["dates"][relevant_idx]
        all_data_dict["pred_idx"] = pred_idx[relevant_idx]

        logger.info(
            f"Dataset split created with sequence_length={self.sequence_length}"
        )
        logger.info(
            f"Total timesteps: {len(all_data_dict['dates'])}, "
            f"Prediction timesteps: {sum(all_data_dict['pred_idx'])}"
        )
        logger.info(
            f"Date range: from {all_data_dict['dates'][0]} to {all_data_dict['dates'][-1]}"
        )

        return all_data_dict

    def _fetch_data(self):
        """Fetches stock data from OpenBB API."""
        # Fetch the list of constituents for the specified index
        constituents = obb.index.constituents(symbol=self.index, provider=self.provider)
        tickers = constituents["symbol"].tolist()

        # If selected_tickers is provided, filter the tickers list
        if self.selected_tickers:
            tickers = [ticker for ticker in tickers if ticker in self.selected_tickers]

        # Initialize an empty DataFrame to store the fetched data
        all_data = pd.DataFrame()

        # Set the overall start and end dates from scraping_date
        overall_start_date = datetime.strptime(self.scraping_date[0], "%Y-%m-%d")
        overall_end_date = datetime.strptime(self.scraping_date[1], "%Y-%m-%d")

        # Verify that we have enough historical data for the training start
        min_required_date = pd.to_datetime(self.train_date[0]) - pd.Timedelta(
            days=self.sequence_length
        )
        if min_required_date < pd.to_datetime(self.scraping_date[0]):
            raise ValueError(
                f"Not enough historical data. Need data from {min_required_date} but scraping starts at {self.scraping_date[0]}"
            )

        # Track the number of API calls
        api_call_count = 0
        api_call_limit = 300

        for ticker in tickers:
            current_start_date = overall_start_date
            ticker_data = pd.DataFrame()

            while current_start_date < overall_end_date:
                # Calculate the end date for the current window
                current_end_date = current_start_date + pd.Timedelta(days=3)
                if current_end_date > overall_end_date:
                    current_end_date = overall_end_date

                # Convert dates to string format for the API call
                start_str = current_start_date.strftime("%Y-%m-%d")
                end_str = current_end_date.strftime("%Y-%m-%d")

                logger.info(f"Fetching data for {ticker} from {start_str} to {end_str}")

                try:
                    data_chunk = obb.equity.price.historical(
                        symbol=ticker,
                        start_date=start_str,
                        end_date=end_str,
                        interval=self.interval,
                        provider=self.provider,
                    )

                    data_chunk["ticker"] = ticker
                    ticker_data = pd.concat([ticker_data, data_chunk])

                except Exception as e:
                    logger.error(f"Failed to fetch data for {ticker}: {e}")
                    break

                current_start_date = current_end_date
                api_call_count += 1

                if api_call_count >= api_call_limit:
                    logger.info("API call limit reached, waiting for 60 seconds...")
                    time.sleep(60)
                    api_call_count = 0

            # Remove duplicates from the ticker data
            ticker_data = ticker_data.drop_duplicates()

            # Save individual ticker data
            ticker_file = os.path.join(
                self.cache_dir,
                self.market,
                self.interval,
                self.index,
                "tickers",
                f"{ticker}.csv",
            )
            os.makedirs(os.path.dirname(ticker_file), exist_ok=True)
            ticker_data.to_csv(ticker_file)

            # Append to all_data
            all_data = pd.concat([all_data, ticker_data])

        # Remove duplicates and save all data
        all_data = all_data.drop_duplicates()
        all_file = os.path.join(
            self.cache_dir, self.market, self.interval, self.index, "ALL.csv"
        )
        os.makedirs(os.path.dirname(all_file), exist_ok=True)
        all_data.to_csv(all_file)

        return all_data

    def _add_time_idx(self, df):
        """Adds a time index to the DataFrame for each group (ticker)."""
        df.index = pd.to_datetime(df.index)
        df["date"] = df.index
        df["month"] = df.index.month
        df["day"] = df.index.day

        def create_time_idx(group):
            group["time_idx"] = pd.factorize(group.index)[0]
            return group

        df = df.groupby("ticker").apply(create_time_idx).reset_index(drop=True)
        return df

    def _check_and_convert_format(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Normalizes numerical columns and includes date information."""
        # Reset index to get date as a column
        df = df.reset_index()

        # Ensure the date column is datetime
        df["date"] = pd.to_datetime(df["date"])

        # Set both date and ticker as index
        df = df.set_index(["date", "ticker"])

        # Resample to 5-minute intervals and compute mean (using 'min' instead of 'T')
        df_resampled = df.resample("5min", level="date").mean()

        # Forward fill any missing values after resampling
        df_resampled = df_resampled.ffill()

        features = pd.DataFrame(index=df_resampled.index)
        features["date"] = df_resampled.index.get_level_values("date")
        features["date_index"] = range(len(df_resampled))

        columns_to_normalize = ["open", "high", "low", "volume", "close"]

        for col in columns_to_normalize:
            if col in df_resampled.columns:
                max_val = df_resampled[col].max()
                if max_val != 0:
                    features[col] = df_resampled[col] / max_val
                else:
                    features[col] = df_resampled[col]

        return features
