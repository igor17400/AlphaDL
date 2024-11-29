import os
import logging
from datetime import datetime
from openbb import obb
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set your OpenBB API key
openbb_api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdXRoX3Rva2VuIjoiYU1zSm5idXl2V3lZY0tmY25jR3NjYnJNWjhkVjJaZVdWcERNRndCOSIsImV4cCI6MTc1OTcyNDQwOX0.2bbtSUb6fY59vRuup1yTUfyLNOwqV04Ea5LnPyBTfTk"
obb.account.login(pat=openbb_api_key)
obb.user.preferences.output_type = "dataframe"

# Configuration
ticker = "AAPL"
train_date = ["2024-10-21", "2024-10-30"]
val_date = ["2024-10-31", "2024-11-04"]
test_date = ["2024-11-05", "2024-11-06"]
interval = "1m"
provider = "fmp"


def fetch_data(ticker, start_date, end_date, interval, provider):
    logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
    data = obb.equity.price.historical(
        symbol=ticker,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        provider=provider,
    )
    logger.info(f"Fetched data head:\n{data.head()}")
    logger.info(f"Fetched data tail:\n{data.tail()}")
    return data


def split_data(data, train_date, val_date, test_date):
    # Convert date strings to datetime objects
    train_start_date, train_end_date = map(
        lambda x: datetime.strptime(x, "%Y-%m-%d"), train_date
    )
    val_start_date, val_end_date = map(
        lambda x: datetime.strptime(x, "%Y-%m-%d"), val_date
    )
    test_start_date, test_end_date = map(
        lambda x: datetime.strptime(x, "%Y-%m-%d"), test_date
    )

    # Split the data
    train_data = data[(data.index >= train_start_date) & (data.index <= train_end_date)]
    val_data = data[(data.index >= val_start_date) & (data.index <= val_end_date)]
    test_data = data[(data.index >= test_start_date) & (data.index <= test_end_date)]

    # Log the size of each dataset
    logger.info(f"Train data size: {train_data.shape}")
    logger.info(f"Validation data size: {val_data.shape}")
    logger.info(f"Test data size: {test_data.shape}")

    return train_data, val_data, test_data


def main():
    # Fetch data for the entire period
    overall_start_date = train_date[0]
    overall_end_date = test_date[1]
    full_data = fetch_data(
        ticker, overall_start_date, overall_end_date, interval, provider
    )

    # Split the data
    train_data, val_data, test_data = split_data(
        full_data, train_date, val_date, test_date
    )

    # Optionally, save the data to CSV for inspection
    train_data.to_csv("train_data.csv")
    val_data.to_csv("val_data.csv")
    test_data.to_csv("test_data.csv")


if __name__ == "__main__":
    main()
