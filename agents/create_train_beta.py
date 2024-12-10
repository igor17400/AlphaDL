import pandas as pd
import glob
import re
from pathlib import Path
import mistune
from bs4 import BeautifulSoup


def parse_markdown_report(file_path, starting_name="stock_impact_analysis_"):
    """Parse markdown report and extract all stock-candidate pairs."""
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Extract date from the filename
    report_date = Path(file_path).stem.split(starting_name)[-1]

    # Extract stock information
    stocks = []
    current_stock = {}
    reason_key = "Why do you think this stock is affected?"

    for line in lines:
        line = line.strip()  # Remove leading and trailing whitespace

        if "Candidate Name" in line:
            if current_stock:  # Save the last stock if exists
                stocks.append(current_stock)
            current_stock = {
                "Candidate Name": line.split(":")[1].replace("**", "").strip()
            }
        elif "Stock Ticker" in line:
            current_stock["Stock Ticker"] = line.split(":")[1].replace("**", "").strip()
        elif "Stock Variation" in line:
            current_stock["Stock Variation"] = (
                line.split(":")[1].replace("**", "").strip()
            )
        elif "Confidence Level" in line:
            current_stock["Confidence Level"] = (
                line.split(":")[1].replace("**", "").strip()
            )
        elif "Impact Strength" in line:
            current_stock["Impact Strength"] = (
                line.split(":")[1].replace("**", "").strip()
            )
        elif "Sentiment" in line:
            current_stock["Sentiment"] = line.split(":")[1].replace("**", "").strip()
        elif reason_key in line:
            current_stock["Reason"] = line[len(reason_key) :].strip()

    # Append the last stock
    if current_stock:
        stocks.append(current_stock)

    # Print debug info only if no stocks were found
    if not stocks:
        print(f"\nError: No stocks found in {file_path}")
        print("*****TEXT CONTENT*****")
        print("".join(lines))
        print("*********************")

    report_data = []

    required_fields = [
        "Candidate Name",
        "Stock Ticker",
        "Confidence Level",
        "Impact Strength",
    ]
    # Process each stock entry
    for stock in stocks:
        # Skip entries with None or N/A values
        if any(stock.get(field) in ["None", "N/A", ""] for field in required_fields):
            print(f"\nSkipped file {file_path} because of NAN fields")
            continue

        # Skip if required fields are missing
        if not all(field in stock for field in required_fields):
            print("--------")
            print(f"\nError: Required fields missing in {file_path}")
            print(stock)
            print("--------")
            continue

        try:
            # Map sentiment to numerical values
            sentiment_mapping = {"Positive": 1, "Negative": -1, "Neutral": 0}
            sentiment_value = sentiment_mapping.get(
                stock.get("Sentiment", "").strip(), 0
            )

            # Map candidates to binary values
            candidate_binary = 1 if "Trump" in stock.get("Candidate Name", "") else 0

            # Append parsed data
            report_data.append(
                {
                    "date_day": report_date,
                    "ticker": stock["Stock Ticker"].strip(),
                    "candidate": candidate_binary,
                    "confidence_level": int(stock["Confidence Level"]),
                    "impact_strength": int(stock["Impact Strength"]),
                    "sentiment": sentiment_value,
                }
            )
        except (ValueError, KeyError, AttributeError):
            # Skip entries that cause conversion errors
            continue

    return report_data


def create_train_beta():
    # Read original training data
    train_df = pd.read_csv("../data/US/1m/sp500/train.csv")
    # Drop duplicate rows based on 'date_day' and 'ticker' columns
    train_df.drop_duplicates(subset=["date", "ticker"], inplace=True)

    # Convert date to datetime and filter for dates before April 2024
    train_df["date"] = pd.to_datetime(train_df["date"])
    train_df = train_df[train_df["date"] < "2024-04-01"]

    # Convert date to string format for merging
    train_df["date_day"] = train_df["date"].dt.strftime("%Y_%m_%d")

    # Parse all report files
    report_data = []
    report_files = glob.glob(
        "../data/US/candidate_news_tracker/output/stock_impact_analysis_*.md"
        # "../data/US/candidate_news_tracker/output/stock_impact_analysis_2024_02_09.md"
    )

    report_dates = []
    for file_path in report_files:
        parsed_data = parse_markdown_report(
            Path(file_path), starting_name="stock_impact_analysis_"
        )
        date = Path(file_path).stem.split("stock_impact_analysis_")[-1]
        report_dates.append(date)
        if parsed_data:
            report_data.extend(parsed_data)

    # Report Dates
    report_dates = sorted(report_dates)  # Sort in ascending order
    print("\nNumber of report dates:", len(report_dates))

    # Report Dataframe
    report_df = pd.DataFrame(report_data)
    print("--- report_df ---")
    print(report_df.shape)
    print(report_df.head())

    # Get sorted unique days from reports
    report_df_dates = sorted(report_df["date_day"].unique())
    print("\nNumber of dataframe report days:", len(report_df_dates))

    # Get sorted unique days from training data
    train_days = sorted(train_df["date_day"].unique())
    print("\nNumber of training days:", len(train_days))

    # Find dates that are in original reports but not in dataframe reports
    missing_in_df_report = set(report_dates) - set(report_df_dates)
    if missing_in_df_report:
        print(
            "\nDates in original reports but missing in dataframe reports:",
            sorted(missing_in_df_report),
        )
    # Find dates that are in training data but not in original reports
    missing_in_original_report = set(train_days) - set(report_dates)
    if missing_in_original_report:
        print(
            "\nDates in training data but missing in original reports:",
            sorted(missing_in_original_report),
        )
    # Find dates that are in training data but not in dataframe reports
    missing_in_report_df = set(train_days) - set(report_df_dates)
    if missing_in_report_df:
        print(
            "\nDates in training data but missing in dataframe reports:",
            sorted(missing_in_report_df),
        )

    # Merge with original training data
    train_beta = train_df.merge(report_df, on=["date_day", "ticker"], how="left")

    # Changing default values
    train_beta.fillna(
        {
            "candidate": -1,
            "confidence_level": -1,
            "impact_strength": -1,
            "sentiment": 0,
        },
        inplace=True,
    )

    print("--- train_df ---")
    print(train_df.shape)
    print(train_df.head())
    print("--- train_beta ---")
    print(train_beta.shape)
    print(train_beta.head())

    # # Save the new dataset
    train_beta.to_csv("./train_beta.csv", index=False)


if __name__ == "__main__":
    create_train_beta()
