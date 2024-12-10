import pandas as pd
import os
from datetime import datetime
import glob


def calculate_daily_variations_from_all(file_path="../data/US/1m/sp500/ALL.csv"):
    """
    Calculate daily variations using the consolidated ALL.csv file
    """
    try:
        # Read the consolidated file
        df = pd.read_csv(file_path)

        # Convert date column to datetime
        df["date"] = pd.to_datetime(df["date"])

        # Get daily first and last prices
        df["date_only"] = df["date"].dt.date
        daily_data = (
            df.groupby(["date_only", "ticker"])
            .agg({"open": "first", "close": "last"})
            .reset_index()
        )

        # Sort by date and ticker to ensure proper order
        daily_data = daily_data.sort_values(['ticker', 'date_only'])
        
        # Calculate variation using previous day's close
        daily_data['prev_close'] = daily_data.groupby('ticker')['close'].shift(1)
        daily_data['variation'] = (
            (daily_data['open'] - daily_data['prev_close']) / daily_data['prev_close']
        ) * 100
        
        # Remove first day for each ticker (no previous close available)
        daily_data = daily_data.dropna(subset=['prev_close'])

        return daily_data

    except Exception as e:
        print(f"Error processing ALL.csv: {e}")
        return None


def calculate_daily_variations_from_individual(directory_path="../data/US/1m/sp500"):
    """
    Calculate daily variations using individual stock files
    """
    try:
        all_daily_data = []
        failed_stocks = []

        # Get all CSV files in the directory
        csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
        total_files = len(csv_files)

        print(f"Processing {total_files} stock files...")

        for file_path in csv_files:
            ticker = os.path.basename(file_path).replace(".csv", "")
            try:
                # Read individual stock file
                df = pd.read_csv(file_path)
                print(f"Processing {ticker}: {df.shape}")

                # Convert date column to datetime
                df["date"] = pd.to_datetime(df["date"])

                # Get daily first and last prices
                df["date_only"] = df["date"].dt.date
                daily_data = (
                    df.groupby("date_only")
                    .agg({"open": "first", "close": "last", "ticker": "first"})
                    .reset_index()
                )

                # Sort by date to ensure proper order
                daily_data = daily_data.sort_values('date_only')
                
                # Calculate variation using previous day's close
                daily_data['prev_close'] = daily_data['close'].shift(1)
                daily_data['variation'] = (
                    (daily_data['open'] - daily_data['prev_close']) / daily_data['prev_close']
                ) * 100
                
                # Remove first day (no previous close available)
                daily_data = daily_data.dropna(subset=['prev_close'])

                all_daily_data.append(daily_data)

            except pd.errors.EmptyDataError:
                failed_stocks.append((ticker, "Empty file"))
                print(f"Error: {ticker} file is empty")
            except pd.errors.ParserError:
                failed_stocks.append((ticker, "File parsing error"))
                print(f"Error: Could not parse {ticker} file")
            except KeyError as e:
                failed_stocks.append((ticker, f"Missing column: {str(e)}"))
                print(f"Error: {ticker} missing required column: {str(e)}")
            except Exception as e:
                failed_stocks.append((ticker, str(e)))
                print(f"Error processing {ticker}: {str(e)}")

        # Print summary of processing results
        print("\n=== Processing Summary ===")
        print(f"Total files attempted: {total_files}")
        print(f"Successfully processed: {len(all_daily_data)}")
        print(f"Failed to process: {len(failed_stocks)}")

        if failed_stocks:
            print("\nFailed stocks and reasons:")
            for ticker, error in failed_stocks:
                print(f"- {ticker}: {error}")

        if not all_daily_data:
            print("\nERROR: No stocks were successfully processed")
            return None

        # Combine all successful stock data
        combined_data = pd.concat(all_daily_data, ignore_index=True)
        return combined_data

    except Exception as e:
        print(f"\nCritical error accessing directory {directory_path}: {e}")
        return None


class DailyVariations:
    def __init__(self):
        self.daily_data = {}
        self.daily_sector_data = {}  # New: Store sector-level data
        self.all_data = None
        self.sp500_info = None
        self._load_sp500_info()
    
    def _load_sp500_info(self):
        """Load SP500 company information"""
        try:
            self.sp500_info = pd.read_csv("../data/US/sp500.csv")
            print(f"Loaded SP500 info: {self.sp500_info.shape}")
        except Exception as e:
            print(f"Error loading SP500 info: {e}")
            self.sp500_info = None

    def store_variations(self, combined_data):
        """Store variations in an easily queryable format"""
        self.all_data = combined_data
        
        # Store data by date for individual stocks
        for date in combined_data["date_only"].unique():
            date_data = combined_data[combined_data["date_only"] == date].copy()
            self.daily_data[date] = date_data
            
            # Calculate sector-level variations
            if self.sp500_info is not None:
                sector_data = self._calculate_sector_variations(date_data)
                self.daily_sector_data[date] = sector_data

    def _calculate_sector_variations(self, date_data):
        """Calculate weighted average variations by sector"""
        # Merge with SP500 info to get sector information
        merged_data = date_data.merge(
            self.sp500_info[["symbol", "sector"]], 
            left_on="ticker", 
            right_on="symbol", 
            how="left"
        )
        
        # Calculate sector variations (weighted by previous closing price)
        sector_data = merged_data.groupby("sector").agg({
            "variation": lambda x: (x * merged_data.loc[x.index, "prev_close"]).sum() / merged_data.loc[x.index, "prev_close"].sum(),
            "open": "sum",  # Total sector market value at open
            "close": "sum",  # Total sector market value at close
            "prev_close": "sum",  # Total sector market value at previous close
            "ticker": "count"  # Number of companies in sector
        }).reset_index()
        
        return sector_data

    def get_variations_for_date(self, date, n=5):
        """
        Get top n positive and negative variations for a specific date
        Returns None, None if date not found
        """
        if date not in self.daily_data:
            print(f"No data available for date: {date}")
            return None, None

        date_data = self.daily_data[date]

        # Get top positive variations
        top_positive = date_data.nlargest(n, "variation")[
            ["ticker", "variation", "open", "close"]
        ]

        # Get top negative variations
        top_negative = date_data.nsmallest(n, "variation")[
            ["ticker", "variation", "open", "close"]
        ]

        return top_positive, top_negative

    def get_sector_variations_for_date(self, date, n=5):
        """Get top n positive and negative sector variations for a specific date"""
        if date not in self.daily_sector_data:
            print(f"No sector data available for date: {date}")
            return None, None

        sector_data = self.daily_sector_data[date]

        # Get top positive variations
        top_positive = sector_data.nlargest(n, "variation")[
            ["sector", "variation", "open", "close", "ticker"]
        ]

        # Get top negative variations
        top_negative = sector_data.nsmallest(n, "variation")[
            ["sector", "variation", "open", "close", "ticker"]
        ]

        return top_positive, top_negative

    def get_available_dates(self):
        """
        Return sorted list of available dates
        """
        return sorted(self.daily_data.keys())

    def generate_daily_report(self, date, top_positive, top_negative):
        """Generate a detailed markdown report for the given date"""
        if self.sp500_info is None:
            print("SP500 information not available")
            return None

        report_lines = [
            f"# Daily Stock Variation Report - {date}\n",
            "## Overview\n",
            "This report highlights the most significant daily variations in stock performance "
            "among S&P 500 companies. Each entry provides detailed information about the company, "
            "its performance, and relevant contextual information.\n",
            "## Top Positive Variations\n"
        ]

        # Process positive variations
        for _, row in top_positive.iterrows():
            report_lines.extend(self._generate_stock_entry(row, "positive"))

        report_lines.append("\n## Top Negative Variations\n")

        # Process negative variations
        for _, row in top_negative.iterrows():
            report_lines.extend(self._generate_stock_entry(row, "negative"))

        # Save report
        os.makedirs("../data/US/daily_reports_stocks", exist_ok=True)
        report_path = f"../data/US/daily_reports_stocks/{date}.md"
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))
        
        return report_path

    def _generate_stock_entry(self, row, variation_type):
        """Generate markdown entry for a single stock"""
        ticker = row["ticker"]
        variation = row["variation"]
        open_price = row["open"]
        close_price = row["close"]
        
        # Get company information
        company_info = self.sp500_info[self.sp500_info["symbol"] == ticker]
        if company_info.empty:
            return [f"### {ticker}\n",
                   f"- Variation: {variation:+.2f}%\n",
                   f"- Opening Price: ${open_price:.2f}\n",
                   f"- Closing Price: ${close_price:.2f}\n",
                   "- Company information not available\n\n"]

        info = company_info.iloc[0]
        
        # Extract the first year if there are multiple years
        founded_year = info['founded'].split('/')[0]
        try:
            years_of_history = 2024 - int(founded_year)
            history_text = f"{years_of_history} years of"
        except (ValueError, TypeError):
            print(f"{ticker} - {info['name']} - {info['founded']}")
            history_text = "significant"  # Fallback text if year calculation fails

        return [
            f"### {ticker} - {info['name']}\n",
            f"**Sector**: {info['sector']} ({info['subSector']})\n",
            f"**Headquarters**: {info['headQuarter']}\n",
            f"**Founded**: {info['founded']}\n",
            f"**Performance Metrics**:\n",
            f"- Daily Variation: {variation:+.2f}%\n",
            f"- Opening Price: ${open_price:.2f}\n",
            f"- Closing Price: ${close_price:.2f}\n",
            f"- Absolute Change: ${close_price - open_price:+.2f}\n\n",
            "**Analysis Context**:\n",
            f"This {variation_type} movement represents a significant change in the company's "
            f"market value. For a company with {info['sector']} sector expertise and "
            f"{history_text} operational history, such movements "
            "may be influenced by sector-specific trends, broader market conditions, "
            "or company-specific developments.\n\n"
        ]

    def generate_daily_sector_report(self, date, top_positive, top_negative):
        """Generate a detailed markdown report for sector performance"""
        report_lines = [
            f"# Daily Sector Variation Report - {date}\n",
            "## Overview\n",
            "This report highlights the most significant daily variations in sector performance "
            "within the S&P 500. Each entry provides detailed information about the sector "
            "and its aggregate performance.\n",
            "## Top Positive Sector Variations\n"
        ]

        # Process positive variations
        for _, row in top_positive.iterrows():
            report_lines.extend(self._generate_sector_entry(row, "positive"))

        report_lines.append("\n## Top Negative Sector Variations\n")

        # Process negative variations
        for _, row in top_negative.iterrows():
            report_lines.extend(self._generate_sector_entry(row, "negative"))

        # Save report
        os.makedirs("../data/US/daily_reports_sectors", exist_ok=True)
        report_path = f"../data/US/daily_reports_sectors/{date}.md"
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))
        
        return report_path

    def _generate_sector_entry(self, row, variation_type):
        """Generate markdown entry for a single sector"""
        sector = row["sector"]
        variation = row["variation"]
        open_value = row["open"]
        close_value = row["close"]
        company_count = row["ticker"]
        
        return [
            f"### {sector}\n",
            f"**Performance Metrics**:\n",
            f"- Daily Variation: {variation:+.2f}%\n",
            f"- Opening Total Value: ${open_value:,.2f}\n",
            f"- Closing Total Value: ${close_value:,.2f}\n",
            f"- Absolute Change: ${close_value - open_value:+,.2f}\n",
            f"- Number of Companies: {company_count}\n\n",
            "**Analysis Context**:\n",
            f"This {variation_type} movement represents a significant change in the sector's "
            f"overall market value. The {sector} sector, comprising {company_count} companies "
            "in this analysis, showed notable movement that may be influenced by "
            "sector-specific developments, broader market conditions, or "
            "macroeconomic factors.\n\n"
        ]


def main():
    # Choose whether to use ALL.csv or individual files
    use_all_csv = True# Set to False to use individual files

    # Calculate daily variations
    if use_all_csv:
        daily_data = calculate_daily_variations_from_all()
    else:
        daily_data = calculate_daily_variations_from_individual()

    if daily_data is None:
        print("Failed to process stock data")
        return

    # Save the processed data to CSV
    output_path = "../data/US/processed/daily_variations.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    daily_data.to_csv(output_path, index=False)
    print(f"\nSaved processed data to: {output_path}")

    # Create variations data structure
    variations = DailyVariations()
    variations.store_variations(daily_data)

    # Process each date and generate reports
    for date in variations.get_available_dates():
        print(f"\nProcessing report for date: {date}")
        
        # Generate stock-level report
        top_positive, top_negative = variations.get_variations_for_date(date)
        if top_positive is not None and top_negative is not None:
            report_path = variations.generate_daily_report(date, top_positive, top_negative)
            print(f"Generated stock report: {report_path}")
        
        # Generate sector-level report
        top_positive_sectors, top_negative_sectors = variations.get_sector_variations_for_date(date)
        if top_positive_sectors is not None and top_negative_sectors is not None:
            sector_report_path = variations.generate_daily_sector_report(
                date, top_positive_sectors, top_negative_sectors
            )
            print(f"Generated sector report: {sector_report_path}")


if __name__ == "__main__":
    main()
