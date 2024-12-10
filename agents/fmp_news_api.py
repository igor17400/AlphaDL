import os
import requests
import pandas as pd
import time
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()  # Load environment variables
FMP_API_KEY = os.getenv("FMP_API")  # Load FMP API key from .env file

def fetch_articles(api_key: str, from_date: str, to_date: str, output_folder: str, limit: int = 750):
    # Convert string dates to datetime objects
    start_date = datetime.strptime(from_date, "%Y-%m-%d")
    end_date = datetime.strptime(to_date, "%Y-%m-%d")
    current_date = start_date
    
    while current_date <= end_date:
        try:
            from_date_str = current_date.strftime("%Y-%m-%d")
            to_date_str = (current_date + timedelta(days=1)).strftime("%Y-%m-%d")
            print("Date: ", from_date_str, to_date_str)
            url = f"https://financialmodelingprep.com/stable/news/general-latest?apikey={api_key}&from={from_date_str}&to={to_date_str}&limit={limit}"
            response = requests.get(url)
            response.raise_for_status()
            articles = response.json()
            
            print(f"Fetched {len(articles)} articles for {from_date_str} to {to_date_str}")

            # Save articles for this day to a CSV file
            if articles:  # Only save if we have articles
                output_file = os.path.join(output_folder, f"{from_date_str}.csv")
                df = pd.DataFrame(articles)
                df.to_csv(output_file, index=False)
                print(f"Articles saved to {output_file}")
            
            # Respect API rate limit - sleep every 700 requests to stay safely under the 750 limit
            if current_date.day % 700 == 0:
                print("Sleeping for 60 seconds to respect API rate limit...")
                time.sleep(60)
            
            # Move to next day
            current_date += timedelta(days=1)
            
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch articles for {from_date_str}: {e}")
            current_date += timedelta(days=1)  # Continue to next day even if there's an error

if __name__ == "__main__":
    api_key = FMP_API_KEY
    if not api_key:
        raise ValueError("FMP_API_KEY environment variable not set.")

    fetch_articles(
        api_key,
        from_date="2024-01-01",
        to_date="2024-11-07",
        output_folder="../data/US/fmp_news_output_per_day"
    )
