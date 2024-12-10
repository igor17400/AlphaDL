from typing import Any, Optional, Type
from pydantic import BaseModel, Field
from crewai_tools.tools.base_tool import BaseTool

import os
import requests
import pandas as pd
import re


class FMPNewsToolSchema(BaseModel):
    """Input for FMPNewsTool."""

    keyword: str = Field(..., description="Keyword to search for in the news articles")


class FMPNewsTool(BaseTool):
    name: str = "Search in FMP news articles"
    description: str = (
        "A tool that can be used to search for news articles mentioning specific keywords from the FMP API."
    )
    args_schema: Type[BaseModel] = FMPNewsToolSchema

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not os.environ.get("FMP_API_KEY"):
            raise ValueError("FMP_API_KEY environment variable not set.")

    def fetch_all_articles(self) -> pd.DataFrame:
        """Fetches news articles from FMP API and returns them as a DataFrame."""
        all_articles = []
        for page in range(2):
            try:
                api_key = os.environ.get("FMP_API_KEY")
                url = f"https://financialmodelingprep.com/api/v4/general_news?page={page}&apikey={api_key}"
                response = requests.get(url)
                response.raise_for_status()
                all_articles.extend(response.json())
            except requests.exceptions.RequestException as e:
                print(f"Failed to fetch page {page}: {e}")
                break

        df = pd.DataFrame(all_articles)
        df.to_csv("./output/fmp_news.csv", index=False)
        return df

    def filter_articles(self, keyword: str) -> pd.DataFrame:
        """Fetches and filters articles containing the specified keyword."""
        articles = self.fetch_all_articles()
        return articles[
            articles["text"].str.contains(keyword, flags=re.IGNORECASE, na=False)
        ]

    def _run(self, **kwargs: Any) -> Any:
        # Validate input using the schema
        input_data = self.args_schema(**kwargs)
        keyword = input_data.keyword

        filtered_articles = self.filter_articles(keyword)
        return filtered_articles.to_dict(orient="records")


# Example usage
if __name__ == "__main__":
    tool = FMPNewsTool()
    results = tool._run(keyword="Donald Trump")
    for article in results:
        print(article["title"], article["url"])
