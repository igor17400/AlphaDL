import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import glob
import argparse


class EntityEndorsementAnalyzer:
    def __init__(self, target_entities: List[str], candidate_name: str = None):
        self.target_entities = target_entities
        self.candidate_name_parts = candidate_name.lower().split() if candidate_name else None

    def process_article(self, article: Dict) -> Optional[List[Dict]]:
        results = []
        
        # Handle potential None or float values
        text = str(article.get("text", "")).lower() if article.get("text") is not None else ""
        title = str(article.get("title", "")).lower() if article.get("title") is not None else ""

        # First check candidate name parts if available
        if self.candidate_name_parts:
            if any(part in text or part in title for part in self.candidate_name_parts):
                results.append({
                    "entity": " ".join(self.candidate_name_parts).title(),
                    **article  # Include all fields from the original article
                })

        # Then check other entities
        for entity in self.target_entities:
            entity_lower = entity.lower()
            if entity_lower in text or entity_lower in title:
                results.append({
                    "entity": entity,
                    **article  # Include all fields from the original article
                })

        return results if results else None


class NewsAnalyzer:
    def __init__(self, input_dir: str, output_dir: str, country: str):
        """
        Initialize the analyzer with input/output directories and country

        Args:
            input_dir: Path to directory containing daily news files
            output_dir: Path to save entity cluster results
            country: Country code (e.g., 'US', 'UK')
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.country = country.upper()

        # Load campaign relations data based on country
        relations_file = Path(
            f"../data/{self.country}/entities_per_candidate/campaign_relations.json"
        )
        if not relations_file.exists():
            raise ValueError(
                f"No campaign relations file found for country: {self.country}"
            )

        with open(relations_file, "r") as f:
            self.campaign_data = json.load(f)

        # Create output directory
        self.create_output_directory()

    def create_output_directory(self):
        """Create necessary output directory for daily reports"""
        daily_reports_dir = self.output_dir / "daily_reports_news"
        daily_reports_dir.mkdir(parents=True, exist_ok=True)

    def process_daily_files(self):
        """Process all daily news files"""
        csv_files = glob.glob(str(self.input_dir / "*.csv"))

        for file_path in csv_files:
            try:
                file_date = Path(file_path).stem
                daily_news = pd.read_csv(file_path)

                # Process each candidate's entities
                daily_results = {}

                for candidate in self.campaign_data["candidates"]:
                    # Extract all related entities for this candidate
                    target_entities = []  # Remove candidate name from target_entities
                    for category in ["associates", "endorsers", "supporters"]:
                        target_entities.extend(
                            [
                                person["name"]
                                for person in candidate["relations"][category]
                            ]
                        )

                    # Pass candidate name separately for flexible matching
                    analyzer = EntityEndorsementAnalyzer(
                        target_entities=target_entities,
                        candidate_name=candidate["full_name"],
                    )
                    candidate_results = []

                    # Process each article
                    for _, article in daily_news.iterrows():
                        article_dict = article.to_dict()
                        results = analyzer.process_article(article_dict)
                        if results:
                            candidate_results.extend(results)

                    if candidate_results:
                        daily_results[candidate["id"]] = {
                            "candidate_data": candidate,
                            "results": candidate_results,
                        }

                if daily_results:
                    self.save_daily_report(file_date, daily_results)
                    print(f"Processed {file_path}")

            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

    def save_daily_report(self, date: str, results: Dict):
        """Save results as a single markdown file with sections for each candidate"""
        daily_reports_dir = self.output_dir / "daily_reports_news"
        markdown_file = daily_reports_dir / f"{date}.md"

        markdown_content = f"# Daily News Analysis Report - {date}\n\n"
        markdown_content += f"Country: {self.country}\n\n"

        # Add sections for each candidate
        for candidate_id, data in results.items():
            candidate = data["candidate_data"]
            articles = data["results"]

            markdown_content += f"Candidate: {candidate['full_name']}\n\n"
            markdown_content += f"Position: {candidate['position']}\n\n"

            # Add related entities section
            markdown_content += "Related Entities:\n\n"
            for category in ["associates", "endorsers", "supporters"]:
                markdown_content += f"{category.title()}:\n\n"
                for relation in candidate["relations"][category]:
                    markdown_content += f"\t•\t{relation['name']} ({relation['role']}): {relation['details']}\n"
                markdown_content += "\n"

            # Group articles by entity
            entity_articles = {}
            for article in articles:
                entity = article['entity']
                if entity not in entity_articles:
                    entity_articles[entity] = []
                entity_articles[entity].append(article)

            # Add news articles section
            markdown_content += "News Articles:\n\n"
            for entity, entity_articles_list in entity_articles.items():
                markdown_content += f"\t•\t{entity} in News:\n"
                for i, article in enumerate(entity_articles_list, 1):
                    markdown_content += f"\t{i}.\tTitle: {article['title']}\n"
                    markdown_content += f"\tPublisher: {article.get('publisher', 'N/A')}\n"
                    markdown_content += f"\tDate: {article['publishedDate']}\n"
                    markdown_content += f"\tSite: {article.get('site', 'N/A')}\n"
                    markdown_content += f"\tImage: {article.get('image', 'N/A')}\n"
                    markdown_content += f"\tText: {article.get('text', 'N/A')}\n"
                    markdown_content += f"\tURL: {article['url']}\n"
                    markdown_content += f"\tSymbol: {article.get('symbol', 'N/A')}\n"
                    markdown_content += "\n"

            markdown_content += "\n"  # Add space between candidates

        # Save the markdown file
        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)


if __name__ == "__main__":
    # Add argument parser
    parser = argparse.ArgumentParser(
        description="Analyze news articles for candidates in a specific country."
    )
    parser.add_argument(
        "-country", type=str, required=True, help="Country code (e.g., US, UK)"
    )

    args = parser.parse_args()

    # Define input and output directories
    input_dir = f"../data/{args.country.upper()}/fmp_news_output_per_day"
    output_dir = f"../data/{args.country.upper()}"

    # Initialize and run analyzer
    news_analyzer = NewsAnalyzer(
        input_dir=input_dir, output_dir=output_dir, country=args.country
    )

    # Process all files
    news_analyzer.process_daily_files()

    print("\nAnalysis complete!")
