import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Set, Tuple
import argparse
import re


class CompanyNewsTracker:
    def __init__(self):
        """Initialize the tracker with S&P 500 company data"""
        # Load S&P 500 companies data
        with open('./sp500.json', 'r') as f:
            sp500_data = json.load(f)
        
        # Process company names into a more structured format
        self.company_data = self._process_company_names(sp500_data)

    def _process_company_names(self, sp500_data: List[Dict]) -> Dict[str, Dict]:
        """
        Process company names to create various matching patterns
        Returns a dict with full names as keys and matching info as values
        """
        company_data = {}
        
        common_suffixes = r'\s+(Inc\.?|Corp\.?|Corporation|Company|Co\.?|Ltd\.?|Limited|LLC|Group|Incorporated|Holdings|Plc)\.?'
        
        for company in sp500_data:
            full_name = company['name']
            symbol = company['symbol']
            
            # Clean and store the original name
            clean_name = re.sub(common_suffixes, '', full_name, flags=re.IGNORECASE).strip()
            
            # Store company information
            company_data[full_name] = {
                'clean_name': clean_name,
                'symbol': symbol,
                'sector': company['sector'],
                'variations': set()
            }
            
            # Add the clean name as a variation
            company_data[full_name]['variations'].add(clean_name.lower())
            
            # Add the stock symbol as a variation with boundaries
            company_data[full_name]['variations'].add(rf"\b{symbol}\b")
            
            # Handle special cases
            if "The " in full_name:
                without_the = full_name.replace("The ", "")
                company_data[full_name]['variations'].add(without_the.lower())
            
            # Add full name with boundaries
            company_data[full_name]['variations'].add(re.escape(full_name.lower()))
            
            # Don't add individual words for companies with location names
            if not any(state in full_name for state in ['Texas', 'Virginia', 'California', 'Florida', 'Georgia']):
                # Only add specific words if they're unique enough
                unique_words = [word for word in clean_name.split() 
                              if len(word) > 3 and word.lower() not in {
                                  'inc', 'corp', 'ltd', 'the', 'and', 'company', 
                                  'group', 'international', 'incorporated'
                              }]
                
                if len(unique_words) == 1:  # Only add if it's a unique identifier
                    company_data[full_name]['variations'].add(rf"\b{unique_words[0]}\b")

        return company_data

    def find_company_mentions(self, text: str) -> List[Tuple[str, str]]:
        """
        Find mentions of S&P 500 companies in text
        Returns list of tuples (full_company_name, matched_text)
        """
        text_lower = text.lower()
        matches = []
        
        for full_name, data in self.company_data.items():
            for pattern in data['variations']:
                # If it's a regex pattern (starts with \b)
                if pattern.startswith('\\b'):
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        matches.append((full_name, pattern))
                        break
                # Otherwise do simple string matching
                elif pattern in text_lower:
                    matches.append((full_name, pattern))
                    break
        
        return matches


class CompanyNewsAnalyzer:
    def __init__(self, candidate: str):
        """
        Initialize the analyzer for a specific candidate
        
        Args:
            candidate: Name of the candidate to analyze
        """
        self.candidate = candidate
        self.tracker = CompanyNewsTracker()
        
        # Setup input/output paths
        self.input_path = Path("entities_cluster") / candidate.lower() / "complete_analysis.csv"
        self.output_dir = Path("company_analysis") / candidate.lower()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_articles(self):
        """Process all articles and track company mentions"""
        # Read the complete analysis file
        df = pd.read_csv(self.input_path)
        
        company_mentions = []
        
        # Process each article
        for _, row in df.iterrows():
            text = f"{row['title']}"  # Using just the title for now
            
            # Find company mentions
            mentioned_companies = self.tracker.find_company_mentions(text)
            
            if mentioned_companies:
                for full_name, matched_text in mentioned_companies:
                    mention = {
                        'company': full_name,
                        'matched_text': matched_text,
                        'candidate': self.candidate,
                        'article_url': row['url'],
                        'article_date': row['date'],
                        'article_title': row['title']
                    }
                    company_mentions.append(mention)

        # Convert results to DataFrame
        if company_mentions:
            results_df = pd.DataFrame(company_mentions)
            
            # Save complete results
            results_df.to_csv(self.output_dir / "company_mentions.csv", index=False)
            
            # Save summary by company
            summary_df = results_df.groupby('company').agg({
                'article_title': 'count'
            }).reset_index()
            summary_df.columns = ['company', 'mention_count']
            summary_df = summary_df.sort_values('mention_count', ascending=False)
            summary_df.to_csv(self.output_dir / "company_mentions_summary.csv", index=False)
            
            return results_df
        
        return pd.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Track S&P 500 company mentions in candidate news articles.')
    parser.add_argument('-cand', '--candidate', 
                       type=str,
                       required=True,
                       choices=['Trump', 'Harris'],
                       help='Candidate name to analyze (Trump or Harris)')

    args = parser.parse_args()

    # Initialize and run analyzer
    analyzer = CompanyNewsAnalyzer(args.candidate)
    results = analyzer.process_articles()

    print("\nAnalysis complete!")
    if not results.empty:
        print(f"Total company mentions found: {len(results)}")
        print("\nTop companies by mention count:")
        company_counts = results['company'].value_counts().head(10)
        print(company_counts)
    else:
        print("No company mentions found in the articles.")
