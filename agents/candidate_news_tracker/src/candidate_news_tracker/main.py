#!/usr/bin/env python
import sys
import warnings
from candidate_news_tracker.crew import CandidateNewsTracker

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """
    Run the crew.
    """
    media_date = "2024-10-30"
    report_date = "2024-10-31"
    inputs = {
        "media_date": media_date,
        "report_date": report_date,
        "reference_candidate_party_one": "Donald Trump",
        "reference_candidate_party_two": "Kamala Harris",
        "reference_associated_entity_party_one": "Vivek Ramaswamy",
        "reference_associated_entity_party_two": "Tim Walz",
        "path_to_csv_sp500": f"../../data/US/sp500.csv",
        "path_to_daily_media": f"../../data/US/daily_reports_news/{media_date}.md",
        "path_to_daily_reports_stocks": f"../../data/US/daily_reports_stocks/{report_date}.md",
        "path_to_daily_reports_sectors": f"../../data/US/daily_reports_sectors/{report_date}.md",
    }
    CandidateNewsTracker().crew().kickoff(inputs=inputs)

def train():
    """Train the crew."""
    inputs = {"topic": "AI LLMs"}
    try:
        CandidateNewsTracker().crew().train(
            n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs
        )
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """Replay the crew execution."""
    try:
        CandidateNewsTracker().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """Test the crew execution."""
    inputs = {"topic": "AI LLMs"}
    try:
        CandidateNewsTracker().crew().test(
            n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs
        )
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
