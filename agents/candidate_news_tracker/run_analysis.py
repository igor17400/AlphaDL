import os
import shutil
import subprocess
from datetime import datetime, timedelta


def update_main_py(media_date, report_date):
    """Update main.py with new dates"""
    main_content = f'''#!/usr/bin/env python
import sys
import warnings
from candidate_news_tracker.crew import CandidateNewsTracker

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """
    Run the crew.
    """
    media_date = "{media_date}"
    report_date = "{report_date}"
    inputs = {{
        "media_date": media_date,
        "report_date": report_date,
        "reference_candidate_party_one": "Donald Trump",
        "reference_candidate_party_two": "Kamala Harris",
        "reference_associated_entity_party_one": "Vivek Ramaswamy",
        "reference_associated_entity_party_two": "Tim Walz",
        "path_to_csv_sp500": f"../../data/US/sp500.csv",
        "path_to_daily_media": f"../../data/US/daily_reports_news/{{media_date}}.md",
        "path_to_daily_reports_stocks": f"../../data/US/daily_reports_stocks/{{report_date}}.md",
        "path_to_daily_reports_sectors": f"../../data/US/daily_reports_sectors/{{report_date}}.md",
    }}
    CandidateNewsTracker().crew().kickoff(inputs=inputs)

def train():
    """Train the crew."""
    inputs = {{"topic": "AI LLMs"}}
    try:
        CandidateNewsTracker().crew().train(
            n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs
        )
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {{e}}")

def replay():
    """Replay the crew execution."""
    try:
        CandidateNewsTracker().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {{e}}")

def test():
    """Test the crew execution."""
    inputs = {{"topic": "AI LLMs"}}
    try:
        CandidateNewsTracker().crew().test(
            n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs
        )
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {{e}}")
'''

    with open("src/candidate_news_tracker/main.py", "w") as f:
        f.write(main_content)


def save_reports_with_date(report_date):
    """Save the output reports with date in filename"""
    output_dir = "../../data/US/candidate_news_tracker/output"
    os.makedirs(output_dir, exist_ok=True)

    # Format date for filename
    date_str = report_date.replace("-", "_")

    # Save stock analysis report
    if os.path.exists("output/stock_impact_analysis.md"):
        shutil.copy(
            "output/stock_impact_analysis.md",
            f"{output_dir}/stock_impact_analysis_{date_str}.md",
        )

    # Save sector analysis report
    if os.path.exists("output/sector_impact_analysis.md"):
        shutil.copy(
            "output/sector_impact_analysis.md",
            f"{output_dir}/sector_impact_analysis_{date_str}.md",
        )


def run_analysis():
    """Run analysis for all January dates"""
    # January 2024 dates to analyze
    report_dates = [
        "01",
        "02",
        "03",
        "04",
        # "05",
        # "06",
        "07",
        "08",
        "09",
        "10",
        "11",
        # "12",
        # "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        # "19",
        # "20",
        "21",
        "22",
        "23",
        "24",
        "25",
        # "26",
        # "27",
        "28",
        "29",
        "30",
        "31",
    ]

    for report_day in report_dates:
        # Calculate media date (previous day)
        report_date = f"2024-10-{report_day}"
        media_date = (
            datetime.strptime(report_date, "%Y-%m-%d") - timedelta(days=1)
        ).strftime("%Y-%m-%d")

        print(f"\nProcessing: Media Date: {media_date}, Report Date: {report_date}")

        try:
            # Update main.py with new dates
            update_main_py(media_date, report_date)

            # Run CrewAI
            subprocess.run(["crewai", "run"], check=True)

            # Save reports with date
            save_reports_with_date(report_date)

            print(f"Successfully processed {report_date}")

        except subprocess.CalledProcessError as e:
            print(f"Error processing {report_date}: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error processing {report_date}: {e}")
            continue


if __name__ == "__main__":
    run_analysis()
