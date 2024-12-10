from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import FileReadTool

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


@CrewBase
class CandidateNewsTracker:
    """CandidateNewsTracker crew"""

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def media_analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config["media_analyzer"],
            tools=[FileReadTool()],
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def stock_impact_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["stock_impact_analyst"],
            tools=[FileReadTool()],
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def sector_impact_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["sector_impact_analyst"],
            tools=[FileReadTool()],
            verbose=True,
            allow_delegation=False,
        )

    @task
    def media_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["media_analysis_task"],
            agent=self.media_analyzer(),
        )

    @task
    def stock_impact_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["stock_impact_analysis_task"],
            agent=self.stock_impact_analyst(),
            output_file="./output/stock_impact_analysis.md",
        )

    @task
    def sector_impact_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["sector_impact_analysis_task"],
            agent=self.sector_impact_analyst(),
            output_file="./output/sector_impact_analysis.md",
        )

    @crew
    def crew(self) -> Crew:
        """Creates the CandidateNewsTracker crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
