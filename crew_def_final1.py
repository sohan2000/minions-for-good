from crewai import Agent, Crew, Task
from logging_config import get_logger

class ResearchCrew:
    def __init__(self, verbose=True, logger=None):
        self.verbose = verbose
        self.logger = logger or get_logger(__name__)
        self.crew = self.create_crew()
        self.logger.info("ResearchCrew initialized")

    def create_crew(self):
        self.logger.info("Creating research crew with agents")
        
        builder = Agent(
            role='Client',
            goal='Approve or reject the builder’s bundled bid, then wait for milestone reports.',
            backstory='Acts as the project owner and funding source.',
            verbose=self.verbose
        )

        client = Agent(
            role='Builder',
            goal='Coordinate suppliers and deliver the best overall deal',
            backstory='Veteran general contractor; summarises quotes & picks winners.',
            verbose=self.verbose
        )

        self.logger.info("Created research and writer agents")

        crew = Crew(
            agents=[builder, client],
            tasks=[
                Task(
                    description=(
                        "Parse the client brief: {input_data}. Extract budget (+10 % stretch), "
                        "deadline and key preferences. Reply in **≤2 sentences**."
                    ),
                    expected_output="Req brief in 2 sentences",
                    agent=builder,
                ),
                Task(
                    description=(
                        "Compare all quotes against budget & preferences. Pick one construction, "
                        "one woodwork and one paint company. Explain each quote **and** why the "
                        "combination is chosen. No strict length limit here."
                    ),
                    expected_output=(
                        "Final deal: chosen suppliers, combined cost/duration, justification."
                    ),
                    agent=builder,
                )
            ]
        )
        self.logger.info("Crew setup completed")
        return crew