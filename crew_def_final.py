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
        
        client = Agent(
            role='Client',
            goal='Approve or reject the builder’s bundled bid, then wait for milestone reports.',
            backstory='Acts as the project owner and funding source.',
            verbose=self.verbose
        )

        builder = Agent(
            role='Content Summarizer',
            goal='Coordinate suppliers and deliver the best overall deal. Manage the whole construction.',
            backstory='Veteran general contractor; summarises quotes & picks winners',
            verbose=self.verbose
        )

        self.logger.info("Created research and writer agents")

        crew = Crew(
            agents=[client, builder],
            tasks=[
                Task(
                    description='Parse the client brief: {text}. Extract budget (+10  percent stretch)',
                    expected_output='Accepted budget and deadline. Given money and time constraints',
                    agent=client
                ),
                Task(
                    description='Give your best construction quote. ',
                    expected_output='Contruction quotes, managing the whole construction. Dividing money.',
                    agent=builder
                )
            ]
        )
        self.logger.info("Crew setup completed")
        return crew