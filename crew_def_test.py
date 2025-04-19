from crewai import Agent, Crew, Task

class ResearchCrew:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.crew = self.create_crew()

    def create_crew(self):
        researcher = Agent(
            role='Research Analyst',
            goal='Find and analyze key information',
            backstory='Expert at extracting information',
            verbose=self.verbose
        )

        writer = Agent(
            role='Content Summarizer',
            goal='Create clear summaries from research',
            backstory='Skilled at transforming complex information',
            verbose=self.verbose
        )

        crew = Crew(
            agents=[researcher, writer],
            tasks=[
                Task(
                    description='Research: {input_data}',
                    expected_output='Detailed research findings about the topic',
                    agent=researcher
                ),
                Task(
                    description='Write summary',
                    expected_output='Clear and concise summary of the research findings',
                    agent=writer
                )
            ]
        )
        return crew