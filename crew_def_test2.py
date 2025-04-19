import os
import litellm
from crewai import Agent, Crew, Task


# Configure litellm to use Ollama
#os.environ["OPENAI_MODEL_NAME"] = "GPT-4o mini"  # Replace with your Ollama server address if needed

class AllAgentsCrew:
    """Multi‑agent crew that gathers multiple quotes (2‑sentence limit per agent)."""
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.crew = self._create_crew()

    # ---------- helpers ----------
    def _mk_company(self, role, backstory):
        return Agent(
            role=role,
            goal="Provide a concise, two‑sentence quote for its trade.",
            backstory=backstory,
            verbose=self.verbose,
        )

    # ---------- crew ----------
    def _create_crew(self) -> Crew:
        # Client
        client_agent = Agent(
            role="Client Agent",
            goal="Define project requirements, request progress reports, and sign a contract with the Builder company.",
            backstory="The Client represents the end-user and initiates the construction project. Responsible for outlining project  \
                requirements, reviewing progress updates, and authorizing key milestones and payments. Interacts directly with the  \
                Master Agent to ensure the project stays aligned with expectations.",
            verbose=True
        )

        # Construction Company
        master_agent = Agent(
            role="Master Agent",
            goal="Oversee the full construction lifecycle, verify inputs, trigger tasks, and report to the Client Agent.",
            backstory="The Master Agent acts as the Builder company's brain, orchestrating the entire project pipeline. " \
                "It manages the Bidding, Scheduling, Payments, and Vision Agents, while communicating directly with the Client. " \
                "It uses insights from Vision AI to determine when to release payments and assess if timelines or budgets need adjustment. " \
                "If delays are detected, it coordinates with the Scheduling Agent to adjust subcontractor timelines accordingly." \
                "It evaluates proposals from subcontractors based on price, timelines, and quality. " \
                "Operates under the guidance of the Master Agent to ensure the best match for the project's needs."\
                "Preemptively schedule time with subcontractor agent based on the progress of the building. eg: schedule time with subcontractor 2 if the subcontractor 1 finishes work or 80 percent completion is met"\
                "Coordinates task assignments based on milestones and AI insights given by Vision Agent"\
                "Once the Vision agent gives a progress of 80 percentage from the contractor, you schedule the next contractor for the next job"\
                "Disburse milestone-based payments to sub-contractors using data from the Vision Agent."\
                "Handles all financial transactions. First receives project funds from the Client Agent, then releases staged payments to subcontractors. The name of Master agent is BOB."\
                "Make sure that the total completion of the project is 100 percent and accordingly money is always disbursed. It can be multiple payments to the subcontractors",
            verbose=True,
        )

        # Subcontractor Agent
        subcontractor_agent1 = Agent(
            role="Subcontractor Agent 1",
            goal="Submit competitive bids to the Bidding Agent and execute tasks when scheduled.",
            backstory="A trusted subcontractor who specializes in foundational construction work. Known for reliability and fair pricing. Competes with Subcontractor Agent 2 during bidding.",
            verbose=True
        )


        subcontractor_agent2 = Agent(
            role="Subcontractor Agent 2",
            goal="Submit bids to the Bidding Agent without exceeding Subcontractor Agent 1’s bid. Execute tasks post-approval. Provide a concise, two‑sentence quote for its trade",
            backstory="A newer subcontractor with a lean team. Offers competitive rates but is still building reputation. Works closely with Subcontractor Agent 1 during phased construction tasks.",
            verbose=True
        )


        # Vision AI
        vision_agent = Agent(
            role="Progress and Budget Tracking Agent",
            goal="Run predictions using and continuously update the Master Agent with insights on progress and budget.",
            backstory="An AI-powered monitoring agent that uses a computer vision model to analyze image stills from construction site cameras."\ 
            "It estimates percent completion, timeline status, and remaining days. Insights are shared with the Master Agent, which in turn informs the Scheduling and Payments Agents.",
            verbose=True
        )

        tasks = [
            Task(
                description='Coordinate project kickoff with all stakeholders with these requirements: {input_data}.',
                expected_output='Kickoff meeting scheduled and stakeholders notified.',
                agent=client_agent
            ),
            Task(
                description='Contract between Client and Master.',
                expected_output='Smart contract created between Client and Master Agent.',
                agent=master_agent
            ),
            Task(
                description='Gather client requirements and feedback.',
                expected_output='Comprehensive list of client requirements.',
                agent=master_agent
            ),
            Task(
                description='Initiate and manage the bidding process.',
                expected_output='Bids collected and evaluated.',
                agent=master_agent
            ),
            Task(
                description='Evaluate bids from subcontractors.',
                expected_output='Winning subcontractor selected based on criteria.',
                agent=master_agent
            ),
            Task(
                description='Execute foundational construction work.',
                expected_output='Foundation phase completed with report.',
                agent=subcontractor_agent1
            ),
            Task(
                description='Execute follow-up construction work.',
                expected_output='Follow-up phase completed with report.',
                agent=subcontractor_agent2
            ),
            Task(
                description='Monitor quality and progress using visual data.',
                expected_output='Quality assurance and progress reports.',
                agent=vision_agent
            ),
            Task(
                description='Handle milestone-based payments and contract updates.',
                expected_output='Payments disbursed and contracts updated. Complete the project with 100 percent completion.',
                agent=master_agent
            ),
        ]


        #crew = Crew(
        #    agents=[client_agent, master_agent, subcontractor_agent1, subcontractor_agent2, vision_agent],
        #    tasks=tasks,
        #    verbose=True
        #)

        agents = [client_agent, master_agent, subcontractor_agent1, subcontractor_agent2, vision_agent]

        return Crew(agents=agents, tasks=tasks)