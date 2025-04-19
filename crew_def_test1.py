# crew_definition.py
from crewai import Agent, Crew, Task

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
                "If delays are detected, it coordinates with the Scheduling Agent to adjust subcontractor timelines accordingly.",
            verbose=True,
            
        )
        
        bidding_agent = Agent(
            role="Bidding Agent",
            goal="Receive requirements from the Master Agent, collect bids from Subcontractor Agents, and select the most optimal bid.",
            backstory="The Bidding Agent manages competitive tendering. It evaluates proposals from subcontractors based on price, timelines, and quality. " \
            "Operates under the guidance of the Master Agent to ensure the best match for the project's needs.",
            verbose=True,
            
        )
        
        scheduling_agent = Agent(
            role='Scheduling Agent',
            goal='Preemptively schedule time with subcontractor agent based on the progress of the building. eg: schedule time with subcontractor 2 if the subcontractor 1 finishes work or 80% completion is met ',
            backstory='Coordinates task assignments based on milestones and AI insights given by Master Agent',
            verbose=True,
            
        )
        
        # Payments AI
        payments_agent = Agent(
            role="Payments Agent",
            goal="Disburse milestone-based payments using data from the Vision Agent.",
            backstory="Handles all financial transactions. First receives project funds from the Client Agent, then releases staged payments to subcontractors based on oversight by the Master Agent.",
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
            goal="Submit bids to the Bidding Agent without exceeding Subcontractor Agent 1’s bid. Execute tasks post-approval.",
            backstory="A newer subcontractor with a lean team. Offers competitive rates but is still building reputation. Works closely with Subcontractor Agent 1 during phased construction tasks.",
            verbose=True
        )
        
        
        # Vision AI
        vision_agent = Agent(
            role="Progress and Budget Tracking Agent",
            goal="Run predictions using ⁠ predict.py ⁠ and continuously update the Master Agent with insights on progress and budget.",
            backstory="An AI-powered monitoring agent that uses a computer vision model (⁠ predict.py ⁠) to analyze image stills from construction site cameras. It estimates percent completion, timeline status, and remaining days. Insights are shared with the Master Agent, which in turn informs the Scheduling and Payments Agents.",
            verbose=True
        )


        # suppliers
        construction_A = self._mk_company("ConstructionCo‑A", "Large firm with bulk discounts")
        construction_B = self._mk_company("ConstructionCo‑B", "Fast‑schedule medium builder")
        construction_C = self._mk_company("ConstructionCo‑C", "Boutique cost‑focused builder")

        wood_A = self._mk_company("WoodworkCo‑A", "Premium hardwood specialist")
        wood_B = self._mk_company("WoodworkCo‑B", "Budget‑friendly engineered‑wood shop")

        paint_A = self._mk_company("PaintCo‑A", "Eco‑friendly painter with 10‑year warranty")
        paint_B = self._mk_company("PaintCo‑B", "High‑volume, low‑cost painter")

        # tasks
        tasks = [
            Task(
                description='Client provides project requirements to Master Agent.',
                expected_output='Client requirements documented.',
                agent=client_agent
            ),
            Task(
                description='Establish smart contract between Client and Master through Payments Agent.',
                expected_output='Smart contract with proportional milestone payments created.',
                agent=payments_agent
            ),
            Task(
                description='Initiate and manage bidding process.',
                expected_output='Bids from subcontractors collected.',
                agent=bidding_agent
            ),
            Task(
                description='Evaluate bids and select subcontractor.',
                expected_output='Best bid selected and forwarded to Payments Agent.',
                agent=master_agent
            ),
            Task(
                description='Establish smart contract between Master and Subcontractor via Payments Agent.',
                expected_output='Smart contract established with proportional payment terms.',
                agent=payments_agent
            ),
            Task(
                description='Create and assign a schedule for subcontractor.',
                expected_output='Timeline with expected progress percentage established.',
                agent=scheduling_agent
            ),
            Task(
                description='Subcontractor executes scheduled task and reports progress.',
                expected_output='Work report with percent completion sent to Master Agent.',
                agent=subcontractor_agent1
            ),
            Task(
                description='Vision Agent analyzes progress from site footage.',
                expected_output='AI-based progress and delay report submitted.',
                agent=vision_agent
            ),
            Task(
                description='Verify subcontractor report with Vision Agent’s report.',
                expected_output='Cross-validated progress data.',
                agent=master_agent
            ),
            Task(
                description='Disburse payment to subcontractor and update smart contract.',
                expected_output='Funds transferred; smart contract updated.',
                agent=payments_agent
            ),
            Task(
                description='Reschedule or assign next subcontractor if progress ≥ 80%.',
                expected_output='Schedule updated; next subcontractor prepped if needed.',
                agent=scheduling_agent
            ),
            Task(
                description='Update client on construction progress.',
                expected_output='Progress report shared with Client Agent.',
                agent=master_agent
            ),
            Task(
                description='Request next installment from client based on progress.',
                expected_output='Funds requested and smart contract updated.',
                agent=payments_agent
            )
        ]


        agents = [client_agent, master_agent, bidding_agent, scheduling_agent, payments_agent, subcontractor_agent1, subcontractor_agent2, vision_agent]
        return Crew(agents=agents, tasks=tasks)
