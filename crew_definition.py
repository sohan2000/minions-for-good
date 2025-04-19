from crewai import Agent, Crew, Task
from typing import List
from logging_config import get_logger

# Masumi tool classes
class MasumiCommunicationTool:
    def send_message(self, recipient_id: str, message: str) -> bool:
        print(f"Sending message to {recipient_id}: {message}")
        return True

class MasumiTransactionTool:
    def initiate_payment(self, recipient_address: str, amount: float, description: str) -> str:
        print(f"Initiating payment of {amount} to {recipient_address} ({description})")
        return "tx_id_12345"

class MasumiContractTool:
    def register_contract(self, contract_details: dict) -> str:
        print(f"Registering contract: {contract_details}")
        return "contract_id_5678"
    def get_contract_status(self, contract_id: str) -> str:
        print(f"Getting contract status for {contract_id}")
        return "active"

class MasumiQueryTool:
    def find_agents_by_capability(self, capability: str) -> List[str]:
        print(f"Finding agents with capability: {capability}")
        if capability == "electrical subcontractor":
            return ["subcontractor_id_1", "subcontractor_id_2"]
        elif capability == "plumbing subcontractor":
            return ["plumbing_sub_1"]
        else:
            return []

class MasumiDataReportingTool:
    def report_data(self, data_type: str, data: dict) -> bool:
        print(f"Reporting data of type {data_type}: {data}")
        return True

class AllAgentsCrew:
    def __init__(self, verbose=True, logger=None):
        self.verbose = verbose
        self.logger = logger or get_logger(__name__)
        self.crew = self.create_crew()
        self.logger.info("AllAgentsCrew initialized")

    def create_crew(self):
        self.logger.info("Initializing Masumi tools")
        masumi_communication_tool = MasumiCommunicationTool()
        masumi_transaction_tool = MasumiTransactionTool()
        masumi_contract_tool = MasumiContractTool()
        masumi_query_tool = MasumiQueryTool()
        masumi_data_reporting_tool = MasumiDataReportingTool()

        self.logger.info("Defining agents")
        master_agent = Agent(
            role='Project Manager/Orchestrator',
            goal='Successfully manage the end-to-end project lifecycle.',
            backstory='Experienced project manager with strong leadership and communication skills.',
            tools=[masumi_communication_tool, masumi_query_tool],
            allow_delegation=True,
            verbose=self.verbose
        )

        client_agent = Agent(
            role='Client Interface',
            goal='Effectively communicate project requirements and provide feedback.',
            backstory='Represents the client\'s interests and needs.',
            tools=[masumi_communication_tool],
            allow_delegation=False,
            verbose=self.verbose
        )

        bidding_agent = Agent(
            role='Bid Management',
            goal='Efficiently manage the bidding process.',
            backstory='Skilled in procurement and vendor management.',
            tools=[masumi_communication_tool, masumi_query_tool],
            allow_delegation=True,
            verbose=self.verbose
        )

        scheduling_agent = Agent(
            role='Project Scheduling and Timeline Management',
            goal='Create and maintain a realistic project schedule.',
            backstory='Experienced in project planning and scheduling.',
            tools=[masumi_communication_tool],
            allow_delegation=False,
            verbose=self.verbose
        )

        payment_agent = Agent(
            role='Financial Transactions and Contract Management',
            goal='Securely handle payments and manage contract agreements.',
            backstory='Detail-oriented and knowledgeable in financial processes.',
            tools=[masumi_communication_tool, masumi_transaction_tool, masumi_contract_tool],
            allow_delegation=False,
            verbose=self.verbose
        )

        subcontractor_agent = Agent(
            role='Specialized Task Execution',
            goal='Efficiently execute specific tasks and provide progress updates.',
            backstory='Represents various specialized skills.',
            tools=[masumi_communication_tool, masumi_data_reporting_tool],
            allow_delegation=False,
            verbose=self.verbose
        )

        vision_agent = Agent(
            role='Quality Assurance and Progress Monitoring',
            goal='Assess work quality and track progress using visual data.',
            backstory='Equipped with image/data analysis capabilities.',
            tools=[masumi_data_reporting_tool],
            allow_delegation=False,
            verbose=self.verbose
        )

        self.logger.info("Agents defined")

        # Example tasks (customize as needed)
        tasks = [
            Task(
                description='Coordinate project kickoff with all stakeholders.',
                expected_output='Kickoff meeting scheduled and stakeholders notified.',
                agent=master_agent
            ),
            Task(
                description='Gather client requirements and feedback.',
                expected_output='Comprehensive list of client requirements.',
                agent=client_agent
            ),
            Task(
                description='Initiate and manage the bidding process.',
                expected_output='Bids collected and evaluated.',
                agent=bidding_agent
            ),
            Task(
                description='Develop and update the project schedule.',
                expected_output='Detailed project timeline.',
                agent=scheduling_agent
            ),
            Task(
                description='Handle payments and contract registration.',
                expected_output='Payments processed and contracts registered.',
                agent=payment_agent
            ),
            Task(
                description='Execute specialized project tasks and report progress.',
                expected_output='Task execution reports.',
                agent=subcontractor_agent
            ),
            Task(
                description='Monitor quality and progress using visual data.',
                expected_output='Quality assurance and progress reports.',
                agent=vision_agent
            ),
        ]

        crew = Crew(
            agents=[
                master_agent, client_agent, bidding_agent, scheduling_agent,
                payment_agent, subcontractor_agent, vision_agent
            ],
            tasks=tasks
        )

        self.logger.info("Crew setup completed")
        return crew