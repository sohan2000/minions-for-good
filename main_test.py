import os
from dotenv import load_dotenv
#from crew_def_test import ResearchCrew
#from crew_definition import AllAgentsCrew

from crew_def_test1 import AllAgentsCrew
#from crew_def_test2 import AllAgentsCrew


# Load environment variables
load_dotenv()

# Retrieve OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def main():
    # Pass input as a dictionary with the key matching the format string
    #input_data = {"input_data": "The impact of AI on the job market"}


    input_data = {
    "input_data": (
        "I am Minion and I am looking for a house. "
        "I want to buy/build a house where the focus is on wood work. "
        "I have the total budget of 100K and want the house in the next two years. "
        "I am open to +10% increase in my budget if my preferences are fulfilled, "
        "but if not then try to keep the budget as minimum as possible."
    )
}

    crew = AllAgentsCrew()
    result = crew.crew.kickoff(input_data)
    
    print("\nCrew Output:\n", result)

if __name__ == "__main__":
    main()