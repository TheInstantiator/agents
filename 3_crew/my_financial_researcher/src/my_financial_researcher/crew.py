from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from the main project's .env file
# This goes up from: my_financial_researcher/src/my_financial_researcher/crew.py
# to: agents/.env
project_root = Path(__file__).parent.parent.parent.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

@CrewBase
class ResearchCrew():
    """Financial Research crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self):
        grok_key = os.getenv("GROK_API_KEY")
        grok_base_url = os.getenv("GROK_BASE_URL")
        print("GROK_API_KEY:", grok_key[:4])
        print("GROK_BASE_URL:", grok_base_url)

        gemini_key = os.getenv("GEMINI_API_KEY")
        gemini_base_url = os.getenv("GEMINI_BASE_URL")
        print("GEMINI_API_KEY:", gemini_key[:4])
        print("GEMINI_BASE_URL:", gemini_base_url)

        if not grok_key:
            raise ValueError("GROK_API_KEY environment variable is not set.")
        
        if not grok_base_url:
            raise ValueError("GROK_BASE_URL environment variable is not set.")
        
        if not gemini_base_url:
            raise ValueError("GOOGLE_BASE_URL environment variable is not set.")
        
        if not gemini_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")

        # Configure the LLM for Grok here (do this once and reuse for agents)
        self.llm_grok = LLM(
            model="grok-4-fast",  # Use the appropriate Grok model name (e.g., grok-4 for Grok 4)
            base_url=grok_base_url,
            api_key=grok_key,  # Replace with your actual key or use os.getenv for security
            temperature=0.7  # Adjust as needed
        )

        self.llm_gemini = LLM(
            model="gemini-2.5-flash",  # Use the appropriate Grok model name (e.g., grok-4 for Grok 4)
            base_url=gemini_base_url,
            api_key=gemini_key,  # Replace with your actual key or use os.getenv for security
            temperature=0.7  # Adjust as needed
        )

        self.llm_ollama = LLM(
            model="deepseek-r1:32b",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            temperature=0.7
        )

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            llm=self.llm_grok,
            verbose=True,
            tools=[SerperDevTool()]
        )

    @agent
    def analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['analyst'],
            llm=self.llm_gemini,  # Assign the Grok LLM here
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
        )

    @task
    def analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['analysis_task'],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Debate crew"""

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )

