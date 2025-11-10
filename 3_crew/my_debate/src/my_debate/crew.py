from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
import os  # Add this import if using environment variables for API key

@CrewBase
class Debate():
    """Debate crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self):
        grok_key = os.getenv("GROK_API_KEY")
        grok_base_url = os.getenv("GROK_BASE_URL")
        print("GROK_API_KEY:", grok_key[:4])
        print("GROK_BASE_URL:", grok_base_url)

        gemini_key = os.getenv("GOOGLE_API_KEY")
        gemini_base_url = os.getenv("GOOGLE_BASE_URL")
        print("GOOGLE_API_KEY:", gemini_key[:4])
        print("GOOGLE_BASE_URL:", gemini_base_url)

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
    def debater(self) -> Agent:
        return Agent(
            config=self.agents_config['debater'],
            llm=self.llm_ollama,  # Assign the Grok LLM here
            verbose=True
        )

    @agent
    def judge(self) -> Agent:
        return Agent(
            config=self.agents_config['judge'],
            llm=self.llm_ollama,  # Assign the Grok LLM here
            verbose=True
        )

    @task
    def propose(self) -> Task:
        return Task(
            config=self.tasks_config['propose'],
        )

    @task
    def oppose(self) -> Task:
        return Task(
            config=self.tasks_config['oppose'],
        )

    @task
    def decide(self) -> Task:
        return Task(
            config=self.tasks_config['decide'],
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

