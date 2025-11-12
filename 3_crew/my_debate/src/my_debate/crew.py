from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

import sys
from pathlib import Path

# Add the 3_crew directory to the path to import shared_llms
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from shared_llms import build_llms

@CrewBase
class Debate():
    """Debate crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self):
        # Build just the LLMs you need for this crew
        llms = build_llms(['grok-4-fast', 'gemini-2.5-flash', 'ollama-gemma27b'])
        self.llm_grok = llms['grok-4-fast']
        self.llm_gemini = llms['gemini-2.5-flash']
        self.llm_ollama = llms['ollama-gemma27b']

    @agent
    def debater(self) -> Agent:
        return Agent(
            config=self.agents_config['debater'],
            llm=self.llm_grok,  # Assign the Grok LLM here
            verbose=True
        )

    @agent
    def judge(self) -> Agent:
        return Agent(
            config=self.agents_config['judge'],
            llm=self.llm_gemini,  # Assign the Grok LLM here
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

