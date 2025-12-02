from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

import sys
from pathlib import Path

# Add the 3_crew directory to the path to import shared_llms
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from shared_llms import build_llms

@CrewBase
class MyCoder():
    """MyCoder crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self):
        # Build just the LLMs you need for this crew
        llms = build_llms(['grok-4-fast', 'gemini-2.5-flash', 'ollama-gemma27b', 'grok-code-fast-1'])
        self.llm_grok = llms['grok-4-fast']
        self.llm_gemini = llms['gemini-2.5-flash']
        self.llm_ollama = llms['ollama-gemma27b']
        self.llm_grokcf1 = llms['grok-code-fast-1']


    @agent
    def coder(self) -> Agent:
        return Agent(
            config=self.agents_config['coder'],
            verbose=True,
            allow_code_execution=True,
            code_execution_mode="safe",  # Uses Docker for safety
            max_execution_time=30, 
            max_retry_limit=3,
            llm=self.llm_grokcf1
    )

    @task
    def coding_task(self) -> Task:
        return Task(
            config=self.tasks_config['coding_task'],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the MyCoder crew"""

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
