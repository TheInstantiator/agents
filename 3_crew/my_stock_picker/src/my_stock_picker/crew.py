from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from pydantic import BaseModel, Field

import sys
from pathlib import Path

# Add the 3_crew directory to the path to import shared_llms
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from shared_llms import build_llms

class TrendingCompany(BaseModel):
    """ A publicly traded company in the news that is trending. """
    name: str = Field(description="Name of the company.")
    ticker: str = Field(description="Stock ticker symbol of the publicly traded company.")
    price: float | None = Field(default=None, description="Current stock price.")
    projected_price: float | None = Field(default=None, description="Projected price of the company in the next 5 years")
    reason: str = Field(description="Reason why the company is trending.")

class TrendingCompanies(BaseModel):
    """ A list of trending companies. """
    companies: list[TrendingCompany] = Field(description="List of trending companies.")

class TrendingCompanyResearch(BaseModel):
    """ Comprehensive company research """
    name: str = Field(description="Company Name")
    market_position: str = Field(description="Current market position and competitive analysis")
    future_outlook: str = Field(description="Future outlook and growth prospects")
    investment_potential: str = Field(description="Investment potential and suitability for investment")

class TrendingCompanyResearchList(BaseModel):
    """ A list of detailed company research. """
    research_list: list[TrendingCompanyResearch] = Field(description="List of comprehensive company research.")

@CrewBase
class MyStockPicker():
    """MyStockPicker crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self, llm_overrides=None):
       
        # Build all available LLMs with rate limiting (10 requests/minute)
        # This prevents hitting API rate limits like Gemini's 10 req/min free tier
        llms = build_llms(
            ['grok-4-fast', 'gemini-2.5-flash', 'ollama-gemma27b'],
            requests_per_minute=10  # Limit API models to 10 req/min (ollama excluded)
        )
        self.llm_grok = llms['grok-4-fast']
        self.llm_gemini = llms['gemini-2.5-flash']
        self.llm_ollama = llms['ollama-gemma27b']

        # Map string names to actual LLM objects
        llm_map = {
            'gemini-2.5-flash': self.llm_gemini,
            'grok-4-fast': self.llm_grok,
            'ollama-gemma27b': self.llm_ollama
        }

        # Set defaults for diverse opinions
        defaults = {
            'manager': 'gemini-2.5-flash',
            'researcher': 'grok-4-fast',      # Grok for thorough research
            'analyst': 'gemini-2.5-flash',    # Gemini for quick analysis
            'rater': 'grok-4-fast'            # Grok again for different perspective
        }

        # Apply overrides if provided
        if llm_overrides:
            defaults.update(llm_overrides)

        # Assign LLMs to roles
        self.manager_llm = llm_map.get(defaults['manager'], self.llm_gemini)
        self.researcher_llm = llm_map.get(defaults['researcher'], self.llm_grok)
        self.analyst_llm = llm_map.get(defaults['analyst'], self.llm_gemini)
        self.rater_llm = llm_map.get(defaults['rater'], self.llm_grok)

    @agent
    def trending_company_finder_grok(self) -> Agent:
        """Grok-powered finder focusing on fundamentals"""
        return Agent(
            config=self.agents_config['trending_company_finder_grok'],
            tools=[SerperDevTool()],
            llm=self.llm_grok,  # Always use Grok for this agent
            verbose=True
        )

    @agent
    def trending_company_finder_gemini(self) -> Agent:
        """Gemini-powered finder focusing on momentum"""
        return Agent(
            config=self.agents_config['trending_company_finder_gemini'],
            tools=[SerperDevTool()],
            llm=self.llm_gemini,  # Always use Gemini for this agent
            verbose=True
        )

    @agent
    def financial_researcher(self) -> Agent:
        """Researcher that analyzes companies from both finders"""
        return Agent(
            config=self.agents_config['financial_researcher'],
            llm=self.llm_grok,  # Use Grok for thorough research
            tools=[SerperDevTool()],
            verbose=True
        )

    @agent
    def stock_rater(self) -> Agent:
        """Rater that scores all researched companies"""
        return Agent(
            config=self.agents_config['stock_rater'],
            llm=self.llm_grok,  # Use Grok for rating
            tools=[SerperDevTool()],
            verbose=True
        )
    
    @task
    def find_trending_companies_grok(self) -> Task:
        """Grok finds fundamental-focused companies"""
        return Task(
            config=self.tasks_config['find_trending_companies_grok'],
            output_pydantic=TrendingCompanies
        )

    @task
    def find_trending_companies_gemini(self) -> Task:
        """Gemini finds momentum-focused companies"""
        return Task(
            config=self.tasks_config['find_trending_companies_gemini'],
            output_pydantic=TrendingCompanies
        )

    @task
    def research_trending_companies(self) -> Task:
        return Task(
            config=self.tasks_config['research_trending_companies'],
            output_pydantic=TrendingCompanyResearchList
        )
    
    @task
    def rate_trending_companies(self) -> Task:
        return Task(
            config=self.tasks_config['rate_trending_companies'],
        )


    @crew
    def crew(self) -> Crew:
        """Creates the MyStockPicker crew"""

        manager = Agent(
            config=self.agents_config['manager'],
            llm=self.manager_llm,  # Use the configured manager LLM (supports fallback)
            allow_delegation=True,
            verbose=True
        )

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical, # must have a manager for hierarchical
            manager_agent=manager, # you could use manager_llm and give it one instead but this is "supposed" to be slightly better
            verbose=True,
        )
