from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from pydantic import BaseModel, Field

import sys
from pathlib import Path
import os

# Add the 3_crew directory to the path to import shared_llms
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from shared_llms import build_llms

from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
import faiss

class FAISSStorage(RAGStorage):
    def __init__(self, embedder_config, type, path):
        # Skip super().__init__ to avoid calling build_embedder
        self.embedder_config = embedder_config
        self.type = type
        self.path = path
        # Manually create the local embeddings
        self.embedding_function = HuggingFaceEmbeddings(model_name=embedder_config['config']['model'])
        # Load or create FAISS vectorstore
        if os.path.exists(path):
            self.vectorstore = FAISS.load_local(path, self.embedding_function, allow_dangerous_deserialization=True)
        else:
            # Create empty index
            dimension = len(self.embedding_function.embed_query("dummy text"))
            index = faiss.IndexFlatL2(dimension)
            self.vectorstore = FAISS(
                embedding_function=self.embedding_function.embed_query,
                index=index,
                docstore=InMemoryDocstore({}),
                index_to_docstore_id={},
            )
        # Set collection_name if needed (default from RAGStorage)
        self.collection_name = f"{type}_memory"

    def search(self, query, **kwargs):
        return self.vectorstore.similarity_search(query, **kwargs)
    
    def add(self, texts, metadatas=None, **kwargs):
        self.vectorstore.add_texts(texts, metadatas, **kwargs)
        self.vectorstore.save_local(self.path)

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
        """
        Initialize the crew with LLMs.

        Args:
            llm_overrides: Dict to override which LLM each configurable agent uses.
                          Useful for fallback scenarios or testing different LLM combinations.

                          Configurable agents (respect overrides):
                          - manager: Coordinates the hierarchical workflow
                          - researcher: Performs deep analysis on trending companies
                          - rater: Scores and rates investment potential

                          Fixed agents (ignore overrides, intentional for diverse perspectives):
                          - trending_company_finder_grok: Always uses Grok (fundamental analysis)
                          - trending_company_finder_gemini: Always uses Gemini (momentum analysis)

                          Example:
                          {
                              'manager': 'gemini-2.5-flash',     # Use Gemini as manager
                              'researcher': 'ollama-gemma27b',   # Use local Ollama for research
                              'rater': 'grok-4-fast'             # Use Grok for rating
                          }

                          Available LLMs: 'gemini-2.5-flash', 'grok-4-fast', 'ollama-gemma27b'

                          Default assignments (if not overridden):
                          - manager: gemini-2.5-flash (fast coordinator)
                          - researcher: grok-4-fast (thorough research)
                          - rater: grok-4-fast (rating & scoring)
        """
        # Build all available LLMs using shared_llms
        # Rate limiting is handled by max_rpm on Agent objects
        llms = build_llms(['grok-4-fast', 'gemini-2.5-flash', 'ollama-gemma27b'])
        self.llm_grok = llms['grok-4-fast']
        self.llm_gemini = llms['gemini-2.5-flash']
        self.llm_ollama = llms['ollama-gemma27b']

        # Map string names to actual LLM objects
        llm_map = {
            'gemini-2.5-flash': self.llm_gemini,
            'grok-4-fast': self.llm_grok,
            'ollama-gemma27b': self.llm_ollama
        }

        # Set defaults for agent roles
        defaults = {
            'manager': 'gemini-2.5-flash',    # Fast coordinator
            'researcher': 'grok-4-fast',      # Thorough research
            'rater': 'grok-4-fast'            # Rating & scoring
        }

        # Apply overrides if provided
        if llm_overrides:
            defaults.update(llm_overrides)

        # Assign LLMs to configurable roles
        self.manager_llm = llm_map.get(defaults['manager'], self.llm_gemini)
        self.researcher_llm = llm_map.get(defaults['researcher'], self.llm_grok)
        self.rater_llm = llm_map.get(defaults['rater'], self.llm_grok)

    @agent
    def trending_company_finder_grok(self) -> Agent:
        """Grok-powered finder focusing on fundamentals"""
        return Agent(
            config=self.agents_config['trending_company_finder_grok'],
            tools=[SerperDevTool()],
            llm=self.llm_grok,  # Always use Grok for this agent
            verbose=True,
            memory=True
        )

    @agent
    def trending_company_finder_gemini(self) -> Agent:
        """Gemini-powered finder focusing on momentum"""
        return Agent(
            config=self.agents_config['trending_company_finder_gemini'],
            tools=[SerperDevTool()],
            llm=self.llm_gemini,  # Always use Gemini for this agent
            verbose=True,
            max_rpm=3,  # 30% of Gemini's 10 req/min quota
            memory=True
        )

    @agent
    def financial_researcher(self) -> Agent:
        """Researcher that analyzes companies from both finders"""
        return Agent(
            config=self.agents_config['financial_researcher'],
            llm=self.researcher_llm,  # Configurable via llm_overrides
            tools=[SerperDevTool()],
            verbose=True,
            max_rpm=3 if self.researcher_llm == self.llm_gemini else None,  # 30% if using Gemini
            memory=True
        )

    @agent
    def stock_rater(self) -> Agent:
        """Rater that scores all researched companies"""
        return Agent(
            config=self.agents_config['stock_rater'],
            llm=self.rater_llm,  # Configurable via llm_overrides
            tools=[SerperDevTool()],
            verbose=True,
            max_rpm=2 if self.rater_llm == self.llm_gemini else None,  # 20% if using Gemini
            memory=True
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
            verbose=True,
            max_rpm=2 if self.manager_llm == self.llm_gemini else None  # 20% of Gemini quota (manager does most coordination)
        )

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical, # must have a manager for hierarchical
            manager_agent=manager, 
            # you could use manager_llm and give it one instead but this is "supposed" to be slightly better
            verbose=True,
            memory=True,
            # Long-term memory for persistent storage across sessions
            long_term_memory = LongTermMemory(
                storage=LTMSQLiteStorage(
                    db_path="./memory/long_term_memory_storage.db"
                )
            ),
            # Short-term memory for current context using RAG with FAISS
            short_term_memory = ShortTermMemory(
                storage = FAISSStorage(
                        embedder_config={
                            "provider": "huggingface",
                            "config": {
                                "model": 'sentence-transformers/all-MiniLM-L6-v2'
                            }
                        },
                        type="short_term",
                        path="./memory/short_term_faiss"
                    )
                ),            
            # Entity memory for tracking key information about entities with FAISS
            entity_memory = EntityMemory(
                storage=FAISSStorage(
                    embedder_config={
                        "provider": "huggingface",
                        "config": {
                            "model": 'sentence-transformers/all-MiniLM-L6-v2'
                        }
                    },
                    type="short_term",
                    path="./memory/entity_faiss"
                )
            ),
        )