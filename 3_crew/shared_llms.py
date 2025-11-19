import os
from pathlib import Path
from dotenv import load_dotenv
from crewai import LLM
import logging

logger = logging.getLogger(__name__)

# Available LLM configurations
AVAILABLE_LLMS = [
    'grok-4',
    'grok-4-fast',
    'gemini-2.5-flash',
    'gemini-2.5-pro',
    'ollama-deepseek32b',
    'ollama-gemma27b',
]

# Load environment variables from the main project's .env file
def load_project_env():
    """Load .env from the agents/ root directory"""
    project_root = Path(__file__).parent.parent
    env_path = project_root / '.env'
    load_dotenv(dotenv_path=env_path)

def list_available_llms():
    """
    Returns a list of all available LLM configuration names.

    Returns:
        List of strings with available LLM names

    Example:
        from shared_llms import list_available_llms
        print("Available LLMs:", list_available_llms())
    """
    return AVAILABLE_LLMS.copy()

def build_llms(llm_list):
    """
    Build LLM instances based on a list of model configuration names.

    Args:
        llm_list: List of model config names like ['grok-4-fast', 'gemini-2.5-flash', 'ollama-deepseek32b']

    Returns:
        Dictionary of LLM instances with same keys as input list

    Example:
        llms = build_llms(['grok-4-fast', 'gemini-2.5-flash', 'ollama-gemma27b'])

    Raises:
        ValueError: If an unknown LLM configuration is requested

    Note:
        For rate limiting, use the max_rpm parameter on Agent objects.
        See USING_MAX_RPM.md for details.
    """
    load_project_env()

    # Validate all requested LLMs are available
    for llm_name in llm_list:
        if llm_name not in AVAILABLE_LLMS:
            available = ', '.join(AVAILABLE_LLMS)
            raise ValueError(
                f"Unknown LLM configuration: '{llm_name}'. "
                f"Available options: {available}"
            )

    llms = {}

    # Grok models
    if 'grok-4' in llm_list:
        grok_key = os.getenv("GROK_API_KEY")
        grok_base_url = os.getenv("GROK_BASE_URL")
        if not grok_key or not grok_base_url:
            raise ValueError("GROK_API_KEY or GROK_BASE_URL not set")
        llms['grok-4'] = LLM(
            model="grok-4",
            base_url=grok_base_url,
            api_key=grok_key,
            temperature=0.7,
            max_retries=10
        )

    if 'grok-4-fast' in llm_list:
        grok_key = os.getenv("GROK_API_KEY")
        grok_base_url = os.getenv("GROK_BASE_URL")
        if not grok_key or not grok_base_url:
            raise ValueError("GROK_API_KEY or GROK_BASE_URL not set")
        llms['grok-4-fast'] = LLM(
            model="grok-4-fast",
            base_url=grok_base_url,
            api_key=grok_key,
            temperature=0.7,
            max_retries=10
        )

    # Gemini models
    if 'gemini-2.5-flash' in llm_list:
        gemini_key = os.getenv("GEMINI_API_KEY")
        gemini_base_url = os.getenv("GEMINI_BASE_URL")
        if not gemini_key or not gemini_base_url:
            raise ValueError("GEMINI_API_KEY or GEMINI_BASE_URL not set")
        llms['gemini-2.5-flash'] = LLM(
            model="gemini-2.5-flash",
            base_url=gemini_base_url,
            api_key=gemini_key,
            temperature=0.7,
            max_retries=10
        )

    if 'gemini-2.5-pro' in llm_list:
        gemini_key = os.getenv("GEMINI_API_KEY")
        gemini_base_url = os.getenv("GEMINI_BASE_URL")
        if not gemini_key or not gemini_base_url:
            raise ValueError("GEMINI_API_KEY or GEMINI_BASE_URL not set")
        llms['gemini-2.5-pro'] = LLM(
            model="gemini-2.5-pro",
            base_url=gemini_base_url,
            api_key=gemini_key,
            temperature=0.7,
            max_retries=10
        )

    # Ollama models
    if 'ollama-deepseek32b' in llm_list:
        llms['ollama-deepseek32b'] = LLM(
            model="deepseek-r1:32b",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            temperature=0.7,
            max_retries=2
        )

    if 'ollama-gemma27b' in llm_list:
        llms['ollama-gemma27b'] = LLM(
            model="gemma3:27B",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            temperature=0.7,
            max_retries=2
        )

    return llms
