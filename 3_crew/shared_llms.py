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
    'grok-code-fast-1',
    'gemini-2.5-flash',
    'gemini-2.5-pro',
    'ollama-deepseek32b',
    'ollama-gemma27b',
    'ollama-deepseek-coder',
    'ollama-qwen-coder'
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

def build_llms(llm_list, requests_per_minute=0):
    """
    Build LLM instances based on a list of model configuration names.

    Args:
        llm_list: List of model config names like ['grok-4-fast', 'gemini-2.5-flash', 'ollama-deepseek32b']
        requests_per_minute: Optional rate limit (requests per minute). If 0, no rate limiting.
                            If > 0, applies to ALL non-ollama models in llm_list.

                            Examples:
                            - 10 = 10 requests/min (1 request every 6 seconds)
                            - 2 = 2 requests/min (1 request every 30 seconds)
                            - 0 = No limit (default, backward compatible)

    Returns:
        Dictionary of LLM instances with same keys as input list

    Example:
        # No rate limiting
        llms = build_llms(['grok-4-fast', 'gemini-2.5-flash'])

        # Rate limit all models to 10 req/min
        llms = build_llms(['grok-4-fast', 'gemini-2.5-flash'], requests_per_minute=10)

        # Rate limit to 2 req/min (gemini free tier)
        llms = build_llms(['gemini-2.5-pro'], requests_per_minute=2)

    Raises:
        ValueError: If an unknown LLM configuration is requested
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
            temperature=0.7
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
            temperature=0.7
        )

    if 'grok-code-fast-1' in llm_list:
        grok_key = os.getenv("GROK_API_KEY")
        grok_base_url = os.getenv("GROK_BASE_URL")
        if not grok_key or not grok_base_url:
            raise ValueError("GROK_API_KEY or GROK_BASE_URL not set")
        llms['grok-code-fast-1'] = LLM(
            model="grok-code-fast-1",
            base_url=grok_base_url,
            api_key=grok_key,
            temperature=0.7
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
            temperature=0.7
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
            temperature=0.7
        )

    # Ollama models
    if 'ollama-deepseek32b' in llm_list:
        llms['ollama-deepseek32b'] = LLM(
            model="deepseek-r1:32b",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            temperature=0.7
        )

    if 'ollama-gemma27b' in llm_list:
        llms['ollama-gemma27b'] = LLM(
            model="gemma3:27B",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            temperature=0.7
        )

    if 'ollama-deepseek-coder' in llm_list:
        llms['ollama-deepseek-coder'] = LLM(
            model="deepseek-coder-v2:16b",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            temperature=0.7
        )

    if 'ollama-qwen-coder' in llm_list:
        llms['ollama-qwen-coder'] = LLM(
            model="qwen3-coder:30b",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            temperature=0.7
        )

    # Apply rate limiting if requested
    if requests_per_minute > 0:
        from rate_limited_llm import RateLimitedLLM

        logger.info(f"Applying rate limiting: {requests_per_minute} requests/minute to non-ollama models")

        # Wrap all non-ollama LLMs with rate limiting
        for model_name, llm_instance in llms.items():
            # Skip ollama models (they're local, no rate limits needed)
            if not model_name.startswith('ollama-'):
                llms[model_name] = RateLimitedLLM(llm_instance, requests_per_minute)
                logger.info(f"  - {model_name}: rate limited to {requests_per_minute} req/min")

    return llms
