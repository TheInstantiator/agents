import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

def get_providers():
    """Returns the API configuration for different LLM providers."""
    return {
        'grok': {
            'base_url': os.getenv("GROK_BASE_URL", "https://api.x.ai/v1"),
            'api_key': os.getenv("GROK_API_KEY")
        },
        'gemini': {
            'base_url': os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
            # Tier 1 key explicitly chosen by user to avoid using standard key:
            'api_key': os.getenv("GEMINI_API_KEYX")
        },
        'ollama': {
            'base_url': os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            'api_key': "ollama"  # ollama requires a key but accepts any string
        }
    }

def get_available_models():
    """Fetches all accessible models from the configured providers."""
    providers = get_providers()
    available_models = {}

    for name, config in providers.items():
        if not config['api_key']:
            print(f"[{name.upper()}] Warning: API KEY missing. Skipping.")
            continue

        try:
            client = OpenAI(base_url=config['base_url'], api_key=config['api_key'])
            models = client.models.list()
            all_models = [m.id for m in models.data]
            available_models[name] = all_models
        except Exception as e:
            print(f"[{name.upper()}] Error fetching models: {e}")

    return available_models

def get_best_model_for_job(job_type="general", available_models=None):
    """
    Selects the best available model for a specific job type.
    Job types could be: 'coding', 'reasoning', 'vision', 'general'
    """
    if available_models is None:
        available_models = get_available_models()

    # Flatten the list of all available models across providers
    flattened_models = []
    for provider_models in available_models.values():
        flattened_models.extend(provider_models)

    # Define heuristics / preferences for different job types (in order of preference)
    preferences = {
        'coding': ['qwen3-coder:480b-cloud', 'grok-code-fast-1', 'grok-3', 'models/gemini-2.5-pro'],
        'reasoning': ['deepseek-r1:32b', 'grok-4-1-fast-reasoning', 'models/gemini-2.5-pro'],
        'vision': ['grok-imagine-image-pro', 'models/gemini-2.5-flash-image', 'models/veo-3.0-generate-001'],
        'general': ['grok-3-mini', 'models/gemini-2.5-flash', 'gemma3:27B', 'qwen3:32b']
    }

    prefs = preferences.get(job_type, preferences['general'])

    # Return the first available model that matches our preferences
    for pref in prefs:
        if pref in flattened_models:
            return pref

    # Fallback to the first available model if preferences don't match
    if flattened_models:
        return flattened_models[0]

    return None

if __name__ == "__main__":
    print("Fetching available models...\n")
    models_dict = get_available_models()

    for provider, models in models_dict.items():
        print(f"--- {provider.upper()} models ({len(models)} available) ---")
        for model in models:
            print(f"  {model}")
        print()

    print("--- Best Models by Job Type ---")
    jobs = ['coding', 'reasoning', 'vision', 'general']
    for job in jobs:
        best = get_best_model_for_job(job, models_dict)
        print(f"Best for '{job:<10}': {best}")