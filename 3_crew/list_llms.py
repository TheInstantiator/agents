#!/usr/bin/env python3
"""
Simple CLI tool to list available LLM configurations
Usage: python list_llms.py (from anywhere in 3_crew hierarchy)
"""

import sys
from pathlib import Path

# Add the 3_crew directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

# Import only the list we need, avoiding heavy dependencies
def get_available_llms():
    """Read the available LLMs directly without importing crewai/dotenv"""
    # Just read the constant from the file
    llms = [
        'grok-4',
        'grok-4-fast',
        'gemini-2.5-flash',
        'gemini-2.5-pro',
        'ollama-deepseek32b',
        'ollama-gemma27b',
    ]
    return llms

if __name__ == "__main__":
    print("Available LLM configurations:")
    print("-" * 40)
    for llm in get_available_llms():
        print(f"  - {llm}")
    print("-" * 40)
    print(f"Total: {len(get_available_llms())} configurations")
