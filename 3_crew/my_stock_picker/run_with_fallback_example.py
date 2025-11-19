#!/usr/bin/env python3
"""
Example: Running MyStockPicker with automatic LLM fallback

This demonstrates the CORRECT way to handle rate limits with CrewAI.
The RateLimitedLLM wrapper doesn't work with CrewAI's LLM class,
so we use crew-level fallback instead.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from crew_runner_with_fallback import run_crew_with_fallback
from my_stock_picker.crew import MyStockPicker


def main():
    """
    Run MyStockPicker with automatic fallback handling.

    This configuration:
    1. Tries Gemini/Grok mix first (best results, but rate-limited)
    2. Falls back to all-Grok if Gemini hits 429
    3. Finally falls back to local Ollama (unlimited, but lower quality)
    """
    print("=" * 60)
    print("Running MyStockPicker with Automatic LLM Fallback")
    print("=" * 60)
    print()

    # Define fallback chain
    llm_configs = [
        {
            # Config 1: Diverse mix (preferred)
            # - Gemini is fast for coordination
            # - Grok is thorough for research and rating
            'manager': 'gemini-2.5-flash',      # Fast coordinator
            'researcher': 'grok-4-fast',        # Thorough research
            'rater': 'grok-4-fast'              # Different perspective
        },
        {
            # Config 2: All Grok (fallback if Gemini hits rate limit)
            'manager': 'grok-4-fast',
            'researcher': 'grok-4-fast',
            'rater': 'grok-4-fast'
        },
        {
            # Config 3: All Ollama (final fallback - unlimited but local)
            'manager': 'ollama-gemma27b',
            'researcher': 'ollama-gemma27b',
            'rater': 'ollama-gemma27b'
        }
    ]

    try:
        result = run_crew_with_fallback(
            crew_class=MyStockPicker,
            llm_configs=llm_configs,
            inputs={},                # Stock picker doesn't need inputs
            max_retries=1             # Don't retry same config, move to next
        )

        print("\n" + "=" * 60)
        print("SUCCESS! Crew completed")
        print("=" * 60)
        print("\nResult:")
        print(result)

        return result

    except Exception as e:
        print("\n" + "=" * 60)
        print("FAILED: All LLM configurations failed")
        print("=" * 60)
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    # Run the crew with fallback
    result = main()
