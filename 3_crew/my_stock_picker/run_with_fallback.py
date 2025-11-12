#!/usr/bin/env python3
"""
Example: Run my_stock_picker with LLM fallback

This script demonstrates running the stock picker crew with automatic
fallback to different LLMs if one fails (e.g., rate limiting).

Usage:
    python run_with_fallback.py
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
from crew_runner_with_fallback import run_crew_with_fallback

# Import your crew
sys.path.append(str(Path(__file__).parent / "src"))
from my_stock_picker.crew import MyStockPicker


def main():
    """Run the stock picker with fallback LLMs"""

    # Define the fallback chain
    # Try Gemini first (fastest), then Grok (most capable), then Ollama (local fallback)
    manager_llms = [
        'gemini-2.5-flash',   # Try this first
        'grok-4-fast',        # Fall back to this if Gemini fails
        'ollama-gemma27b'     # Final fallback (local, always available)
    ]

    # Get user input
    sector = input("Enter a sector to research (e.g., technology, healthcare): ")

    print(f"\nStarting stock picker for sector: {sector}")
    print(f"LLM fallback chain: {' -> '.join(manager_llms)}\n")

    try:
        result = run_crew_with_fallback(
            crew_class=MyStockPicker,
            manager_llms=manager_llms,
            inputs={'sector': sector},
            max_retries=2  # Retry each LLM up to 2 times before moving to next
        )

        print("\n" + "="*80)
        print("CREW COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\nFinal Output:\n{result.raw}")

        print("\n" + "="*80)
        print("Output files created:")
        print("  - output/trending_companies.json")
        print("  - output/research_report.json")
        print("  - output/decision.md")
        print("="*80)

    except Exception as e:
        print("\n" + "="*80)
        print("CREW FAILED WITH ALL LLMS")
        print("="*80)
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
