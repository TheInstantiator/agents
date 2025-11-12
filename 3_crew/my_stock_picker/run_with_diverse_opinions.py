#!/usr/bin/env python3
"""
Example: Run my_stock_picker with diverse LLM opinions and intelligent fallback

This demonstrates:
1. Using different LLMs for different agents (diverse perspectives)
2. Automatic fallback if any LLM fails
3. Strategic configuration ordering

Usage:
    python run_with_diverse_opinions.py
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
    """Run the stock picker with diverse LLM opinions and fallback"""

    # Define fallback chain with diverse opinions
    llm_configs = [
        {
            # CONFIGURATION 1: Diverse Opinions (Ideal)
            # Use Grok for research (thorough), Gemini for analysis (fast)
            # This gives you different "perspectives" on the stocks
            'manager': 'gemini-2.5-flash',    # Fast coordinator
            'researcher': 'grok-4-fast',       # Deep research with Grok
            'analyst': 'gemini-2.5-flash',     # Quick analysis with Gemini
            'rater': 'grok-4-fast'             # Different rating perspective
        },
        {
            # CONFIGURATION 2: Fallback if Grok unavailable
            # All Gemini - still works but less diversity
            'manager': 'gemini-2.5-flash',
            'researcher': 'gemini-2.5-flash',
            'analyst': 'gemini-2.5-flash',
            'rater': 'gemini-2.5-flash'
        },
        {
            # CONFIGURATION 3: Final fallback (all local)
            # Use local Ollama - always available, no rate limits
            'manager': 'ollama-gemma27b',
            'researcher': 'ollama-gemma27b',
            'analyst': 'ollama-gemma27b',
            'rater': 'ollama-gemma27b'
        }
    ]

    # Get user input
    sector = input("Enter a sector to research (e.g., technology, healthcare): ")

    print(f"\nStarting stock picker for sector: {sector}")
    print("\nLLM Configuration Strategy:")
    print("  1. DIVERSE: Grok (research) + Gemini (analysis) for different perspectives")
    print("  2. FALLBACK: All Gemini if Grok unavailable")
    print("  3. LOCAL: All Ollama (always works)\n")

    try:
        result = run_crew_with_fallback(
            crew_class=MyStockPicker,
            llm_configs=llm_configs,
            inputs={'sector': sector},
            max_retries=2
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
        print("CREW FAILED WITH ALL CONFIGURATIONS")
        print("="*80)
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
