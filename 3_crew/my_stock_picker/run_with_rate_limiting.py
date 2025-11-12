#!/usr/bin/env python3
"""
Example: Run my_stock_picker with rate limiting to avoid API quota errors

This demonstrates how to use rate limiting to stay within API limits like:
- Gemini 2.5 Flash: 10 requests/minute (free tier)
- Gemini 2.5 Pro: 2 requests/minute (free tier)

The rate limiting is transparent - crews work exactly the same, but requests
are automatically throttled to stay within limits.

Usage:
    python run_with_rate_limiting.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from my_stock_picker.crew import MyStockPicker


def main():
    """Run the stock picker with rate limiting enabled"""

    print("=" * 80)
    print("MY STOCK PICKER - WITH RATE LIMITING")
    print("=" * 80)

    # Get user input
    sector = input("\nEnter a sector to research (e.g., technology, healthcare): ")

    print(f"\nStarting stock picker for sector: {sector}")
    print("\nRate Limiting Configuration:")
    print("  - Gemini 2.5 Flash: 10 requests/minute limit")
    print("  - Grok 4 Fast: 10 requests/minute limit")
    print("  - Ollama: No limit (local)")
    print("\nNote: Requests will be automatically throttled to stay within limits.")
    print("      You'll see log messages when rate limiting is active.\n")

    try:
        # Initialize crew
        # The rate limiting is configured inside MyStockPicker.__init__()
        # See crew.py for implementation
        crew_instance = MyStockPicker()

        # Run the crew
        print("Starting crew execution...")
        print("-" * 80)

        result = crew_instance.crew().kickoff(inputs={'sector': sector})

        print("-" * 80)
        print("\n" + "=" * 80)
        print("CREW COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nFinal Output:\n{result.raw}")

        print("\n" + "=" * 80)
        print("Output files created:")
        print("  - output/trending_companies_grok.json")
        print("  - output/trending_companies_gemini.json")
        print("  - output/research_report.json")
        print("  - output/decision.md")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print("CREW FAILED")
        print("=" * 80)
        print(f"Error: {e}")

        # Check if it's a rate limit error (shouldn't happen with rate limiting enabled)
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['rate limit', 'quota', '429', 'resource_exhausted']):
            print("\n⚠️  Rate limit error occurred despite rate limiting!")
            print("    This might mean:")
            print("    1. Rate limit is set too high for your tier")
            print("    2. Other applications are using the same API key")
            print("    3. Multiple concurrent agents exceeded combined limit")
            print("\n    Try reducing requests_per_minute in crew.py")

        sys.exit(1)


if __name__ == "__main__":
    main()
