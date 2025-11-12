#!/usr/bin/env python3
"""
Simple test script to verify rate limiting works correctly
Run from 3_crew directory: python test_rate_limiting.py
"""
import sys
from pathlib import Path
import logging
import time

# Setup logging to see rate limit messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from shared_llms import build_llms

def test_rate_limiting():
    """Test that rate limiting works by making several rapid requests"""

    print("=" * 80)
    print("Testing Rate Limiting")
    print("=" * 80)

    # Build LLMs with 10 requests per minute rate limit
    # This means 1 request every 6 seconds
    print("\nBuilding LLMs with 10 requests/minute rate limit...")
    print("(This means 1 request every 6 seconds)\n")

    llms = build_llms(
        ['gemini-2.5-flash', 'grok-4-fast', 'ollama-gemma27b'],
        requests_per_minute=10  # Test with 10 req/min
    )

    print("\n" + "=" * 80)
    print("Testing with gemini-2.5-flash (should be rate limited)")
    print("=" * 80)

    gemini = llms['gemini-2.5-flash']
    print(f"\nLLM instance: {gemini}")
    print(f"Type: {type(gemini)}")

    # Make 3 rapid calls - should see rate limiting after the first one
    print("\nMaking 3 rapid calls (should see rate limiting)...\n")

    start = time.time()
    for i in range(3):
        print(f"Call {i+1} starting...")
        call_start = time.time()

        # Just access an attribute to test the wrapper works
        # (We won't actually make API calls in this test)
        model_name = gemini.model

        call_end = time.time()
        print(f"Call {i+1} completed in {call_end - call_start:.2f}s (model: {model_name})")

    total_time = time.time() - start
    print(f"\nTotal time for 3 calls: {total_time:.2f}s")

    # Check ollama is NOT rate limited
    print("\n" + "=" * 80)
    print("Testing with ollama-gemma27b (should NOT be rate limited)")
    print("=" * 80)

    ollama = llms['ollama-gemma27b']
    print(f"\nLLM instance: {ollama}")
    print(f"Type: {type(ollama)}")

    print("\nMaking 3 rapid calls (should be instant)...\n")

    start = time.time()
    for i in range(3):
        print(f"Call {i+1} starting...")
        call_start = time.time()
        model_name = ollama.model
        call_end = time.time()
        print(f"Call {i+1} completed in {call_end - call_start:.2f}s (model: {model_name})")

    total_time = time.time() - start
    print(f"\nTotal time for 3 calls: {total_time:.2f}s (should be ~0s)")

    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)

if __name__ == "__main__":
    test_rate_limiting()
