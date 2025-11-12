#!/usr/bin/env python3
"""
Debug test to understand what's happening with rate limiter
"""
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rate_limited_llm import RateLimitedLLM


class MockLLM:
    """Mock LLM for testing"""
    def __init__(self, model_name):
        self.model = model_name
        self.call_count = 0

    def call(self, *args, **kwargs):
        self.call_count += 1
        print(f"    MockLLM.call() invoked (count: {self.call_count})")
        return f"Response {self.call_count}"


# Create mock and wrapper
mock_llm = MockLLM("test-model")
limited_llm = RateLimitedLLM(mock_llm, requests_per_minute=10)

print("Testing call method:")
print(f"limited_llm.call method: {limited_llm.call}")
print(f"Type: {type(limited_llm.call)}")

print("\nMaking first call (should consume 1 token)...")
start = time.time()
result1 = limited_llm.call("test 1")
elapsed1 = time.time() - start
print(f"First call took {elapsed1:.2f}s, result: {result1}")

print(f"\nTokens after first call: {limited_llm._tokens:.2f}")

print("\nMaking second call (should wait for token)...")
start = time.time()
result2 = limited_llm.call("test 2")
elapsed2 = time.time() - start
print(f"Second call took {elapsed2:.2f}s, result: {result2}")

print(f"\nTokens after second call: {limited_llm._tokens:.2f}")
