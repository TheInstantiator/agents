#!/usr/bin/env python3
"""
Unit test for rate limiter without requiring full CrewAI setup
Tests just the RateLimitedLLM wrapper logic
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
        return f"Response {self.call_count}"


def test_rate_limiting():
    """Test that rate limiting properly throttles requests"""

    print("=" * 80)
    print("Unit Test: Rate Limiting Logic")
    print("=" * 80)

    # Create mock LLM
    mock_llm = MockLLM("test-model")

    # Wrap with rate limiter: 10 requests/minute = 1 request every 6 seconds
    print("\nCreating rate-limited wrapper: 10 requests/minute")
    print("This means 1 request every 6 seconds\n")

    limited_llm = RateLimitedLLM(mock_llm, requests_per_minute=10)

    # Test that attributes are accessible
    print(f"Model name: {limited_llm.model}")
    assert limited_llm.model == "test-model", "Attribute access failed"

    # Make 3 rapid calls
    print("\nMaking 3 rapid calls:")
    print("  - Call 1: should be instant (tokens available)")
    print("  - Call 2: should wait ~6 seconds (rate limited)")
    print("  - Call 3: should wait ~6 seconds (rate limited)\n")

    start = time.time()
    call_times = []

    for i in range(3):
        call_start = time.time()
        result = limited_llm.call(f"test message {i}")
        call_end = time.time()

        elapsed = call_end - call_start
        call_times.append(elapsed)

        print(f"Call {i+1}: {result} (took {elapsed:.2f}s)")

    total_time = time.time() - start

    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Expected: ~12s (0s + 6s + 6s)")
    print(f"Call 1: {call_times[0]:.2f}s (expected ~0s)")
    print(f"Call 2: {call_times[1]:.2f}s (expected ~6s)")
    print(f"Call 3: {call_times[2]:.2f}s (expected ~6s)")

    # Verify timing (with some tolerance)
    assert call_times[0] < 1.0, "First call should be instant"
    assert 5.0 < call_times[1] < 7.0, f"Second call should wait ~6s, got {call_times[1]:.2f}s"
    assert 5.0 < call_times[2] < 7.0, f"Third call should wait ~6s, got {call_times[2]:.2f}s"

    print("\n✓ All tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    test_rate_limiting()
