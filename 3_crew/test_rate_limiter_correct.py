#!/usr/bin/env python3
"""
Correct unit test for rate limiter
The token bucket starts FULL, so first N requests are instant
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

    # Wrap with rate limiter: 2 requests/minute = 1 request every 30 seconds
    # Start with 2 tokens (bucket starts full)
    print("\nCreating rate-limited wrapper: 2 requests/minute")
    print("This means 1 request every 30 seconds")
    print("Token bucket starts with 2 tokens (full)\n")

    limited_llm = RateLimitedLLM(mock_llm, requests_per_minute=2)

    # Test that attributes are accessible
    print(f"Model name: {limited_llm.model}")
    assert limited_llm.model == "test-model", "Attribute access failed"

    # Make 4 rapid calls
    print("\nMaking 4 rapid calls:")
    print("  - Call 1: should be instant (token 1/2)")
    print("  - Call 2: should be instant (token 2/2)")
    print("  - Call 3: should wait ~30 seconds (bucket empty, wait for refill)")
    print("  - Call 4: should wait ~30 seconds (bucket empty again)\n")

    start = time.time()
    call_times = []

    for i in range(4):
        call_start = time.time()
        result = limited_llm.call(f"test message {i}")
        call_end = time.time()

        elapsed = call_end - call_start
        total_elapsed = call_end - start
        call_times.append(elapsed)

        print(f"Call {i+1}: {result} (took {elapsed:.2f}s, total: {total_elapsed:.2f}s)")

    total_time = time.time() - start

    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Expected: ~60s (0s + 0s + 30s + 30s)")
    print(f"Call 1: {call_times[0]:.2f}s (expected ~0s)")
    print(f"Call 2: {call_times[1]:.2f}s (expected ~0s)")
    print(f"Call 3: {call_times[2]:.2f}s (expected ~30s)")
    print(f"Call 4: {call_times[3]:.2f}s (expected ~30s)")

    # Verify timing (with some tolerance)
    assert call_times[0] < 1.0, "First call should be instant"
    assert call_times[1] < 1.0, "Second call should be instant"
    assert 28.0 < call_times[2] < 32.0, f"Third call should wait ~30s, got {call_times[2]:.2f}s"
    assert 28.0 < call_times[3] < 32.0, f"Fourth call should wait ~30s, got {call_times[3]:.2f}s"

    print("\n✓ All tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    print("NOTE: This test will take ~60 seconds to complete (testing rate limiting)\n")
    test_rate_limiting()
