"""
Rate limiter utility for controlling API request frequency
Uses token bucket algorithm to throttle requests
"""
import time
import threading
from typing import Dict


class RateLimiter:
    """
    Token bucket rate limiter for API calls.

    Example:
        # Allow 10 requests per minute
        limiter = RateLimiter(max_requests=10, time_window=60)

        # Before making API call
        limiter.acquire()  # Blocks if rate limit would be exceeded
        # ... make API call ...
    """

    def __init__(self, max_requests: int, time_window: float = 60.0):
        """
        Args:
            max_requests: Maximum number of requests allowed
            time_window: Time window in seconds (default 60 for per-minute)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.tokens = max_requests
        self.last_refill = time.time()
        self.lock = threading.Lock()

        # Calculate refill rate (tokens per second)
        self.refill_rate = max_requests / time_window

    def acquire(self, block: bool = True) -> bool:
        """
        Acquire a token to make a request.

        Args:
            block: If True, wait until a token is available. If False, return immediately.

        Returns:
            True if token acquired, False if not available (only when block=False)
        """
        while True:
            with self.lock:
                self._refill_tokens()

                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return True

                if not block:
                    return False

                # Calculate wait time for next token
                wait_time = (1.0 - self.tokens) / self.refill_rate

            # Sleep outside the lock
            time.sleep(min(wait_time, 1.0))

    def _refill_tokens(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill

        # Add tokens based on elapsed time
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.max_requests, self.tokens + new_tokens)
        self.last_refill = now


class MultiModelRateLimiter:
    """
    Rate limiter that manages different limits for different models.

    Example:
        limiter = MultiModelRateLimiter({
            'gemini-2.5-flash': (10, 60),    # 10 req/min
            'gemini-2.5-pro': (2, 60),       # 2 req/min
            'grok-4-fast': (100, 60),        # 100 req/min
        })

        # Before API call
        limiter.acquire('gemini-2.5-pro')
        # ... make API call ...
    """

    def __init__(self, model_limits: Dict[str, tuple[int, float]]):
        """
        Args:
            model_limits: Dict mapping model name to (max_requests, time_window)
                         Example: {'gemini-2.5-pro': (2, 60)}
        """
        self.limiters = {
            model: RateLimiter(max_req, window)
            for model, (max_req, window) in model_limits.items()
        }

    def acquire(self, model_name: str, block: bool = True) -> bool:
        """
        Acquire rate limit token for a specific model.

        Args:
            model_name: Name of the model (must be in model_limits)
            block: If True, wait until token available

        Returns:
            True if acquired, False if not available (only when block=False)
        """
        if model_name not in self.limiters:
            # No rate limit configured for this model
            return True

        return self.limiters[model_name].acquire(block=block)
