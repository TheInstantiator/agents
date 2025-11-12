"""
Rate-limited wrapper for CrewAI LLM instances
Provides transparent rate limiting using token bucket algorithm
"""
import time
import threading
import logging
from typing import Any

logger = logging.getLogger(__name__)


class RateLimitedLLM:
    """
    Wrapper for CrewAI LLM that adds rate limiting using token bucket algorithm.

    This wrapper is transparent - it can be used anywhere a regular LLM is expected.

    Example:
        # Limit to 10 requests per minute
        llm = RateLimitedLLM(base_llm, requests_per_minute=10)

        # Use normally - rate limiting happens automatically
        agent = Agent(llm=llm, ...)
    """

    def __init__(self, base_llm: Any, requests_per_minute: int):
        """
        Args:
            base_llm: The underlying LLM instance to wrap
            requests_per_minute: Maximum requests per minute (must be > 0)
        """
        self._base_llm = base_llm
        self._requests_per_minute = requests_per_minute
        self._time_window = 60.0  # 1 minute in seconds

        # Token bucket state
        self._tokens = float(requests_per_minute)
        self._last_refill = time.time()
        self._lock = threading.Lock()

        # Calculate refill rate (tokens per second)
        self._refill_rate = requests_per_minute / self._time_window

        # Track model name for logging
        self._model_name = getattr(base_llm, 'model', 'unknown')

        logger.info(
            f"Rate limiting enabled for {self._model_name}: "
            f"{requests_per_minute} req/min "
            f"(1 request every {self._time_window / requests_per_minute:.1f}s)"
        )

    def _acquire_token(self):
        """Acquire a token before making a request (blocks if necessary)"""
        while True:
            with self._lock:
                self._refill_tokens()

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return

                # Calculate wait time for next token
                wait_time = (1.0 - self._tokens) / self._refill_rate

            # Log the wait
            logger.info(
                f"Rate limit: waiting {wait_time:.1f}s before next {self._model_name} call "
                f"({self._requests_per_minute} req/min limit)"
            )

            # Sleep outside the lock
            time.sleep(min(wait_time, 1.0))

    def _refill_tokens(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self._last_refill

        # Add tokens based on elapsed time
        new_tokens = elapsed * self._refill_rate
        self._tokens = min(self._requests_per_minute, self._tokens + new_tokens)
        self._last_refill = now

    def call(self, *args, **kwargs):
        """Make a rate-limited call to the LLM"""
        self._acquire_token()
        return self._base_llm.call(*args, **kwargs)

    def __getattr__(self, name):
        """
        Delegate all other attribute access to the base LLM.
        This makes the wrapper transparent - it behaves like the underlying LLM.
        """
        return getattr(self._base_llm, name)

    def __repr__(self):
        return f"RateLimitedLLM({self._base_llm!r}, {self._requests_per_minute} req/min)"
