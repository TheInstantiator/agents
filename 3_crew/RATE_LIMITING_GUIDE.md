# Rate Limiting Guide

## Overview

The `build_llms()` function now supports optional rate limiting to prevent hitting API rate limits like Gemini's 10 requests/minute or 2 requests/minute for free tiers.

## How It Works

- Uses **token bucket algorithm** for smooth rate limiting
- Bucket starts FULL, so initial requests are instant
- After tokens are exhausted, requests wait for refill
- **Ollama models are never rate limited** (they're local)
- **Only applies to non-ollama models** when enabled

## Usage

### No Rate Limiting (Default)

```python
# Backward compatible - no rate limiting
llms = build_llms(['grok-4-fast', 'gemini-2.5-flash', 'ollama-gemma27b'])
```

### Rate Limiting for All Models

```python
# Limit all non-ollama models to 10 requests/minute
# This means 1 request every 6 seconds
llms = build_llms(
    ['grok-4-fast', 'gemini-2.5-flash', 'ollama-gemma27b'],
    requests_per_minute=10
)
```

### Rate Limiting for Free Tier

```python
# Limit to 2 requests/minute (Gemini free tier)
# This means 1 request every 30 seconds
llms = build_llms(
    ['gemini-2.5-pro', 'ollama-gemma27b'],
    requests_per_minute=2
)
```

## Common Rate Limits

| Service | Free Tier | Paid Tier |
|---------|-----------|-----------|
| Gemini 2.5 Flash | 10 req/min | Higher |
| Gemini 2.5 Pro | 2 req/min | Higher |
| Grok | Varies | Varies |
| Ollama | Unlimited (local) | N/A |

## How Token Bucket Works

```
requests_per_minute = 10
→ 10 tokens in bucket
→ 1 token refills every 6 seconds

Request pattern:
Call 1:  Instant (9 tokens left)
Call 2:  Instant (8 tokens left)
...
Call 10: Instant (0 tokens left)
Call 11: Wait 6 seconds for 1 token
Call 12: Wait 6 seconds for 1 token
```

## Example: My Stock Picker with Rate Limiting

Update [crew.py](my_stock_picker/src/my_stock_picker/crew.py):

```python
def __init__(self, llm_overrides=None):
    # Build LLMs with rate limiting for Gemini free tier
    llms = build_llms(
        ['grok-4-fast', 'gemini-2.5-flash', 'ollama-gemma27b'],
        requests_per_minute=10  # Limit to 10 req/min for Gemini
    )
    self.llm_grok = llms['grok-4-fast']
    self.llm_gemini = llms['gemini-2.5-flash']
    self.llm_ollama = llms['ollama-gemma27b']

    # Rest of initialization...
```

## Logging

Rate limiting events are logged:

```
INFO - Rate limiting enabled for gemini-2.5-flash: 10 req/min (1 request every 6.0s)
INFO - Rate limiting enabled for grok-4-fast: 10 req/min (1 request every 6.0s)
INFO - Rate limit: waiting 5.8s before next gemini-2.5-flash call (10 req/min limit)
```

## Error You're Trying to Avoid

Without rate limiting, you'll see errors like:

```
Exception: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429,
'message': 'You exceeded your current quota, please check your plan and billing details.
Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests,
limit: 10. Please retry in 46.862038852s.'}}
```

With rate limiting enabled, requests automatically wait instead of failing.

## Testing

Run the unit test to verify rate limiting works:

```bash
cd ~/projects/agents/3_crew
python3 test_rate_limiter_correct.py
```

This test takes ~60 seconds and verifies:
- First 2 calls are instant (bucket has tokens)
- Subsequent calls wait for token refill
- Timing is accurate (30s per request for 2 req/min limit)

## Choosing the Right Limit

1. **Check your API limits** in provider dashboard
2. **Add safety margin**: Use 90% of limit (e.g., 9 req/min instead of 10)
3. **Consider concurrent agents**: If you have 2 agents both using Gemini, set limit to 5 req/min each
4. **Test first**: Run with higher limit, reduce if you hit errors

## Files

- [shared_llms.py](shared_llms.py) - Main configuration with `build_llms()`
- [rate_limited_llm.py](rate_limited_llm.py) - Wrapper implementation
- [test_rate_limiter_correct.py](test_rate_limiter_correct.py) - Unit test

## Notes

- **Ollama is never limited**: Local models don't need rate limiting
- **Transparent wrapper**: Crews don't need any code changes
- **Thread-safe**: Uses locking for concurrent access
- **Backward compatible**: Default `requests_per_minute=0` means no limiting
