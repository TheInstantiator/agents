# Enable Rate Limiting for My Stock Picker

## Quick Start

To enable rate limiting, simply add the `requests_per_minute` parameter to `build_llms()` in [crew.py](src/my_stock_picker/crew.py):

### Before (No Rate Limiting)

```python
def __init__(self, llm_overrides=None):
    # Build all available LLMs
    llms = build_llms(['grok-4-fast', 'gemini-2.5-flash', 'ollama-gemma27b'])
    self.llm_grok = llms['grok-4-fast']
    self.llm_gemini = llms['gemini-2.5-flash']
    self.llm_ollama = llms['ollama-gemma27b']
```

### After (With Rate Limiting)

```python
def __init__(self, llm_overrides=None):
    # Build all available LLMs with 10 req/min limit
    llms = build_llms(
        ['grok-4-fast', 'gemini-2.5-flash', 'ollama-gemma27b'],
        requests_per_minute=10  # Limit to 10 requests/minute
    )
    self.llm_grok = llms['grok-4-fast']
    self.llm_gemini = llms['gemini-2.5-flash']
    self.llm_ollama = llms['ollama-gemma27b']
```

## What Changes

1. **Gemini Flash** (used by trending_company_finder_gemini, manager):
   - Limited to 10 requests/minute
   - 1 request every 6 seconds

2. **Grok Fast** (used by trending_company_finder_grok, researcher, rater):
   - Limited to 10 requests/minute
   - 1 request every 6 seconds

3. **Ollama Gemma 27B** (fallback):
   - **NOT limited** (local model, no API limits)

## Running with Rate Limiting

Use the provided example script:

```bash
cd ~/projects/agents/3_crew/my_stock_picker
python run_with_rate_limiting.py
```

Or use the fallback runner:

```bash
python run_with_diverse_opinions.py
```

## Recommended Limits by Tier

### Gemini Free Tier
- **Flash**: 10 requests/minute
- **Pro**: 2 requests/minute

```python
# For Flash (recommended for most users)
llms = build_llms(['gemini-2.5-flash'], requests_per_minute=10)

# For Pro (more conservative)
llms = build_llms(['gemini-2.5-pro'], requests_per_minute=2)
```

### Safety Margin (Recommended)

If you're still hitting limits, reduce by 10-20%:

```python
# Use 8-9 req/min instead of 10 for safety margin
llms = build_llms(['gemini-2.5-flash'], requests_per_minute=9)
```

## Architecture Considerations

My Stock Picker uses **hierarchical process** with:
- 1 manager (Gemini Flash)
- 2 finders (1 Grok, 1 Gemini)
- 1 researcher (Grok)
- 1 rater (Grok)

This means multiple agents may be active simultaneously. If you have:
- 2 agents using Gemini concurrently
- 10 req/min limit
→ Each agent gets ~5 req/min effectively

**Recommendation**: Start with 10 req/min and reduce if you still hit errors.

## Logging

You'll see rate limiting activity in logs:

```
INFO - Applying rate limiting: 10 requests/minute to non-ollama models
INFO - Rate limiting enabled for gemini-2.5-flash: 10 req/min (1 request every 6.0s)
INFO - Rate limiting enabled for grok-4-fast: 10 req/min (1 request every 6.0s)
INFO - Rate limit: waiting 4.2s before next gemini-2.5-flash call (10 req/min limit)
```

## Troubleshooting

### Still Getting 429 Errors?

1. **Reduce the limit**:
   ```python
   requests_per_minute=8  # More conservative
   ```

2. **Check for other applications** using the same API key

3. **Consider using different models**:
   ```python
   # Use Ollama (local, unlimited) as fallback
   llm_configs = [
       {'manager': 'gemini-2.5-flash', 'researcher': 'grok-4-fast'},
       {'manager': 'ollama-gemma27b', 'researcher': 'ollama-gemma27b'}
   ]
   ```

### Crew Taking Too Long?

Rate limiting will slow down execution. For 10 req/min:
- 30 requests = ~3 minutes minimum
- 60 requests = ~6 minutes minimum

This is expected and necessary to stay within API limits.

### Disable Rate Limiting

Simply don't pass `requests_per_minute` (or set to 0):

```python
llms = build_llms(['gemini-2.5-flash'])  # No limit
```

## See Also

- [Rate Limiting Guide](../RATE_LIMITING_GUIDE.md) - Complete documentation
- [crew.py](src/my_stock_picker/crew.py) - Main crew file to edit
- [run_with_rate_limiting.py](run_with_rate_limiting.py) - Example runner
