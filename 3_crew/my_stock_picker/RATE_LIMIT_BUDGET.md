# Rate Limit Budget for my_stock_picker

## The Problem

Gemini free tier has a **10 requests/minute** limit that is **shared across ALL agents** using Gemini. If multiple agents use Gemini and each has `max_rpm=8`, you'll exceed the quota (8 + 8 = 16 > 10).

## The Solution: Budget Your Quota

Distribute the 10 req/min across all Gemini agents so the total doesn't exceed 10.

## Current Configuration

### Agents Using Gemini (by default)

| Agent | LLM | max_rpm | % of Quota | Usage Pattern |
|-------|-----|---------|------------|---------------|
| **manager** | gemini (default) | 2 | 20% | Heavy coordinator |
| **trending_company_finder_gemini** | gemini (fixed) | 3 | 30% | Moderate search |
| **financial_researcher** | grok (default) | 3 if gemini | 30% | Only if overridden |
| **stock_rater** | grok (default) | 2 if gemini | 20% | Only if overridden |

**Total when all use Gemini:** 2 + 3 + 3 + 2 = **10 req/min** (exactly at limit)

### Agents Using Grok (no rate limit concerns)

| Agent | LLM | max_rpm |
|-------|-----|---------|
| **trending_company_finder_grok** | grok (fixed) | None (Grok has higher limit) |
| **financial_researcher** | grok (default) | None |
| **stock_rater** | grok (default) | None |

## Why This Distribution?

### Manager: 2 req/min (20%)
- **Most active agent** in hierarchical mode
- Coordinates all tasks
- Reads outputs from all workers
- Makes ~15-20 requests per workflow
- **Conservative limit** prevents bottleneck

### trending_company_finder_gemini: 3 req/min (30%)
- **Fixed to Gemini** (intentional design)
- Searches web for trending stocks
- Makes ~5-8 requests per task
- **Medium priority**

### financial_researcher: 3 req/min (30%) if using Gemini
- **Defaults to Grok** (unlimited)
- Only limited if overridden to Gemini
- Deep analysis requires multiple requests
- **Medium priority**

### stock_rater: 2 req/min (20%) if using Gemini
- **Defaults to Grok** (unlimited)
- Only limited if overridden to Gemini
- Simple scoring, fewer requests
- **Lower priority**

## LLM Override Impact

### Default Configuration (Recommended)
```python
crew = MyStockPicker()  # No overrides
```

**Gemini users:**
- manager: 2 req/min
- trending_company_finder_gemini: 3 req/min
**Total Gemini:** 5 req/min (50% of quota)

**Grok users:**
- trending_company_finder_grok: unlimited
- financial_researcher: unlimited
- stock_rater: unlimited

✅ **Well within Gemini limit!**

### Override: All Gemini (Testing)
```python
crew = MyStockPicker(llm_overrides={
    'manager': 'gemini-2.5-flash',
    'researcher': 'gemini-2.5-flash',
    'rater': 'gemini-2.5-flash'
})
```

**Gemini users:**
- manager: 2 req/min
- trending_company_finder_gemini: 3 req/min
- financial_researcher: 3 req/min
- stock_rater: 2 req/min
**Total Gemini:** 10 req/min (100% of quota)

⚠️ **At the limit - might be slow!**

### Override: Manager Ollama, Workers Gemini
```python
crew = MyStockPicker(llm_overrides={
    'manager': 'ollama-gemma27b',     # Unlimited!
    'researcher': 'gemini-2.5-flash',
    'rater': 'gemini-2.5-flash'
})
```

**Gemini users:**
- trending_company_finder_gemini: 3 req/min
- financial_researcher: 3 req/min
- stock_rater: 2 req/min
**Total Gemini:** 8 req/min (80% of quota)

**Ollama users:**
- manager: unlimited (local)

✅ **Excellent distribution!**

## How max_rpm Works

### Per-Agent Rate Limiting

```python
@agent
def my_agent(self) -> Agent:
    return Agent(
        llm=self.llm_gemini,
        max_rpm=3,  # This agent limited to 3 req/min
        ...
    )
```

- CrewAI tracks requests **per agent**
- If agent exceeds `max_rpm`, it **waits** until next minute window
- Each agent's counter is independent
- But all agents share the same API quota!

### Example Timeline

```
00:00 - Manager makes 2 requests (2 total)
00:10 - Gemini finder makes 3 requests (5 total)
00:20 - Researcher makes 3 requests (8 total)
00:30 - Rater makes 2 requests (10 total)  ← At Gemini limit!
00:40 - Manager tries to make request → BLOCKED by Gemini (429)
01:00 - New minute starts, counters reset
```

With `max_rpm` properly set, agents wait before exceeding their individual limits, preventing 429 errors.

## Debugging Rate Limits

### Enable Verbose Mode

```python
@agent
def my_agent(self) -> Agent:
    return Agent(
        llm=self.llm_gemini,
        max_rpm=3,
        verbose=True,  # Shows rate limit activity
        ...
    )
```

Output shows:
```
[Agent] Waiting 15s before next request (max_rpm limit)...
```

### Check Which LLM is Used

```python
# In your code
print(f"Manager LLM: {self.manager_llm}")
print(f"Researcher LLM: {self.researcher_llm}")
print(f"Rater LLM: {self.rater_llm}")
```

### Monitor API Usage

- Gemini: https://ai.dev/usage?tab=rate-limit
- Check current usage and remaining quota

## Recommendations

### For Development/Testing

Use Ollama for most agents to avoid rate limits:

```python
crew = MyStockPicker(llm_overrides={
    'manager': 'ollama-gemma27b',      # Local, unlimited
    'researcher': 'ollama-gemma27b',   # Local, unlimited
    'rater': 'ollama-gemma27b'         # Local, unlimited
})
```

Only `trending_company_finder_gemini` uses Gemini (3 req/min).

### For Production

Use the default configuration:
- Manager: Gemini (2 req/min) - fast coordination
- Finders: Gemini (3 req/min) + Grok (unlimited) - diverse opinions
- Researcher: Grok (unlimited) - thorough analysis
- Rater: Grok (unlimited) - consistent rating

Total Gemini: 5 req/min (50% of quota) - safe margin

### If Still Hitting 429 Errors

1. **Wait longer between runs**
   ```bash
   crewai run
   sleep 70  # Wait 70 seconds
   crewai run
   ```

2. **Reduce max_rpm further**
   ```python
   max_rpm=1  # Very conservative
   ```

3. **Use fallback runner**
   ```python
   from crew_runner_with_fallback import run_crew_with_fallback

   result = run_crew_with_fallback(
       crew_class=MyStockPicker,
       llm_configs=[
           {'manager': 'gemini-2.5-flash', 'researcher': 'grok-4-fast', 'rater': 'grok-4-fast'},
           {'manager': 'ollama-gemma27b', 'researcher': 'ollama-gemma27b', 'rater': 'ollama-gemma27b'}
       ]
   )
   ```

4. **Switch to Ollama entirely**
   ```python
   crew = MyStockPicker(llm_overrides={
       'manager': 'ollama-gemma27b',
       'researcher': 'ollama-gemma27b',
       'rater': 'ollama-gemma27b'
   })
   ```

## Summary

- ✅ **Budget quota** across all Gemini agents (total ≤ 10 req/min)
- ✅ **Set max_rpm** on every agent that might use Gemini
- ✅ **Use conditional max_rpm** for configurable agents
- ✅ **Default config** uses 5 req/min (50% margin)
- ✅ **Use Ollama** for development to avoid limits
- ✅ **Enable verbose** mode to see rate limiting in action

**Current configuration is optimized to avoid 429 errors while maintaining good performance!**
