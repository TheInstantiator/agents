# Using CrewAI's Built-in Rate Limiting (max_rpm)

## Overview

CrewAI provides built-in rate limiting through the `max_rpm` (maximum requests per minute) parameter on Agent objects. This handles API rate limits automatically without requiring custom wrappers.

## Usage

### Setting max_rpm on Agents

```python
from crewai import Agent

@agent
def my_agent(self) -> Agent:
    return Agent(
        config=self.agents_config['my_agent'],
        llm=self.llm_gemini,
        max_rpm=8,  # Limit to 8 requests per minute
        verbose=True
    )
```

### Setting max_rpm on Manager Agent

```python
@crew
def crew(self) -> Crew:
    manager = Agent(
        config=self.agents_config['manager'],
        llm=self.manager_llm,
        allow_delegation=True,
        verbose=True,
        max_rpm=8  # Limit manager to 8 requests per minute
    )

    return Crew(
        agents=self.agents,
        tasks=self.tasks,
        process=Process.hierarchical,
        manager_agent=manager,
        verbose=True
    )
```

### Conditional Rate Limiting

Only apply rate limiting to specific LLMs:

```python
@agent
def my_agent(self) -> Agent:
    return Agent(
        config=self.agents_config['my_agent'],
        llm=self.llm_gemini,
        # Only limit Gemini, not Ollama
        max_rpm=8 if self.llm_gemini else None,
        verbose=True
    )
```

## Common Rate Limits

| Service | Model | Free Tier Limit | Recommended max_rpm |
|---------|-------|-----------------|---------------------|
| **Gemini** | 2.5 Flash | 10 req/min | 8 (80% safety margin) |
| **Gemini** | 2.5 Pro | 2 req/min | 2 |
| **Grok** | grok-4 | Varies by account | Check your tier |
| **Grok** | grok-4-fast | Varies by account | Check your tier |
| **Ollama** | All models | Unlimited (local) | None needed |

**Pro Tip:** Set `max_rpm` to 80-90% of your actual API limit to provide a safety margin.

## Real-World Example

From the `my_stock_picker` crew:

```python
@agent
def trending_company_finder_gemini(self) -> Agent:
    return Agent(
        config=self.agents_config['trending_company_finder_gemini'],
        tools=[SerperDevTool()],
        llm=self.llm_gemini,
        verbose=True,
        max_rpm=8  # Gemini free tier is 10/min, use 8 for safety
    )

@crew
def crew(self) -> Crew:
    manager = Agent(
        config=self.agents_config['manager'],
        llm=self.manager_llm,
        allow_delegation=True,
        verbose=True,
        # Conditionally limit based on which LLM is used
        max_rpm=8 if self.manager_llm == self.llm_gemini else None
    )

    return Crew(
        agents=self.agents,
        tasks=self.tasks,
        process=Process.hierarchical,
        verbose=True,
        manager_agent=manager
    )
```

## Best Practices

### 1. Per-Agent Rate Limiting

Rate limits are **per-agent**, not global:

```python
# Each agent has its own rate limit
agent1 = Agent(..., max_rpm=8)  # 8 req/min
agent2 = Agent(..., max_rpm=8)  # 8 req/min
# Total crew can do up to 16 req/min (if agents run in parallel)
```

### 2. Local Models Don't Need Limits

Only set `max_rpm` for API-based models:

```python
# API model - needs rate limiting
agent_gemini = Agent(
    llm=self.llm_gemini,
    max_rpm=8
)

# Local model - no rate limiting needed
agent_ollama = Agent(
    llm=self.llm_ollama,
    max_rpm=None  # or just omit this parameter
)
```

### 3. Hierarchical Process Considerations

In hierarchical process, the manager agent makes many requests:

```python
# Manager coordinates everything, so it needs rate limiting too
manager = Agent(
    config=self.agents_config['manager'],
    llm=self.llm_gemini,
    max_rpm=8  # Important for hierarchical process!
)
```

### 4. Combine with LLM Overrides

Mix rate-limited API models with unlimited local models:

```python
crew = MyStockPicker(llm_overrides={
    'manager': 'gemini-2.5-flash',  # Fast coordinator (rate limited)
    'researcher': 'ollama-gemma27b',  # Local research (unlimited)
    'rater': 'grok-4-fast'  # API rating (rate limited)
})
```

## What Happens When Rate Limit is Hit?

CrewAI automatically:
1. **Detects** when rate limit is exceeded
2. **Waits** until the next minute window
3. **Retries** the request automatically
4. **Continues** execution seamlessly

No custom retry logic needed!

## Debugging Rate Limits

Enable verbose mode to see rate limiting in action:

```python
agent = Agent(
    config=self.agents_config['my_agent'],
    llm=self.llm_gemini,
    max_rpm=8,
    verbose=True  # Shows when rate limiting kicks in
)

crew = Crew(
    agents=self.agents,
    tasks=self.tasks,
    verbose=True  # Shows crew-level rate limit handling
)
```

## Alternatives to Rate Limiting

### 1. Use Local Models (Ollama)

No rate limits at all:

```python
crew = MyStockPicker(llm_overrides={
    'manager': 'ollama-gemma27b',
    'researcher': 'ollama-gemma27b',
    'rater': 'ollama-gemma27b'
})
# Unlimited requests, works offline
```

### 2. Use Fallback Chain

Automatically switch to different LLMs if one is exhausted:

```python
from crew_runner_with_fallback import run_crew_with_fallback

result = run_crew_with_fallback(
    crew_class=MyStockPicker,
    llm_configs=[
        {'manager': 'gemini-2.5-flash', 'researcher': 'grok-4-fast'},
        {'manager': 'grok-4-fast', 'researcher': 'grok-4-fast'},
        {'manager': 'ollama-gemma27b', 'researcher': 'ollama-gemma27b'}
    ]
)
```

See [crew_runner_with_fallback.py](crew_runner_with_fallback.py) for details.

## Common Issues

### Issue: Still Hitting 429 Errors

**Cause:** Multiple agents sharing the same API quota

**Solution:** Lower the `max_rpm` value or use local models for some agents:

```python
# Before: Too aggressive
agent1 = Agent(llm=self.llm_gemini, max_rpm=10)  # At the limit
agent2 = Agent(llm=self.llm_gemini, max_rpm=10)  # At the limit
# Total: 20 req/min, exceeds Gemini's 10/min quota!

# After: More conservative
agent1 = Agent(llm=self.llm_gemini, max_rpm=5)   # Half the quota
agent2 = Agent(llm=self.llm_ollama, max_rpm=None)  # Unlimited local
# Total: Only 5 req/min on Gemini API
```

### Issue: Crew Running Too Slowly

**Cause:** `max_rpm` set too low

**Solution:** Increase `max_rpm` closer to your actual API limit, or use faster local models:

```python
# Speed up by using local models
crew = MyStockPicker(llm_overrides={
    'manager': 'ollama-gemma27b',  # Fast local coordination
    'researcher': 'gemini-2.5-flash',  # Keep API for critical work
    'rater': 'ollama-gemma27b'  # Fast local rating
})
```

## Summary

- ✅ Use `max_rpm` parameter on Agent objects for rate limiting
- ✅ Set to 80-90% of actual API limit for safety margin
- ✅ Only set on API-based agents, not local models
- ✅ Consider per-agent limits (not global)
- ✅ Enable verbose mode for debugging
- ✅ Combine with fallback chains for resilience

**Next Steps:**
- See [LLM_OVERRIDE_EXAMPLES.md](my_stock_picker/LLM_OVERRIDE_EXAMPLES.md) for LLM configuration examples
- See [crew_runner_with_fallback.py](crew_runner_with_fallback.py) for automatic failover