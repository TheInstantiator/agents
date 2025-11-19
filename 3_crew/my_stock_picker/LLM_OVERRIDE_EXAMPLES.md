# LLM Override Examples

The `MyStockPicker` crew supports overriding which LLM is used for specific agents via the `llm_overrides` parameter.

## Overview

### Configurable Agents

These agents **respect** the `llm_overrides` parameter:

- **manager**: Coordinates the hierarchical workflow
- **researcher**: Performs deep analysis on trending companies
- **rater**: Scores and rates investment potential

### Fixed Agents

These agents have **hardcoded** LLMs (intentional for diverse perspectives):

- **trending_company_finder_grok**: Always uses Grok for fundamental analysis
- **trending_company_finder_gemini**: Always uses Gemini for momentum analysis

## Available LLMs

- `'gemini-2.5-flash'` - Fast, good for coordination and quick analysis
- `'grok-4-fast'` - Thorough, good for research and deep analysis
- `'ollama-gemma27b'` - Local model, unlimited requests, good for testing

## Usage Examples

### Default Configuration (No Overrides)

```python
from my_stock_picker.crew import MyStockPicker

# Use default LLM assignments
crew = MyStockPicker()

# Defaults:
# - manager: gemini-2.5-flash
# - researcher: grok-4-fast
# - rater: grok-4-fast
```

### Use Ollama for Everything (Free, Local)

```python
# Good for testing without hitting API limits
crew = MyStockPicker(llm_overrides={
    'manager': 'ollama-gemma27b',
    'researcher': 'ollama-gemma27b',
    'rater': 'ollama-gemma27b'
})
```

### Use Only Gemini (When Grok is Down)

```python
# Fallback scenario when Grok API is unavailable
crew = MyStockPicker(llm_overrides={
    'manager': 'gemini-2.5-flash',
    'researcher': 'gemini-2.5-flash',  # Override researcher to use Gemini
    'rater': 'gemini-2.5-flash'        # Override rater to use Gemini
})
```

### Mix: Gemini Manager + Grok Researcher + Ollama Rater

```python
# Custom combination for cost optimization
crew = MyStockPicker(llm_overrides={
    'manager': 'gemini-2.5-flash',     # Fast coordinator
    'researcher': 'grok-4-fast',       # Thorough research (API)
    'rater': 'ollama-gemma27b'         # Rating locally (free)
})
```

### Test Different Researchers

```python
# Compare Grok vs Gemini for research quality
crew_grok = MyStockPicker(llm_overrides={
    'researcher': 'grok-4-fast'
})

crew_gemini = MyStockPicker(llm_overrides={
    'researcher': 'gemini-2.5-flash'
})

crew_ollama = MyStockPicker(llm_overrides={
    'researcher': 'ollama-gemma27b'
})
```

## Practical Use Cases

### 1. Cost Optimization

Use expensive models only where needed:

```python
crew = MyStockPicker(llm_overrides={
    'manager': 'ollama-gemma27b',      # Simple coordination - use local
    'researcher': 'grok-4-fast',       # Critical research - use best API
    'rater': 'ollama-gemma27b'         # Scoring logic - use local
})
```

### 2. Rate Limit Management

Control request rates using Agent `max_rpm` parameter:

```python
# Set in crew.py agents
@agent
def my_agent(self) -> Agent:
    return Agent(
        config=self.agents_config['my_agent'],
        llm=self.llm_gemini,
        max_rpm=8  # Limit to 8 requests per minute
    )

# Or shift load to unlimited local model
crew = MyStockPicker(llm_overrides={
    'manager': 'ollama-gemma27b',
    'researcher': 'grok-4-fast',
    'rater': 'ollama-gemma27b'
})
```

### 3. Service Outage Fallback

When Grok service is down:

```python
crew = MyStockPicker(llm_overrides={
    'manager': 'gemini-2.5-flash',
    'researcher': 'gemini-2.5-flash',  # Fallback to Gemini
    'rater': 'gemini-2.5-flash'
})
```

### 4. Development/Testing

Fast iteration with local models:

```python
crew = MyStockPicker(llm_overrides={
    'manager': 'ollama-gemma27b',
    'researcher': 'ollama-gemma27b',
    'rater': 'ollama-gemma27b'
})

# No API costs, no rate limits, instant testing
result = crew.kickoff(inputs={...})
```

## Why Some Agents Don't Accept Overrides

The two finder agents (`trending_company_finder_grok` and `trending_company_finder_gemini`) are **intentionally hardcoded** to:

- **Get diverse perspectives**: Grok finds fundamental-focused companies, Gemini finds momentum-focused companies
- **Ensure quality**: Each LLM has different strengths in analysis style
- **Maintain workflow intent**: The crew is designed for these specific viewpoints

If you need both finders to use the same LLM, you would need to modify the crew design itself.

## Rate Limiting

Use CrewAI's built-in `max_rpm` parameter on Agent objects to control request rates.

See [USING_MAX_RPM.md](../../USING_MAX_RPM.md) for details.

## Notes

- Invalid LLM names in overrides will fall back to defaults (with warning logged)
- Empty dict `{}` is same as no overrides - uses defaults
- Partial overrides are fine - only specified agents are overridden
