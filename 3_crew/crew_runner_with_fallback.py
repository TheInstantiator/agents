"""
Run a crew with LLM fallback logic for improved resilience.

This runner provides automatic failover between different LLM configurations,
useful for handling service outages or general API failures.

Note: For rate limiting, use the max_rpm parameter on Agent objects instead.
See USING_MAX_RPM.md for details.
"""

def run_crew_with_fallback(crew_class, llm_configs, inputs=None, max_retries=3):
    """
    Run a crew with automatic fallback to different LLM configurations if one fails.

    This is useful for:
    - Handling service outages (e.g., Grok API down)
    - Testing different LLM combinations
    - Ensuring availability through fallback chains

    Args:
        crew_class: The crew class to instantiate (e.g., MyStockPicker)
        llm_configs: List of LLM configuration dicts to try in order

                     Example 1 (Simple - same LLM for all):
                     [
                         {'manager': 'gemini-2.5-flash', 'researcher': 'gemini-2.5-flash', ...},
                         {'manager': 'grok-4-fast', 'researcher': 'grok-4-fast', ...},
                         {'manager': 'ollama-gemma27b', 'researcher': 'ollama-gemma27b', ...}
                     ]

                     Example 2 (Diverse - different LLMs per agent):
                     [
                         {  # Try this config first (diverse opinions)
                             'manager': 'gemini-2.5-flash',
                             'researcher': 'grok-4-fast',
                             'analyst': 'gemini-2.5-flash',
                             'rater': 'grok-4-fast'
                         },
                         {  # Fallback if Grok is down (all Gemini)
                             'manager': 'gemini-2.5-flash',
                             'researcher': 'gemini-2.5-flash',
                             'analyst': 'gemini-2.5-flash',
                             'rater': 'gemini-2.5-flash'
                         },
                         {  # Final fallback (all local)
                             'manager': 'ollama-gemma27b',
                             'researcher': 'ollama-gemma27b',
                             'analyst': 'ollama-gemma27b',
                             'rater': 'ollama-gemma27b'
                         }
                     ]

        inputs: Input dictionary for the crew
        max_retries: Maximum retries per config

    Returns:
        CrewOutput if successful

    Raises:
        Exception if all configs fail

    Example:
        from my_stock_picker.crew import MyStockPicker

        # Try diverse opinions first, fallback to single LLM if needed
        result = run_crew_with_fallback(
            crew_class=MyStockPicker,
            llm_configs=[
                {  # Diverse: Grok for research, Gemini for analysis
                    'manager': 'gemini-2.5-flash',
                    'researcher': 'grok-4-fast',
                    'analyst': 'gemini-2.5-flash',
                    'rater': 'grok-4-fast'
                },
                {  # Fallback: All Gemini if Grok unavailable
                    'manager': 'gemini-2.5-flash',
                    'researcher': 'gemini-2.5-flash',
                    'analyst': 'gemini-2.5-flash',
                    'rater': 'gemini-2.5-flash'
                },
                {  # Final: All local Ollama
                    'manager': 'ollama-gemma27b',
                    'researcher': 'ollama-gemma27b',
                    'analyst': 'ollama-gemma27b',
                    'rater': 'ollama-gemma27b'
                }
            ],
            inputs={'sector': 'technology'}
        )
    """
    import logging
    from time import sleep

    logger = logging.getLogger(__name__)

    if inputs is None:
        inputs = {}

    last_error = None

    for config_idx, llm_config in enumerate(llm_configs, 1):
        config_name = f"Config {config_idx}: {llm_config}"
        logger.info(f"Attempting to run crew with {config_name}...")

        for attempt in range(max_retries):
            try:
                # Instantiate crew with the specific LLM configuration
                crew_instance = crew_class(llm_overrides=llm_config)

                # Run the crew
                result = crew_instance.crew().kickoff(inputs=inputs)

                logger.info(f"SUCCESS: Crew completed with {config_name}")
                return result

            except Exception as e:
                error_msg = str(e).lower()
                last_error = e

                # Check if it's a potentially transient error worth retrying
                is_transient = any(keyword in error_msg for keyword in [
                    'rate limit', 'quota', 'overloaded', '429', 'too many requests',
                    '503', 'service unavailable', 'timeout', 'connection'
                ])

                if is_transient:
                    logger.warning(f"Transient error on {config_name} (attempt {attempt + 1}/{max_retries}): {e}")

                    if attempt < max_retries - 1:
                        # Wait before retry (exponential backoff)
                        wait_time = 2 ** attempt
                        logger.info(f"Waiting {wait_time}s before retry...")
                        sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Max retries reached for {config_name}, moving to next config")
                        break
                else:
                    # Non-transient error, move to next config immediately
                    logger.error(f"Error with {config_name}: {e}")
                    break

    # If we get here, all configs failed
    logger.error(f"All LLM configurations failed. Last error: {last_error}")
    raise Exception(f"Crew failed with all LLM configurations. Last error: {last_error}")
