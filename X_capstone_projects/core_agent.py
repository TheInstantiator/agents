import os
import time
import json
import re
from typing import List, Dict, Type, Optional
from pydantic import BaseModel, Field
import litellm

# Disable telemetry
litellm.telemetry = False

class AgentResponse(BaseModel):
    """Default fallback Pydantic model if none is provided."""
    thoughts: str = Field(description="Internal reasoning before answering.")
    final_answer: str = Field(description="The final response returned to the user.")

class CoreAgent:

    tokens_used: int = 0
    cost: float = 0.0
    total_response_time: float = 0.0
    
    def __init__(
        self, 
        agent_id: str, 
        system_prompt: str, 
        litellm_kwargs: dict, 
        one_shot: bool = True,
        response_model: Optional[Type[BaseModel]] = AgentResponse
    ):
        """
        Initialize the CoreAgent.
        
        Args:
            agent_id: A unique identifier for saving/loading state and DB namespacing.
            system_prompt: The instructions that define the agent's behavior.
            litellm_kwargs: Dictionary containing LLM config (model, api_key, api_base, etc.)
            one_shot: If True, bypasses memory persistence routines.
            response_model: The Pydantic model dictating the exact JSON output format required.
        """
        self.agent_id = agent_id
        self.system_prompt = system_prompt
        self.litellm_kwargs = litellm_kwargs.copy()
        
        self.config_model_name = self.litellm_kwargs.pop("_config_model_name", self.litellm_kwargs.get("model", "unknown"))
        
        self.one_shot = one_shot
        self.response_model = response_model
        
        # L1 Memory (Short-Term Working Memory)
        self.working_memory: List[Dict[str, str]] = []

    @staticmethod
    def load_litellm_kwargs_from_config(agent_name: str, config_path: str = "config.json") -> dict:
        """
        Utility method to load kwargs directly from a JSON config file.
        Normally, you would pass these in during __init__, but this is here for convenience.
        """
        # Resolve the path relative to wherever this script is called from
        script_dir = os.path.dirname(os.path.abspath(__file__))
        abs_config_path = os.path.join(script_dir, config_path)
        
        with open(abs_config_path, "r") as f:
            config = json.load(f)
            for model in config.get("model_list", []):
                if model.get("model_name") == agent_name:
                    params = model.get("litellm_params", {}).copy()
                    
                    # Resolve os.environ/ variables dynamically
                    for k, v in params.items():
                        if isinstance(v, str) and v.startswith("os.environ/"):
                            env_var = v.split("/")[1]
                            params[k] = os.getenv(env_var, "")
                    params["_config_model_name"] = agent_name
                    return params
        raise ValueError(f"Agent {agent_name} not found in config {abs_config_path}")

    def _clean_json_output(self, text: str) -> str:
        """Extract JSON from potential markdown blocks or surrounding text."""
        # 0. Strip deepseek <think> blocks if present
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # 1. Try stripping markdown blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if json_match:
            return json_match.group(1)
            
        # 2. Try finding the first { and last }
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return text[first_brace:last_brace+1]
            
        return text

    async def ask(self, user_input: str, response_model: Optional[Type[BaseModel]] = None, max_retries: int = 3) -> BaseModel | str:
        """
        The main generation loop (Phase 1).
        Takes the user input, combines it with the system prompt, and calls the LLM.
        """
        # Determine which response model to use (specific call vs default)
        active_model = response_model if response_model is not None else self.response_model

        # 0. Manage Working Memory Context Window (Memory Pruning)
        max_working_memory_items = 20  # Keep the last 20 messages (e.g. 10 user/10 assistant)
        if len(self.working_memory) >= max_working_memory_items:
            self.working_memory = self.working_memory[-max_working_memory_items:]

        # 1. Update Working Memory
        self.working_memory.append({"role": "user", "content": user_input})
        
        # 2. Build the Payload
        dynamic_prompt = self.system_prompt
        payload = [{"role": "system", "content": dynamic_prompt}] + self.working_memory

        # 3. Prepare Litellm Arguments
        base_generate_kwargs = self.litellm_kwargs.copy()
        
        # Enforce JSON structure if a Pydantic model was provided
        if active_model:
            model_name = base_generate_kwargs.get("model", "").lower()
            if "deepseek" in model_name:
                # DeepSeek MUST use <think> blocks and breaks under Ollama's strict JSON schema mode.
                # Fallback to prompt-injected JSON instructions so it can think freely before outputting JSON.
                schema_str = json.dumps(active_model.model_json_schema())
                dynamic_prompt += f"\n\nYou MUST output valid JSON that strictly conforms to this JSON schema. Output the JSON inside a ```json block after your thoughts:\n{schema_str}"
                payload[0]["content"] = dynamic_prompt
            else:
                # Native Structured Outputs: Pass the Pydantic model directly.
                # LiteLLM translates this to a valid JSON Schema engine enforcement
                base_generate_kwargs["response_format"] = active_model

        # Initialize messages for potentially multiple attempts
        current_messages = payload.copy()
            
        # 4. Call the LLM (Async)
        for attempt in range(max_retries):
            # Build current arguments safely
            generate_kwargs = base_generate_kwargs.copy()
            generate_kwargs["messages"] = current_messages
            
            print(f"[{generate_kwargs['model']}] Thinking..." + (f" (Attempt {attempt+1}/{max_retries})" if attempt > 0 else ""))
            
            start_time = time.time()
            response = await litellm.acompletion(**generate_kwargs)
            self.total_response_time += (time.time() - start_time)
            
            # Accumulate Tokens
            try:
                if response.usage:
                    if hasattr(response.usage, 'total_tokens'):
                        self.tokens_used += response.usage.total_tokens
                    elif isinstance(response.usage, dict) and 'total_tokens' in response.usage:
                        self.tokens_used += response.usage.get('total_tokens', 0)
            except Exception as e:
                print(f"Token accumulation error: {e}")

            # Accumulate Cost
            try:
                run_cost = litellm.completion_cost(completion_response=response)
                if run_cost:
                    self.cost += run_cost
            except Exception:
                pass
                
            raw_output = response.choices[0].message.content
            
            # 5. Parse, Validate, and return the result
            if active_model:
                cleaned_output = self._clean_json_output(raw_output)
                try:
                    parsed_output = active_model.model_validate_json(cleaned_output)
                    
                    self.working_memory.append({"role": "assistant", "content": raw_output})
                    if self.one_shot: self.working_memory.clear()
                    
                    return parsed_output
                except Exception as e:
                    print(f"Error validating JSON: {e}\nRaw Output: {raw_output}")
                    if attempt == max_retries - 1:
                        raise e
                    else:
                        print(f"Retrying {self.agent_id} JSON generation...")
                        # Append the failure to the context so it corrects itself without permanent mutation
                        current_messages.append({"role": "assistant", "content": raw_output})
                        
                        if "response_format" in base_generate_kwargs:
                            print(f"[{self.agent_id}] Disabling Native Structured Outputs and falling back to prompt injection...")
                            del base_generate_kwargs["response_format"]
                            schema_str = json.dumps(active_model.model_json_schema())
                            current_messages.append({"role": "user", "content": f"Your strict JSON engine failed. Error: {e}. \n\nI have disabled strict JSON mode. You MUST now output valid JSON that strictly conforms to this JSON schema. Put it inside a ```json block:\n{schema_str}"})
                        else:
                            current_messages.append({"role": "user", "content": f"You output invalid JSON. Error: {e}. Please strictly adhere to the schema."})
                        continue
            
            else:
                # Fallback for completely unstructured text
                self.working_memory.append({"role": "assistant", "content": raw_output})
                if self.one_shot: self.working_memory.clear()
                
                return raw_output

# --- PHASE 1 TEST BLOCK ---
if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    
    # Load env vars for API keys
    load_dotenv(override=True)
    if os.getenv("GROK_API_KEY") and not os.getenv("XAI_API_KEY"):
        os.environ["XAI_API_KEY"] = os.getenv("GROK_API_KEY")
        
    # Map the custom Gemini key if present
    if os.getenv("GEMINI_API_KEYX") and not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEYX")

    async def main():
        # Load kwargs dynamically from config.json to ensure litellm has the correct provider routing
        use_llm = "phi-agent"  # Default fallback
        model_override = None
        try:
            from show_available_models import get_available_models
            available = get_available_models()
            
            # Prioritize local gemma4 / gemma model as requested
            if "ollama" in available and available["ollama"]:
                ollama_models = available["ollama"]
                
                # Find the exact local name (e.g. gemma:latest or gemma4:e4b)
                gemma_match = [m for m in ollama_models if "e4b" in m.lower()]
                phi_match = [m for m in ollama_models if "phi" in m.lower()]
                qwen_match = [m for m in ollama_models if "qwen" in m.lower()]
                llama_match = [m for m in ollama_models if "llama" in m.lower()]
                
                if gemma_match:
                    use_llm = "gemma-agent"
                    model_override = "ollama/" + gemma_match[0]
                elif phi_match:
                    use_llm = "phi-agent"
                    model_override = "ollama/" + phi_match[0]
                elif qwen_match:
                    use_llm = "qwen-agent"
                    model_override = "ollama/" + qwen_match[0]
                elif llama_match:
                    use_llm = "llama-agent"
                    model_override = "ollama/" + llama_match[0]
            elif "gemini" in available and available["gemini"]:
                use_llm = "gemini-low"
        except Exception as e:
            print(f"Note: Dynamic model lookup failed, defaulting to phi-agent. Error: {e}")

        print(f"Selected test agent configuration: {use_llm}")
        test_kwargs = CoreAgent.load_litellm_kwargs_from_config(use_llm)
        if model_override:
            print(f"Overriding model string to match actual running version: {model_override}")
            test_kwargs["model"] = model_override
            
        test_kwargs["temperature"] = 0.0

        # Initialize Phase 1 Agent
        agent = CoreAgent(
            agent_id=f"test_agent - {use_llm}",
            system_prompt= """
            You are a strategic combat advisor in the area of air to air space superiority.  
            You are also a tactician and strategist with an emphasis on unconventional thinking and solutions.  
            You are also a historian with an emphasis on military history and tactics.  
            You are also a psychologist with an emphasis on group dynamics and decision making under pressure.
            """,
            litellm_kwargs=test_kwargs,
            one_shot=True # We don't save memory to disk yet
        )
        
        print("\n--- Sending Prompt to Agent ---")
        prompt = """
        If you were in a space fight against a ship that had a weapons system and an engine system.  
        Your ship is identical.  You could also just win with straight weapon damage destroying 
        the ship as your ships are not armored.  What would you do, target a system or just 
        destroy the ship?  Each ships have light shields that can absorb a few hits before depletion.
        Your ships have weapons on each side that each take 3 hits to destory.  Hits will reduce 
        the rate of fire of the weapons system as it degrades.  The engines take 5 hits to destroy and will 
        reduce the speed of the ship as it degrages.  After the shields go down, you can destroy 
        the ship with 16 hits.  6 hits to the weapons, 5 for the engines, and 5 for the rest.  Targeting
        weapons has a 35 percent chance to hit.  Targeting engines has a 55 percent chance to hit.  Hitting 
        in general has a 65 percent chance to hit.  
        """
        # print(f"User: {prompt}")
        
        # Call the agent
        response_obj = await agent.ask(prompt)
        
        print("\n--- Agent Response ---")
        print(f"THOUGHTS: {response_obj.thoughts}")
        print(f"ANSWER: {response_obj.final_answer}")
        
        print(f"\n[Tracking] Tokens Used: {agent.tokens_used} | Est. Cost: ${agent.cost:.6f} | Total Time: {agent.total_response_time:.2f}s")

    asyncio.run(main())
