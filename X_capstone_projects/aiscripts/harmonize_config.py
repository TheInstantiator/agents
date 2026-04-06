import json
import os

config_path = "/home/ouar/projects/agents/X_capstone_projects/config.json"
with open(config_path, "r") as f:
    config = json.load(f)

for model in config["model_list"]:
    params = model["litellm_params"]
    if "ollama" not in params.get("model", ""):
        # Update param dict for Grok/Gemini to match Phi/Gemma best practices
        params["max_input_tokens"] = 16384
        params["k_specific"] = 10
        params["temperature"] = 0.0
        
        # Keep reasoning models with one_shot_judge = False, others True
        if "think" in model["model_name"] or "reasoning" in params.get("model", ""):
            params["one_shot_judge"] = False
        else:
            params["one_shot_judge"] = True

with open(config_path, "w") as f:
    json.dump(config, f, indent=4)
