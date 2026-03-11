import json
import os

config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")

with open(config_path, "r") as f:
    config = json.load(f)

for model in config.get("model_list", []):
    # Set default K values
    # We use k=10 for local (ollama) and k=3 for paid APIs
    if "ollama" in model["litellm_params"].get("model", ""):
        model["litellm_params"]["k_specific"] = 10
    else:
        model["litellm_params"]["k_specific"] = 3

with open(config_path, "w") as f:
    json.dump(config, f, indent=4)

print("Successfully updated config.json with k_specific values!")
