import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROK_API_KEY"),
    base_url=os.getenv("GROK_BASE_URL", "https://api.x.ai/v1")
)

model_name = "grok-4-fast"

print(f"Testing model: {model_name}")

try:
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "Hello, what is 2+2?"}],
    )
    print("Success!")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Error with {model_name}: {e}")
