import asyncio
import os
from dotenv import load_dotenv
import litellm

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)
if os.getenv("GROK_API_KEY") and not os.getenv("XAI_API_KEY"):
    os.environ["XAI_API_KEY"] = os.getenv("GROK_API_KEY")

async def test():
    response = await litellm.acompletion(
        model="xai/grok-3-mini",
        messages=[{"role": "user", "content": "Say hello in one word."}]
    )
    print("Response usage:", response.usage)
    print("Has hasattr total_tokens?", hasattr(response.usage, 'total_tokens'))
    try:
        if response.usage:
            print("Total tokens property:", getattr(response.usage, 'total_tokens', 'N/A'))
            if hasattr(response.usage, 'dict'):
                print("Total tokens dict dict:", response.usage.dict().get("total_tokens"))
    except Exception as e:
        print("Error accessing property:", e)
    
    try:
        cost = litellm.completion_cost(completion_response=response)
        print("Calculated Cost:", cost)
    except Exception as e:
        print("Cost exception:", e)

asyncio.run(test())
