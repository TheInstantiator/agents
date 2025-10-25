import os
import sys
from openai import OpenAI
from dotenv import load_dotenv


# Load .env file from the project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

class GrokAgent:
    def __init__(self, model="grok-4"):  # Use "grok-3" for free tier if available
        self.model = model
        self.conversation_history = []  # List of dicts: [{"role": "user/system", "content": "..."}]


    def add_message(self, role, content):
        """Add a message to the conversation history."""
        self.conversation_history.append({"role": role, "content": content})


    def chat(self, user_input, stream=False):
        """Send a message to Grok and get a response."""
        self.add_message("user", user_input)
       
        # Prepare messages for the API (include system prompt for agent behavior)
        system_prompt = "You are Grok, a helpful and witty AI built by xAI. Respond concisely and truthfully."
        messages = [{"role": "system", "content": system_prompt}] + self.conversation_history
       
        client = OpenAI(
            api_key=os.getenv("GROK_API_KEY"),
            base_url="https://api.x.ai/v1"
        )
       
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,  # Adjust for creativity (0-1)
            stream=stream
        )
       
        if stream:
            # Stream the response token by token
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content
            print()  # New line after streaming
            self.add_message("assistant", full_response)
            return full_response
        else:
            # Non-streaming: Get full response
            full_response = response.choices[0].message.content
            self.add_message("assistant", full_response)
            return full_response


def fetch_play_by_play(game_id):
    """Fetch the first 10 plays from play_by_play table for the given game_id."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)  # Use dict cursor for easier access
        query = """
            SELECT * FROM play_by_play
            WHERE game_id = %s
            ORDER BY play_id
            LIMIT 10
        """
        cursor.execute(query, (game_id,))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        sys.exit(1)


def interpret_detail(agent, detail):
    """Use the Grok agent to interpret a single detail string."""
    prompt = f"""
    Analyze this NFL play detail: '{detail}'
   
    Determine if it's a play or something else (e.g., coin toss, kickoff, timeout).
   
    If it's a play:
    - Type: pass, run, or other (e.g., punt, field goal).
    - For pass: thrownBy: name, receivedBy: name, direction: direction, yards: yards. Tackled by: names. Penalty: [if any, who committed, yards, accepted/declined, else none].
    - For run: ranBy: name, direction: direction, yards: yards. Tackled by: names. Penalty: [if any, who committed, yards, accepted/declined, else none].
    - For other: Provide a concise summary including tackled by if applicable. Penalty: [if any, who committed, yards, accepted/declined, else none].
   
    If non-play: Type: non-play. Details: concise summary. Penalty: [if any, who committed, yards, accepted/declined, else none].
   
    Respond strictly in this structured format (no extra text):
    Type: [pass/run/other/non-play]
    Details: [structured as above]
    Penalty: [details or none]
    """
    return agent.chat(prompt, stream=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grok Agent for NFL Play Analysis")
    parser.add_argument("--gameID", type=str, help="Game ID to analyze (e.g., 202109120cin)")
    args = parser.parse_args()


    agent = GrokAgent(model="grok-4-fast")  # Or "grok-3"


    if args.gameID:
        print(f"Fetching plays for game ID: {args.gameID}")
        plays = fetch_play_by_play(args.gameID)
        if not plays:
            print("No plays found for this game ID.")
            sys.exit(0)
       
        for play in plays:
            detail = play.get('detail')  # Assuming 'detail' is the column name
            if detail:
                print(f"\nRaw Detail: {detail}")
                interpretation = interpret_detail(agent, detail)
                print(f"Interpretation:\n{interpretation}")
    else:
        print("Grok Agent ready! Enter a gameID (e.g., 202109120cin) or type 'quit' to exit.")
        while True:
            user_input = input("\nEnter gameID: ")
            if user_input.lower() == "quit":
                break
            # Treat input as gameID and analyze
            print(f"Fetching plays for game ID: {user_input}")
            plays = fetch_play_by_play(user_input)
            if not plays:
                print("No plays found for this game ID. Try another.")
                continue
           
            for play in plays:
                detail = play.get('detail')  # Assuming 'detail' is the column name
                if detail:
                    print(f"\nRaw Detail: {detail}")
                    interpretation = interpret_detail(agent, detail)
                    print(f"Interpretation:\n{interpretation}")
