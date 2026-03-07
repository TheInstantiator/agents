# D&D Capstone Project Overview

This document explains the architecture and execution flow of the D&D Capstone project under the `2_openai` directory. The application uses AI agents to generate and assemble a balanced Dungeons & Dragons adventuring party.

## Project Structure

The project is divided into several clear components:

- **`constants.py`**: The "rulebook" of the application. It holds static game data such as character advancement levels, standard array attributes for various classes, and hit dice information. It contains no logic, only structural constants ensuring consistency with D&D rules.
- **`config.json`**: The model configuration file. It provides the setup definitions mapping directly into LiteLLM, outlining API keys, URLs, and model strings for various AI agents (like Grok, Gemini, and locally run Ollama models).
- **`party_state.json`**: The persistence layer. It stores the generated D&D party in JSON format once a valid team has been assembled. If this file exists, the application will load the party from it rather than generating a new one.
- **`recruiter.py`**: The "Brain" and entry point of the application. It imports `litellm` directly, parsing configurations to orchestrate the AI agents that recruit individual characters and judge the overall balance of the completed party.
- **`factory.py`** (noted implicitly): Handles the creation of character objects and serialization/deserialization of the party.

## Execution Flow: `uv run recruiter.py`

When you execute `uv run recruiter.py`, a multi-agent orchestrated process begins. Here is a step-by-step breakdown of what happens under the hood:

### 1. The Persistence Check

The `main()` function starts by checking if `party_state.json` already exists.

- If it does, the application bypasses the AI generation process, loads the existing party members, prints them to the console, and exits.
- If it does not exist, the application begins Phase 2: The Recruitment Loop.

### 2. The AI Recruitment Loop

The application attempts to build a balanced party of 5 members. It allows up to 3 overall attempts to get it right.

For each of the 5 party slots:

- A call is made to the **Recruiter Agent** (`recruit_member`). To ensure variety and leverage different AI strengths, the application **rotates between multiple agents** (e.g., `grok-agent-tier1-think`, `gemini-agent-tier1-think`, `gemini-agent-vanilla`, `grok-agent-vanilla`) in a round-robin fashion for each new character created.
- The AI is provided with a set of D&D class guidelines, the current state of the party (who has been recruited so far), and any previous feedback.
- The AI uses LiteLLM's `response_format` feature to confidently output a structured `CharacterIdentity` (Name, Species, Class, Backstory, etc.).
- Using the returned identity and base stats dynamically pulled from `constants.py`, a final player character object is instantiated and appended to the growing party list.

### 3. The AI Evaluation (Judge)

Once 5 members are recruited, the application calls the **Judge Agent** (`evaluate_party`).

- The Judge looks at the proposed party to ensure it meets strict "Balanced Party Standards" requiring specific roles (Healer, Frontline, Stealth, Arcane).
- It returns an evaluation determining if the party is valid, alongside feedback reasoning.

### 4. Verdict & Storage

- **If Approved:** The successful party is saved to `party_state.json` so it will be remembered on the next execution. The loop breaks, and the application shows the final party.
- **If Rejected:** The Judge's feedback is saved, the current party is wiped, and the script iterates to Attempt #2 (up to a max of 3), feeding the critical feedback to the next Recruiter Agent so it learns from the previous failure.

### 5. Cost and Token Tracking

Throughout the generation process, `litellm` intercepts each response. By inspecting the token usage returned by the AI provider, the application maintains a running tally of total tokens consumed and total API cost. This final cost breakdown is aggregated and printed right after the final party assembles.

## Sequence Diagram

```mermaid
sequenceDiagram
    participant User
    participant Main as recruiter.py
    participant Recruiter as Recruiter Agent
    participant Judge as Judge Agent
    participant File as party_state.json
    
    User->>Main: uv run recruiter.py
    Main->>File: Check for party_state.json
    
    alt File Exists
        File-->>Main: Load saved party
        Main-->>User: Print loaded heroes & Exit
    else File Does Not Exist
        loop Up to 3 Attempts
            loop 5 Times
                Main->>Recruiter: recruit_member(Current Party Summary, Feedback)
                Note right of Recruiter: Utilizes litellm structured response
                Recruiter-->>Main: Returns CharacterIdentity
                Main->>Main: Apply stats, track costs, & add to party
            end
            
            Main->>Judge: evaluate_party(Full 5-member Party)
            Judge-->>Main: EvaluationResult (is_valid, feedback)
            Main->>Main: Track costs
            
            alt is_valid == True
                Main->>File: save_party()
                break
            else is_valid == False
                Main->>Main: Store feedback for next attempt
            end
        end
        Main-->>User: Print assembled final party and total API costs
    end
```
