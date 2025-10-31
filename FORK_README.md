Setup:
Clone the repo:
https://github.com/ed-donner/agents.git
Fork the repo:
https://github.com/TheInstantiator/agents.git
Setup git go push changes to the fork on agents-main and pull new changes from the original
VSCode is the IDE
Extensions
Python
Jupyter
UV Package Manager is the python package manager
Setup .env with GROK_API_KEY, GEMINI_API_KEY, and OPENAI_KEY
Deeseek you need to give some money to see api key

OpenAI calls for other APIs
GROK_BASE_URL = "https://api.x.ai/v1"
grok = OpenAI(base_url=GROK_BASE_URL, api_key=grok_api_key)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
gemini = OpenAI(base_url=GEMINI_BASE_URL, api_key=gemini_api_key)

AI Instructions:
I am in the clone repo, setup the fork it exists.  The agents-main branch exists in origin.
Set it up so I push my changes to agents-main and pull or fetch upstream from the ed-donner original repo.
Then verify that uv is installed otherwise install it.  You are in wsl linux.
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh

## System Prompts 
# System Prompt (system_prompt):

Acts like giving the AI its job description
Tells it to be an NFL injury report analyst
Defines exactly what information to look for (name, injury, status)
Specifies how to format the output (markdown tables)
This stays consistent across all injury report requests

# User Prompt (user_prompt_prefix):

Gives the specific task for this particular website
Tells it to ignore non-injury content
Specifies what to do with the website content that follows
Gets combined with the actual website content
Messages List:

Combines both prompts in the format OpenAI expects
Always has system message first, then user message

Git Flow:
I have forked a github training branch to my github
I want to make my changes in the branch agents-main 
I want to get changes from the forked main sometimes

# fetch latest from upstream
git fetch upstream

# ensure you're on agents-main
git checkout agents-main

# merge upstream/main
git merge upstream/main
# or rebase:
# git rebase upstream/main

# push merged changes to your fork
git push origin agents-main