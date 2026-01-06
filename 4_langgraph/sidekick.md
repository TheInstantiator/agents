# Sidekick

**The sidekick module is composed of 3 tools:**

1. `sidekick_tools.py`
   This file contains the tools that the sidekick can use.

   - playwright - perform web actions
   - serper - perform web search actions
   - pushover - provide push notifications to the user
   - file tools - access the file system
   - wikipedia - perform wikipedia actions
   - python repl - perform python actions, create and execute python code
   **Not in a docker container**

2. `sidekick.py`
   This file contains the sidekick agent.  It has the sidekick class, the prompts, and the graph.

3. `sidekick.md`
   This file contains the sidekick's description.
