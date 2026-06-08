# D&D Capstone: "Kobayashi Maru" Survival Scenario Design

## 1. Overview
The D&D Capstone game will be an unwinnable, "hold your ground" survival scenario, inspired by Star Trek's Kobayashi Maru. The 5-member adventuring party will face endless waves of enemies with escalating difficulty. The objective is not victory, but to survive as long as possible before the party is inevitably overwhelmed and defeated.

## 2. Agent Swarm Architecture
The game relies on a multi-agent swarm interacting within a shared game state:

*   **Dungeon Master (DM) Agent:** The core orchestrator. The DM narrates the environment, controls monster behavior and spawns, adjudicates rules (via the RAG MCP/Tool), and resolves combat mechanics (rolls, damage, saving throws).
*   **Player Agents (4-5):** Autonomous agents controlling the individual party members. They must strategize, manage class resources (spells, abilities), and decide on actions during their turn.
*   **Human-in-the-Loop Option:** The system supports a hybrid play mode. The human user can opt to assume control of 1 character. If selected, autonomous agents will pilot the remaining 4 characters. If the user opts out, the swarm pilots all 5 characters autonomously.
*   **Storyboard Agent:** A specialized bridge for AI Video Production. At the conclusion of a turn (or round), the DM passes the narrative summary to the Storyboard Agent. This agent's sole job is to translate the gameplay events into highly visual, structured storyboard prompts designed to feed directly into an AI video generation pipeline.

## 3. Map & Environment Design
*   **Setting - The Magical Gladiator Arena:** The tactical map is a sprawling, magical gladiator arena.
    *   **The Gates:** The party begins in a confined starting area. When the match begins, a massive gate rises, allowing them to enter the main arena. A mirrored, ominous gate sits on the far opposite side, from which enemies spawn.
    *   **The Spectators:** The arena stands are packed with a roaring crowd, overseen by a prominent, ruler-type figure watching the carnage unfold from a raised dais.
    *   **Tactical Terrain (D&D Flavor):** The arena is littered with classic D&D environmental elements to enable tactical play. This includes pillars for cover, difficult terrain (like mud or debris), magical hazards, elevation changes, and interactive objects to reward strategic positioning.
*   **Wave Mechanics:** The DM will spawn enemies from the far gate in distinct waves. Each successive wave scales up in CR (Challenge Rating), introducing deadlier monsters or overwhelming numbers.

## 4. Gameplay Loop
1.  **Initiative:** The GameManager establishes the turn order for all agents, human players, and monsters.
2.  **Turn Execution:** The active entity (Player Agent, Human, or DM controlling monsters) declares their action, movement, and bonus action. The DM resolves the mechanics using the D&D RAG database.
3.  **Visual Translation:** The narrative results of the turn are handed to the Storyboard Agent to generate the video prompts.
4.  **Escalation & Resting:** When the current wave of enemies is wiped out, the DM evaluates pacing. Between certain waves, the party is allowed to take **Short Rests** (to spend hit dice and recover specific abilities). At a critical milestone, they are granted the opportunity for one **Full Camp (Long Rest)** to fully reset before the final, most brutal stages of the Kobayashi Maru. Once rest periods conclude (or if none are granted), the next wave begins.
5.  **Termination:** The game ends only when the entire 5-member party has fallen.
