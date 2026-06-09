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
4.  **Escalation, Resting & Leveling:** Recovery phases are woven dynamically into wave progression mechanics to reward survival:
    *   **Alternating 5-Wave Rest Rewards:** Every 5 waves, the party is awarded a recovery phase that alternates in utility:
        *   **Short Rests (Waves 5, 15, 25, etc.):** Allows characters to spend hit dice and recover short-rest features.
        *   **Full Camps / Long Rests (Waves 10, 20, 30, etc.):** Grants a complete reset of spell slots, health, and clears debuffs.
    *   **Leveling (Top-off Reward):** Leveling up occurs at key milestone waves, acting as a complete "top-off" reward that instantly refills all health, spell slots/points, and class resources.
    Once recovery and level-up phases conclude, the next wave begins.
5.  **Termination:** The game ends only when the entire 5-member party has fallen.

---

## 5. Shared Game State Specification
The shared game state is persisted dynamically in `party_state.json` to keep track of combat stats, grid positions, and class resources. Both Player Agents and the DM query and mutate this JSON state:

```json
{
  "active_wave": 1,
  "round_number": 1,
  "combat_state": "out_of_combat",
  "arena_grid_dimensions": [20, 20],
  "party": [
    {
      "name": "Sir Kaelen",
      "actual_class": "Paladin",
      "level": 1,
      "max_hp": 12,
      "current_hp": 12,
      "armor_class": 16,
      "position": [3, 2],
      "is_alive": true,
      "spell_slots": {
        "1st_level": 0
      },
      "class_resources": {
        "lay_on_hands_points": 5,
        "hit_dice_remaining": 1
      }
    }
  ],
  "monsters": [
    {
      "monster_id": "goblin_01",
      "name": "Goblin Scout",
      "max_hp": 7,
      "current_hp": 7,
      "armor_class": 13,
      "position": [15, 18],
      "is_alive": true
    }
  ]
}
```

## 6. Storyboard Agent Output Specification
At the conclusion of each combat turn, the DM passes narrative updates to the Storyboard Agent. The Storyboard Agent compiles these into structured, video-generation-ready prompts according to this schema:

```json
{
  "scene_id": "wave_01_round_02_turn_03",
  "narrative_action": "Sir Kaelen swings his broadsword, splitting the goblin's wooden shield in half as static sparks burst across the dirt.",
  "storyboard_frames": [
    {
      "frame_sequence": 1,
      "image_prompt": "A close-up shot of a human paladin in polished steel armor swinging a glowing broadsword downward, striking a goblin's round wooden shield. Splinters of wood and crackling yellow magical electricity fly through the air, set in a dusty gladiator arena, detailed fantasy art style.",
      "camera_angle": "low angle close-up",
      "motion_direction": "camera tracks the sword swing downward",
      "duration_seconds": 3.0,
      "visual_vibe": "dramatic, high-contrast cinematic lighting"
    }
  ]
}
```

