# factory.py
# This file is the "Assembly Line" for your characters. It defines what a character 
# looks like (using Pydantic models) and provides functions to build, save, and load them.

import json
import os
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

# --- DATA STRUCTURES (Pydantic Models) ---
# We use Pydantic to ensure our data is always in the correct format. 
# If we try to create a character with a missing field, Python will raise an error.

class Stats(BaseModel):
    """The six core attributes of a D&D character."""
    STR: int
    DEX: int
    CON: int
    INT: int
    WIS: int
    CHA: int

class CharacterIdentity(BaseModel):
    """The 'flavor' and identify of a character (Name, Bio, etc.)."""
    thoughts: str = Field(default="", description="Your internal reasoning for picking this class and role.")
    name: str
    species: str
    background: str
    classic_role: str  # e.g., "HEALER", "FRONTLINE"
    actual_class: str  # e.g., "Cleric", "Fighter"
    backstory: str = Field(description="A 1-3 sentence origin story.")

class EvaluationResult(BaseModel):
    """The format the 'Judge Agent' uses to tell us if a party is balanced."""
    thoughts: str = Field(default="", description="Step by step reasoning evaluating the party balance.")
    is_valid: bool
    feedback: str
    suggested_replacements: list[str] | None = None

class PlayerCharacter(BaseModel):
    """The final completed character object, including both stats and identity."""
    identity: CharacterIdentity
    stats: Stats
    hp: int
    ac: int
    level: int = 1
    xp: int = 0
    proficiency_bonus: int = 2
    inventory: List[str] = []
    is_alive: bool = True
    played_by: str = ""


# --- HELPERS ---

def get_modifier(score: int) -> int:
    """Calculates a D&D modifier from an attribute score. (Score 10 = +0, 12 = +1, etc.)"""
    return (score - 10) // 2

def create_pc_from_choice(identity: CharacterIdentity, stats_dict: Dict, played_by: str = "") -> PlayerCharacter:
    """
    This function acts as the 'Factory'. It takes the raw identity (from the AI) 
    and stats (from constants.py) and calculates health and defense.
    """
    from constants import CLASS_HIT_DIE
    
    # Calculate health: Level 1 HP = Max Hit Die + Constitution Modifier
    con_mod = get_modifier(stats_dict["CON"])
    base_hp = CLASS_HIT_DIE.get(identity.actual_class, 8)
    
    # Calculate defense: Armor Class (AC) = 10 + Dexterity Modifier (Very simplified!)
    dex_mod = get_modifier(stats_dict["DEX"])
    base_ac = 10 + dex_mod
    
    # Return the fully assembled PlayerCharacter object
    return PlayerCharacter(
        identity=identity,
        stats=Stats(**stats_dict),
        hp=base_hp + con_mod,
        ac=base_ac,
        played_by=played_by
    )


# --- PERSISTENCE (Saving and Loading) ---

def save_party(party_list: List[PlayerCharacter], filename: str = 'party_state.json'):
    """Converts the list of heroes into JSON and saves it to a file."""
    # model_dump() turns the Pydantic object into a regular Python dictionary
    party_data = [char.model_dump() for char in party_list]
    with open(filename, 'w') as f:
        json.dump(party_data, f, indent=4)

def load_party(filename: str = 'party_state.json') -> List[PlayerCharacter]:
    """Reads a JSON file and turns it back into a list of PlayerCharacter objects."""
    if not os.path.exists(filename):
        return []
    with open(filename, 'r') as f:
        data = json.load(f)
    # The double asterisk (**) 'unpacks' the dictionary so Pydantic can read it
    return [PlayerCharacter(**item) for item in data]


# --- TEST BLOCK ---
# This code ONLY runs if you execute 'python factory.py' directly. 
# It's used for testing to make sure the library works.
if __name__ == "__main__":
    from constants import CLASS_ATTRIBUTES

    identity = CharacterIdentity(
        name="Sir Kaelen",
        species="Human",
        background="Noble",
        classic_role="Paladin",
        actual_class="Paladin",
        backstory="A fallen knight seeking redemption."
    )
    stats = CLASS_ATTRIBUTES["Paladin"]
    paladin = create_pc_from_choice(identity, stats)
    
    print("--- CHARACTER CREATED ---")
    print(f"Name: {paladin.identity.name} | HP: {paladin.hp} | AC: {paladin.ac}")