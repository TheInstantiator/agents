# constants.py
# This file holds "static" game data. Think of it as the rulebook for your application.
# It doesn't perform any logic; it just stores the numbers and names that the rest of the 
# system uses to stay consistent with D&D 2024 rules.

# --- CHARACTER ADVANCEMENT TABLE ---
# This dictionary tracks what a character gets as they level up.
# Key: The Level number (1-10)
# Value: Another dictionary containing the XP needed for that level and the Proficiency Bonus.
CHARACTER_ADVANCEMENT = {
    1:  {"xp_required": 0,       "proficiency_bonus": 2},
    2:  {"xp_required": 300,     "proficiency_bonus": 2},
    3:  {"xp_required": 900,     "proficiency_bonus": 2},
    4:  {"xp_required": 2700,    "proficiency_bonus": 2},
    5:  {"xp_required": 6500,    "proficiency_bonus": 3},
    6:  {"xp_required": 14000,   "proficiency_bonus": 3},
    7:  {"xp_required": 23000,   "proficiency_bonus": 3},
    8:  {"xp_required": 34000,   "proficiency_bonus": 3},
    9:  {"xp_required": 48000,   "proficiency_bonus": 4},
    10: {"xp_required": 64000,   "proficiency_bonus": 4},
}

# --- THE STANDARD ARRAY (Attributes) ---
# In D&D, beginners often use a "Standard Array" of numbers: 15, 14, 13, 12, 10, 8.
# This dictionary pre-assigns those numbers based on the character's class.
# For example, a Barbarian wants their highest score (15) in Strength (STR) 
# and their lowest (8) in Charisma (CHA).
CLASS_ATTRIBUTES = {
    "Barbarian": {"STR": 15, "DEX": 13, "CON": 14, "INT": 10, "WIS": 12, "CHA": 8},
    "Bard":      {"STR": 8,  "DEX": 14, "CON": 12, "INT": 13, "WIS": 10, "CHA": 15},
    "Cleric":    {"STR": 14, "DEX": 8,  "CON": 13, "INT": 10, "WIS": 15, "CHA": 12},
    "Druid":     {"STR": 8,  "DEX": 12, "CON": 14, "INT": 13, "WIS": 15, "CHA": 10},
    "Fighter":   {"STR": 15, "DEX": 14, "CON": 13, "INT": 8,  "WIS": 10, "CHA": 12},
    "Monk":      {"STR": 12, "DEX": 15, "CON": 13, "INT": 10, "WIS": 14, "CHA": 8},
    "Paladin":   {"STR": 15, "DEX": 10, "CON": 13, "INT": 8,  "WIS": 12, "CHA": 14},
    "Ranger":    {"STR": 12, "DEX": 15, "CON": 13, "INT": 8,  "WIS": 14, "CHA": 10},
    "Rogue":     {"STR": 12, "DEX": 15, "CON": 13, "INT": 14, "WIS": 10, "CHA": 8},
    "Sorcerer":  {"STR": 10, "DEX": 13, "CON": 14, "INT": 8,  "WIS": 12, "CHA": 15},
    "Warlock":   {"STR": 8,  "DEX": 14, "CON": 13, "INT": 12, "WIS": 10, "CHA": 15},
    "Wizard":    {"STR": 8,  "DEX": 12, "CON": 13, "INT": 15, "WIS": 14, "CHA": 10},
}

# --- HIT DIE TABLE ---
# A Hit Die determines how much health (HP) a character starts with and gains per level.
# D12 is the highest (Barbarians are tough!), and D6 is the lowest (Wizards are squishy).
CLASS_HIT_DIE = {
    "Barbarian": 12,
    "Fighter": 10, "Paladin": 10, "Ranger": 10,
    "Cleric": 8, "Druid": 8, "Bard": 8, "Rogue": 8, "Warlock": 8, "Monk": 8,
    "Wizard": 6, "Sorcerer": 6
}