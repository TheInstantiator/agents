import os
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

query = "How much does a Longsword cost and what damage does it deal?"

doc_table = """Longsword 1d8 Slashing Versatile (1d10) Sap 3 lb. 15 GP

Maul 2d6 Bludgeoning Heavy, Two-Handed Topple 10 lb. 10 GP

Morningstar 1d8 Piercing     - Sap 4 lb. 15 GP

Pike 1d10 Piercing Heavy, Reach, Two-Handed Push 18 lb. 5 GP

Rapier 1d8 Piercing Finesse Vex 2 lb. 25 GP"""

doc_hobgoblin = """Actions

_**Longsword.**_ _Melee Attack Roll:_ +3, reach 5 ft. _Hit:_ 12
(2d10 + 1) Slashing damage.

_**Longbow.**_ _Ranged Attack Roll:_ +3, range 150/600 ft.
_Hit:_ 5 (1d8 + 1) Piercing damage plus 7 (3d4) Poison damage.
### **Hobgoblin Captain**

_Medium Fey (Goblinoid), Lawful Evil_"""

doc_rod = """**Button 1.** A fiery blade sprouts from the end opposite the rod’s flanged head. The flames shed
Bright Light in a 40-foot radius and Dim Light for
an additional 40 feet, and the blade functions as
a magic Longsword or Shortsword (your choice)
that deals an extra 2d6 Fire damage on a hit."""

pairs = [
    [query, doc_table],
    [query, doc_hobgoblin],
    [query, doc_rod]
]

scores = cross_encoder.predict(pairs)

print(f"Table Row Score: {scores[0]}")
print(f"Hobgoblin Score: {scores[1]}")
print(f"Rod Score: {scores[2]}")
