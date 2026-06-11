import asyncio
from recruiter import evaluate_party
from factory import CharacterIdentity, create_pc_from_choice
from constants import CLASS_ATTRIBUTES

async def run_test():
    # Construct the exact scenario that failed previously due to index-based strictness
    roster = [
        ("Eira Shadowglow", "Cleric"),
        ("Thorn Windrunner", "Ranger"),
        ("Alaric Thorne", "Wizard"),
        ("Kaelen Stonehand", "Fighter"),
        ("Lysandra Moonwhistle", "Bard")
    ]
    
    party = []
    for name, class_name in roster:
        identity = CharacterIdentity(
            name=name,
            species="Human",
            background="Hero",
            classic_role=class_name,
            actual_class=class_name,
            backstory="A brave adventurer."
        )
        stats = CLASS_ATTRIBUTES.get(class_name)
        pc = create_pc_from_choice(identity, stats)
        party.append(pc)
        
    print("--- EVALUATING HOLISTIC PARTY ---")
    party_summary = "\n".join([f"- {p.identity.name}: {p.identity.actual_class}" for p in party])
    print(party_summary)
    
    # Use the exact same agent logic ("gemma-agent") defined in recruiter.py
    eval_result = await evaluate_party("gemma-agent", party)
    
    print(f"\n[JUDGE THOUGHTS]: {eval_result.thoughts}")
    if eval_result.is_valid:
        print(f"\n✅ JUDGE APPROVED: {eval_result.feedback}")
    else:
        print(f"\n❌ JUDGE REJECTED: {eval_result.feedback}")
        if eval_result.replacements:
            print(f"Suggested replacements: {eval_result.replacements}")

if __name__ == "__main__":
    asyncio.run(run_test())
