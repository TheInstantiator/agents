import json
import asyncio
import sys
import os
from typing import List, Dict
from pydantic import BaseModel, Field

# ========================= PATHS =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.append(os.path.join(SCRIPT_DIR, ".."))

from core_agent import CoreAgent

# Import the exact pipeline logic from the app rather than duplicating it!
from core_rag_pipeline import answer_dnd_question, answer_kwargs

# ========================= MODELS =========================
class AnswerEvalVerdict(BaseModel):
    thoughts: str = Field(description="Reasoning for the verdict.")
    is_correct: bool = Field(description="Whether the generated answer matches the expected answer factually.")

# ========================= AGENTS =========================
# The Judge is the only agent unique to tests, it doesn't exist in the core app.
answer_judge = CoreAgent(
    agent_id="answer_judge",
    system_prompt="You are an impartial evaluator. Compare the Generated Answer to the Expected Answer. Determine if the Generated Answer contains the core information required by the Expected Answer. CRITICAL RULES: 1) Keep your 'thoughts' incredibly concise (1-2 sentences maximum). 2) DO NOT penalize the Generated Answer for containing EXTRA helpful info, as long as the core Expected facts are present. If the core facts match, output is_correct: true.",
    litellm_kwargs=CoreAgent.load_litellm_kwargs_from_config("phi-agent"),  # Use Phi for rigid binary logic judging
    response_model=AnswerEvalVerdict,
    one_shot=True
)

# ========================= HELPERS =========================
class MultiLogger:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

# ========================= MAIN EVAL =========================
async def run_pre_release_eval(dataset: List[Dict], dataset_name: str, debug_log_file):
    print(f"\n=== PRE-RELEASE EVALUATION: {dataset_name} ===\n")
    total_correct = 0
    total_questions = len(dataset)
    
    # Store failure details for debug log
    for i, item in enumerate(dataset):
        try:
            original_query = item['question']
            expected_answer = item.get('expected_answer', '')

            print(f"\nQ{i+1}: {original_query}")
            print(f"Expected Answer: {expected_answer}\n")

            final_answer_text = ""
            final_thoughts = ""
            
            # Consume the EXACT same generative loop as the Streamlit UI
            async for event in answer_dnd_question(original_query):
                if event["type"] == "classification":
                    print(f"  [Classifier] Question categorized as: {event.get('category')} (Passes allowed: {event.get('max_passes')})")
                
                elif event["type"] == "expansion":
                    print(f"  [Decomposer] Generated {len(event.get('sub_queries', []))} sub-queries: {event.get('sub_queries')}")
                    print(f"  [HyDE] Excerpt: {event.get('hyde_text', '')[:150]}...")
                    
                elif event["type"] == "retrieval_stats":
                    # We might need to add this yield to core_rag_pipeline if you want these specific prints,
                    # but for now we skip or read status.
                    pass
                    
                elif event["type"] == "status":
                    print(f"  [Status] {event['message']}")
                    
                elif event["type"] == "initial_answer":
                    final_thoughts = event.get("thoughts", "")
                    
                elif event["type"] == "critic_eval":
                    YELLOW = "\033[0;33m"
                    RESET = "\033[0m"
                    print(f"  {YELLOW}[Critic Pass {event['pass']} Thoughts]: {event['thoughts']}{RESET}")
                    print(f"  {YELLOW}[Critic Pass {event['pass']} Follow-up]: {event.get('follow_up', '')}{RESET}")
                    
                elif event["type"] == "merge_update":
                    pass # Handled on 'done'
                    
                elif event["type"] == "done":
                    final_answer_text = event["final_answer"]

            # Visual formatting for the final answer
            GREEN_BOLD = "\033[1;32m"
            CYAN = "\033[0;36m"
            RESET = "\033[0m"
            
            print(f"\n{CYAN}THOUGHTS:{RESET}")
            print(f"{CYAN}{final_thoughts}{RESET}")
            print(f"\n{GREEN_BOLD}{'='*60}")
            print(f"FINAL ANSWER:")
            print(f"{final_answer_text}")
            print(f"{'='*60}{RESET}\n")

            # Final Judge
            verdict = await answer_judge.ask(
                f"Question: {original_query}\n\nExpected Answer: {expected_answer}\nGenerated Answer: {final_answer_text}\nIs the Generated Answer factually correct based on the Expected Answer?"
            )

            if verdict.is_correct:
                total_correct += 1
                print(f"  ✅ Correct\n  [Judge]: {verdict.thoughts}")
            else:
                print(f"  ❌ Incorrect\n  [Judge]: {verdict.thoughts}")
                # Log failure to debug log
                debug_info = f"Dataset: {dataset_name}\nQuestion: {original_query}\nExpected: {expected_answer}\nGenerated: {final_answer_text}\nJudge Thoughts: {verdict.thoughts}\n"
                debug_log_file.write(f"{'#'*40}\n{debug_info}\n")
                debug_log_file.flush()

        except Exception as e:
            print(f"  💥 ERROR processing question {i+1}: {e}")
            query_text = item.get('question', 'Unknown') if isinstance(item, dict) else 'Unknown'
            debug_log_file.write(f"{'#'*40}\nERROR on Question {i+1}: {query_text}\nException: {e}\n")
            debug_log_file.flush()
            continue

    score_pct = (total_correct / total_questions * 100) if total_questions > 0 else 0
    print(f"\nFinal Score for {dataset_name}: {total_correct}/{total_questions} ({score_pct:.1f}%)")
    return total_correct, total_questions

if __name__ == "__main__":
    FILES_TO_RUN = [
        "golden_dataset_original.json",
        "golden_dataset_gemini.json",
        "golden_dataset_grok.json",
        "golden_dataset_phi.json"
    ]
    
    OUTPUT_LOG_PATH = os.path.join(SCRIPT_DIR, "eval_pre_release_output_log.txt")
    SUM_LOG_PATH = os.path.join(SCRIPT_DIR, "eval_pre_release_sum_log.txt")
    DEBUG_LOG_PATH = os.path.join(SCRIPT_DIR, "eval_pre_release_debug_log.txt")
    
    # Open files
    out_f = open(OUTPUT_LOG_PATH, "w")
    sum_f = open(SUM_LOG_PATH, "w")
    debug_f = open(DEBUG_LOG_PATH, "w")
    
    # Redirect stdout to both console and output log
    original_stdout = sys.stdout
    sys.stdout = MultiLogger(original_stdout, out_f)
    
    try:
        sum_f.write("=== EVALUATION SUMMARY ===\n\n")
        sum_f.flush()
        
        for filename in FILES_TO_RUN:
            file_path = os.path.join(SCRIPT_DIR, filename)
            if not os.path.exists(file_path):
                print(f"⚠️ Warning: File {filename} not found. Skipping.")
                continue
                
            with open(file_path, 'r') as f:
                dataset = json.load(f)
            
            # Run the eval
            correct, total = asyncio.run(run_pre_release_eval(dataset, filename, debug_f))
            
            # Write to summary log
            pct = (correct / total * 100) if total > 0 else 0
            sum_f.write(f"{filename}: {correct}/{total} ({pct:.1f}%)\n")
            sum_f.flush()
            
        print("\nAll evaluations complete.")
    except KeyboardInterrupt:
        print("\nEvaluation stopped by user.")
        sum_f.write("\n[Evaluation interrupted before completion]\n")
        sum_f.flush()
        
    finally:
        sys.stdout = original_stdout
        out_f.close()
        sum_f.close()
        debug_f.close()