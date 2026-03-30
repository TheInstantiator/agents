import streamlit as st
import asyncio
from core_rag_pipeline import route_query_with_history

st.set_page_config(page_title="D&D Swarm AI", page_icon="🐉", layout="centered")

st.title("🐉 D&D Multi-Agent Swarm")
st.markdown("Ask any complex or highly specific question about D&D 5e mechanics, stats, or rules.")

# CSS to make the final answer pop with a vibrant green outline & dark bg
css = """
<style>
.final-answer {
    background-color: var(--secondary-background-color);
    color: var(--text-color);
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #00ff00;
    font-size: 1.1em;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display all previous messages in conversational UI format
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("e.g. How does stealth and hiding actually work?")

if query:
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
        
    with st.chat_message("assistant"):
        status_text = st.empty()
        expanders_container = st.container()
        
        async def run_pipeline():
            # Build history string (last 6 messages to keep context lean)
            history_str = ""
            for msg in st.session_state.chat_history[-7:-1]:
                history_str += f"{msg['role'].upper()}: {msg['content']}\n"
                
            final_ans = ""
            with st.spinner("Waking up Swarm Engine..."):
                async for event in route_query_with_history(query, history_str):
                    
                    if event["type"] == "status":
                        status_text.info(f"🔄 **{event['message']}**")
                        
                    elif event["type"] == "done_from_cache":
                        status_text.empty()
                        with expanders_container.expander("⚡ Router Memory Intercept", expanded=True):
                            st.write("**Router Thoughts:**")
                            st.caption(event['thoughts'])
                            st.success("The answer was found instantly in the chat memory! No database search needed.")
                        st.markdown("### Answer from Memory")
                        st.markdown(f'<div class="final-answer">\n\n{event["final_answer"]}\n\n</div>', unsafe_allow_html=True)
                        st.caption(f"**Metrics:** Tokens: {event.get('tokens', 0)} | Cost: ${event.get('cost', 0.0):.6f}")
                        final_ans = event["final_answer"]
                        
                    elif event["type"] == "classification":
                        with expanders_container.expander("📌 Question Classification", expanded=True):
                            st.write(f"**Category Identified:** {event['category']}")
                            st.write(f"**Critic Loops Allowed:** {event['max_passes']}")
                            
                    elif event["type"] == "expansion":
                        with expanders_container.expander("🧩 Decomposer & HyDE Output", expanded=False):
                            st.write("**Sub-Queries from Decomposer:**")
                            for sq in event['sub_queries']:
                                st.markdown(f"- {sq}")
                            st.write("**HyDE Hallucinated Draft Search:**")
                            st.info(event['hyde_text'])
                            
                    elif event["type"] == "initial_answer":
                        with expanders_container.expander("✍️ Initial Draft Answer", expanded=False):
                            st.write("**Agent Thoughts:**")
                            st.caption(event['thoughts'])
                            st.write("**Draft Response:**")
                            st.markdown(event['answer'])
                            
                    elif event["type"] == "critic_eval":
                        with expanders_container.expander(f"⚖️ Critic Evaluation (Pass {event['pass']})", expanded=True):
                            st.write("**Critic Thoughts:**")
                            st.caption(event['thoughts'])
                            if event['is_complete'] or not event.get('follow_up'):
                                st.success("✅ The Critic has verified the answer is factually complete!")
                            else:
                                st.warning(f"❌ Missing facts detected. Launching full retrieval pipeline for follow-up query:\n> **{event['follow_up']}**")
                                
                    elif event["type"] == "merge_update":
                        with expanders_container.expander(f"🔄 Merger Output (Pass {event['pass']})", expanded=False):
                            st.write("The Merger Agent combined the new facts into the running answer:")
                            st.markdown(event['answer'])
                            
                    elif event["type"] == "done":
                        status_text.empty()
                        st.success("🎉 Swarm process completed successfully!")
                        st.markdown("### Final Official Answer")
                        st.markdown(f'<div class="final-answer">\n\n{event["final_answer"]}\n\n</div>', unsafe_allow_html=True)
                        st.caption(f"**Swarm Metrics:** Tokens: {event.get('tokens', 0)} | Cost: ${event.get('cost', 0.0):.6f}")
                        final_ans = event["final_answer"]
                        
            # Save assistant's answer to memory
            if final_ans:
                st.session_state.chat_history.append({"role": "assistant", "content": final_ans})

        asyncio.run(run_pipeline())
