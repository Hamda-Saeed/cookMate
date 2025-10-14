"""
CookMate Streamlit UI
Web interface for the voice-driven recipe assistant
"""

import streamlit as st
import time
from cookmate_rag import CookMateRAG,ConversationState
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

# Page config
st.set_page_config(
    page_title="CookMate Assistant",
    page_icon="🍳",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .step-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #FF6B6B;
        margin: 1rem 0;
    }
    .ingredient-list {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'cookmate' not in st.session_state:
    with st.spinner("🔥 Heating up CookMate..."):
        st.session_state.cookmate = CookMateRAG(use_whisper=True)
        st.session_state.messages = []
        st.session_state.stats = {
            'queries': 0,
            'avg_latency': 0,
            'latencies': []
        }

# Header
st.markdown('<h1 class="main-header">🍳 CookMate Assistant</h1>', unsafe_allow_html=True)
st.markdown("##### Your AI-powered cooking companion with voice & context awareness")

# Sidebar
with st.sidebar:
    st.header("📊 Session Info")
    
    # Current recipe status
    state = st.session_state.cookmate.state
    if state.current_recipe_name:
        st.success(f"**Current Recipe:** {state.current_recipe_name}")
        progress = state.current_step / state.total_steps if state.total_steps > 0 else 0
        st.progress(progress)
        st.write(f"Step {state.current_step} of {state.total_steps}")
    else:
        st.info("No active recipe")
    
    st.divider()
    
    # Statistics
    st.header("📈 Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Queries", st.session_state.stats['queries'])
    with col2:
        avg_lat = st.session_state.stats['avg_latency']
        st.metric("Avg Latency", f"{avg_lat:.2f}s")
    
    # Latency chart
    if st.session_state.stats['latencies']:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=st.session_state.stats['latencies'],
            mode='lines+markers',
            name='Latency',
            line=dict(color='#FF6B6B', width=2)
        ))
        fig.update_layout(
            title="Response Latency",
            yaxis_title="Seconds",
            xaxis_title="Query #",
            height=200,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Available recipes
    st.header("📚 Available Recipes")
    recipes = st.session_state.cookmate.recipes
    for recipe_id, recipe in recipes.items():
        with st.expander(recipe['name']):
            st.write(f"⏱️ Prep: {recipe.get('prep_time', 'N/A')}")
            st.write(f"👨‍🍳 Cook: {recipe.get('cook_time', 'N/A')}")
            st.write(f"🍽️ Serves: {recipe.get('servings', 'N/A')}")
            if st.button(f"Start {recipe['name']}", key=f"start_{recipe_id}"):
                query = f"Start {recipe['name']}"
                st.session_state.messages.append({"role": "user", "content": query})
                response, latency = st.session_state.cookmate.process_query(query)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.stats['queries'] += 1
                st.session_state.stats['latencies'].append(latency)
                st.session_state.stats['avg_latency'] = sum(st.session_state.stats['latencies']) / len(st.session_state.stats['latencies'])
                st.rerun()
    
    st.divider()
    
    # Quick actions
    st.header("⚡ Quick Actions")
    if st.button("🔄 Reset Session"):
        st.session_state.cookmate.state = ConversationState()
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.header("💬 Chat with CookMate")

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.chat_input("Ask me anything about cooking...")

# In your Streamlit app, modify the voice section:
with col2:
    voice_mode = st.checkbox("🎤 Voice", help="Enable voice mode (requires microphone)")
    
    # If voice is enabled, show a message
    if voice_mode:
        st.warning("⚠️ Voice mode in browser requires additional setup. For full voice features, run the desktop version.")

# Process user input
if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Show thinking message
    with st.spinner("🤔 Thinking..."):
        try:
            response, latency = st.session_state.cookmate.process_query(user_input)
        except Exception as e:
            response = f"Sorry, I encountered an error: {str(e)}. Please try again!"
            latency = 0
    
    # Update stats
    st.session_state.stats['queries'] += 1
    if latency > 0:
        st.session_state.stats['latencies'].append(latency)
        st.session_state.stats['avg_latency'] = sum(st.session_state.stats['latencies']) / len(st.session_state.stats['latencies'])
    
    # Add assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.rerun()

# Quick suggestion buttons
st.divider()
st.subheader("💡 Try these commands:")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("▶️ Next Step"):
        user_input = "What's next?"
        st.session_state.messages.append({"role": "user", "content": user_input})
        response, latency = st.session_state.cookmate.process_query(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.stats['queries'] += 1
        st.session_state.stats['latencies'].append(latency)
        st.session_state.stats['avg_latency'] = sum(st.session_state.stats['latencies']) / len(st.session_state.stats['latencies'])
        st.rerun()

with col2:
    if st.button("🔁 Repeat"):
        user_input = "Repeat that"
        st.session_state.messages.append({"role": "user", "content": user_input})
        response, latency = st.session_state.cookmate.process_query(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.stats['queries'] += 1
        st.session_state.stats['latencies'].append(latency)
        st.session_state.stats['avg_latency'] = sum(st.session_state.stats['latencies']) / len(st.session_state.stats['latencies'])
        st.rerun()

with col3:
    if st.button("◀️ Previous"):
        user_input = "Go back"
        st.session_state.messages.append({"role": "user", "content": user_input})
        response, latency = st.session_state.cookmate.process_query(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.stats['queries'] += 1
        st.session_state.stats['latencies'].append(latency)
        st.session_state.stats['avg_latency'] = sum(st.session_state.stats['latencies']) / len(st.session_state.stats['latencies'])
        st.rerun()

with col4:
    if st.button("📝 Ingredients"):
        user_input = "What ingredients do I need?"
        st.session_state.messages.append({"role": "user", "content": user_input})
        response, latency = st.session_state.cookmate.process_query(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.stats['queries'] += 1
        st.session_state.stats['latencies'].append(latency)
        st.session_state.stats['avg_latency'] = sum(st.session_state.stats['latencies']) / len(st.session_state.stats['latencies'])
        st.rerun()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🍳 CookMate - Powered by RAG, Whisper, Ollama & ChromaDB</p>
    <p><small>Created for NLP Project | BCS-5A | NUCES Lahore</small></p>
</div>
""", unsafe_allow_html=True)