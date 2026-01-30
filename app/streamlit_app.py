"""Streamlit chat UI for the procurement assistant."""

import streamlit as st

from app.agent import run_agent

st.set_page_config(page_title="Procurement Assistant", page_icon="📊", layout="wide")
st.title("AI Procurement Assistant")
st.caption("Ask questions about California state purchase order data (2012–2015)")

# --- Sidebar with example queries ---
with st.sidebar:
    st.header("Example Queries")
    examples = [
        "How many orders were created in 2013?",
        "Which quarter had the highest total spending?",
        "What are the top 10 most frequently ordered items?",
        "Show me the total spending by department for fiscal year 2014-2015",
        "Which suppliers have the most orders?",
        "What is the average order value by acquisition type?",
        "How many orders used CalCard?",
        "What are the distinct fiscal years in the dataset?",
    ]
    for example in examples:
        if st.button(example, use_container_width=True):
            st.session_state["pending_query"] = example

# --- Chat state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle input: either from sidebar button or chat input
query = st.chat_input("Ask a question about procurement data...")

if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")

if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Run agent and show response
    with st.chat_message("assistant"):
        with st.spinner("Querying data..."):
            response = run_agent(query, st.session_state.messages[:-1])
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
