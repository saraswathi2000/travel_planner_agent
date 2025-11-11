import streamlit as st
import os
import json
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory

from tools import (
    load_sheet_data, 
    find_flights, 
    find_hotels,
    save_message_to_sheet,
    load_chat_history_from_sheet,
    clear_sheet_history,
    get_all_sessions
)

# Configuration

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]


SHEET_NAME = "AI_Agent_data"

# Page configuration
st.set_page_config(
    page_title="🧳 Travel Planner Agent",
    page_icon="✈️",
    layout="wide"
)

# Initialize LLM
@st.cache_resource
def get_llm():
    return ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",
        temperature=0.3
    )

llm = get_llm()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

if "session_id" not in st.session_state:
    st.session_state.session_id = "default"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "history_loaded" not in st.session_state:
    st.session_state.history_loaded = False


# ============== HELPER FUNCTIONS ==============

def format_chat_history() -> str:
    """Format chat history as a readable string"""
    messages = st.session_state.chat_history.messages
    if not messages:
        return "No previous conversation."

    formatted = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"Assistant: {msg.content}")

    return "\n".join(formatted)


def restore_chat_history(session_id: str = "default"):
    """Restore chat history from Google Sheets into memory"""
    messages = load_chat_history_from_sheet(SHEET_NAME, session_id)
    
    st.session_state.chat_history.clear()
    st.session_state.messages = []
    
    for msg in messages:
        role = msg.get("Role", "")
        content = msg.get("Content", "")
        
        if role == "user":
            st.session_state.chat_history.add_user_message(content)
            st.session_state.messages.append({"role": "user", "content": content})
        elif role == "assistant":
            st.session_state.chat_history.add_ai_message(content)
            st.session_state.messages.append({"role": "assistant", "content": content})
    
    return len(messages)


def clear_history(session_id: str = "default"):
    """Clear chat history from both memory and Google Sheets"""
    st.session_state.chat_history.clear()
    st.session_state.messages = []
    clear_sheet_history(SHEET_NAME, session_id)


# ============== LLM CHAINS ==============

# Step 1 — Extract trip details
extract_prompt = ChatPromptTemplate.from_template("""
You are an assistant that extracts structured travel details from a user's request.

Conversation so far:
{chat_history}

User Request:
{user_text}

Return a valid JSON with keys:
origin_city, destination_city, start_date, end_date, trip_length_days, budget_usd, interests.

If anything is missing, set it to null.
""")

extract_chain = RunnableSequence(extract_prompt | llm | StrOutputParser())


# Step 2 — Summarize
summary_prompt = ChatPromptTemplate.from_template("""
You are a helpful travel planner. Use the chat history and structured data below
to write a friendly, detailed itinerary.

Chat History:
{chat_history}

Structured Data:
{final_state}

Include:
- Flights summary
- Hotel recommendation
- Day-by-day plan (3–5 days)
- Budget breakdown

If there are no flight or hotel options available, suggest alternatives and explain the situation.
""")

summary_chain = RunnableSequence(summary_prompt | llm | StrOutputParser())


# ============== TOOL EXECUTION ==============

def simulate_tool_calls(structured_json_str: str) -> Dict[str, Any]:
    """Simulate tool calls to fetch flight and hotel data"""
    try:
        structured_data = json.loads(structured_json_str)
    except json.JSONDecodeError:
        structured_data = {}

    origin = structured_data.get("origin_city") or ""
    destination = structured_data.get("destination_city") or ""
    start_date = structured_data.get("start_date")
    budget = structured_data.get("budget_usd")
    nights = structured_data.get("trip_length_days") or 3
    budget_per_night = None

    if budget:
        try:
            budget_per_night = (float(budget) * 0.4) / nights
        except Exception:
            pass

    try:
        flights_df = load_sheet_data(SHEET_NAME, "flights_data")
        hotels_df = load_sheet_data(SHEET_NAME, "hotels_data")

        flights = find_flights(flights_df, origin, destination, prefer_date=start_date, budget_usd=budget)
        hotels = find_hotels(hotels_df, destination, budget_per_night=budget_per_night)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        flights = []
        hotels = []

    structured_data["tool_results"] = {
        "flight_options": flights,
        "hotel_options": hotels
    }

    return {"final_state": json.dumps(structured_data, indent=2)}


# ============== MAIN AGENT ==============

def run_agent(user_text: str, session_id: str = "default") -> str:
    """Main agent workflow with memory"""

    # Add user message to chat history (memory)
    st.session_state.chat_history.add_user_message(user_text)
    
    # Save user message to Google Sheets
    save_message_to_sheet(SHEET_NAME, "user", user_text, session_id)

    # Format chat history for the prompt
    formatted_history = format_chat_history()

    # Extract structured data
    with st.spinner("Extracting travel details..."):
        structured_data = extract_chain.invoke({
            "user_text": user_text,
            "chat_history": formatted_history
        })

    # Simulate tool calls
    with st.spinner(" Finding flights and hotels..."):
        tool_output = simulate_tool_calls(structured_data)

    # Generate final summary
    with st.spinner(" Creating your plan"):
        final_output = summary_chain.invoke({
            "final_state": tool_output["final_state"],
            "chat_history": formatted_history
        })

    # Save assistant reply to chat history (memory)
    st.session_state.chat_history.add_ai_message(final_output)
    
    # Save assistant message to Google Sheets
    save_message_to_sheet(SHEET_NAME, "assistant", final_output, session_id)

    return final_output


# ============== STREAMLIT UI ==============

# Sidebar
with st.sidebar:
    # st.title("🧳 Travel Planner")
    st.markdown("---")
    
    # Session Management
    # st.subheader("📝 Session Management")
    
    # # Get all available sessions
    # all_sessions = get_all_sessions(SHEET_NAME)
    # if not all_sessions:
    #     all_sessions = ["default"]
    
    # # Session selector
    # selected_session = st.selectbox(
    #     "Select Session",
    #     options=all_sessions,
    #     index=all_sessions.index(st.session_state.session_id) if st.session_state.session_id in all_sessions else 0
    # )
    
    # # New session input
    # new_session = st.text_input("Or create new session:", placeholder="e.g., paris-trip-2025")
    
    # if st.button("Switch/Create Session"):
    #     if new_session:
    #         st.session_state.session_id = new_session
    #         st.session_state.history_loaded = False
    #         st.rerun()
    #     elif selected_session != st.session_state.session_id:
    #         st.session_state.session_id = selected_session
    #         st.session_state.history_loaded = False
    #         st.rerun()
    
    # st.markdown("---")
    
    # Action buttons
    st.subheader("Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(" Restore History"):
            count = restore_chat_history(st.session_state.session_id)
            st.success(f"Restored {count} messages")
            st.rerun()
    
    with col2:
        if st.button("Clear History"):
            clear_history(st.session_state.session_id)
            st.success("History cleared!")
            st.rerun()
    
    st.markdown("---")
    
    # Current session info
    st.info(f"**Current Session:** {st.session_state.session_id}")
    st.caption(f"**Total Messages:** {len(st.session_state.messages)}")

# Main content
st.title(" Travel Planner Agent")
# st.markdown("Plan your perfect trip with AI assistance!")

# Load history on first run
if not st.session_state.history_loaded:
    with st.spinner("Loading conversation history..."):
        count = restore_chat_history(st.session_state.session_id)
        if count > 0:
            st.success(f"Loaded {count} previous messages")
    st.session_state.history_loaded = True

# Display chat messages
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Where would you like to go? "): #(e.g., 'Plan a 5-day trip to Paris from NYC for $2000')
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add to messages
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get bot response
    try:
        response = run_agent(prompt, st.session_state.session_id)
        
        # Display assistant message
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Add to messages
        st.session_state.messages.append({"role": "assistant", "content": response})
        
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
# st.caption("💡 Tip: Start a new session for each trip to keep conversations organized!")
