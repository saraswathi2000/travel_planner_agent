import streamlit as st
import os
import json
from typing import Dict, Any
import re

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
    page_title="Travel Planner Agent",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #202123;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #ececf1;
    }
    
    .stButton button {
        width: 100%;
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.1);
        background-color: transparent;
        color: white;
        padding: 10px;
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        background-color: rgba(255,255,255,0.1);
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stChatInput {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

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

if "all_sessions" not in st.session_state:
    st.session_state.all_sessions = ["default"]



def check_input_guardrails(user_input: str) -> tuple[bool, str]:
    """
    Check if user input violates guardrails
    Returns: (is_valid, error_message)
    """
    
    if len(user_input.strip()) < 3:
        return False, "Please provide more details about your travel plans."
    
    if len(user_input) > 1000:
        return False, " Your message is too long. Please keep it under 1000 characters."
    
    # inappropriate_keywords = [
    #     'hack', 'exploit', 'illegal', 'drugs', 'weapons',
    #     'violence', 'terrorism', 'steal', 'fraud'
    # ]
    
    # user_input_lower = user_input.lower()
    # for keyword in inappropriate_keywords:
    #     if keyword in user_input_lower:
    #         return False, " I'm a travel planning assistant. Please ask travel-related questions only."
    
    # # Check 3: Non-travel queries (basic detection)
    non_travel_keywords = [
        'recipe', 'code', 'program', 'python', 'javascript', 
        'medicine', 'disease', 'legal advice', 'financial advice',
        'homework', 'essay', 'write me'
    ]
    
    # Only flag if it's clearly not travel-related
    travel_keywords = [
        'trip', 'travel', 'flight', 'hotel', 'vacation', 'visit',
        'tour', 'destination', 'booking', 'budget', 'itinerary',
        'airport', 'plan', 'go to', 'want to', 'going to'
    ]
    
    has_travel_context = any(keyword in user_input_lower for keyword in travel_keywords)
    has_non_travel = any(keyword in user_input_lower for keyword in non_travel_keywords)
    
    if has_non_travel and not has_travel_context:
        return False, "I'm specialized in travel planning. Please ask me about trips, destinations, flights, hotels, or travel activities."
    
    # Check 4: Spam detection (repeated characters/words)
    if re.search(r'(.)\1{10,}', user_input):  # 10+ repeated characters
        return False, " Please provide a valid travel query."
    
    return True, ""


def check_output_guardrails(ai_response: str, user_input: str) -> tuple[bool, str]:
    """
    Check if AI response is appropriate and travel-related
    Returns: (is_valid, error_message)
    """
    
    # Check 1: Response length
    if len(ai_response.strip()) < 50:
        return False, " Response too short. Let me provide more details."
    
    # Check 2: Check if response is actually about travel
    travel_indicators = [
        'flight', 'hotel', 'destination', 'trip', 'travel',
        'itinerary', 'budget', 'vacation', 'visit', 'tour',
        'day', 'activities', 'airport', 'accommodation'
    ]
    
    response_lower = ai_response.lower()
    has_travel_content = any(indicator in response_lower for indicator in travel_indicators)
    
    if not has_travel_content and len(st.session_state.messages) > 0:
        return False, " Let me refocus on your travel plans. Could you tell me more about your trip?"
    
    harmful_patterns = [
        'illegal', 'dangerous', 'unsafe', 'scam', 'fraud'
    ]
    
    if any(pattern in response_lower for pattern in harmful_patterns):
        return False, " I apologize, but I can only provide safe and legal travel advice."
    
    return True, ""


def validate_budget(budget_str: str) -> tuple[bool, str]:
    """Validate budget input"""
    try:
        budget = float(budget_str)
        if budget < 100:
            return False, " Budget seems too low for a realistic trip. Please provide a budget of at least $100."
        if budget > 1000000:
            return False, " Budget seems unrealistically high. Please provide a more reasonable budget."
        return True, ""
    except:
        return True, ""  # Let LLM handle parsing


def sanitize_input(user_input: str) -> str:
    """Sanitize user input"""
    # Remove excessive whitespace
    user_input = ' '.join(user_input.split())
    
    # Remove any potential code injection attempts
    user_input = user_input.replace('<script>', '').replace('</script>', '')
    user_input = user_input.replace('<?php', '').replace('?>', '')
    
    return user_input.strip()




def get_available_destinations() -> str:
    """Get list of available destinations from database"""
    try:
        flights_df = load_sheet_data(SHEET_NAME, "flights_data")
        hotels_df = load_sheet_data(SHEET_NAME, "hotels_data")
        
        # Get unique destinations
        flight_destinations = set(flights_df['destination'].unique()) if 'destination' in flights_df.columns else set()
        hotel_destinations = set(hotels_df['city'].unique()) if 'city' in hotels_df.columns else set()
        
        all_destinations = flight_destinations.union(hotel_destinations)
        
        if all_destinations:
            return ", ".join(sorted(all_destinations))
        return "various destinations"
    except:
        return "various destinations"


def format_chat_history() -> str:

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


def load_all_sessions():
    """Load all available sessions"""
    sessions = get_all_sessions(SHEET_NAME)
    if not sessions:
        sessions = ["default"]
    st.session_state.all_sessions = sessions
    return sessions


def switch_session(session_id: str):
    """Switch to a different session"""
    st.session_state.session_id = session_id
    st.session_state.history_loaded = False
    restore_chat_history(session_id)


def create_new_session():
    """Create a new session"""
    import time
    new_session_id = f"trip_{int(time.time())}"
    st.session_state.session_id = new_session_id
    st.session_state.chat_history.clear()
    st.session_state.messages = []
    st.session_state.history_loaded = True
    if new_session_id not in st.session_state.all_sessions:
        st.session_state.all_sessions.append(new_session_id)



# System guardrail prompt
system_guardrail = """
CRITICAL SYSTEM INSTRUCTIONS - YOU MUST FOLLOW THESE:

1. YOU ARE ONLY A TRAVEL PLANNING ASSISTANT
   - Only answer questions about travel, trips, destinations, flights, hotels, activities
   - Politely decline any non-travel questions
   - Do not provide information on: coding, medicine, legal advice, financial advice, homework, etc.

2. SAFETY AND ETHICS
   - Never suggest illegal activities
   - Always prioritize traveler safety
   - Do not provide information about dangerous locations without warnings
   - Do not help with fraudulent activities

3. STAY IN SCOPE
   - Focus on: destinations, flights, hotels, itineraries, budgets, activities
   - Redirect off-topic questions back to travel planning

4. BE HELPFUL BUT CAUTIOUS
   - Provide realistic travel advice
   - Acknowledge limitations (e.g., "I can't book flights, but I can help you find options")
   - Don't make up information about flights, hotels, or prices not in the data

If user asks non-travel questions, respond with:
"I'm specialized in travel planning. I can help you with trip planning, destinations, flights, hotels, and activities. How can I assist with your travel plans?"
"""

extract_prompt = ChatPromptTemplate.from_template(system_guardrail + """

You are an assistant that extracts structured travel details from a user's request.

Conversation so far:
{chat_history}

User Request:
{user_text}

IMPORTANT INSTRUCTIONS:
1. If this is a NEW trip request, extract all available details
2. If this is a FOLLOW-UP question about an existing trip, preserve previous trip details
3. If user asks NON-TRAVEL questions, set query_type to "off_topic"

Return a valid JSON with keys:
origin_city, destination_city, start_date, end_date, trip_length_days, budget_usd, interests, query_type, needs_clarification, clarification_question, missing_fields

query_type options: "new_trip", "providing_details", "hotel_query", "flight_query", "activity_query", "budget_query", "general_query", "modification", "off_topic"

If query_type is "off_topic", leave all other fields null.
""")

extract_chain = RunnableSequence(extract_prompt | llm | StrOutputParser())

summary_prompt = ChatPromptTemplate.from_template(system_guardrail + """

You are a helpful travel planner. Use the chat history and structured data below to provide a contextual response.

Chat History:
{chat_history}

Structured Data:
{final_state}

User's Current Request:
{user_text}

CRITICAL DATA CONSTRAINT:
- You can ONLY provide information about flights and hotels that are present in the {structured_data["tool_results"]} data
- If tool_results shows empty flight_options or hotel_options, you MUST inform the user that we don't have data for that route/destination
- DO NOT make up or suggest flights, hotels, prices, or itineraries if the data is not in tool_results
- DO NOT provide general travel advice about destinations not in our database
- If no data is available, clearly state: "I don't have flight/hotel information for this route in my database."

RESPONSE INSTRUCTIONS:
1. **Check tool_results first**: If flight_options or hotel_options are empty, inform user immediately
2. **Only use provided data**: Never suggest options not in the tool_results
3. **If data is missing**: Apologize and ask if they'd like to search for a different route/destination that we have data for
4. **If it's a new trip with data**: Provide itinerary using ONLY the flights and hotels from tool_results
5. **If it's a follow-up**: Focus on the specific aspect, but only if data exists
6. **If query_type is "off_topic"**: Politely redirect to travel planning

Example responses when NO data available:
- "I apologize, but I don't have flight information for the route from [origin] to [destination] in my database. Would you like to try a different route?"
- "Unfortunately, I don't have hotel data for [destination] at the moment. Can I help you with another destination?"
- "I currently don't have information for this specific travel route. The destinations I can help with are based on available flight and hotel data in my system."

Stay focused, truthful, and only provide information that exists in tool_results.
""")

summary_chain = RunnableSequence(summary_prompt | llm | StrOutputParser())




def simulate_tool_calls(structured_json_str: str) -> Dict[str, Any]:
    """Simulate tool calls to fetch flight and hotel data"""
    try:
        structured_data = json.loads(structured_json_str)
    except json.JSONDecodeError:
        structured_data = {}

    # Check if query is off-topic
    if structured_data.get("query_type") == "off_topic":
        return {"final_state": json.dumps(structured_data, indent=2)}

    origin = structured_data.get("origin_city") or ""
    destination = structured_data.get("destination_city") or ""
    start_date = structured_data.get("start_date")
    budget = structured_data.get("budget_usd")
    nights = structured_data.get("trip_length_days") or 3
    budget_per_night = None

    if budget:
        try:
            budget = float(budget)
            # Budget validation
            if budget < 100 or budget > 1000000:
                structured_data["budget_warning"] = "Budget seems unusual. Please verify."
            else:
                budget_per_night = (budget * 0.4) / nights
        except Exception:
            pass

    flights = []
    hotels = []
    data_availability = {
        "flights_available": False,
        "hotels_available": False,
        "origin": origin,
        "destination": destination
    }

    try:
        # Only search if we have origin and destination
        if origin and destination:
            flights_df = load_sheet_data(SHEET_NAME, "flights_data")
            hotels_df = load_sheet_data(SHEET_NAME, "hotels_data")

            flights = find_flights(flights_df, origin, destination, prefer_date=start_date, budget_usd=budget)
            hotels = find_hotels(hotels_df, destination, budget_per_night=budget_per_night)
            
            # Track data availability
            data_availability["flights_available"] = len(flights) > 0
            data_availability["hotels_available"] = len(hotels) > 0
            
    except Exception as e:
        # Keep flights and hotels as empty lists
        pass

    structured_data["tool_results"] = {
        "flight_options": flights,
        "hotel_options": hotels,
        "data_availability": data_availability
    }
    
    # Add explicit message if no data found
    if not flights and not hotels and origin and destination:
        structured_data["no_data_message"] = f"No flight or hotel data available for travel from {origin} to {destination}"
    elif not flights and origin and destination:
        structured_data["no_flight_data_message"] = f"No flight data available from {origin} to {destination}"
    elif not hotels and destination:
        structured_data["no_hotel_data_message"] = f"No hotel data available for {destination}"

    return {"final_state": json.dumps(structured_data, indent=2)}


def run_agent(user_text: str, session_id: str = "default") -> str:
    """Main agent workflow with guardrails"""

    user_text = sanitize_input(user_text)
    
    # Check input guardrails
    is_valid, error_msg = check_input_guardrails(user_text)
    if not is_valid:
        return error_msg
    
    # Rate limiting (max 50 messages per session)
    if len(st.session_state.messages) >= 100:
        return " You've reached the maximum number of messages for this session. Please start a new trip."
    
    st.session_state.chat_history.add_user_message(user_text)
    save_message_to_sheet(SHEET_NAME, "user", user_text, session_id)

    formatted_history = format_chat_history()

    try:
        with st.spinner(" Analyzing your request..."):
            structured_data = extract_chain.invoke({
                "user_text": user_text,
                "chat_history": formatted_history
            })

        with st.spinner(" Finding best options..."):
            tool_output = simulate_tool_calls(structured_data)

        with st.spinner(" Preparing your plan..."):
            final_output = summary_chain.invoke({
                "final_state": tool_output["final_state"],
                "chat_history": formatted_history,
                "user_text": user_text
            })
        
        # Check output guardrails
        is_valid, error_msg = check_output_guardrails(final_output, user_text)
        if not is_valid:
            final_output = "I'm here to help with your travel planning. What destination are you interested in?"

        st.session_state.chat_history.add_ai_message(final_output)
        save_message_to_sheet(SHEET_NAME, "assistant", final_output, session_id)

        return final_output
        
    except Exception as e:
        error_response = " I encountered an error processing your request. Please try rephrasing your question."
        st.session_state.chat_history.add_ai_message(error_response)
        save_message_to_sheet(SHEET_NAME, "assistant", error_response, session_id)
        return error_response


with st.sidebar:
    if st.button(" Clear Current Trip", use_container_width=True):
        clear_history(st.session_state.session_id)
        st.success(" Cleared!")
        st.rerun()



st.title(" Travel Planner")
# st.caption("Plan your perfect trip with AI assistance")

# Load history on first run
if not st.session_state.history_loaded:
    with st.spinner("Loading conversation..."):
        count = restore_chat_history(st.session_state.session_id)
    st.session_state.history_loaded = True

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input with guardrails
if prompt := st.chat_input("Where would you like to go? "):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get bot response with guardrails
    try:
        response = run_agent(prompt, st.session_state.session_id)
        
        with st.chat_message("assistant"):
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        
    except Exception as e:
        error_msg = " Something went wrong. Please try again."
        st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
























# import streamlit as st
# import os
# import json
# from typing import Dict, Any
# import re

# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnableSequence
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_community.chat_message_histories import ChatMessageHistory

# from tools import (
#     load_sheet_data, 
#     find_flights, 
#     find_hotels,
#     save_message_to_sheet,
#     load_chat_history_from_sheet,
#     clear_sheet_history,
#     get_all_sessions
# )

# # Configuration
# OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
# SHEET_NAME = "AI_Agent_data"

# # Page configuration
# st.set_page_config(
#     page_title="Travel Planner Agent",
#     page_icon="✈️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS
# st.markdown("""
# <style>
#     [data-testid="stSidebar"] {
#         background-color: #202123;
#     }
    
#     [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
#         color: #ececf1;
#     }
    
#     .stButton button {
#         width: 100%;
#         border-radius: 8px;
#         border: 1px solid rgba(255,255,255,0.1);
#         background-color: transparent;
#         color: white;
#         padding: 10px;
#         transition: all 0.2s;
#     }
    
#     .stButton button:hover {
#         background-color: rgba(255,255,255,0.1);
#     }
    
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
    
#     .stChatInput {
#         border-radius: 12px;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Initialize LLM
# @st.cache_resource
# def get_llm():
#     return ChatOpenAI(
#         openai_api_key=OPENAI_API_KEY,
#         model="gpt-4o-mini",
#         temperature=0.3
#     )

# llm = get_llm()

# # Initialize session state
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = ChatMessageHistory()

# if "session_id" not in st.session_state:
#     st.session_state.session_id = "default"

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# if "history_loaded" not in st.session_state:
#     st.session_state.history_loaded = False

# if "all_sessions" not in st.session_state:
#     st.session_state.all_sessions = ["default"]


# # ============== GUARDRAILS ==============

# def check_input_guardrails(user_input: str) -> tuple[bool, str]:
#     """
#     Check if user input violates guardrails
#     Returns: (is_valid, error_message)
#     """
    
#     # Check 1: Input length
#     if len(user_input.strip()) < 3:
#         return False, "Please provide more details about your travel plans."
    
#     if len(user_input) > 1000:
#         return False, "Your message is too long. Please keep it under 1000 characters."
    
#     # Check 2: Inappropriate content (basic filters)
#     inappropriate_keywords = [
#         'hack', 'exploit', 'illegal', 'drugs', 'weapons',
#         'violence', 'terrorism', 'steal', 'fraud'
#     ]
    
#     user_input_lower = user_input.lower()
#     for keyword in inappropriate_keywords:
#         if keyword in user_input_lower:
#             return False, "I'm a travel planning assistant. Please ask travel-related questions only."
    
#     # Check 3: Non-travel queries (basic detection)
#     non_travel_keywords = [
#         'recipe', 'code', 'program', 'python', 'javascript', 
#         'medicine', 'disease', 'legal advice', 'financial advice',
#         'homework', 'essay', 'write me'
#     ]
    
#     # Only flag if it's clearly not travel-related
#     travel_keywords = [
#         'trip', 'travel', 'flight', 'hotel', 'vacation', 'visit',
#         'tour', 'destination', 'booking', 'budget', 'itinerary',
#         'airport', 'plan', 'go to', 'want to', 'going to'
#     ]
    
#     has_travel_context = any(keyword in user_input_lower for keyword in travel_keywords)
#     has_non_travel = any(keyword in user_input_lower for keyword in non_travel_keywords)
    
#     if has_non_travel and not has_travel_context:
#         return False, "I'm specialized in travel planning. Please ask me about trips, destinations, flights, hotels, or travel activities."
    
#     # Check 4: Spam detection (repeated characters/words)
#     if re.search(r'(.)\1{10,}', user_input):  # 10+ repeated characters
#         return False, "Please provide a valid travel query."
    
#     return True, ""


# def check_output_guardrails(ai_response: str, user_input: str) -> tuple[bool, str]:
#     """
#     Check if AI response is appropriate and travel-related
#     Returns: (is_valid, error_message)
#     """
    
#     # Check 1: Response length
#     if len(ai_response.strip()) < 50:
#         return False, "Response too short. Let me provide more details."
    
#     # Check 2: Check if response is actually about travel
#     travel_indicators = [
#         'flight', 'hotel', 'destination', 'trip', 'travel',
#         'itinerary', 'budget', 'vacation', 'visit', 'tour',
#         'day', 'activities', 'airport', 'accommodation'
#     ]
    
#     response_lower = ai_response.lower()
#     has_travel_content = any(indicator in response_lower for indicator in travel_indicators)
    
#     if not has_travel_content and len(st.session_state.messages) > 0:
#         # Allow initial greetings, but subsequent messages should be travel-related
#         return False, "Let me refocus on your travel plans. Could you tell me more about your trip?"
    
#     # Check 3: Ensure no harmful/inappropriate content in response
#     harmful_patterns = [
#         'illegal', 'dangerous', 'unsafe', 'scam', 'fraud'
#     ]
    
#     if any(pattern in response_lower for pattern in harmful_patterns):
#         return False, " I apologize, but I can only provide safe and legal travel advice."
    
#     return True, ""


# def validate_budget(budget_str: str) -> tuple[bool, str]:
#     """Validate budget input"""
#     try:
#         budget = float(budget_str)
#         if budget < 100:
#             return False, "Budget seems too low for a realistic trip. Please provide a budget of at least $100."
#         if budget > 1000000:
#             return False, "Budget seems unrealistically high. Please provide a more reasonable budget."
#         return True, ""
#     except:
#         return True, ""  # Let LLM handle parsing


# def sanitize_input(user_input: str) -> str:
#     """Sanitize user input"""
#     # Remove excessive whitespace
#     user_input = ' '.join(user_input.split())
    
#     # Remove any potential code injection attempts
#     user_input = user_input.replace('<script>', '').replace('</script>', '')
#     user_input = user_input.replace('<?php', '').replace('?>', '')
    
#     return user_input.strip()


# # ============== HELPER FUNCTIONS ==============

# def format_chat_history() -> str:
#     """Format chat history as a readable string"""
#     messages = st.session_state.chat_history.messages
#     if not messages:
#         return "No previous conversation."

#     formatted = []
#     for msg in messages:
#         if isinstance(msg, HumanMessage):
#             formatted.append(f"User: {msg.content}")
#         elif isinstance(msg, AIMessage):
#             formatted.append(f"Assistant: {msg.content}")

#     return "\n".join(formatted)


# def restore_chat_history(session_id: str = "default"):
#     """Restore chat history from Google Sheets into memory"""
#     messages = load_chat_history_from_sheet(SHEET_NAME, session_id)
    
#     st.session_state.chat_history.clear()
#     st.session_state.messages = []
    
#     for msg in messages:
#         role = msg.get("Role", "")
#         content = msg.get("Content", "")
        
#         if role == "user":
#             st.session_state.chat_history.add_user_message(content)
#             st.session_state.messages.append({"role": "user", "content": content})
#         elif role == "assistant":
#             st.session_state.chat_history.add_ai_message(content)
#             st.session_state.messages.append({"role": "assistant", "content": content})
    
#     return len(messages)


# def clear_history(session_id: str = "default"):
#     """Clear chat history from both memory and Google Sheets"""
#     st.session_state.chat_history.clear()
#     st.session_state.messages = []
#     clear_sheet_history(SHEET_NAME, session_id)


# def load_all_sessions():
#     """Load all available sessions"""
#     sessions = get_all_sessions(SHEET_NAME)
#     if not sessions:
#         sessions = ["default"]
#     st.session_state.all_sessions = sessions
#     return sessions


# def switch_session(session_id: str):
#     """Switch to a different session"""
#     st.session_state.session_id = session_id
#     st.session_state.history_loaded = False
#     restore_chat_history(session_id)


# def create_new_session():
#     """Create a new session"""
#     import time
#     new_session_id = f"trip_{int(time.time())}"
#     st.session_state.session_id = new_session_id
#     st.session_state.chat_history.clear()
#     st.session_state.messages = []
#     st.session_state.history_loaded = True
#     if new_session_id not in st.session_state.all_sessions:
#         st.session_state.all_sessions.append(new_session_id)


# # ============== LLM CHAINS WITH GUARDRAILS ==============

# # System guardrail prompt
# system_guardrail = """
# CRITICAL SYSTEM INSTRUCTIONS - YOU MUST FOLLOW THESE:

# 1. YOU ARE ONLY A TRAVEL PLANNING ASSISTANT
#    - Only answer questions about travel, trips, destinations, flights, hotels, activities
#    - Politely decline any non-travel questions
#    - Do not provide information on: coding, medicine, legal advice, financial advice, homework, etc.

# 2. SAFETY AND ETHICS
#    - Never suggest illegal activities
#    - Always prioritize traveler safety
#    - Do not provide information about dangerous locations without warnings
#    - Do not help with fraudulent activities

# 3. STAY IN SCOPE
#    - Focus on: destinations, flights, hotels, itineraries, budgets, activities
#    - Redirect off-topic questions back to travel planning

# 4. BE HELPFUL BUT CAUTIOUS
#    - Provide realistic travel advice
#    - Acknowledge limitations (e.g., "I can't book flights, but I can help you find options")
#    - Don't make up information about flights, hotels, or prices not in the data

# If user asks non-travel questions, respond with:
# "I'm specialized in travel planning. I can help you with trip planning, destinations, flights, hotels, and activities. How can I assist with your travel plans?"
# """

# extract_prompt = ChatPromptTemplate.from_template(system_guardrail + """

# You are an assistant that extracts structured travel details from a user's request.

# Conversation so far:
# {chat_history}

# User Request:
# {user_text}

# IMPORTANT INSTRUCTIONS:
# 1. If this is a NEW trip request, extract all available details
# 2. If this is a FOLLOW-UP question about an existing trip, preserve previous trip details
# 3. If user asks NON-TRAVEL questions, set query_type to "off_topic"

# Return a valid JSON with keys:
# origin_city, destination_city, start_date, end_date, trip_length_days, budget_usd, interests, query_type, needs_clarification, clarification_question, missing_fields

# query_type options: "new_trip", "providing_details", "hotel_query", "flight_query", "activity_query", "budget_query", "general_query", "modification", "off_topic"

# If query_type is "off_topic", leave all other fields null.
# """)

# extract_chain = RunnableSequence(extract_prompt | llm | StrOutputParser())

# summary_prompt = ChatPromptTemplate.from_template(system_guardrail + """

# You are a helpful travel planner. Use the chat history and structured data below to provide a contextual response.

# Chat History:
# {chat_history}

# Structured Data:
# {final_state}

# User's Current Request:
# {user_text}

# RESPONSE INSTRUCTIONS:
# 1. **Analyze what the user is specifically asking for**
# 2. **If it's a new trip**: Provide complete itinerary (flights, hotels, activities, budget)
# 3. **If it's a follow-up about specific aspect**: Focus ONLY on that aspect
# 4. **If query_type is "off_topic"**: Politely redirect to travel planning

# Stay focused, relevant, and helpful. Only discuss travel-related topics.
# """)

# summary_chain = RunnableSequence(summary_prompt | llm | StrOutputParser())


# # ============== TOOL EXECUTION ==============

# def simulate_tool_calls(structured_json_str: str) -> Dict[str, Any]:
#     """Simulate tool calls to fetch flight and hotel data"""
#     try:
#         structured_data = json.loads(structured_json_str)
#     except json.JSONDecodeError:
#         structured_data = {}

#     # Check if query is off-topic
#     if structured_data.get("query_type") == "off_topic":
#         return {"final_state": json.dumps(structured_data, indent=2)}

#     origin = structured_data.get("origin_city") or ""
#     destination = structured_data.get("destination_city") or ""
#     start_date = structured_data.get("start_date")
#     budget = structured_data.get("budget_usd")
#     nights = structured_data.get("trip_length_days") or 3
#     budget_per_night = None

#     if budget:
#         try:
#             budget = float(budget)
#             # Budget validation
#             if budget < 100 or budget > 1000000:
#                 structured_data["budget_warning"] = "Budget seems unusual. Please verify."
#             else:
#                 budget_per_night = (budget * 0.4) / nights
#         except Exception:
#             pass

#     try:
#         flights_df = load_sheet_data(SHEET_NAME, "flights_data")
#         hotels_df = load_sheet_data(SHEET_NAME, "hotels_data")

#         flights = find_flights(flights_df, origin, destination, prefer_date=start_date, budget_usd=budget)
#         hotels = find_hotels(hotels_df, destination, budget_per_night=budget_per_night)
#     except Exception as e:
#         flights = []
#         hotels = []

#     structured_data["tool_results"] = {
#         "flight_options": flights,
#         "hotel_options": hotels
#     }

#     return {"final_state": json.dumps(structured_data, indent=2)}




# def run_agent(user_text: str, session_id: str = "default") -> str:
#     """Main agent workflow with guardrails"""
    
#     # Sanitize input
#     user_text = sanitize_input(user_text)
    
#     # Check input guardrails
#     is_valid, error_msg = check_input_guardrails(user_text)
#     if not is_valid:
#         return error_msg
    
#     # Rate limiting (max 50 messages per session)
#     if len(st.session_state.messages) >= 100:
#         return "You've reached the maximum number of messages for this session. Please start a new trip."
    
#     st.session_state.chat_history.add_user_message(user_text)
#     save_message_to_sheet(SHEET_NAME, "user", user_text, session_id)

#     formatted_history = format_chat_history()

#     try:
#         with st.spinner("Analyzing your request..."):
#             structured_data = extract_chain.invoke({
#                 "user_text": user_text,
#                 "chat_history": formatted_history
#             })

#         with st.spinner("Finding best options..."):
#             tool_output = simulate_tool_calls(structured_data)

#         with st.spinner("Preparing your plan..."):
#             final_output = summary_chain.invoke({
#                 "final_state": tool_output["final_state"],
#                 "chat_history": formatted_history,
#                 "user_text": user_text
#             })
        
#         # Check output guardrails
#         is_valid, error_msg = check_output_guardrails(final_output, user_text)
#         if not is_valid:
#             final_output = "I'm here to help with your travel planning. What destination are you interested in?"

#         st.session_state.chat_history.add_ai_message(final_output)
#         save_message_to_sheet(SHEET_NAME, "assistant", final_output, session_id)

#         return final_output
        
#     except Exception as e:
#         error_response = " I encountered an error processing your request. Please try rephrasing your question."
#         st.session_state.chat_history.add_ai_message(error_response)
#         save_message_to_sheet(SHEET_NAME, "assistant", error_response, session_id)
#         return error_response


# # ============== SIDEBAR ==============

# with st.sidebar:
#     if st.button("🗑️ Clear Current Trip", use_container_width=True):
#         clear_history(st.session_state.session_id)
#         st.success(" Cleared!")
#         st.rerun()


# # ============== MAIN CONTENT ==============

# st.title(" Travel Planner")
# # st.caption("Plan your perfect trip with AI assistance")

# # Load history on first run
# if not st.session_state.history_loaded:
#     with st.spinner("Loading conversation..."):
#         count = restore_chat_history(st.session_state.session_id)
#     st.session_state.history_loaded = True

# # Display chat messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Chat input with guardrails
# if prompt := st.chat_input("Where would you like to go?"):
#     # Display user message
#     with st.chat_message("user"):
#         st.markdown(prompt)
    
#     st.session_state.messages.append({"role": "user", "content": prompt})
    
#     # Get bot response with guardrails
#     try:
#         response = run_agent(prompt, st.session_state.session_id)
        
#         with st.chat_message("assistant"):
#             st.markdown(response)
        
#         st.session_state.messages.append({"role": "assistant", "content": response})
        
#     except Exception as e:
#         error_msg = "⚠️ Something went wrong. Please try again."
#         st.error(error_msg)
#         st.session_state.messages.append({"role": "assistant", "content": error_msg})









# import streamlit as st
# import os
# import json
# from typing import Dict, Any

# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnableSequence
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_community.chat_message_histories import ChatMessageHistory

# from tools import (
#     load_sheet_data, 
#     find_flights, 
#     find_hotels,
#     save_message_to_sheet,
#     load_chat_history_from_sheet,
#     clear_sheet_history,
#     get_all_sessions
# )

# # Configuration
# OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
# SHEET_NAME = "AI_Agent_data"

# # Page configuration
# st.set_page_config(
#     page_title="Travel Planner Agent",
#     page_icon="✈️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for ChatGPT-style sidebar
# st.markdown("""
# <style>
#     /* Sidebar styling */
#     [data-testid="stSidebar"] {
#         background-color: #202123;
#     }
    
#     [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
#         color: #ececf1;
#     }
    
#     /* Button styling */
#     .stButton button {
#         width: 100%;
#         border-radius: 8px;
#         border: 1px solid rgba(255,255,255,0.1);
#         background-color: transparent;
#         color: white;
#         padding: 10px;
#         transition: all 0.2s;
#     }
    
#     .stButton button:hover {
#         background-color: rgba(255,255,255,0.1);
#     }
    
#     /* Session item styling */
#     .session-item {
#         padding: 12px;
#         margin: 4px 0;
#         border-radius: 8px;
#         cursor: pointer;
#         background-color: rgba(255,255,255,0.05);
#         color: #ececf1;
#         border: 1px solid transparent;
#         transition: all 0.2s;
#     }
    
#     .session-item:hover {
#         background-color: rgba(255,255,255,0.1);
#         border: 1px solid rgba(255,255,255,0.2);
#     }
    
#     .session-item-active {
#         background-color: rgba(255,255,255,0.15);
#         border: 1px solid rgba(255,255,255,0.3);
#     }
    
#     /* Hide Streamlit branding */
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
    
#     /* Chat input styling */
#     .stChatInput {
#         border-radius: 12px;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Initialize LLM
# @st.cache_resource
# def get_llm():
#     return ChatOpenAI(
#         openai_api_key=OPENAI_API_KEY,
#         model="gpt-4o-mini",
#         temperature=0.3
#     )

# llm = get_llm()

# # Initialize session state
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = ChatMessageHistory()

# if "session_id" not in st.session_state:
#     st.session_state.session_id = "default"

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# if "history_loaded" not in st.session_state:
#     st.session_state.history_loaded = False

# if "all_sessions" not in st.session_state:
#     st.session_state.all_sessions = ["default"]




# def format_chat_history() -> str:
#     """Format chat history as a readable string"""
#     messages = st.session_state.chat_history.messages
#     if not messages:
#         return "No previous conversation."

#     formatted = []
#     for msg in messages:
#         if isinstance(msg, HumanMessage):
#             formatted.append(f"User: {msg.content}")
#         elif isinstance(msg, AIMessage):
#             formatted.append(f"Assistant: {msg.content}")

#     return "\n".join(formatted)


# def restore_chat_history(session_id: str = "default"):
#     """Restore chat history from Google Sheets into memory"""
#     messages = load_chat_history_from_sheet(SHEET_NAME, session_id)
    
#     st.session_state.chat_history.clear()
#     st.session_state.messages = []
    
#     for msg in messages:
#         role = msg.get("Role", "")
#         content = msg.get("Content", "")
        
#         if role == "user":
#             st.session_state.chat_history.add_user_message(content)
#             st.session_state.messages.append({"role": "user", "content": content})
#         elif role == "assistant":
#             st.session_state.chat_history.add_ai_message(content)
#             st.session_state.messages.append({"role": "assistant", "content": content})
    
#     return len(messages)


# def clear_history(session_id: str = "default"):
#     """Clear chat history from both memory and Google Sheets"""
#     st.session_state.chat_history.clear()
#     st.session_state.messages = []
#     clear_sheet_history(SHEET_NAME, session_id)


# def load_all_sessions():
#     """Load all available sessions"""
#     sessions = get_all_sessions(SHEET_NAME)
#     if not sessions:
#         sessions = ["default"]
#     st.session_state.all_sessions = sessions
#     return sessions


# def switch_session(session_id: str):
#     """Switch to a different session"""
#     st.session_state.session_id = session_id
#     st.session_state.history_loaded = False
#     restore_chat_history(session_id)


# def create_new_session():
#     """Create a new session"""
#     import time
#     new_session_id = f"trip_{int(time.time())}"
#     st.session_state.session_id = new_session_id
#     st.session_state.chat_history.clear()
#     st.session_state.messages = []
#     st.session_state.history_loaded = True
#     if new_session_id not in st.session_state.all_sessions:
#         st.session_state.all_sessions.append(new_session_id)




# extract_prompt = ChatPromptTemplate.from_template("""
# You are an assistant that extracts structured travel details from a user's request.

# Conversation so far:
# {chat_history}

# User Request:
# {user_text}

# IMPORTANT INSTRUCTIONS:
# 1. If this is a NEW trip request, extract all available details
# 2. If this is a FOLLOW-UP question about an existing trip, preserve previous trip details and only update what changed
# 3. Look at the chat history to understand context
# 4. Check if this is providing missing information from a previous request

# REQUIRED FIELDS FOR A COMPLETE TRIP:
# - origin_city (where traveling from)
# - destination_city (where traveling to)
# - start_date (departure date)
# - end_date (return date) OR trip_length_days
# - budget_usd (total budget)

# Return a valid JSON with these keys:
# - origin_city: string or null
# - destination_city: string or null
# - start_date: string (YYYY-MM-DD format) or null
# - end_date: string (YYYY-MM-DD format) or null
# - trip_length_days: integer or null
# - budget_usd: number or null
# - interests: list of strings or null
# - query_type: string (see below)
# - missing_fields: list of missing required fields (empty list if all present)
# - needs_clarification: boolean (true if critical info is missing)
# - clarification_question: string (specific question to ask user if needs_clarification is true)

# QUERY TYPES:
# - "new_trip": User wants to plan a new trip
# - "providing_details": User is providing missing information from previous request
# - "hotel_query": Asking about hotels
# - "flight_query": Asking about flights
# - "activity_query": Asking about activities/things to do
# - "budget_query": Asking about budget/costs
# - "general_query": General travel question
# - "modification": Modifying existing trip details

# LOGIC FOR MISSING FIELDS:
# 1. For a NEW trip request, identify which required fields are missing
# 2. If ANY required field is missing, set needs_clarification to true
# 3. Generate a natural, conversational clarification_question that asks for the FIRST missing field
# 4. List ALL missing fields in the missing_fields array
# 5. If this is a follow-up providing missing info, set query_type to "providing_details"

# Examples:

# Example 1 - Missing budget:
# User: "I want to travel to New York on November 1st"
# Response:
# {{
#   "origin_city": null,
#   "destination_city": "New York",
#   "start_date": "2025-11-01",
#   "end_date": null,
#   "trip_length_days": null,
#   "budget_usd": null,
#   "interests": null,
#   "query_type": "new_trip",
#   "missing_fields": ["origin_city", "trip_length_days", "budget_usd"],
#   "needs_clarification": true,
#   "clarification_question": "Great! I'd love to help you plan your trip to New York. Where will you be traveling from?"
# }}

# Example 2 - Providing missing info:
# Previous context: Asked about origin city
# User: "From Chicago"
# Response:
# {{
#   "origin_city": "Chicago",
#   "destination_city": "New York",
#   "start_date": "2025-11-01",
#   "end_date": null,
#   "trip_length_days": null,
#   "budget_usd": null,
#   "interests": null,
#   "query_type": "providing_details",
#   "missing_fields": ["trip_length_days", "budget_usd"],
#   "needs_clarification": true,
#   "clarification_question": "Perfect! How many days are you planning to stay in New York?"
# }}

# Example 3 - Complete information:
# User: "I want to go from Chicago to New York from Nov 1-5, budget $2000"
# Response:
# {{
#   "origin_city": "Chicago",
#   "destination_city": "New York",
#   "start_date": "2025-11-01",
#   "end_date": "2025-11-05",
#   "trip_length_days": 5,
#   "budget_usd": 2000,
#   "interests": null,
#   "query_type": "new_trip",
#   "missing_fields": [],
#   "needs_clarification": false,
#   "clarification_question": null
# }}

# Example 4 - Follow-up question:
# User: "Tell me more about hotels"
# Response:
# {{
#   "origin_city": "[keep from history]",
#   "destination_city": "[keep from history]",
#   "start_date": "[keep from history]",
#   "end_date": "[keep from history]",
#   "trip_length_days": "[keep from history]",
#   "budget_usd": "[keep from history]",
#   "interests": null,
#   "query_type": "hotel_query",
#   "missing_fields": [],
#   "needs_clarification": false,
#   "clarification_question": null
# }}

# CLARIFICATION QUESTION PRIORITIES (ask in this order):
# 1. origin_city: "Where will you be traveling from?"
# 2. trip_length_days/end_date: "How many days are you planning to stay?" or "When do you plan to return?"
# 3. budget_usd: "What's your total budget for this trip?"
# 4. interests (optional): "Are there any specific activities or interests you'd like to focus on?"

# Make clarification questions natural, friendly, and conversational.
# """)

# extract_chain = RunnableSequence(extract_prompt | llm | StrOutputParser())

# summary_prompt = ChatPromptTemplate.from_template("""
# You are a helpful travel planner. Use the chat history and structured data below to provide a contextual response.

# Chat History:
# {chat_history}

# Structured Data:
# {final_state}

# User's Current Request:
# {user_text}

# IMPORTANT INSTRUCTIONS:
# 1. **Analyze what the user is specifically asking for** in their current request
# 2. **If the user asks about a specific aspect** (hotels, flights, activities, budget, etc.), focus ONLY on that aspect
# 3. **If it's a new trip request**, provide a complete itinerary with:
#    - Flights summary
#    - Hotel recommendations
#    - Day-by-day plan (3-5 days)
#    - Budget breakdown
# 4. **If it's a follow-up question** about hotels, ONLY discuss hotels
# 5. **If it's a follow-up question** about flights, ONLY discuss flights
# 6. **If it's a follow-up question** about activities, ONLY discuss activities
# 7. **If it's a clarification or modification**, address ONLY what changed

# Examples:
# - "I want to go to Tokyo" → Full itinerary
# - "Tell me more about hotels" → Only hotel details
# - "What about cheaper flights?" → Only flight options
# - "What can I do on day 3?" → Only day 3 activities
# - "Can you suggest budget hotels?" → Only budget hotel options

# Keep responses concise and relevant to what the user asked. Don't repeat information already discussed unless specifically requested.
# """)

# summary_chain = RunnableSequence(summary_prompt | llm | StrOutputParser())




# def simulate_tool_calls(structured_json_str: str) -> Dict[str, Any]:
#     """Simulate tool calls to fetch flight and hotel data"""
#     try:
#         structured_data = json.loads(structured_json_str)
#     except json.JSONDecodeError:
#         structured_data = {}

#     origin = structured_data.get("origin_city") or ""
#     destination = structured_data.get("destination_city") or ""
#     start_date = structured_data.get("start_date")
#     budget = structured_data.get("budget_usd")
#     nights = structured_data.get("trip_length_days") or 3
#     budget_per_night = None

#     if budget:
#         try:
#             budget_per_night = (float(budget) * 0.4) / nights
#         except Exception:
#             pass

#     try:
#         flights_df = load_sheet_data(SHEET_NAME, "flights_data")
#         hotels_df = load_sheet_data(SHEET_NAME, "hotels_data")

#         flights = find_flights(flights_df, origin, destination, prefer_date=start_date, budget_usd=budget)
#         hotels = find_hotels(hotels_df, destination, budget_per_night=budget_per_night)
#     except Exception as e:
#         st.error(f"Error loading data: {e}")
#         flights = []
#         hotels = []

#     structured_data["tool_results"] = {
#         "flight_options": flights,
#         "hotel_options": hotels
#     }

#     return {"final_state": json.dumps(structured_data, indent=2)}


# # ============== MAIN AGENT ==============

# def run_agent(user_text: str, session_id: str = "default") -> str:
#     """Main agent workflow with memory"""

#     st.session_state.chat_history.add_user_message(user_text)
#     save_message_to_sheet(SHEET_NAME, "user", user_text, session_id)

#     formatted_history = format_chat_history()

#     with st.spinner("Extracting travel details..."):
#         structured_data = extract_chain.invoke({
#             "user_text": user_text,
#             "chat_history": formatted_history
#         })

#     with st.spinner("Finding flights and hotels..."):
#         tool_output = simulate_tool_calls(structured_data)

#     with st.spinner("Creating your plan..."):
#         final_output = summary_chain.invoke({
#             "final_state": tool_output["final_state"],
#             "chat_history": formatted_history,
#             "user_text": user_text  # ← Add this
#         })

#     st.session_state.chat_history.add_ai_message(final_output)
#     save_message_to_sheet(SHEET_NAME, "assistant", final_output, session_id)

#     return final_output

# # ============== SIDEBAR (ChatGPT Style) ==============

# with st.sidebar:
# #     # New Chat Button (like ChatGPT)
# #     st.markdown("### ")
# #     if st.button("➕ New Trip", use_container_width=True):
# #         create_new_session()
# #         st.rerun()
    
# #     st.markdown("---")
    
# #     # Chat History Section
# #     st.markdown("### 📚 Your Trips")
    
# #     # Load all sessions
# #     all_sessions = load_all_sessions()
    
# #     # Display sessions as clickable items
# #     for session in all_sessions:
# #         # Get first message preview if available
# #         session_messages = load_chat_history_from_sheet(SHEET_NAME, session)
# #         preview = "New conversation"
        
# #         if session_messages:
# #             first_msg = session_messages[0].get("Content", "")
# #             preview = first_msg[:40] + "..." if len(first_msg) > 40 else first_msg
        
# #         # Determine if this is the active session
# #         is_active = session == st.session_state.session_id
        
# #         # Create button for each session
# #         col1, col2 = st.columns([5, 1])
        
# #         with col1:
# #             if st.button(
# #                 f"💬 {preview}", 
# #                 key=f"session_{session}",
# #                 use_container_width=True,
# #                 type="primary" if is_active else "secondary"
# #             ):
# #                 switch_session(session)
# #                 st.rerun()
        
# #         with col2:
# #             if st.button("🗑️", key=f"delete_{session}", help="Delete this trip"):
# #                 clear_sheet_history(SHEET_NAME, session)
# #                 if session in st.session_state.all_sessions:
# #                     st.session_state.all_sessions.remove(session)
# #                 if session == st.session_state.session_id:
# #                     create_new_session()
# #                 st.rerun()
    
# #     st.markdown("---")
    
# #     # Settings Section
# #     st.markdown("### ⚙️ Settings")
    
# #     if st.button("🔄 Refresh History", use_container_width=True):
# #         count = restore_chat_history(st.session_state.session_id)
# #         st.success(f"✅ Loaded {count} messages")
# #         st.rerun()
    
#     if st.button("Clear Current Trip", use_container_width=True):
#         clear_history(st.session_state.session_id)
#         st.success("Cleared!")
#         st.rerun()
    
# #     # Footer info
# #     st.markdown("---")
# #     st.caption(f"📝 Current: {st.session_state.session_id}")
# #     st.caption(f"💬 Messages: {len(st.session_state.messages)}")


# # ============== MAIN CONTENT ==============

# # Header
# st.title("Travel Planner")
# # st.caption("Plan your perfect trip with AI assistance")

# # Load history on first run
# if not st.session_state.history_loaded:
#     with st.spinner("Loading conversation..."):
#         count = restore_chat_history(st.session_state.session_id)
#     st.session_state.history_loaded = True

# # Display chat messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Chat input
# if prompt := st.chat_input("Where would you like to go? "):
#     # Display user message
#     with st.chat_message("user"):
#         st.markdown(prompt)
    
#     st.session_state.messages.append({"role": "user", "content": prompt})
    
#     # Get bot response
#     try:
#         response = run_agent(prompt, st.session_state.session_id)
        
#         with st.chat_message("assistant"):
#             st.markdown(response)
        
#         st.session_state.messages.append({"role": "assistant", "content": response})
        
#     except Exception as e:
#         st.error(f"Error: {str(e)}")

















# ========================================================================================================
# import streamlit as st
# import os
# import json
# from typing import Dict, Any

# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnableSequence
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_community.chat_message_histories import ChatMessageHistory

# from tools import (
#     load_sheet_data, 
#     find_flights, 
#     find_hotels,
#     save_message_to_sheet,
#     load_chat_history_from_sheet,
#     clear_sheet_history,
#     get_all_sessions
# )

# # Configuration

# OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]


# SHEET_NAME = "AI_Agent_data"

# # Page configuration
# st.set_page_config(
#     page_title="🧳 Travel Planner Agent",
#     page_icon="✈️",
#     layout="wide"
# )

# # Initialize LLM
# @st.cache_resource
# def get_llm():
#     return ChatOpenAI(
#         openai_api_key=OPENAI_API_KEY,
#         model="gpt-4o-mini",
#         temperature=0.3
#     )

# llm = get_llm()

# # Initialize session state
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = ChatMessageHistory()

# if "session_id" not in st.session_state:
#     st.session_state.session_id = "default"

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# if "history_loaded" not in st.session_state:
#     st.session_state.history_loaded = False


# # ============== HELPER FUNCTIONS ==============

# def format_chat_history() -> str:
#     """Format chat history as a readable string"""
#     messages = st.session_state.chat_history.messages
#     if not messages:
#         return "No previous conversation."

#     formatted = []
#     for msg in messages:
#         if isinstance(msg, HumanMessage):
#             formatted.append(f"User: {msg.content}")
#         elif isinstance(msg, AIMessage):
#             formatted.append(f"Assistant: {msg.content}")

#     return "\n".join(formatted)


# def restore_chat_history(session_id: str = "default"):
#     """Restore chat history from Google Sheets into memory"""
#     messages = load_chat_history_from_sheet(SHEET_NAME, session_id)
    
#     st.session_state.chat_history.clear()
#     st.session_state.messages = []
    
#     for msg in messages:
#         role = msg.get("Role", "")
#         content = msg.get("Content", "")
        
#         if role == "user":
#             st.session_state.chat_history.add_user_message(content)
#             st.session_state.messages.append({"role": "user", "content": content})
#         elif role == "assistant":
#             st.session_state.chat_history.add_ai_message(content)
#             st.session_state.messages.append({"role": "assistant", "content": content})
    
#     return len(messages)


# def clear_history(session_id: str = "default"):
#     """Clear chat history from both memory and Google Sheets"""
#     st.session_state.chat_history.clear()
#     st.session_state.messages = []
#     clear_sheet_history(SHEET_NAME, session_id)


# # ============== LLM CHAINS ==============

# # Step 1 — Extract trip details
# extract_prompt = ChatPromptTemplate.from_template("""
# You are an assistant that extracts structured travel details from a user's request.

# Conversation so far:
# {chat_history}

# User Request:
# {user_text}

# Return a valid JSON with keys:
# origin_city, destination_city, start_date, end_date, trip_length_days, budget_usd, interests.

# If anything is missing, set it to null.
# """)

# extract_chain = RunnableSequence(extract_prompt | llm | StrOutputParser())


# # Step 2 — Summarize
# summary_prompt = ChatPromptTemplate.from_template("""
# You are a helpful travel planner. Use the chat history and structured data below
# to write a friendly, detailed itinerary.

# Chat History:
# {chat_history}

# Structured Data:
# {final_state}

# Include:
# - Flights summary
# - Hotel recommendation
# - Day-by-day plan (3–5 days)
# - Budget breakdown

# If there are no flight or hotel options available, suggest alternatives and explain the situation.
# """)

# summary_chain = RunnableSequence(summary_prompt | llm | StrOutputParser())


# # ============== TOOL EXECUTION ==============

# def simulate_tool_calls(structured_json_str: str) -> Dict[str, Any]:
#     """Simulate tool calls to fetch flight and hotel data"""
#     try:
#         structured_data = json.loads(structured_json_str)
#     except json.JSONDecodeError:
#         structured_data = {}

#     origin = structured_data.get("origin_city") or ""
#     destination = structured_data.get("destination_city") or ""
#     start_date = structured_data.get("start_date")
#     budget = structured_data.get("budget_usd")
#     nights = structured_data.get("trip_length_days") or 3
#     budget_per_night = None

#     if budget:
#         try:
#             budget_per_night = (float(budget) * 0.4) / nights
#         except Exception:
#             pass

#     try:
#         flights_df = load_sheet_data(SHEET_NAME, "flights_data")
#         hotels_df = load_sheet_data(SHEET_NAME, "hotels_data")

#         flights = find_flights(flights_df, origin, destination, prefer_date=start_date, budget_usd=budget)
#         hotels = find_hotels(hotels_df, destination, budget_per_night=budget_per_night)
#     except Exception as e:
#         st.error(f"Error loading data: {e}")
#         flights = []
#         hotels = []

#     structured_data["tool_results"] = {
#         "flight_options": flights,
#         "hotel_options": hotels
#     }

#     return {"final_state": json.dumps(structured_data, indent=2)}


# # ============== MAIN AGENT ==============

# def run_agent(user_text: str, session_id: str = "default") -> str:
#     """Main agent workflow with memory"""

#     # Add user message to chat history (memory)
#     st.session_state.chat_history.add_user_message(user_text)
    
#     # Save user message to Google Sheets
#     save_message_to_sheet(SHEET_NAME, "user", user_text, session_id)

#     # Format chat history for the prompt
#     formatted_history = format_chat_history()

#     # Extract structured data
#     with st.spinner("Extracting travel details..."):
#         structured_data = extract_chain.invoke({
#             "user_text": user_text,
#             "chat_history": formatted_history
#         })

#     # Simulate tool calls
#     with st.spinner(" Finding flights and hotels..."):
#         tool_output = simulate_tool_calls(structured_data)

#     # Generate final summary
#     with st.spinner(" Creating your plan"):
#         final_output = summary_chain.invoke({
#             "final_state": tool_output["final_state"],
#             "chat_history": formatted_history
#         })

#     # Save assistant reply to chat history (memory)
#     st.session_state.chat_history.add_ai_message(final_output)
    
#     # Save assistant message to Google Sheets
#     save_message_to_sheet(SHEET_NAME, "assistant", final_output, session_id)

#     return final_output


# # ============== STREAMLIT UI ==============

# # Sidebar
# # with st.sidebar:
#     # st.title("🧳 Travel Planner")
#     # st.markdown("---")
    
#     # Session Management
#     # st.subheader("📝 Session Management")
    
#     # # Get all available sessions
#     # all_sessions = get_all_sessions(SHEET_NAME)
#     # if not all_sessions:
#     #     all_sessions = ["default"]
    
#     # # Session selector
#     # selected_session = st.selectbox(
#     #     "Select Session",
#     #     options=all_sessions,
#     #     index=all_sessions.index(st.session_state.session_id) if st.session_state.session_id in all_sessions else 0
#     # )
    
#     # # New session input
#     # new_session = st.text_input("Or create new session:", placeholder="e.g., paris-trip-2025")
    
#     # if st.button("Switch/Create Session"):
#     #     if new_session:
#     #         st.session_state.session_id = new_session
#     #         st.session_state.history_loaded = False
#     #         st.rerun()
#     #     elif selected_session != st.session_state.session_id:
#     #         st.session_state.session_id = selected_session
#     #         st.session_state.history_loaded = False
#     #         st.rerun()
    
#     # st.markdown("---")
    
#     # Action buttons
#     # st.subheader("Actions")
    
#     # col1, col2 = st.columns(2)
    
#     # with col1:
#     #     if st.button(" Restore History"):
#     #         count = restore_chat_history(st.session_state.session_id)
#     #         st.success(f"Restored {count} messages")
#     #         st.rerun()
    
#     # with col2:
#     #     if st.button("Clear History"):
#     #         clear_history(st.session_state.session_id)
#     #         st.success("History cleared!")
#     #         st.rerun()
    
#     # st.markdown("---")
    
#     # # Current session info
#     # st.info(f"**Current Session:** {st.session_state.session_id}")
#     # st.caption(f"**Total Messages:** {len(st.session_state.messages)}")

# # Main content
# st.title(" Travel Planner Agent")
# # st.markdown("Plan your perfect trip with AI assistance!")

# # Load history on first run
# if not st.session_state.history_loaded:
#     with st.spinner("Loading conversation history..."):
#         count = restore_chat_history(st.session_state.session_id)
#         if count > 0:
#             st.success(f"Loaded {count} previous messages")
#     st.session_state.history_loaded = True

# # Display chat messages
# chat_container = st.container()

# with chat_container:
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

# # Chat input
# if prompt := st.chat_input("Where would you like to go? "): #(e.g., 'Plan a 5-day trip to Paris from NYC for $2000')
#     # Display user message
#     with st.chat_message("user"):
#         st.markdown(prompt)
    
#     # Add to messages
#     st.session_state.messages.append({"role": "user", "content": prompt})
    
#     # Get bot response
#     try:
#         response = run_agent(prompt, st.session_state.session_id)
        
#         # Display assistant message
#         with st.chat_message("assistant"):
#             st.markdown(response)
        
#         # Add to messages
#         st.session_state.messages.append({"role": "assistant", "content": response})
        
#     except Exception as e:
#         st.error(f"Error: {str(e)}")

# # Footer
# st.markdown("---")
# # st.caption("💡 Tip: Start a new session for each trip to keep conversations organized!")



# =========================================================================================================



























# import streamlit as st
# import os
# import json
# from typing import Dict, Any

# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnableSequence
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_community.chat_message_histories import ChatMessageHistory

# from tools import (
#     load_sheet_data, 
#     find_flights, 
#     find_hotels,
#     save_message_to_sheet,
#     load_chat_history_from_sheet,
#     clear_sheet_history,
#     get_all_sessions
# )

# # Configuration
# OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
# SHEET_NAME = "AI_Agent_data"

# # Page configuration
# st.set_page_config(
#     page_title="Travel Planner Agent",
#     page_icon="✈️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for ChatGPT-style sidebar
# st.markdown("""
# <style>
#     /* Sidebar styling */
#     [data-testid="stSidebar"] {
#         background-color: #202123;
#     }
    
#     [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
#         color: #ececf1;
#     }
    
#     /* Button styling */
#     .stButton button {
#         width: 100%;
#         border-radius: 8px;
#         border: 1px solid rgba(255,255,255,0.1);
#         background-color: transparent;
#         color: white;
#         padding: 10px;
#         transition: all 0.2s;
#     }
    
#     .stButton button:hover {
#         background-color: rgba(255,255,255,0.1);
#     }
    
#     /* Session item styling */
#     .session-item {
#         padding: 12px;
#         margin: 4px 0;
#         border-radius: 8px;
#         cursor: pointer;
#         background-color: rgba(255,255,255,0.05);
#         color: #ececf1;
#         border: 1px solid transparent;
#         transition: all 0.2s;
#     }
    
#     .session-item:hover {
#         background-color: rgba(255,255,255,0.1);
#         border: 1px solid rgba(255,255,255,0.2);
#     }
    
#     .session-item-active {
#         background-color: rgba(255,255,255,0.15);
#         border: 1px solid rgba(255,255,255,0.3);
#     }
    
#     /* Hide Streamlit branding */
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
    
#     /* Chat input styling */
#     .stChatInput {
#         border-radius: 12px;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Initialize LLM
# @st.cache_resource
# def get_llm():
#     return ChatOpenAI(
#         openai_api_key=OPENAI_API_KEY,
#         model="gpt-4o-mini",
#         temperature=0.3
#     )

# llm = get_llm()

# # Initialize session state
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = ChatMessageHistory()

# if "session_id" not in st.session_state:
#     st.session_state.session_id = "default"

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# if "history_loaded" not in st.session_state:
#     st.session_state.history_loaded = False

# if "all_sessions" not in st.session_state:
#     st.session_state.all_sessions = ["default"]


# # ============== HELPER FUNCTIONS ==============

# def format_chat_history() -> str:
#     """Format chat history as a readable string"""
#     messages = st.session_state.chat_history.messages
#     if not messages:
#         return "No previous conversation."

#     formatted = []
#     for msg in messages:
#         if isinstance(msg, HumanMessage):
#             formatted.append(f"User: {msg.content}")
#         elif isinstance(msg, AIMessage):
#             formatted.append(f"Assistant: {msg.content}")

#     return "\n".join(formatted)


# def restore_chat_history(session_id: str = "default"):
#     """Restore chat history from Google Sheets into memory"""
#     messages = load_chat_history_from_sheet(SHEET_NAME, session_id)
    
#     st.session_state.chat_history.clear()
#     st.session_state.messages = []
    
#     for msg in messages:
#         role = msg.get("Role", "")
#         content = msg.get("Content", "")
        
#         if role == "user":
#             st.session_state.chat_history.add_user_message(content)
#             st.session_state.messages.append({"role": "user", "content": content})
#         elif role == "assistant":
#             st.session_state.chat_history.add_ai_message(content)
#             st.session_state.messages.append({"role": "assistant", "content": content})
    
#     return len(messages)


# def clear_history(session_id: str = "default"):
#     """Clear chat history from both memory and Google Sheets"""
#     st.session_state.chat_history.clear()
#     st.session_state.messages = []
#     clear_sheet_history(SHEET_NAME, session_id)


# def load_all_sessions():
#     """Load all available sessions"""
#     sessions = get_all_sessions(SHEET_NAME)
#     if not sessions:
#         sessions = ["default"]
#     st.session_state.all_sessions = sessions
#     return sessions


# def switch_session(session_id: str):
#     """Switch to a different session"""
#     st.session_state.session_id = session_id
#     st.session_state.history_loaded = False
#     restore_chat_history(session_id)


# def create_new_session():
#     """Create a new session"""
#     import time
#     new_session_id = f"trip_{int(time.time())}"
#     st.session_state.session_id = new_session_id
#     st.session_state.chat_history.clear()
#     st.session_state.messages = []
#     st.session_state.history_loaded = True
#     if new_session_id not in st.session_state.all_sessions:
#         st.session_state.all_sessions.append(new_session_id)


# # ============== LLM CHAINS ==============

# extract_prompt = ChatPromptTemplate.from_template("""
# You are an assistant that extracts structured travel details from a user's request.

# Conversation so far:
# {chat_history}

# User Request:
# {user_text}

# Return a valid JSON with keys:
# origin_city, destination_city, start_date, end_date, trip_length_days, budget_usd, interests.

# If anything is missing, set it to null.
# """)

# extract_chain = RunnableSequence(extract_prompt | llm | StrOutputParser())

# summary_prompt = ChatPromptTemplate.from_template("""
# You are a helpful travel planner. Use the chat history and structured data below
# to write a friendly, detailed itinerary.

# Chat History:
# {chat_history}

# Structured Data:
# {final_state}

# Include:
# - Flights summary
# - Hotel recommendation
# - Day-by-day plan (3–5 days)
# - Budget breakdown

# If there are no flight or hotel options available, suggest alternatives and explain the situation.
# """)

# summary_chain = RunnableSequence(summary_prompt | llm | StrOutputParser())


# # ============== TOOL EXECUTION ==============

# def simulate_tool_calls(structured_json_str: str) -> Dict[str, Any]:
#     """Simulate tool calls to fetch flight and hotel data"""
#     try:
#         structured_data = json.loads(structured_json_str)
#     except json.JSONDecodeError:
#         structured_data = {}

#     origin = structured_data.get("origin_city") or ""
#     destination = structured_data.get("destination_city") or ""
#     start_date = structured_data.get("start_date")
#     budget = structured_data.get("budget_usd")
#     nights = structured_data.get("trip_length_days") or 3
#     budget_per_night = None

#     if budget:
#         try:
#             budget_per_night = (float(budget) * 0.4) / nights
#         except Exception:
#             pass

#     try:
#         flights_df = load_sheet_data(SHEET_NAME, "flights_data")
#         hotels_df = load_sheet_data(SHEET_NAME, "hotels_data")

#         flights = find_flights(flights_df, origin, destination, prefer_date=start_date, budget_usd=budget)
#         hotels = find_hotels(hotels_df, destination, budget_per_night=budget_per_night)
#     except Exception as e:
#         st.error(f"Error loading data: {e}")
#         flights = []
#         hotels = []

#     structured_data["tool_results"] = {
#         "flight_options": flights,
#         "hotel_options": hotels
#     }

#     return {"final_state": json.dumps(structured_data, indent=2)}


# # ============== MAIN AGENT ==============

# def run_agent(user_text: str, session_id: str = "default") -> str:
#     """Main agent workflow with memory"""

#     st.session_state.chat_history.add_user_message(user_text)
#     save_message_to_sheet(SHEET_NAME, "user", user_text, session_id)

#     formatted_history = format_chat_history()

#     with st.spinner("🔍 Extracting travel details..."):
#         structured_data = extract_chain.invoke({
#             "user_text": user_text,
#             "chat_history": formatted_history
#         })

#     with st.spinner("✈️ Finding flights and hotels..."):
#         tool_output = simulate_tool_calls(structured_data)

#     with st.spinner("📝 Creating your itinerary..."):
#         final_output = summary_chain.invoke({
#             "final_state": tool_output["final_state"],
#             "chat_history": formatted_history
#         })

#     st.session_state.chat_history.add_ai_message(final_output)
#     save_message_to_sheet(SHEET_NAME, "assistant", final_output, session_id)

#     return final_output


# # ============== SIDEBAR (ChatGPT Style) ==============

# with st.sidebar:
#     # New Chat Button (like ChatGPT)
#     st.markdown("### ")
#     if st.button("➕ New Trip", use_container_width=True):
#         create_new_session()
#         st.rerun()
    
#     st.markdown("---")
    
#     # Chat History Section
#     st.markdown("### 📚 Your Trips")
    
#     # Load all sessions
#     all_sessions = load_all_sessions()
    
#     # Display sessions as clickable items
#     for session in all_sessions:
#         # Get first message preview if available
#         session_messages = load_chat_history_from_sheet(SHEET_NAME, session)
#         preview = "New conversation"
        
#         if session_messages:
#             first_msg = session_messages[0].get("Content", "")
#             preview = first_msg[:40] + "..." if len(first_msg) > 40 else first_msg
        
#         # Determine if this is the active session
#         is_active = session == st.session_state.session_id
        
#         # Create button for each session
#         col1, col2 = st.columns([5, 1])
        
#         with col1:
#             if st.button(
#                 f"💬 {preview}", 
#                 key=f"session_{session}",
#                 use_container_width=True,
#                 type="primary" if is_active else "secondary"
#             ):
#                 switch_session(session)
#                 st.rerun()
        
#         with col2:
#             if st.button("🗑️", key=f"delete_{session}", help="Delete this trip"):
#                 clear_sheet_history(SHEET_NAME, session)
#                 if session in st.session_state.all_sessions:
#                     st.session_state.all_sessions.remove(session)
#                 if session == st.session_state.session_id:
#                     create_new_session()
#                 st.rerun()
    
#     st.markdown("---")
    
#     # Settings Section
#     st.markdown("### ⚙️ Settings")
    
#     if st.button("🔄 Refresh History", use_container_width=True):
#         count = restore_chat_history(st.session_state.session_id)
#         st.success(f"✅ Loaded {count} messages")
#         st.rerun()
    
#     if st.button("🗑️ Clear Current Trip", use_container_width=True):
#         clear_history(st.session_state.session_id)
#         st.success("✅ Cleared!")
#         st.rerun()
    
#     # Footer info
#     st.markdown("---")
#     st.caption(f"📝 Current: {st.session_state.session_id}")
#     st.caption(f"💬 Messages: {len(st.session_state.messages)}")


# # ============== MAIN CONTENT ==============

# # Header
# st.title("✈️ AI Travel Planner")
# st.caption("Plan your perfect trip with AI assistance")

# # Load history on first run
# if not st.session_state.history_loaded:
#     with st.spinner("Loading conversation..."):
#         count = restore_chat_history(st.session_state.session_id)
#     st.session_state.history_loaded = True

# # Display chat messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Chat input
# if prompt := st.chat_input("Where would you like to go? (e.g., 'Plan a 5-day trip to Paris from NYC')"):
#     # Display user message
#     with st.chat_message("user"):
#         st.markdown(prompt)
    
#     st.session_state.messages.append({"role": "user", "content": prompt})
    
#     # Get bot response
#     try:
#         response = run_agent(prompt, st.session_state.session_id)
        
#         with st.chat_message("assistant"):
#             st.markdown(response)
        
#         st.session_state.messages.append({"role": "assistant", "content": response})
        
#     except Exception as e:
#         st.error(f"❌ Error: {str(e)}")









