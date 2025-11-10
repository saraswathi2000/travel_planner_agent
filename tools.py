import gspread
import pandas as pd
from google.oauth2.service_account import Credentials
from typing import Dict, List, Any
from datetime import datetime
import streamlit as st
import os

# Setting up Google Sheets
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# Check if running on Streamlit Cloud or locally
if "gcp_service_account" in st.secrets:
    # Running on Streamlit Cloud - use secrets
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=SCOPES
    )
else:
    # Running locally - use JSON file
    SERVICE_ACCOUNT_FILE = r"C:\Users\saras\Desktop\AIAgent\candidate-database-421805-c2e928dc9e6e.json"
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)

gc = gspread.authorize(creds)

# Constants
CHAT_HISTORY_TAB = "chat_history"


# ============== DATA LOADING FUNCTIONS ==============

def load_sheet_data(sheet_name: str, worksheet_name: str) -> pd.DataFrame:
    """Load data from a Google Sheet worksheet into a pandas DataFrame"""
    sh = gc.open(sheet_name)
    ws = sh.worksheet(worksheet_name)
    data = ws.get_all_records()
    df = pd.DataFrame(data)
    return df


def find_flights(
    df: pd.DataFrame, 
    origin: str, 
    destination: str, 
    prefer_date: str = None, 
    budget_usd: float = None, 
    max_results: int = 3
) -> List[Dict[str, Any]]:
    """Find flights matching the criteria"""
    q = df[
        (df['origin'].str.contains(origin, case=False, na=False)) & 
        (df['destination'].str.contains(destination, case=False, na=False))
    ]
    if budget_usd:
        q = q[q['price_in_dollars'] <= float(budget_usd)]
    q = q.sort_values('price_in_dollars')
    return q.head(max_results).to_dict(orient='records')


def find_hotels(
    df: pd.DataFrame, 
    city: str, 
    budget_per_night: float = None, 
    max_results: int = 3
) -> List[Dict[str, Any]]:
    """Find hotels matching the criteria"""
    q = df[df['city'].str.contains(city, case=False, na=False)]
    if budget_per_night:
        q = q[q['price_per_night_in_dollars'] <= float(budget_per_night)]
    q = q.sort_values('price_per_night_in_dollars')
    return q.head(max_results).to_dict(orient='records')


# ============== CHAT HISTORY MANAGEMENT ==============

def save_message_to_sheet(
    sheet_name: str,
    role: str, 
    content: str, 
    session_id: str = "default"
) -> bool:
    """
    Save a single message to Google Sheets
    
    Args:
        sheet_name: Name of the Google Sheet
        role: Either 'user' or 'assistant'
        content: The message content
        session_id: Identifier for the conversation session
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        spreadsheet = gc.open(sheet_name)
        
        # Try to get existing worksheet, create if doesn't exist
        try:
            worksheet = spreadsheet.worksheet(CHAT_HISTORY_TAB)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(
                title=CHAT_HISTORY_TAB,
                rows=1000,
                cols=5
            )
            # Add headers
            worksheet.append_row([
                "Timestamp",
                "Session ID",
                "Role",
                "Content",
                "Message ID"
            ])
        
        # Append new message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message_id = f"{session_id}_{timestamp.replace(' ', '_').replace(':', '-')}"
        
        worksheet.append_row([
            timestamp,
            session_id,
            role,
            content,
            message_id
        ])
        
        return True
        
    except Exception as e:
        print(f"⚠️ Error saving to Google Sheets: {e}")
        return False


def load_chat_history_from_sheet(
    sheet_name: str,
    session_id: str = "default"
) -> List[Dict[str, str]]:
    """
    Load chat history from Google Sheets for a specific session
    
    Args:
        sheet_name: Name of the Google Sheet
        session_id: Identifier for the conversation session
    
    Returns:
        List of message dictionaries with 'Role' and 'Content' keys
    """
    try:
        spreadsheet = gc.open(sheet_name)
        
        try:
            worksheet = spreadsheet.worksheet(CHAT_HISTORY_TAB)
        except gspread.exceptions.WorksheetNotFound:
            return []
        
        # Get all records
        records = worksheet.get_all_records()
        
        # Filter by session_id and sort by timestamp
        session_messages = [
            msg for msg in records 
            if msg.get("Session ID") == session_id
        ]
        
        # Sort by timestamp
        session_messages.sort(key=lambda x: x.get("Timestamp", ""))
        
        return session_messages
        
    except Exception as e:
        print(f"⚠️ Error loading from Google Sheets: {e}")
        return []


def clear_sheet_history(
    sheet_name: str,
    session_id: str = "default"
) -> bool:
    """
    Clear chat history from Google Sheets for a specific session
    
    Args:
        sheet_name: Name of the Google Sheet
        session_id: Identifier for the conversation session
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        spreadsheet = gc.open(sheet_name)
        worksheet = spreadsheet.worksheet(CHAT_HISTORY_TAB)
        
        # Get all records
        records = worksheet.get_all_records()
        
        # Find rows to delete (in reverse order to maintain indices)
        rows_to_delete = []
        for i, record in enumerate(records, start=2):  # start=2 because row 1 is header
            if record.get("Session ID") == session_id:
                rows_to_delete.append(i)
        
        # Delete rows in reverse order
        for row_num in reversed(rows_to_delete):
            worksheet.delete_rows(row_num)
        
        return True
        
    except Exception as e:
        print(f"⚠️ Error clearing Google Sheets: {e}")
        return False


def get_all_sessions(sheet_name: str) -> List[str]:
    """
    Get list of all unique session IDs from the chat history
    
    Args:
        sheet_name: Name of the Google Sheet
    
    Returns:
        List of unique session IDs
    """
    try:
        spreadsheet = gc.open(sheet_name)
        
        try:
            worksheet = spreadsheet.worksheet(CHAT_HISTORY_TAB)
        except gspread.exceptions.WorksheetNotFound:
            return []
        
        # Get all records
        records = worksheet.get_all_records()
        
        # Extract unique session IDs
        sessions = list(set(msg.get("Session ID") for msg in records))
        sessions.sort()
        
        return sessions
        
    except Exception as e:
        print(f"⚠️ Error getting sessions: {e}")
        return []
