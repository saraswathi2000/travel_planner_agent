import gspread
import pandas as pd
from google.oauth2.service_account import Credentials
from typing import Dict, List, Any
from datetime import datetime
import pytz
import streamlit as st
import os

# Setting up Google Sheets
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# Use Streamlit secrets for credentials
try:
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=SCOPES
    )
    print("✅ Loaded credentials from Streamlit secrets")
except Exception as e:
    print(f"❌ Error loading credentials: {e}")
    print("Make sure 'gcp_service_account' is configured in Streamlit secrets!")
    raise

gc = gspread.authorize(creds)

# Constants
CHAT_HISTORY_TAB = "chat_history"


# ============== DATA LOADING FUNCTIONS ==============

def load_sheet_data(sheet_name: str, worksheet_name: str) -> pd.DataFrame:
    """Load data from a Google Sheet worksheet into a pandas DataFrame"""
    print(f"\n{'='*60}")
    print(f"📂 LOADING SHEET DATA")
    print(f"  Sheet: '{sheet_name}'")
    print(f"  Worksheet: '{worksheet_name}'")
    
    try:
        sh = gc.open(sheet_name)
        print(f"  ✅ Opened spreadsheet: {sh.title}")
        
        # List all worksheets
        all_worksheets = [ws.title for ws in sh.worksheets()]
        print(f"  📋 Available worksheets: {all_worksheets}")
        
        ws = sh.worksheet(worksheet_name)
        print(f"  ✅ Found worksheet: {ws.title}")
        
        data = ws.get_all_records()
        print(f"  📊 Loaded {len(data)} rows")
        
        df = pd.DataFrame(data)
        print(f"  📊 DataFrame shape: {df.shape}")
        print(f"  📊 Columns: {df.columns.tolist()}")
        
        # Show first few rows
        if len(df) > 0:
            print(f"\n  📄 First 3 rows:")
            for idx, row in df.head(3).iterrows():
                print(f"    Row {idx}: {dict(row)}")
        else:
            print(f"  ⚠️ WARNING: DataFrame is EMPTY!")
        
        print(f"{'='*60}\n")
        return df
        
    except Exception as e:
        print(f"  ❌ ERROR loading sheet: {e}")
        print(f"{'='*60}\n")
        raise


def find_flights(
    df: pd.DataFrame, 
    origin: str, 
    destination: str, 
    prefer_date: str = None, 
    budget_usd: float = None, 
    max_results: int = 3
) -> List[Dict[str, Any]]:
    """Find flights matching the criteria"""
    # Clean and strip input
    origin = str(origin).strip() if origin else ""
    destination = str(destination).strip() if destination else ""
    
    print(f"\n{'='*60}")
    print(f"SEARCH REQUEST:")
    print(f"  Origin: '{origin}'")
    print(f"  Destination: '{destination}'")
    print(f"  Budget: {budget_usd}")
    print(f"{'='*60}")
    
    if not origin or not destination:
        print("❌ Empty origin or destination")
        return []
    
    # Check DataFrame
    print(f"DataFrame info:")
    print(f"  Total rows: {len(df)}")
    print(f"  Columns: {df.columns.tolist()}")
    
    if 'origin' not in df.columns or 'destination' not in df.columns:
        print(f"❌ Missing columns! Available: {df.columns.tolist()}")
        return []
    
    # Show ALL data in the sheet
    print(f"\n📊 ALL DATA IN FLIGHTS SHEET:")
    for idx, row in df.head(10).iterrows():
        print(f"  Row {idx}: '{row.get('origin', 'N/A')}' → '{row.get('destination', 'N/A')}' (${row.get('price_in_dollars', 'N/A')})")
    
    # Try exact matching first
    print(f"\n🔍 Searching with case-insensitive contains...")
    
    # Check each condition separately
    origin_matches = df['origin'].astype(str).str.strip().str.lower().str.contains(origin.lower(), na=False, regex=False)
    dest_matches = df['destination'].astype(str).str.strip().str.lower().str.contains(destination.lower(), na=False, regex=False)
    
    print(f"  Origins matching '{origin}': {origin_matches.sum()}")
    print(f"  Destinations matching '{destination}': {dest_matches.sum()}")
    
    # Combined filter
    q = df[origin_matches & dest_matches]
    
    print(f"  ✅ Combined matches: {len(q)}")
    
    if len(q) > 0:
        print(f"\n✅ FOUND {len(q)} FLIGHTS:")
        for idx, row in q.iterrows():
            print(f"  - {row.get('airline', 'N/A')}: {row.get('origin', 'N/A')} → {row.get('destination', 'N/A')} (${row.get('price_in_dollars', 'N/A')})")
    else:
        print(f"\n❌ NO MATCHES FOUND")
        print(f"Tried searching:")
        print(f"  - Origin contains: '{origin}' (case-insensitive)")
        print(f"  - Destination contains: '{destination}' (case-insensitive)")
    
    if budget_usd and 'price_in_dollars' in df.columns and len(q) > 0:
        before_budget = len(q)
        q = q[q['price_in_dollars'] <= float(budget_usd)]
        print(f"  After budget filter: {len(q)} (removed {before_budget - len(q)})")
    
    if 'price_in_dollars' in df.columns and len(q) > 0:
        q = q.sort_values('price_in_dollars')
    
    results = q.head(max_results).to_dict(orient='records')
    print(f"\n📤 Returning {len(results)} results")
    print(f"{'='*60}\n")
    
    return results


def find_hotels(
    df: pd.DataFrame, 
    city: str, 
    budget_per_night: float = None, 
    max_results: int = 3
) -> List[Dict[str, Any]]:
    """Find hotels matching the criteria"""
    # Clean and strip input
    city = str(city).strip() if city else ""
    
    print(f"DEBUG find_hotels: Looking for city '{city}'")
    print(f"DEBUG find_hotels: DataFrame shape: {df.shape}")
    print(f"DEBUG find_hotels: Columns: {df.columns.tolist()}")
    
    if not city:
        print("DEBUG find_hotels: Empty city")
        return []
    
    # Check if required column exists
    if 'city' not in df.columns:
        print(f"DEBUG find_hotels: Missing 'city' column. Available: {df.columns.tolist()}")
        return []
    
    # Print sample data
    if len(df) > 0:
        print(f"DEBUG find_hotels: Sample cities: {df['city'].head(5).tolist()}")
    
    # Filter by city with case-insensitive search
    q = df[df['city'].astype(str).str.strip().str.contains(city, case=False, na=False, regex=False)]
    
    print(f"DEBUG find_hotels: Found {len(q)} hotels after filtering")
    
    if budget_per_night:
        if 'price_per_night_in_dollars' in df.columns:
            q = q[q['price_per_night_in_dollars'] <= float(budget_per_night)]
            print(f"DEBUG find_hotels: {len(q)} hotels within budget")
    
    if 'price_per_night_in_dollars' in df.columns and len(q) > 0:
        q = q.sort_values('price_per_night_in_dollars')
    
    results = q.head(max_results).to_dict(orient='records')
    print(f"DEBUG find_hotels: Returning {len(results)} results")
    
    return results


# ============== CHAT HISTORY MANAGEMENT ==============

def save_message_to_sheet(
    sheet_name: str,
    role: str, 
    content: str, 
    session_id: str = "default"
) -> bool:
    """
    Save a single message to Google Sheets with Indian Standard Time
    
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
                "Timestamp (IST)",
                "Session ID",
                "Role",
                "Content",
                "Message ID"
            ])
        
        # Get Indian Standard Time
        ist = pytz.timezone('Asia/Kolkata')
        timestamp = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
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




# import gspread
# import pandas as pd
# from google.oauth2.service_account import Credentials
# from typing import Dict, List, Any
# from datetime import datetime
# import pytz
# import streamlit as st
# import os

# # Setting up Google Sheets
# SCOPES = [
#     "https://www.googleapis.com/auth/spreadsheets",
#     "https://www.googleapis.com/auth/drive"
# ]

# # Check if running on Streamlit Cloud or locally
# if "gcp_service_account" in st.secrets:
#     # Running on Streamlit Cloud - use secrets
#     creds = Credentials.from_service_account_info(
#         st.secrets["gcp_service_account"],
#         scopes=SCOPES
#     )
# else:
#     # Running locally - use JSON file
#     SERVICE_ACCOUNT_FILE = r"C:\Users\saras\Desktop\AIAgent\candidate-database-421805-c2e928dc9e6e.json"
#     creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)

# gc = gspread.authorize(creds)

# # Constants
# CHAT_HISTORY_TAB = "chat_history"


# # ============== DATA LOADING FUNCTIONS ==============

# def load_sheet_data(sheet_name: str, worksheet_name: str) -> pd.DataFrame:
#     """Load data from a Google Sheet worksheet into a pandas DataFrame"""
#     sh = gc.open(sheet_name)
#     ws = sh.worksheet(worksheet_name)
#     data = ws.get_all_records()
#     df = pd.DataFrame(data)
#     return df


# def find_flights(
#     df: pd.DataFrame, 
#     origin: str, 
#     destination: str, 
#     prefer_date: str = None, 
#     budget_usd: float = None, 
#     max_results: int = 3
# ) -> List[Dict[str, Any]]:
#     """Find flights matching the criteria"""
#     # Clean and strip input
#     origin = str(origin).strip() if origin else ""
#     destination = str(destination).strip() if destination else ""
    
#     print(f"DEBUG find_flights: Looking for '{origin}' -> '{destination}'")
#     print(f"DEBUG find_flights: DataFrame shape: {df.shape}")
#     print(f"DEBUG find_flights: Columns: {df.columns.tolist()}")
    
#     if not origin or not destination:
#         print("DEBUG find_flights: Empty origin or destination")
#         return []
    
#     # Check if required columns exist
#     if 'origin' not in df.columns or 'destination' not in df.columns:
#         print(f"DEBUG find_flights: Missing required columns. Available: {df.columns.tolist()}")
#         return []
    
#     # Print sample data
#     if len(df) > 0:
#         print(f"DEBUG find_flights: Sample origins: {df['origin'].head(3).tolist()}")
#         print(f"DEBUG find_flights: Sample destinations: {df['destination'].head(3).tolist()}")
    
#     # Filter by origin and destination with case-insensitive search
#     q = df[
#         (df['origin'].astype(str).str.strip().str.contains(origin, case=False, na=False, regex=False)) & 
#         (df['destination'].astype(str).str.strip().str.contains(destination, case=False, na=False, regex=False))
#     ]
    
#     print(f"DEBUG find_flights: Found {len(q)} flights after filtering")
    
#     if budget_usd:
#         if 'price_in_dollars' in df.columns:
#             q = q[q['price_in_dollars'] <= float(budget_usd)]
#             print(f"DEBUG find_flights: {len(q)} flights within budget")
    
#     if 'price_in_dollars' in df.columns and len(q) > 0:
#         q = q.sort_values('price_in_dollars')
    
#     results = q.head(max_results).to_dict(orient='records')
#     print(f"DEBUG find_flights: Returning {len(results)} results")
    
#     return results


# def find_hotels(
#     df: pd.DataFrame, 
#     city: str, 
#     budget_per_night: float = None, 
#     max_results: int = 3
# ) -> List[Dict[str, Any]]:
#     """Find hotels matching the criteria"""
#     # Clean and strip input
#     city = str(city).strip() if city else ""
    
#     print(f"DEBUG find_hotels: Looking for city '{city}'")
#     print(f"DEBUG find_hotels: DataFrame shape: {df.shape}")
#     print(f"DEBUG find_hotels: Columns: {df.columns.tolist()}")
    
#     if not city:
#         print("DEBUG find_hotels: Empty city")
#         return []
    
#     # Check if required column exists
#     if 'city' not in df.columns:
#         print(f"DEBUG find_hotels: Missing 'city' column. Available: {df.columns.tolist()}")
#         return []
    
#     # Print sample data
#     if len(df) > 0:
#         print(f"DEBUG find_hotels: Sample cities: {df['city'].head(5).tolist()}")
    
#     # Filter by city with case-insensitive search
#     q = df[df['city'].astype(str).str.strip().str.contains(city, case=False, na=False, regex=False)]
    
#     print(f"DEBUG find_hotels: Found {len(q)} hotels after filtering")
    
#     if budget_per_night:
#         if 'price_per_night_in_dollars' in df.columns:
#             q = q[q['price_per_night_in_dollars'] <= float(budget_per_night)]
#             print(f"DEBUG find_hotels: {len(q)} hotels within budget")
    
#     if 'price_per_night_in_dollars' in df.columns and len(q) > 0:
#         q = q.sort_values('price_per_night_in_dollars')
    
#     results = q.head(max_results).to_dict(orient='records')
#     print(f"DEBUG find_hotels: Returning {len(results)} results")
    
#     return results




# def save_message_to_sheet(
#     sheet_name: str,
#     role: str, 
#     content: str, 
#     session_id: str = "default"
# ) -> bool:
#     """
#     Save a single message to Google Sheets with Indian Standard Time
    
#     Args:
#         sheet_name: Name of the Google Sheet
#         role: Either 'user' or 'assistant'
#         content: The message content
#         session_id: Identifier for the conversation session
    
#     Returns:
#         bool: True if successful, False otherwise
#     """
#     try:
#         spreadsheet = gc.open(sheet_name)
        
#         # Try to get existing worksheet, create if doesn't exist
#         try:
#             worksheet = spreadsheet.worksheet(CHAT_HISTORY_TAB)
#         except gspread.exceptions.WorksheetNotFound:
#             worksheet = spreadsheet.add_worksheet(
#                 title=CHAT_HISTORY_TAB,
#                 rows=1000,
#                 cols=5
#             )
#             # Add headers
#             worksheet.append_row([
#                 "Timestamp (IST)",
#                 "Session ID",
#                 "Role",
#                 "Content",
#                 "Message ID"
#             ])
        
#         # Get Indian Standard Time
#         ist = pytz.timezone('Asia/Kolkata')
#         timestamp = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
#         message_id = f"{session_id}_{timestamp.replace(' ', '_').replace(':', '-')}"
        
#         worksheet.append_row([
#             timestamp,
#             session_id,
#             role,
#             content,
#             message_id
#         ])
        
#         return True
        
#     except Exception as e:
#         print(f" Error saving to Google Sheets: {e}")
#         return False


# def load_chat_history_from_sheet(
#     sheet_name: str,
#     session_id: str = "default"
# ) -> List[Dict[str, str]]:
#     """
#     Load chat history from Google Sheets for a specific session
    
#     Args:
#         sheet_name: Name of the Google Sheet
#         session_id: Identifier for the conversation session
    
#     Returns:
#         List of message dictionaries with 'Role' and 'Content' keys
#     """
#     try:
#         spreadsheet = gc.open(sheet_name)
        
#         try:
#             worksheet = spreadsheet.worksheet(CHAT_HISTORY_TAB)
#         except gspread.exceptions.WorksheetNotFound:
#             return []
        
#         # Get all records
#         records = worksheet.get_all_records()
        
#         # Filter by session_id and sort by timestamp
#         session_messages = [
#             msg for msg in records 
#             if msg.get("Session ID") == session_id
#         ]
        
#         # Sort by timestamp
#         session_messages.sort(key=lambda x: x.get("Timestamp", ""))
        
#         return session_messages
        
#     except Exception as e:
#         print(f"⚠️ Error loading from Google Sheets: {e}")
#         return []


# def clear_sheet_history(
#     sheet_name: str,
#     session_id: str = "default"
# ) -> bool:
#     """
#     Clear chat history from Google Sheets for a specific session
    
#     Args:
#         sheet_name: Name of the Google Sheet
#         session_id: Identifier for the conversation session
    
#     Returns:
#         bool: True if successful, False otherwise
#     """
#     try:
#         spreadsheet = gc.open(sheet_name)
#         worksheet = spreadsheet.worksheet(CHAT_HISTORY_TAB)
        
#         # Get all records
#         records = worksheet.get_all_records()
        
#         # Find rows to delete (in reverse order to maintain indices)
#         rows_to_delete = []
#         for i, record in enumerate(records, start=2):  # start=2 because row 1 is header
#             if record.get("Session ID") == session_id:
#                 rows_to_delete.append(i)
        
#         # Delete rows in reverse order
#         for row_num in reversed(rows_to_delete):
#             worksheet.delete_rows(row_num)
        
#         return True
        
#     except Exception as e:
#         print(f"⚠️ Error clearing Google Sheets: {e}")
#         return False


# def get_all_sessions(sheet_name: str) -> List[str]:

#     try:
#         spreadsheet = gc.open(sheet_name)
        
#         try:
#             worksheet = spreadsheet.worksheet(CHAT_HISTORY_TAB)
#         except gspread.exceptions.WorksheetNotFound:
#             return []
        
#         # Get all records
#         records = worksheet.get_all_records()
        
#         # Extract unique session IDs
#         sessions = list(set(msg.get("Session ID") for msg in records))
#         sessions.sort()
        
#         return sessions
        
#     except Exception as e:
#         print(f"⚠️ Error getting sessions: {e}")
#         return []
