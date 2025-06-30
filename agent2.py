# -*- coding: utf-8 -*-
"""Gmail Agent for Processing Business Documents

This module handles downloading and processing of business-related PDF attachments from Gmail,
with enhanced duplicate detection and logging capabilities.
"""

import os
import re
import base64
import pandas as pd
import hashlib
import time
import requests
import grpc.aio
import asyncio
import atexit
from datetime import datetime, timedelta
from typing import Union, List, Dict, Any, Optional, Type
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from langchain.tools import tool, Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent, AgentOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_community.agent_toolkits import GmailToolkit
from langchain_community.tools.gmail.utils import build_resource_service, get_gmail_credentials
from langchain_core.runnables import RunnablePassthrough
from langchain_core.agents import AgentAction, AgentFinish
from langchain.memory import ConversationBufferMemory
from langchain.agents.format_scratchpad import format_log_to_str
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain.tools.base import BaseTool
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from pathlib import Path
import signal

# Global variables for cleanup management
_cleanup_done = False
_event_loop = None

def get_event_loop():
    """Get or create an event loop."""
    global _event_loop
    try:
        if _event_loop is None or _event_loop.is_closed():
            _event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(_event_loop)
        return _event_loop
    except Exception as e:
        print(f"Error getting event loop: {e}")
        return asyncio.new_event_loop()

def cleanup_resources():
    """Clean up resources before exit."""
    global _cleanup_done, _event_loop
    if _cleanup_done:
        return
    
    try:
        if _event_loop is not None and not _event_loop.is_closed():
            # Create a new event loop for cleanup if needed
            cleanup_loop = _event_loop if not _event_loop.is_closed() else asyncio.new_event_loop()
            asyncio.set_event_loop(cleanup_loop)
            
            try:
                # Shutdown gRPC
                grpc.aio.shutdown_grpc_aio()
            except Exception as e:
                print(f"Warning: Error during gRPC shutdown: {e}")
            
            try:
                # Close the event loop
                cleanup_loop.close()
            except Exception as e:
                print(f"Warning: Error closing event loop: {e}")
            
            _cleanup_done = True
    except Exception as e:
        print(f"Warning: Error during cleanup: {e}")

# Register cleanup handlers
atexit.register(cleanup_resources)
signal.signal(signal.SIGINT, lambda s, f: cleanup_resources())

# Initialize event loop
_event_loop = get_event_loop()

# Initialize gRPC for async operations
try:
    grpc.aio.init_grpc_aio()
except Exception as e:
    print(f"Warning: Error initializing gRPC: {e}")

# Load environment variables and configuration
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

def validate_gemini_api_key(api_key):
    """Validate the Gemini API key."""
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash"
    headers = {"x-goog-api-key": api_key}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 401:
            raise ValueError("Invalid or expired API key. Please renew your Gemini API key.")
        elif response.status_code == 429:
            print("Warning: API rate limit reached. The script will implement retry logic.")
        elif response.status_code != 200:
            raise ValueError(f"API key validation failed with status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to validate API key: {str(e)}")

# Validate API key before proceeding
try:
    validate_gemini_api_key(api_key)
except ValueError as e:
    print(f"Error: {str(e)}")
    print("Please update your API key in the .env file and try again.")
    exit(1)

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
# BASE_DIR = os.getcwd()  # gets the current working directory
# SAVE_PATH = os.path.join(BASE_DIR, 'download_email')
# LOG_FILE = os.path.join(BASE_DIR, "email_download_log.xlsx")

# Define base paths using pathlib for cross-platform compatibility
BASE_DIR = Path(__file__).parent
SAVE_PATH = BASE_DIR / "download_email"
LOG_FILE = SAVE_PATH / "email_download_log.xlsx"

# Create necessary directories
SAVE_PATH.mkdir(parents=True, exist_ok=True)

# Initialize Gemini LLM with retry logic
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception)
)
def create_llm():
    """Create LLM instance with retry logic."""
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.7,
        top_p=0.95,
        max_output_tokens=2048,
        model_kwargs={
            "retry_on_failure": True
        }
    )

# Initialize LLM with retry logic
try:
    llm = create_llm()
except Exception as e:
    print(f"Failed to initialize LLM after retries: {str(e)}")
    print("Please check your API key and try again later.")
    exit(1)

# Create email classification prompt template with retry logic
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception)
)
def classify_email(subject, body):
    """Classify email with retry logic."""
    try:
        # Directly use the LLM to get classification
        prompt = email_classification_prompt.format(subject=subject, body=body)
        response = llm.invoke(prompt)
        
        # Extract classification from response
        if hasattr(response, 'content'):
            classification = response.content.strip().upper()
        elif isinstance(response, str):
            classification = response.strip().upper()
        else:
            print(f"Unexpected response type: {type(response)}")
            return "UNRELATED"
        
        # Strict classification validation - only allow ENQUIRY or QUOTATION
        if classification == "ENQUIRY":
            return "ENQUIRY"
        elif classification == "QUOTATION":
            return "QUOTATION"
        else:
            print(f"Not a valid enquiry or quotation email: {classification}")
            return "UNRELATED"
            
    except Exception as e:
        print(f"Classification attempt failed: {str(e)}")
        return "UNRELATED"

# Create email classification prompt template
email_classification_prompt = PromptTemplate(
    input_variables=["subject", "body"],
    template="""
    Analyze the following email subject and body to determine if it is SPECIFICALLY related to an Enquiry or Quotation.
    
    Subject: {subject}
    Body: {body}
    
    Task: Carefully analyze the content and determine if this email is:
    1. ENQUIRY - The email is EXPLICITLY requesting:
       - Product/service information
       - Technical specifications
       - Pricing details
       - Availability inquiry
       - Request for proposal
    
    2. QUOTATION - The email is EXPLICITLY:
       - Sending a price quote
       - Providing a formal proposal
       - Discussing pricing terms
       - Responding to a quote request
       - Containing pricing information
    
    3. UNRELATED - The email:
       - Does not explicitly mention or discuss enquiries or quotations
       - Is about general communication
       - Contains unrelated attachments or information
       - Is unclear or ambiguous about its purpose
    
    Rules:
    - Do NOT classify based on just the presence of attachments
    - The content must EXPLICITLY indicate it's an enquiry or quotation
    - When in doubt, classify as UNRELATED
    - Do not make assumptions - rely only on explicit content
    
    Respond with ONLY one of these exact words: ENQUIRY, QUOTATION, or UNRELATED
    """
)

# Create classification chain with retry logic
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def classify_email(subject, body):
    """Classify email with retry logic for rate limits."""
    try:
        result = classification_chain.invoke({"subject": subject, "body": body})
        return result.strip().upper()
    except Exception as e:
        print(f"Error in email classification: {str(e)}")
        raise

# Create classification chain using the new pattern
def get_classification_response(inputs):
    """Get classification response and extract the content."""
    response = llm.invoke(email_classification_prompt.format(**inputs))
    if hasattr(response, 'content'):
        return response.content.strip().upper()
    elif isinstance(response, str):
        return response.strip().upper()
    else:
        print(f"Unexpected response type: {type(response)}")
        return "UNKNOWN"

classification_chain = RunnablePassthrough() | get_classification_response

# Define agent creation and execution functions
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception)
)
def create_agent(llm, tools, prompt):
    """Create agent with retry logic."""
    try:
        return create_react_agent(
            llm=llm,
            tools=tools,
            prompt=prompt
        )
    except Exception as e:
        print(f"Error creating agent: {str(e)}")
        raise

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception)
)
def execute_agent(agent_executor, tools_str, tool_names):
    """Execute agent with retry logic."""
    try:
        return agent_executor.invoke({
            "input": "Download PDF attachments from Gmail that are related to business Enquiry or Quotation.",
            "agent_scratchpad": [],
            "tools": tools_str,
            "tool_names": ", ".join(tool_names)
        })
    except Exception as e:
        print(f"Error executing agent: {str(e)}")
        raise

def extract_email_body(msg_data):
    """Extract email body from the message data."""
    def get_body_from_part(part):
        if part.get('mimeType') == 'text/plain':
            data = part.get('body', {}).get('data', '')
            if data:
                return base64.urlsafe_b64decode(data).decode('utf-8')
        elif part.get('mimeType') == 'text/html':
            data = part.get('body', {}).get('data', '')
            if data:
                html_text = base64.urlsafe_b64decode(data).decode('utf-8')
                text = re.sub('<[^<]+?>', '', html_text)
                return text
        return None

    body_parts = []
    
    if msg_data['payload'].get('mimeType') == 'text/plain':
        data = msg_data['payload'].get('body', {}).get('data', '')
        if data:
            body_parts.append(base64.urlsafe_b64decode(data).decode('utf-8'))
    
    def process_parts(parts):
        for part in parts:
            if 'parts' in part:
                process_parts(part['parts'])
            body = get_body_from_part(part)
            if body:
                body_parts.append(body)
    
    if 'parts' in msg_data['payload']:
        process_parts(msg_data['payload']['parts'])
    
    return '\n'.join(body_parts)

def get_gmail_service():
    """Initialize Gmail service with credentials."""
    creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    return build('gmail', 'v1', credentials=creds)

def initialize_log_file():
    """
    Initialize the Excel log file with a structured format for document archival.
    
    The log file tracks the following metadata categories:
    1. Document Identity:
        - subject: Email subject
        - email_id: Unique email identifier
        - thread_id: Email thread identifier
        - sender: Email sender
        
    2. Processing Timeline:
        - first_inbox_msg: Initial processing timestamp
        - last_check_date: Latest verification timestamp
        - download_date: Document download timestamp
        - duplicate_check_date: Last duplicate verification
        
    3. File Management:
        - count_download: Total attachments count
        - list_name_count: Unique filename count
        - attachment_names: List of attachment names
        - file_paths: Storage locations
        - original_filenames: Original file names
        - res_path: Result file path location
        
    4. Data Integrity:
        - message_hash: Email content hash
        - file_hashes: Attachment content hashes
        - unique_file_ids: Generated unique identifiers
        
    5. Processing Status:
        - process_status: Current processing state
        - classification: Document type classification
        - duplicate_status: Duplicate check result
        - res_status: Result processing status
    """
    try:
        if not LOG_FILE.exists():
            # Create initial DataFrame with proper column types
            df = pd.DataFrame({
                # Document Identity
                'subject': pd.Series(dtype='str'),
                'email_id': pd.Series(dtype='str'),
                'thread_id': pd.Series(dtype='str'),
                'sender': pd.Series(dtype='str'),
                
                # Processing Timeline
                'first_inbox_msg': pd.Series(dtype='datetime64[ns]'),
                'last_check_date': pd.Series(dtype='datetime64[ns]'),
                'download_date': pd.Series(dtype='datetime64[ns]'),
                'duplicate_check_date': pd.Series(dtype='datetime64[ns]'),
                
                # File Management
                'count_download': pd.Series(dtype='int64'),
                'list_name_count': pd.Series(dtype='int64'),
                'attachment_names': pd.Series(dtype='str'),
                'file_paths': pd.Series(dtype='str'),
                'original_filenames': pd.Series(dtype='str'),
                'res_path': pd.Series(dtype='str'),
                
                # Data Integrity
                'message_hash': pd.Series(dtype='str'),
                'file_hashes': pd.Series(dtype='str'),
                'unique_file_ids': pd.Series(dtype='str'),
                
                # Processing Status
                'process_status': pd.Series(dtype='str'),
                'classification': pd.Series(dtype='str'),
                'duplicate_status': pd.Series(dtype='str'),
                'res_status': pd.Series(dtype='str')
            })
            
            # Ensure the directory exists
            LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with proper date formatting
            with pd.ExcelWriter(LOG_FILE, engine='openpyxl', datetime_format='YYYY-MM-DD HH:MM:SS') as writer:
                df.to_excel(writer, index=False)
            print(f"Created new structured log file: {LOG_FILE}")
            return df
        else:
            # Load existing file
            df = pd.read_excel(LOG_FILE, parse_dates=[
                'first_inbox_msg', 'last_check_date', 'download_date', 'duplicate_check_date'
            ])
            return df
    except Exception as e:
        print(f"Error initializing log file: {e}")
        # Create empty DataFrame with proper structure as fallback
        return pd.DataFrame(columns=[
            'subject', 'email_id', 'thread_id', 'sender',
            'first_inbox_msg', 'last_check_date', 'download_date', 'duplicate_check_date',
            'count_download', 'list_name_count', 'attachment_names', 'file_paths',
            'original_filenames', 'res_path', 'message_hash', 'file_hashes',
            'unique_file_ids', 'process_status', 'classification', 'duplicate_status',
            'res_status'
        ])

def load_log_data():
    """Load existing log data from Excel file."""
    try:
        if LOG_FILE.exists():
            df = pd.read_excel(LOG_FILE)
            if 'file_hash' not in df.columns:
                df['file_hash'] = ''
            if 'unique_file_id' not in df.columns:
                df['unique_file_id'] = ''
            return df
        else:
            return initialize_log_file()
    except Exception as e:
        print(f"Error loading log data: {e}")
        return pd.DataFrame()

def save_log_data(df):
    """Save log data to Excel file."""
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(LOG_FILE, index=False)
        print(f"Log data saved to {LOG_FILE}")
    except Exception as e:
        print(f"Error saving log data: {e}")

def generate_message_hash(msg_data):
    """Generate unique hash for message to detect duplicates."""
    headers = msg_data['payload'].get('headers', [])
    subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '')
    message_id = msg_data.get('id', '')
    thread_id = msg_data.get('threadId', '')
    return hashlib.md5(f"{message_id}_{thread_id}_{subject}".encode()).hexdigest()

def generate_file_hash(file_data):
    """Generate hash of file content to detect identical files."""
    return hashlib.sha256(file_data).hexdigest()

def generate_unique_file_id(filename, file_hash, email_id):
    """Generate unique identifier for a file based on name, content and email."""
    return f"{filename}_{email_id}_{file_hash[:16]}"

def is_file_already_downloaded(log_df, filename, file_data, email_id, thread_id=None):
    """
    Check if file has already been downloaded using multiple criteria.
    Returns: (bool, str) - (is_duplicate, reason)
    """
    if log_df.empty:
        return False, "No previous downloads"

    file_hash = generate_file_hash(file_data)
    unique_file_id = generate_unique_file_id(filename, file_hash, email_id)

    # Check for exact content match (identical files)
    exact_matches = log_df[log_df['file_hashes'].apply(lambda x: file_hash in str(x).split(',') if pd.notna(x) else False)]
    if not exact_matches.empty:
        match = exact_matches.iloc[0]
        return True, (
            f"Identical file content already exists (downloaded on {match['download_date']} "
            f"from {match['sender']}, subject: {match['subject']})"
        )

    # Check for same file in the same email thread
    if thread_id:
        thread_matches = log_df[
            (log_df['thread_id'] == thread_id) & 
            (log_df['attachment_names'].apply(lambda x: filename in str(x).split(',') if pd.notna(x) else False))
        ]
        if not thread_matches.empty:
            match = thread_matches.iloc[0]
            return True, (
                f"Same filename already downloaded in this email thread "
                f"(original download: {match['download_date']}, subject: {match['subject']})"
            )

    # Check for similar filenames with different content
    similar_names = log_df[log_df['attachment_names'].apply(lambda x: filename in str(x).split(',') if pd.notna(x) else False)]
    if not similar_names.empty:
        return True, f"File with same name already exists (even with different content): {filename}"

    # Check if this exact combination of file and email has been processed
    unique_matches = log_df[log_df['unique_file_ids'].apply(lambda x: unique_file_id in str(x).split(',') if pd.notna(x) else False)]
    if not unique_matches.empty:
        match = unique_matches.iloc[0]
        return True, (
            f"This exact file from this email has already been processed "
            f"(original download: {match['download_date']})"
        )

    return False, "File is new"

def should_process_email(subject, body, has_pdf_attachment):
    """Determine if an email should be processed based on content analysis."""
    if not has_pdf_attachment:
        return False, {
            'decision': 'SKIP',
            'reason': 'No PDF attachment present',
            'classification': 'UNRELATED',
            'confidence': 0.0
        }

    try:
        classification = classify_email(subject, body)
        
        # Strict classification check
        if classification == "ENQUIRY" or classification == "QUOTATION":
            return True, {
                'decision': 'PROCESS',
                'reason': f'Email classified as {classification}',
                'classification': classification,
                'confidence': 1.0
            }
        
        # Any other classification results in skip
        return False, {
            'decision': 'SKIP',
            'reason': 'Email not related to enquiry or quotation',
            'classification': 'UNRELATED',
            'confidence': 0.0
        }
    except Exception as e:
        print(f"Warning: Classification failed - {str(e)}")
        return False, {
            'decision': 'SKIP',
            'reason': 'Classification failed - skipping download',
            'classification': 'UNRELATED',
            'confidence': 0.0
        }

def update_download_log(log_df, new_row):
    """Update the download log with new entry and save."""
    try:
        # Ensure lists are stored as comma-separated strings
        for field in ['attachment_names', 'file_paths', 'file_hashes', 'unique_file_ids', 'original_filenames']:
            if isinstance(new_row.get(field), list):
                new_row[field] = ','.join(str(item) for item in new_row[field])

        # Add duplicate check information
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        new_row['duplicate_check_date'] = current_time
        new_row['duplicate_status'] = 'unique'
        new_row['download_date'] = current_time  # Ensure download date is set
        
        # Initialize result processing fields if not present
        if 'res_status' not in new_row:
            new_row['res_status'] = 'pending'
        if 'res_path' not in new_row:
            new_row['res_path'] = ''
        
        # Update tracking fields
        new_row['last_check_date'] = current_time
        
        # Handle count_download and list_name_count for existing email
        if 'email_id' in new_row:
            existing_email = log_df[log_df['email_id'] == new_row['email_id']]
            if not existing_email.empty:
                # Get existing counts and names
                existing_count = existing_email['count_download'].iloc[0] if 'count_download' in existing_email else 0
                existing_names = set()
                if 'original_filenames' in existing_email and pd.notna(existing_email['original_filenames'].iloc[0]):
                    existing_names = set(str(existing_email['original_filenames'].iloc[0]).split(','))
                
                # Get new filenames (use original filenames, not the renamed ones)
                new_names = set()
                if 'original_filenames' in new_row:
                    if isinstance(new_row['original_filenames'], list):
                        new_names = set(str(name) for name in new_row['original_filenames'])
                    elif isinstance(new_row['original_filenames'], str):
                        new_names = set(new_row['original_filenames'].split(','))
                
                # Update counts
                new_row['count_download'] = existing_count + len(new_names)
                new_row['list_name_count'] = len(existing_names.union(new_names))
                new_row['first_inbox_msg'] = existing_email.iloc[0]['first_inbox_msg']
                
                # Merge file information
                for field in ['file_paths', 'file_hashes', 'unique_file_ids', 'original_filenames']:
                    if field in existing_email and pd.notna(existing_email[field].iloc[0]):
                        existing_values = set(str(existing_email[field].iloc[0]).split(','))
                        new_values = set(str(new_row.get(field, '')).split(','))
                        new_row[field] = ','.join(sorted(existing_values.union(new_values)))
            else:
                # New email - set initial counts using original filenames
                if 'original_filenames' in new_row:
                    if isinstance(new_row['original_filenames'], list):
                        new_row['count_download'] = len(new_row['original_filenames'])
                        new_row['list_name_count'] = len(set(new_row['original_filenames']))
                    elif isinstance(new_row['original_filenames'], str):
                        filenames = new_row['original_filenames'].split(',')
                        new_row['count_download'] = len(filenames)
                        new_row['list_name_count'] = len(set(filenames))
                    else:
                        new_row['count_download'] = 0
                        new_row['list_name_count'] = 0
                new_row['first_inbox_msg'] = current_time
        
        # Remove old entry if exists
        if 'email_id' in new_row:
            log_df = log_df[log_df['email_id'] != new_row['email_id']]
        
        # Convert date strings to datetime objects
        for date_field in ['first_inbox_msg', 'last_check_date', 'download_date', 'duplicate_check_date']:
            if date_field in new_row and isinstance(new_row[date_field], str):
                try:
                    new_row[date_field] = pd.to_datetime(new_row[date_field])
                except:
                    new_row[date_field] = pd.to_datetime(current_time)
        
        # Update the DataFrame
        log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save to Excel with proper date formatting
        with pd.ExcelWriter(LOG_FILE, engine='openpyxl', datetime_format='YYYY-MM-DD HH:MM:SS') as writer:
            log_df.to_excel(writer, index=False)
            
        return log_df
    except Exception as e:
        print(f"Error updating download log: {e}")
        return log_df

def monitor_gmail_for_new_attachments_with_logging() -> str:
    """Monitor Gmail for new emails with document attachments in the last 24 hours with enhanced duplicate prevention."""
    try:
        # Initialize logging
        initialize_log_file()
        log_df = load_log_data()
       
        service = get_gmail_service()
       
        # Search for recent emails with attachments (last 24 hours)
        yesterday = datetime.now() - timedelta(days=1)
        query = f'has:attachment after:{yesterday.strftime("%Y/%m/%d")}'
        print(f"Searching for emails with query: {query}")
       
        results = service.users().messages().list(userId='me', q=query).execute()
        messages = results.get('messages', [])
 
        if not messages:
            return "📭 No new emails with attachments found in the last 24 hours."
 
        downloaded = []
        skipped = []
        processed_emails = []
        skip_details = []
       
        for msg in messages:
            try:
                msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
               
                # Extract email information
                headers = msg_data['payload'].get('headers', [])
                subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
                sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown Sender')
                date_header = next((h['value'] for h in headers if h['name'] == 'Date'), '')
                thread_id = msg_data.get('threadId', '')
                
                print(f"\nProcessing email from: {sender}")
                print(f"Subject: {subject}")
                
                # Extract body and classify immediately
                body = extract_email_body(msg_data)
                classification = classify_email(subject, body)
                
                # Strict classification check - only proceed if explicitly ENQUIRY or QUOTATION
                if classification not in ["ENQUIRY", "QUOTATION"]:
                    skip_reason = f"Not a valid enquiry or quotation email (got {classification})"
                    print(f"Skipping email: {skip_reason}")
                    skipped.append(f"'{subject}' from {sender} - {skip_reason}")
                    skip_details.append(f"  • {subject}: {skip_reason}")
                    continue

                print(f"Email classified as: {classification}")
               
                # Generate message hash
                message_hash = generate_message_hash(msg_data)
               
                # Process attachments
                parts = msg_data['payload'].get('parts', [])
                if not parts and msg_data['payload'].get('filename'):
                    parts = [msg_data['payload']]
               
                document_attachments = []
                for part in parts:
                    filename = part.get('filename', '')
                    if filename and filename.lower().endswith('.pdf'):
                        document_attachments.append(filename)
               
                if not document_attachments:
                    skip_reason = "No PDF attachments found"
                    print(f"Skipping email: {skip_reason}")
                    skipped.append(f"'{subject}' from {sender} - {skip_reason}")
                    skip_details.append(f"  • {subject}: {skip_reason}")
                    continue
               
                # Check for existing thread processing
                if thread_id:
                    thread_entries = log_df[log_df['thread_id'] == thread_id]
                    if not thread_entries.empty:
                        existing_entry = thread_entries.iloc[0]
                        if existing_entry['classification'] == classification:
                            skip_reason = "Thread already processed with same classification"
                            print(f"Skipping: {skip_reason}")
                            skipped.append(f"'{subject}' - {skip_reason}")
                            skip_details.append(f"  • {subject}: {skip_reason}")
                            continue

                # Download new attachments with enhanced checking
                successfully_downloaded = []
                file_paths = []
                file_hashes = []
                unique_file_ids = []
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
               
                for part in parts:
                    filename = part.get('filename', '')
                    if filename in document_attachments:
                        try:
                            # Get file data for duplicate checking
                            if 'body' not in part or 'attachmentId' not in part['body']:
                                print(f"Skipping {filename}: Invalid attachment data")
                                continue
                            
                            attachment = service.users().messages().attachments().get(
                                userId='me', messageId=msg['id'], id=part['body']['attachmentId']
                            ).execute()
                            file_data = base64.urlsafe_b64decode(attachment['data'])
 
                            # Enhanced duplicate check
                            is_duplicate, reason = is_file_already_downloaded(log_df, filename, file_data, msg['id'], thread_id)
                           
                            if is_duplicate:
                                print(f"Skipping duplicate: {reason}")
                                skipped.append(filename)
                                skip_details.append(f"  • {filename}: {reason}")
                                continue
 
                            # Generate unique filename with classification prefix
                            base_name, ext = os.path.splitext(filename)
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            final_filename = f"{classification.lower()}_{timestamp}_{filename}"
                            file_path = SAVE_PATH / final_filename
                            
                            # Ensure unique filename
                            counter = 1
                            while file_path.exists():
                                final_filename = f"{classification.lower()}_{timestamp}_{base_name}_{counter}{ext}"
                                file_path = SAVE_PATH / final_filename
                                counter += 1
                           
                            # Write file
                            file_path.write_bytes(file_data)
                           
                            # Generate file metadata
                            file_hash = generate_file_hash(file_data)
                            unique_file_id = generate_unique_file_id(filename, file_hash, msg['id'])
                           
                            successfully_downloaded.append(final_filename)
                            file_paths.append(str(file_path))
                            file_hashes.append(file_hash)
                            unique_file_ids.append(unique_file_id)
                            downloaded.append(f"{filename} -> {final_filename}")
                            print(f"Successfully downloaded: {final_filename}")
                           
                        except Exception as e:
                            print(f"Error downloading {filename}: {e}")
                            continue

                if successfully_downloaded:
                    # Log the download with enhanced metadata
                    new_row = {
                        'subject': subject,
                        'first_inbox_msg': date_header,
                        'last_check_date': current_time,
                        'download_date': current_time,
                        'count_download': len(successfully_downloaded),
                        'list_name_count': ', '.join(successfully_downloaded),
                        'email_id': msg['id'],
                        'thread_id': thread_id,
                        'sender': sender,
                        'attachment_names': ', '.join(successfully_downloaded),
                        'file_paths': ', '.join(file_paths),
                        'message_hash': message_hash,
                        'file_hashes': ', '.join(file_hashes),
                        'unique_file_ids': ', '.join(unique_file_ids),
                        'process_status': 'downloaded',
                        'classification': classification,
                        'res_status': 'pending',
                        'res_path': '',
                        'duplicate_check_date': current_time,
                        'duplicate_status': 'unique'
                    }
                    
                    log_df = update_download_log(log_df, new_row)
                    processed_emails.append(f"📧 {sender}: {subject}")
               
            except Exception as e:
                print(f"Error processing message: {e}")
                continue
       
        # Generate detailed summary
        summary = []
        summary.append("🔍 Gmail Monitoring Complete")
        summary.append("=" * 40)
       
        if downloaded:
            summary.append(f"\n✅ Downloaded {len(downloaded)} new attachments:")
            for file in downloaded:
                summary.append(f"   • {file}")
       
        if skipped:
            summary.append(f"\n⏭️ Skipped {len(skipped)} attachments:")
            summary.extend(skip_details)
       
        if processed_emails:
            summary.append(f"\n📨 Processed emails:")
            for email in processed_emails:
                summary.append(f"   {email}")
       
        summary.append(f"\n📊 Log file: {LOG_FILE}")
        summary.append(f"💾 Save path: {SAVE_PATH}")
       
        return '\n'.join(summary) if downloaded or skipped else "📭 No new document attachments found in recent emails."
       
    except Exception as e:
        return f"❌ Error monitoring emails: {str(e)}"


def download_gmail_attachments_with_logging() -> str:
    """Download PDF attachments from business-related Enquiry or Quotation emails only."""
    try:
        print("Starting email attachment download process...")
        initialize_log_file()
        log_df = load_log_data()
        service = get_gmail_service()
        
        # Search for emails with PDF attachments from the last 30 days
        date_limit = (datetime.now() - timedelta(days=30)).strftime('%Y/%m/%d')
        query = f'has:attachment filename:pdf after:{date_limit}'
        print(f"Searching for emails with query: {query}")
        
        try:
            results = service.users().messages().list(userId='me', q=query).execute()
            messages = results.get('messages', [])
        except Exception as e:
            print(f"Error searching emails: {str(e)}")
            return "❌ Failed to search emails. Please check your connection and try again."
        
        if not messages:
            return "📭 No new emails with PDF attachments found in the last 30 days."

        downloaded = []
        skipped = []
        processed_emails = []
        skip_details = []

        for msg in messages:
            try:
                msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
                headers = msg_data['payload'].get('headers', [])
                subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
                sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown Sender')
                date_header = next((h['value'] for h in headers if h['name'] == 'Date'), '')
                thread_id = msg_data.get('threadId', '')
                
                print(f"\nProcessing email from: {sender}")
                print(f"Subject: {subject}")

                # Extract email body and classify immediately
                body = extract_email_body(msg_data)
                classification = classify_email(subject, body)
                
                # Strict classification check - only proceed if explicitly ENQUIRY or QUOTATION
                if classification not in ["ENQUIRY", "QUOTATION"]:
                    skip_reason = f"Not a valid enquiry or quotation email (got {classification})"
                    print(f"Skipping email: {skip_reason}")
                    skipped.append(f"'{subject}' from {sender} - {skip_reason}")
                    skip_details.append(f"  • {subject}: {skip_reason}")
                    continue

                print(f"Email classified as: {classification}")

                # Check for existing thread processing
                if thread_id:
                    thread_entries = log_df[log_df['thread_id'] == thread_id]
                    if not thread_entries.empty:
                        existing_entry = thread_entries.iloc[0]
                        if existing_entry['classification'] == classification:
                            skip_reason = "Thread already processed with same classification"
                            print(f"Skipping: {skip_reason}")
                            skipped.append(f"'{subject}' - {skip_reason}")
                            skip_details.append(f"  • {subject}: {skip_reason}")
                            continue

                # Only check for PDF attachments after classification is confirmed
                parts = msg_data['payload'].get('parts', [])
                if not parts:
                    parts = [msg_data['payload']] if msg_data['payload'].get('filename') else []
                
                pdf_parts = [part for part in parts if part.get('filename', '').lower().endswith('.pdf')]
                if not pdf_parts:
                    skip_reason = "No PDF attachments found"
                    print(f"Skipping email: {skip_reason}")
                    skipped.append(f"'{subject}' from {sender} - {skip_reason}")
                    skip_details.append(f"  • {subject}: {skip_reason}")
                    continue

                # Process attachments only if email is properly classified
                successfully_downloaded = []
                file_paths = []
                file_hashes = []
                unique_file_ids = []
                message_hash = generate_message_hash(msg_data)
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                for part in pdf_parts:
                    filename = part.get('filename')
                    if not filename:
                        continue
                    
                    print(f"Processing attachment: {filename}")
                    
                    try:
                        # Validate attachment data
                        if 'body' not in part or 'attachmentId' not in part['body']:
                            print(f"Skipping {filename}: Invalid attachment data")
                            continue
                        
                        # Get attachment data
                        attachment = service.users().messages().attachments().get(
                            userId='me', messageId=msg['id'], id=part['body']['attachmentId']
                        ).execute()
                        file_data = base64.urlsafe_b64decode(attachment['data'])
                        
                        # Check for duplicates
                        is_duplicate, reason = is_file_already_downloaded(log_df, filename, file_data, msg['id'], thread_id)
                        if is_duplicate:
                            print(f"Skipping duplicate: {reason}")
                            skipped.append(filename)
                            skip_details.append(f"  • {filename}: {reason}")
                            continue
                        
                        # Generate unique filename with classification prefix
                        base_name, ext = os.path.splitext(filename)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        final_filename = f"{classification.lower()}_{timestamp}_{filename}"
                        file_path = SAVE_PATH / final_filename
                        
                        # Ensure unique filename
                        counter = 1
                        while file_path.exists():
                            final_filename = f"{classification.lower()}_{timestamp}_{base_name}_{counter}{ext}"
                            file_path = SAVE_PATH / final_filename
                            counter += 1
                        
                        # Write file
                        file_path.write_bytes(file_data)
                        
                        # Generate file metadata
                        file_hash = generate_file_hash(file_data)
                        unique_file_id = generate_unique_file_id(filename, file_hash, msg['id'])
                        
                        successfully_downloaded.append(final_filename)
                        file_paths.append(str(file_path))
                        file_hashes.append(file_hash)
                        unique_file_ids.append(unique_file_id)
                        downloaded.append(f"{filename} -> {final_filename}")
                        print(f"Successfully downloaded: {final_filename}")
                        
                    except Exception as e:
                        print(f"Error downloading {filename}: {e}")
                        continue

                if successfully_downloaded:
                    # Log the download with enhanced metadata
                    new_row = {
                        'subject': subject,
                        'first_inbox_msg': date_header,
                        'last_check_date': current_time,
                        'download_date': current_time,  # Ensure download date is set
                        'count_download': len(successfully_downloaded),
                        'list_name_count': ', '.join(successfully_downloaded),
                        'email_id': msg['id'],
                        'thread_id': thread_id,
                        'sender': sender,
                        'attachment_names': ', '.join(successfully_downloaded),
                        'file_paths': ', '.join(file_paths),
                        'message_hash': message_hash,
                        'file_hashes': ', '.join(file_hashes),
                        'unique_file_ids': ', '.join(unique_file_ids),
                        'process_status': 'downloaded',
                        'classification': classification,
                        'res_status': 'pending',
                        'res_path': '',
                        'duplicate_check_date': current_time,
                        'duplicate_status': 'unique'
                    }
                    
                    log_df = update_download_log(log_df, new_row)
                    processed_emails.append(f"📧 {sender}: {subject}")

            except Exception as e:
                print(f"Error processing message: {e}")
                continue

        # Generate detailed summary
        summary = []
        summary.append("🔍 Gmail Download Complete")
        summary.append("=" * 40)
        
        if downloaded:
            summary.append(f"\n✅ Downloaded {len(downloaded)} new attachments:")
            for file in downloaded:
                summary.append(f"   • {file}")
        
        if skipped:
            summary.append(f"\n⏭️ Skipped {len(skipped)} attachments:")
            summary.extend(skip_details)
        
        if processed_emails:
            summary.append(f"\n📨 Processed emails:")
            for email in processed_emails:
                summary.append(f"   {email}")
        
        summary.append(f"\n📊 Log file: {LOG_FILE}")
        summary.append(f"💾 Save path: {SAVE_PATH}")
        
        return '\n'.join(summary)

    except Exception as e:
        error_msg = f"❌ Error in download process: {str(e)}"
        print(f"\nError: {error_msg}")
        return error_msg


def view_download_log() -> str:
    """View the current download log statistics with duplicate prevention details."""
    try:
        if not LOG_FILE.exists():
            return "📄 No log file found. No downloads recorded yet."
       
        log_df = pd.read_excel(LOG_FILE)
       
        if log_df.empty:
            return "📄 Log file is empty. No downloads recorded yet."
       
        # Generate statistics
        total_downloads = len(log_df)
        unique_senders = log_df['sender'].nunique() if 'sender' in log_df.columns else 0
        total_files = log_df['count_download'].sum() if 'count_download' in log_df.columns else 0
        unique_threads = log_df['thread_id'].nunique() if 'thread_id' in log_df.columns else 0
       
       
        # File hash statistics (if available)
        unique_files_by_content = 0
        if 'file_hash' in log_df.columns:
            all_hashes = []
            for _, row in log_df.iterrows():
                if pd.notna(row['file_hash']) and row['file_hash']:
                    hashes = str(row['file_hash']).split(', ')
                    all_hashes.extend(hashes)
            unique_files_by_content = len(set(all_hashes)) if all_hashes else 0
       
        # Recent downloads (last 7 days)
        if 'download_date' in log_df.columns:
            log_df['download_date'] = pd.to_datetime(log_df['download_date'], errors='coerce')
            recent = log_df[log_df['download_date'] > (datetime.now() - timedelta(days=7))]
            recent_count = len(recent)
        else:
            recent_count = 0
       
        summary = []
        summary.append("📊 Enhanced Download Log Summary")
        summary.append("=" * 40)
        summary.append(f"📧 Total email entries: {total_downloads}")
        summary.append(f"📎 Total files downloaded: {total_files}")
        summary.append(f"🔗 Unique email threads: {unique_threads}")
        summary.append(f"📄 Unique files by content: {unique_files_by_content}")
        summary.append(f"👥 Unique senders: {unique_senders}")
        summary.append(f"🕐 Recent downloads (7 days): {recent_count}")
        summary.append(f"📁 Log file location: {LOG_FILE}")
       
        if not log_df.empty and len(log_df) > 0:
            summary.append("\n📋 Recent entries:")
            for _, row in log_df.tail(5).iterrows():
                subject = row.get('subject', 'N/A')
                sender = row.get('sender', 'N/A')
                files = row.get('list_name_count', 'N/A')
                thread_id = row.get('thread_id', 'N/A')[:8] + '...' if row.get('thread_id') else 'N/A'
                summary.append(f"   • {subject[:40]}... from {sender}")
                summary.append(f"     Files: {files} | Thread: {thread_id}")
       
        return '\n'.join(summary)
       
    except Exception as e:
        return f"❌ Error reading log: {str(e)}"


def clear_duplicate_entries() -> str:
    """Clean up duplicate entries in the log file based on file content."""
    try:
        if not LOG_FILE.exists():
            return "📄 No log file found."
       
        log_df = pd.read_excel(LOG_FILE, parse_dates=[
            'first_inbox_msg', 'last_check_date', 'download_date', 'duplicate_check_date'
        ])
       
        if log_df.empty:
            return "📄 Log file is empty."
       
        original_count = len(log_df)
        
        # First pass: Remove exact duplicates based on file content
        if 'file_hashes' in log_df.columns:
            # Split file_hashes into sets for comparison
            log_df['hash_set'] = log_df['file_hashes'].apply(
                lambda x: set(str(x).split(',')) if pd.notna(x) else set()
            )
            
            # Keep first occurrence of each unique hash set
            seen_hashes = set()
            duplicate_indices = []
            
            for idx, row in log_df.iterrows():
                current_hashes = row['hash_set']
                if not current_hashes:
                    continue
                    
                if current_hashes.issubset(seen_hashes):
                    duplicate_indices.append(idx)
                else:
                    seen_hashes.update(current_hashes)
            
            # Remove duplicates
            log_df = log_df.drop(duplicate_indices)
            log_df = log_df.drop('hash_set', axis=1)
        
        # Second pass: Remove duplicates based on email thread and filenames
        if 'thread_id' in log_df.columns and 'original_filenames' in log_df.columns:
            # Group by thread_id and check for duplicate filenames
            thread_groups = log_df.groupby('thread_id')
            duplicate_indices = []
            
            for _, group in thread_groups:
                if len(group) <= 1:
                    continue
                    
                # Convert filename strings to sets
                filename_sets = group['original_filenames'].apply(
                    lambda x: set(str(x).split(',')) if pd.notna(x) else set()
                )
                
                # Check each row against previous rows in the same thread
                seen_files = set()
                for idx, files in filename_sets.items():
                    if not files:
                        continue
                        
                    if files.issubset(seen_files):
                        duplicate_indices.append(idx)
                    else:
                        seen_files.update(files)
            
            # Remove duplicates
            log_df = log_df.drop(duplicate_indices)
        
        # Third pass: Remove duplicates based on unique_file_ids
        if 'unique_file_ids' in log_df.columns:
            # Split unique_file_ids into sets
            log_df['id_set'] = log_df['unique_file_ids'].apply(
                lambda x: set(str(x).split(',')) if pd.notna(x) else set()
            )
            
            # Keep first occurrence of each unique ID
            seen_ids = set()
            duplicate_indices = []
            
            for idx, row in log_df.iterrows():
                current_ids = row['id_set']
                if not current_ids:
                    continue
                    
                if current_ids.issubset(seen_ids):
                    duplicate_indices.append(idx)
                else:
                    seen_ids.update(current_ids)
            
            # Remove duplicates
            log_df = log_df.drop(duplicate_indices)
            log_df = log_df.drop('id_set', axis=1)
        
        cleaned_count = len(log_df)
        removed_count = original_count - cleaned_count
        
        if removed_count > 0:
            # Save cleaned data with proper date formatting
            with pd.ExcelWriter(LOG_FILE, engine='openpyxl', datetime_format='YYYY-MM-DD HH:MM:SS') as writer:
                log_df.to_excel(writer, index=False)
            return f"✅ Cleaned log file: Removed {removed_count} duplicate entries. {cleaned_count} entries remain."
        else:
            return "✅ No duplicate entries found in log file."
       
    except Exception as e:
        return f"❌ Error cleaning log: {str(e)}"


def agent_main():
    """Main function to initialize and run the Gmail processing agent."""
    try:
        print("Initializing Gmail Processing Agent...")
        
        # Initialize Gmail credentials
        try:
            credentials = get_gmail_credentials(
                token_file="token.json",
                scopes=SCOPES,
                client_secrets_file="credentials.json"
            )
            
            api_resource = build_resource_service(credentials=credentials)
            print("✅ Gmail credentials initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize Gmail credentials: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "status": "error",
                "message": "Gmail initialization failed",
                "error": error_msg
            }

        try:
            # Execute functions in sequence with proper error handling
            results = []
            
            print("\n1. Monitoring Gmail for new attachments...")
            try:
                monitor_result = monitor_gmail_for_new_attachments_with_logging()
                results.append(("Monitor", "success", monitor_result))
                print("✅ Monitoring completed successfully")
            except Exception as e:
                results.append(("Monitor", "error", str(e)))
                print(f"❌ Monitoring failed: {str(e)}")
            
            print("\n2. Downloading relevant PDF attachments...")
            try:
                download_result = download_gmail_attachments_with_logging()
                results.append(("Download", "success", download_result))
                print("✅ Download completed successfully")
            except Exception as e:
                results.append(("Download", "error", str(e)))
                print(f"❌ Download failed: {str(e)}")
            
            print("\n3. Checking download log...")
            try:
                log_result = view_download_log()
                results.append(("Log View", "success", log_result))
                print("✅ Log check completed successfully")
            except Exception as e:
                results.append(("Log View", "error", str(e)))
                print(f"❌ Log check failed: {str(e)}")
            
            print("\n4. Cleaning up duplicate entries...")
            try:
                cleanup_result = clear_duplicate_entries()
                results.append(("Cleanup", "success", cleanup_result))
                print("✅ Cleanup completed successfully")
            except Exception as e:
                results.append(("Cleanup", "error", str(e)))
                print(f"❌ Cleanup failed: {str(e)}")
            
            # Generate final report
            summary = []
            summary.append("\n=== Final Execution Report ===")
            
            all_successful = all(result[1] == "success" for result in results)
            
            for step, status, message in results:
                if status == "success":
                    summary.append(f"\n✅ {step}:")
                    summary.append(message)
                else:
                    summary.append(f"\n❌ {step} failed:")
                    summary.append(f"Error: {message}")
            
            final_output = "\n".join(summary)
            
            return {
                "status": "success" if all_successful else "partial_success",
                "output": final_output,
                "error": None if all_successful else "Some steps failed, see output for details"
            }
            
        except Exception as e:
            error_msg = f"Error in execution: {str(e)}"
            print(f"\n❌ {error_msg}")
            return {
                "status": "error",
                "message": "Execution failed",
                "error": error_msg
            }

    except Exception as e:
        error_msg = f"Error in setup: {str(e)}"
        print(f"\n❌ {error_msg}")
        
        if "API key" in str(e):
            print("\nAPI key error. Please:")
            print("1. Check if GEMINI_API_KEY is set in your .env file")
            print("2. Verify the API key is valid")
            print("3. Generate a new API key if needed")
        
        return {
            "status": "error",
            "message": "Setup failed",
            "error": error_msg
        }

if __name__ == "__main__":
    try:
        # Run the main function
        result = agent_main()
        if result["status"] != "success":
            print("\nExecution completed with issues:")
            print(result["error"])
        else:
            print("\nExecution completed successfully!")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
    finally:
        cleanup_resources()
        print("\nExecution finished.")
