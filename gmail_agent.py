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
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from langchain.tools import tool, Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.agent_toolkits import GmailToolkit
from langchain_community.tools.gmail.utils import build_resource_service, get_gmail_credentials
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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
SAVE_PATH = r"/mnt/c/Users/Imran/OneDrive - Ahana Systems and Solutions (P) Ltd/Desktop/Demo/steer_document_processing_poc/demo_app/backend/Agent_AI/download_email"
LOG_FILE = os.path.join(SAVE_PATH, "email_download_log.xlsx")

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
            return "UNKNOWN"
        
        # Validate classification
        if classification in ["ENQUIRY", "QUOTATION", "UNRELATED"]:
            return classification
        else:
            print(f"Invalid classification received: {classification}")
            return "UNKNOWN"
            
    except Exception as e:
        print(f"Classification attempt failed: {str(e)}")
        raise

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
    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame(columns=[
            # Document Identity
            'subject',
            'email_id',
            'thread_id',
            'sender',
            
            # Processing Timeline
            'first_inbox_msg',
            'last_check_date',
            'download_date',
            'duplicate_check_date',
            
            # File Management
            'count_download',
            'list_name_count',
            'attachment_names',
            'file_paths',
            'original_filenames',
            'res_path',           # New field: Result file path location
            
            # Data Integrity
            'message_hash',
            'file_hashes',
            'unique_file_ids',
            
            # Processing Status
            'process_status',
            'classification',
            'duplicate_status',
            'res_status'         # New field: Result processing status
        ])
        df.to_excel(LOG_FILE, index=False)
        print(f"Created new structured log file: {LOG_FILE}")
    return LOG_FILE

def load_log_data():
    """Load existing log data from Excel file."""
    try:
        if os.path.exists(LOG_FILE):
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
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
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
    exact_matches = log_df[log_df['file_hashes'].str.contains(file_hash, na=False)]
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
            (log_df['attachment_names'].str.contains(re.escape(filename), na=False))
        ]
        if not thread_matches.empty:
            match = thread_matches.iloc[0]
            return True, (
                f"Same filename already downloaded in this email thread "
                f"(original download: {match['download_date']}, subject: {match['subject']})"
            )

    # Check for similar filenames with different content
    similar_names = log_df[log_df['attachment_names'].str.contains(re.escape(filename), na=False)]
    if not similar_names.empty:
        # If same filename exists but with different content, add a note
        print(f"Note: Found file with same name but different content: {filename}")

    # Check if this exact combination of file and email has been processed
    unique_matches = log_df[log_df['unique_file_ids'].str.contains(unique_file_id, na=False)]
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
            'classification': 'N/A',
            'confidence': 0.0
        }

    try:
        classification = classify_email(subject, body)
        
        if classification in ["ENQUIRY", "QUOTATION"]:
            return True, {
                'decision': 'PROCESS',
                'reason': f'Email classified as {classification}',
                'classification': classification,
                'confidence': 1.0
            }
        elif classification == "UNKNOWN":
            return False, {
                'decision': 'SKIP',
                'reason': 'Email classification is UNKNOWN - skipping download',
                'classification': 'UNKNOWN',
                'confidence': 0.0
            }
        else:
            return False, {
                'decision': 'SKIP',
                'reason': 'Email not clearly related to enquiry or quotation',
                'classification': 'UNRELATED',
                'confidence': 0.0
            }
    except Exception as e:
        print(f"Warning: Classification failed - {str(e)}")
        return False, {
            'decision': 'SKIP',
            'reason': 'Classification failed - skipping download',
            'classification': 'UNKNOWN',
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
        new_row['duplicate_check_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        new_row['duplicate_status'] = 'unique'
        
        # Update tracking fields
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Handle count_download and list_name_count for existing email
        if 'email_id' in new_row:
            existing_email = log_df[log_df['email_id'] == new_row['email_id']]
            if not existing_email.empty:
                # Get existing counts
                existing_count = existing_email['count_download'].iloc[0] if 'count_download' in existing_email else 0
                existing_names = set()
                if 'original_filenames' in existing_email:
                    existing_names = set(str(existing_email['original_filenames'].iloc[0]).split(','))
                
                # Get new filenames
                new_names = set()
                if isinstance(new_row.get('original_filenames'), list):
                    new_names = set(str(name) for name in new_row['original_filenames'])
                elif isinstance(new_row.get('original_filenames'), str):
                    new_names = set(new_row['original_filenames'].split(','))
                
                # Update counts
                new_row['count_download'] = existing_count + len(new_names)
                new_row['list_name_count'] = len(existing_names.union(new_names))
                new_row['first_inbox_msg'] = existing_email.iloc[0]['first_inbox_msg']
            else:
                # New email - set initial counts
                if isinstance(new_row.get('original_filenames'), list):
                    new_row['count_download'] = len(new_row['original_filenames'])
                    new_row['list_name_count'] = len(set(new_row['original_filenames']))
                elif isinstance(new_row.get('original_filenames'), str):
                    filenames = new_row['original_filenames'].split(',')
                    new_row['count_download'] = len(filenames)
                    new_row['list_name_count'] = len(set(filenames))
                else:
                    new_row['count_download'] = 0
                    new_row['list_name_count'] = 0
                new_row['first_inbox_msg'] = current_time
        
        # Update last_check_date
        new_row['last_check_date'] = current_time
        
        # Remove old entry if exists
        if 'email_id' in new_row:
            log_df = log_df[log_df['email_id'] != new_row['email_id']]
        
        # Update the DataFrame
        log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save to Excel
        save_log_data(log_df)
        return log_df
    except Exception as e:
        print(f"Error updating download log: {e}")
        return log_df

@tool
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
        
        results = service.users().messages().list(
            userId='me',
            q=query
        ).execute()
        
        messages = results.get('messages', [])
        if not messages:
            return "No emails with PDF attachments found in the last 30 days."

        print(f"Found {len(messages)} emails to process")
        downloaded_files = []
        skipped_files = []
        duplicate_files = []
        processed_count = 0
        total_download_count = 0
        unique_files_count = 0

        for msg in messages:
            try:
                msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
                headers = msg_data['payload'].get('headers', [])
                subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
                sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown Sender')
                
                print(f"\nProcessing email {processed_count + 1}/{len(messages)}")
                print(f"From: {sender}")
                print(f"Subject: {subject}")

                body = extract_email_body(msg_data)
                
                # Check for PDF attachments
                parts = msg_data['payload'].get('parts', [])
                if not parts:
                    parts = [msg_data['payload']] if msg_data['payload'].get('filename') else []
                
                pdf_parts = [part for part in parts if part.get('filename', '').lower().endswith('.pdf')]
                has_pdf = len(pdf_parts) > 0
                
                should_process, process_info = should_process_email(subject, body, has_pdf)
                
                if not should_process:
                    print(f"Skipping: {process_info['reason']}")
                    skipped_files.append(f"'{subject}' from {sender} - {process_info['reason']}")
                    continue

                print(f"Email classified as: {process_info['classification']}")
                print(f"Found {len(pdf_parts)} PDF attachments")
                
                # Track attachments for this email
                email_attachments = []
                email_file_paths = []
                email_file_hashes = []
                email_unique_ids = []
                email_original_names = []
                
                for part in pdf_parts:
                    filename = part.get('filename')
                    if not filename:
                        continue
                        
                    print(f"Processing attachment: {filename}")
                    
                    if 'body' in part and 'attachmentId' in part['body']:
                        attachment = service.users().messages().attachments().get(
                            userId='me',
                            messageId=msg['id'],
                            id=part['body']['attachmentId']
                        ).execute()
                        
                        file_data = base64.urlsafe_b64decode(attachment['data'])
                        is_duplicate, duplicate_reason = is_file_already_downloaded(
                            log_df, filename, file_data, msg['id'], msg_data.get('threadId')
                        )
                        
                        if is_duplicate:
                            print(f"Skipping duplicate: {duplicate_reason}")
                            duplicate_files.append(f"'{filename}' from {sender} - {duplicate_reason}")
                            continue

                        file_hash = generate_file_hash(file_data)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        unique_filename = f"{process_info['classification'].lower()}_{timestamp}_{filename}"
                        file_path = os.path.join(SAVE_PATH, unique_filename)
                        
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)
                        with open(file_path, 'wb') as f:
                            f.write(file_data)
                        
                        print(f"Downloaded: {filename} -> {unique_filename}")
                        
                        # Collect attachment information
                        email_attachments.append(filename)
                        email_file_paths.append(file_path)
                        email_file_hashes.append(file_hash)
                        email_unique_ids.append(generate_unique_file_id(filename, file_hash, msg['id']))
                        email_original_names.append(filename)
                        downloaded_files.append(f"{filename} -> {unique_filename}")
                        total_download_count += 1

                # Only create log entry if files were downloaded
                if email_attachments:
                    unique_files_count += len(set(email_original_names))
                    new_row = {
                        'subject': subject,
                        'sender': sender,
                        'email_id': msg['id'],
                        'thread_id': msg_data.get('threadId'),
                        'attachment_names': email_attachments,
                        'file_paths': email_file_paths,
                        'file_hashes': email_file_hashes,
                        'unique_file_ids': email_unique_ids,
                        'download_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'process_status': 'downloaded',
                        'classification': process_info['classification'],
                        'original_filenames': email_original_names,
                        'res_path': file_path
                    }
                    
                    log_df = update_download_log(log_df, new_row)

                processed_count += 1
                print(f"Progress: {processed_count}/{len(messages)} emails processed")
                print(f"Total downloads so far: {total_download_count}")
                print(f"Unique files so far: {unique_files_count}")

            except Exception as e:
                print(f"Error processing email: {str(e)}")
                continue

        summary = [
            "\nDownload Summary",
            "=" * 20,
            f"\nProcessed: {processed_count} emails",
            f"Total Downloads: {total_download_count} files",
            f"Unique Files: {unique_files_count} files",
            f"Skipped: {len(skipped_files)} files",
            f"Duplicates Found: {len(duplicate_files)} files",
            "\nDownloaded Files:",
            *[f"- {file}" for file in downloaded_files],
            "\nSkipped Files:",
            *[f"- {file}" for file in skipped_files],
            "\nDuplicate Files:",
            *[f"- {file}" for file in duplicate_files],
            f"\nFiles saved to: {SAVE_PATH}"
        ]

        return "\n".join(summary)

    except Exception as e:
        error_msg = f"Error in download process: {str(e)}"
        print(f"\nError: {error_msg}")
        return error_msg

def agent_main():
    """Main function to initialize and run the Gmail processing agent."""
    try:
        print("Initializing Gmail Processing Agent...")
        
        # Initialize the download tool
        download_tool = Tool(
            name="download_attachments",
            description="Downloads PDF attachments from Gmail that are related to business Enquiry or Quotation",
            func=download_gmail_attachments_with_logging
        )

        tools = [download_tool]

        # Prepare tool information
        tool_names = [tool.name for tool in tools]
        tool_descriptions = [f"{tool.name}: {tool.description}" for tool in tools]
        tools_str = "\n".join(tool_descriptions)

        # Create a simpler prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Gmail processing assistant that helps download relevant PDF attachments.
            Your task is to download PDF attachments from emails that are related to business Enquiry or Quotation.
            
            Available tools: {tools}
            Tool Names: {tool_names}
            
            To use a tool, respond with:
            ```
            Action: download_attachments
            Action Input: None
            ```
            """),
            ("human", "{input}"),
            ("ai", "{agent_scratchpad}")
        ])

        try:
            # Create the agent using retry-enabled function
            agent = create_agent(llm, tools, prompt)
            
            # Create the agent executor
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                max_iterations=1,
                early_stopping_method="force"
            )

            # Execute the agent using retry-enabled function
            response = execute_agent(agent_executor, tools_str, tool_names)
            result = response.get("output", "Email processing completed")
            
            print("\nFinal Status:")
            print("Status: success")
            print("Message:", result)
            
            return {
                "status": "success",
                "message": result
            }

        except Exception as e:
            error_msg = f"Error in agent execution: {str(e)}"
            print("\nFinal Status:")
            print("Status: error")
            print("Message:", error_msg)
            
            if "429" in str(e) or "rate limit" in str(e).lower():
                print("\nRate limit reached. Please try again later or:")
                print("1. Check your API quota in Google Cloud Console")
                print("2. Consider upgrading your API plan")
                print("3. Implement request rate limiting in your code")
            elif "401" in str(e) or "invalid" in str(e).lower():
                print("\nAPI key is invalid or expired. Please:")
                print("1. Check your API key in the .env file")
                print("2. Ensure the key is properly set in the environment")
                print("3. Generate a new API key if needed")
            
            return {
                "status": "error",
                "message": "Agent execution failed",
                "error": error_msg
            }

    except Exception as e:
        error_msg = f"Error in agent setup: {str(e)}"
        print("\nFinal Status:")
        print("Status: error")
        print("Message: Agent setup failed")
        print("Error Details:", error_msg)
        
        if "API key" in str(e):
            print("\nAPI key error. Please:")
            print("1. Check if GEMINI_API_KEY is set in your .env file")
            print("2. Verify the API key is valid")
            print("3. Generate a new API key if needed")
        
        return {
            "status": "error",
            "message": "Agent setup failed",
            "error": error_msg
        }

# if __name__ == "__main__":
#     # Execute the main function
#     try:
#         result = agent_main()
#         if result["status"] == "error":
#             print("\nError occurred during execution:")
#             print(result["error"])
#     except Exception as e:
#         print(f"\nUnexpected error: {str(e)}")



















