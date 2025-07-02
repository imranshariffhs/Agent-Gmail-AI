"""Gmail Agent for Processing Business Documents

This module handles downloading and processing of business-related PDF attachments from Gmail,
with enhanced duplicate detection and logging capabilities.
"""

import asyncio
import atexit
import base64
import hashlib
import os
import re
import signal
from datetime import datetime, timedelta
from pathlib import Path

import grpc.aio
import pandas as pd
import requests
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from langchain.agents import create_react_agent
from langchain_community.tools.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)
from langchain_core.prompts import (
    PromptTemplate,
)
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from logger import logger

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
        logger.error("Error getting event loop: %s", e)
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
                logger.warning("Warning: Error during gRPC shutdown: %s", e)

            try:
                # Close the event loop
                cleanup_loop.close()
            except Exception as e:
                logger.warning("Warning: Error closing event loop: %s", e)

            _cleanup_done = True
    except Exception as e:
        logger.warning("Warning: Error during cleanup: %s", e)

# Register cleanup handlers
atexit.register(cleanup_resources)
signal.signal(signal.SIGINT, lambda s, f: cleanup_resources())

# Initialize event loop
_event_loop = get_event_loop()

# Initialize gRPC for async operations
try:
    grpc.aio.init_grpc_aio()
except Exception as e:
    logger.warning("Warning: Error initializing gRPC: %s", e)

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
            logger.warning("Warning: API rate limit reached. The script will implement retry logic.")
        elif response.status_code != 200:
            raise ValueError(f"API key validation failed with status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to validate API key: {str(e)}") from e

# Validate API key before proceeding
try:
    validate_gemini_api_key(api_key)
except ValueError as e:
    logger.error("Error: %s", str(e))
    logger.info("Please update your API key in the .env file and try again.")
    exit(1)

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
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
    retry=retry_if_exception_type(Exception),
)
def create_llm():
    """Create LLM instance with retry logic."""
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.7,
        top_p=0.95,
        max_output_tokens=2048,
        model_kwargs={"retry_on_failure": True},
    )


# Initialize LLM with retry logic
try:
    llm = create_llm()
except Exception as e:
    logger.error("Failed to initialize LLM after retries: %s", str(e))
    logger.info("Please check your API key and try again later.")
    exit(1)


# Create email classification prompt template
email_classification_prompt = PromptTemplate(
    input_variables=["subject", "body"],
    template="""
    Analyze the following email subject and body to determine if it is SPECIFICALLY related to an 
    Enquiry or Quotation.
    
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
    """,
)


# Create classification chain with retry logic
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def classify_email(subject, body):
    """Classify email with retry logic for rate limits."""
    try:
        result = classification_chain.invoke({"subject": subject, "body": body})
        return result.strip().upper()
    except Exception as e:
        logger.warning("Error in email classification: %s", str(e))
        raise


# Create classification chain using the new pattern
def get_classification_response(inputs):
    """Get classification response and extract the content."""
    response = llm.invoke(email_classification_prompt.format(**inputs))
    if hasattr(response, "content"):
        return response.content.strip().upper()
    elif isinstance(response, str):
        return response.strip().upper()
    else:
        logger.warning("Unexpected response type: %s", type(response))
        return "UNKNOWN"


classification_chain = RunnablePassthrough() | get_classification_response


# Define agent creation and execution functions
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception),
)
def create_agent(llm, tools, prompt):
    """Create agent with retry logic."""
    try:
        return create_react_agent(llm=llm, tools=tools, prompt=prompt)
    except Exception as e:
        logger.error("Error creating agent: %s", str(e))
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception),
)
def execute_agent(agent_executor, tools_str, tool_names):
    """Execute agent with retry logic."""
    try:
        return agent_executor.invoke(
            {
                "input": "Download PDF attachments from Gmail that are related to business Enquiry or Quotation.",
                "agent_scratchpad": [],
                "tools": tools_str,
                "tool_names": ", ".join(tool_names),
            }
        )
    except Exception as e:
        logger.error("Error executing agent: %s", str(e))
        raise


def _get_body_from_part(part):
    if part.get("mimeType") == "text/plain":
        data = part.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data).decode("utf-8")
    elif part.get("mimeType") == "text/html":
        data = part.get("body", {}).get("data", "")
        if data:
            html_text = base64.urlsafe_b64decode(data).decode("utf-8")
            text = re.sub("<[^<]+?>", "", html_text)
            return text
    return None


def _process_parts(parts, body_parts):
    for part in parts:
        if "parts" in part:
            _process_parts(part["parts"], body_parts)
        body = _get_body_from_part(part)
        if body:
            body_parts.append(body)


def extract_email_body(msg_data):
    """Extract email body from the message data."""
    body_parts = []
    if msg_data["payload"].get("mimeType") == "text/plain":
        data = msg_data["payload"].get("body", {}).get("data", "")
        if data:
            body_parts.append(base64.urlsafe_b64decode(data).decode("utf-8"))
    if "parts" in msg_data["payload"]:
        _process_parts(msg_data["payload"]["parts"], body_parts)
    return "\n".join(body_parts)


def get_gmail_service():
    """Initialize Gmail service with credentials."""
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    return build("gmail", "v1", credentials=creds)


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
            df = pd.DataFrame(
                {
                    # Document Identity
                    "subject": pd.Series(dtype="str"),
                    "email_id": pd.Series(dtype="str"),
                    "thread_id": pd.Series(dtype="str"),
                    "sender": pd.Series(dtype="str"),
                    # Processing Timeline
                    "first_inbox_msg": pd.Series(dtype="datetime64[ns]"),
                    "last_check_date": pd.Series(dtype="datetime64[ns]"),
                    "download_date": pd.Series(dtype="datetime64[ns]"),
                    "duplicate_check_date": pd.Series(dtype="datetime64[ns]"),
                    # File Management
                    "count_download": pd.Series(dtype="int64"),
                    "list_name_count": pd.Series(dtype="int64"),
                    "attachment_names": pd.Series(dtype="str"),
                    "file_paths": pd.Series(dtype="str"),
                    "original_filenames": pd.Series(dtype="str"),
                    "res_path": pd.Series(dtype="str"),
                    "markdown_status": pd.Series(dtype="str"),
                    # Data Integrity
                    "message_hash": pd.Series(dtype="str"),
                    "file_hashes": pd.Series(dtype="str"),
                    "unique_file_ids": pd.Series(dtype="str"),
                    # Processing Status
                    "process_status": pd.Series(dtype="str"),
                    "classification": pd.Series(dtype="str"),
                    "duplicate_status": pd.Series(dtype="str"),
                    "res_status": pd.Series(dtype="str"),
                }
            )

            # Ensure the directory exists
            LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

            # Save with proper date formatting
            with pd.ExcelWriter(LOG_FILE, engine="openpyxl", datetime_format="YYYY-MM-DD HH:MM:SS") as writer:
                df.to_excel(writer, index=False)
            logger.info("Created new structured log file: %s", LOG_FILE)
            return df
        else:
            # Load existing file
            df = pd.read_excel(
                LOG_FILE,
                parse_dates=[
                    "first_inbox_msg",
                    "last_check_date",
                    "download_date",
                    "duplicate_check_date",
                ],
            )
            return df
    except Exception as e:
        logger.error("Error initializing log file: %s", e)
        # Create empty DataFrame with proper structure as fallback
        return pd.DataFrame(
            columns=[
                "subject",
                "email_id",
                "thread_id",
                "sender",
                "first_inbox_msg",
                "last_check_date",
                "download_date",
                "duplicate_check_date",
                "count_download",
                "list_name_count",
                "attachment_names",
                "file_paths",
                "original_filenames",
                "res_path",
                "message_hash",
                "file_hashes",
                "unique_file_ids",
                "process_status",
                "classification",
                "duplicate_status",
                "res_status",
            ]
        )


def load_log_data():
    """Load existing log data from Excel file."""
    try:
        if LOG_FILE.exists():
            df = pd.read_excel(LOG_FILE)
            if "file_hash" not in df.columns:
                df["file_hash"] = ""
            if "unique_file_id" not in df.columns:
                df["unique_file_id"] = ""
            return df
        else:
            return initialize_log_file()
    except Exception as e:
        logger.error("Error loading log data: %s", e)
        return pd.DataFrame()


def save_log_data(df):
    """Save log data to Excel file."""
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(LOG_FILE, index=False)
        logger.info("Log data saved to %s", LOG_FILE)
    except Exception as e:
        logger.error("Error saving log data: %s", e)


def generate_message_hash(msg_data):
    """Generate unique hash for message to detect duplicates."""
    headers = msg_data["payload"].get("headers", [])
    subject = next((h["value"] for h in headers if h["name"] == "Subject"), "")
    message_id = msg_data.get("id", "")
    thread_id = msg_data.get("threadId", "")
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
    exact_matches = log_df[
        log_df["file_hashes"].apply(lambda x: file_hash in str(x).split(",") if pd.notna(x) else False)
    ]
    if not exact_matches.empty:
        match = exact_matches.iloc[0]
        return True, (
            f"Identical file content already exists (downloaded on {match['download_date']} "
            f"from {match['sender']}, subject: {match['subject']})"
        )

    # Check for same file in the same email thread
    if thread_id:
        thread_matches = log_df[
            (log_df["thread_id"] == thread_id)
            & (log_df["attachment_names"].apply(lambda x: filename in str(x).split(",") if pd.notna(x) else False))
        ]
        if not thread_matches.empty:
            match = thread_matches.iloc[0]
            return True, (
                f"Same filename already downloaded in this email thread "
                f"(original download: {match['download_date']}, subject: {match['subject']})"
            )

    # Check for similar filenames with different content
    similar_names = log_df[
        log_df["attachment_names"].apply(lambda x: filename in str(x).split(",") if pd.notna(x) else False)
    ]
    if not similar_names.empty:
        return (
            True,
            f"File with same name already exists (even with different content): {filename}",
        )

    # Check if this exact combination of file and email has been processed
    unique_matches = log_df[
        log_df["unique_file_ids"].apply(lambda x: unique_file_id in str(x).split(",") if pd.notna(x) else False)
    ]
    if not unique_matches.empty:
        match = unique_matches.iloc[0]
        return True, (
            f"This exact file from this email has already been processed (original download: {match['download_date']})"
        )

    return False, "File is new"


def should_process_email(subject, body, has_pdf_attachment):
    """Determine if an email should be processed based on content analysis."""
    if not has_pdf_attachment:
        return False, {
            "decision": "SKIP",
            "reason": "No PDF attachment present",
            "classification": "UNRELATED",
            "confidence": 0.0,
        }

    try:
        classification = classify_email(subject, body)

        # Strict classification check
        if classification == "ENQUIRY" or classification == "QUOTATION":
            return True, {
                "decision": "PROCESS",
                "reason": f"Email classified as {classification}",
                "classification": classification,
                "confidence": 1.0,
            }

        # Any other classification results in skip
        return False, {
            "decision": "SKIP",
            "reason": "Email not related to enquiry or quotation",
            "classification": "UNRELATED",
            "confidence": 0.0,
        }
    except Exception as e:
        logger.warning("Warning: Classification failed - %s", str(e))
        return False, {
            "decision": "SKIP",
            "reason": "Classification failed - skipping download",
            "classification": "UNRELATED",
            "confidence": 0.0,
        }


def _normalize_list_fields_in_new_row(new_row):
    for field in [
        "attachment_names",
        "file_paths",
        "file_hashes",
        "unique_file_ids",
        "original_filenames",
    ]:
        if isinstance(new_row.get(field), list):
            new_row[field] = ",".join(str(item) for item in new_row[field])
    return new_row


def _initialize_result_processing_fields(new_row):
    if "res_status" not in new_row:
        if "attachment_names" in new_row and new_row["attachment_names"]:
            pdf_count = len(str(new_row["attachment_names"]).split(","))
            new_row["res_status"] = ",".join(["pending"] * pdf_count)
        else:
            new_row["res_status"] = "pending"
    if "res_path" not in new_row:
        if "attachment_names" in new_row and new_row["attachment_names"]:
            pdf_count = len(str(new_row["attachment_names"]).split(","))
            new_row["res_path"] = ",".join([""] * pdf_count)
        else:
            new_row["res_path"] = ""
    return new_row


def _merge_existing_email_counts_and_info(existing_email, new_row):
    existing_count = existing_email["count_download"].iloc[0] if "count_download" in existing_email else 0
    existing_names = set()
    if "original_filenames" in existing_email and pd.notna(existing_email["original_filenames"].iloc[0]):
        existing_names = set(str(existing_email["original_filenames"].iloc[0]).split(","))
    new_names = set()
    if "original_filenames" in new_row:
        if isinstance(new_row["original_filenames"], list):
            new_names = set(str(name) for name in new_row["original_filenames"])
        elif isinstance(new_row["original_filenames"], str):
            new_names = set(new_row["original_filenames"].split(","))
    new_row["count_download"] = existing_count + len(new_names)
    new_row["list_name_count"] = len(existing_names.union(new_names))
    new_row["first_inbox_msg"] = existing_email.iloc[0]["first_inbox_msg"]
    for field in [
        "file_paths",
        "file_hashes",
        "unique_file_ids",
        "original_filenames",
    ]:
        if field in existing_email and pd.notna(existing_email[field].iloc[0]):
            existing_values = set(str(existing_email[field].iloc[0]).split(","))
            new_values = set(str(new_row.get(field, "")).split(","))
            new_row[field] = ",".join(sorted(existing_values.union(new_values)))
    return new_row


def _init_new_email_counts_and_info(new_row, current_time):
    if "original_filenames" in new_row:
        if isinstance(new_row["original_filenames"], list):
            new_row["count_download"] = len(new_row["original_filenames"])
            new_row["list_name_count"] = len(set(new_row["original_filenames"]))
        elif isinstance(new_row["original_filenames"], str):
            filenames = new_row["original_filenames"].split(",")
            new_row["count_download"] = len(filenames)
            new_row["list_name_count"] = len(set(filenames))
        else:
            new_row["count_download"] = 0
            new_row["list_name_count"] = 0
    new_row["first_inbox_msg"] = current_time
    return new_row


def _update_counts_and_merge_info(log_df, new_row, current_time):
    if "email_id" in new_row:
        existing_email = log_df[log_df["email_id"] == new_row["email_id"]]
        if not existing_email.empty:
            new_row = _merge_existing_email_counts_and_info(existing_email, new_row)
        else:
            new_row = _init_new_email_counts_and_info(new_row, current_time)
    return new_row


def _convert_date_strings_to_datetime(new_row, current_time):
    for date_field in [
        "first_inbox_msg",
        "last_check_date",
        "download_date",
        "duplicate_check_date",
    ]:
        if date_field in new_row and isinstance(new_row[date_field], str):
            try:
                new_row[date_field] = pd.to_datetime(new_row[date_field])
            except Exception:
                new_row[date_field] = pd.to_datetime(current_time)
    return new_row


def _remove_old_entry_and_update_df(log_df, new_row):
    if "email_id" in new_row:
        log_df = log_df[log_df["email_id"] != new_row["email_id"]]
    log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)
    return log_df


def _save_log_df_to_excel(log_df):
    with pd.ExcelWriter(LOG_FILE, engine="openpyxl", datetime_format="YYYY-MM-DD HH:MM:SS") as writer:
        log_df.to_excel(writer, index=False)


def update_download_log(log_df, new_row):
    """Update the download log with new entry and save."""
    try:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Normalize list fields
        new_row = _normalize_list_fields_in_new_row(new_row)
        # Add duplicate check information
        new_row["duplicate_check_date"] = current_time
        new_row["duplicate_status"] = "unique"
        new_row["download_date"] = current_time
        # Initialize result processing fields
        new_row = _initialize_result_processing_fields(new_row)
        # Update tracking fields
        new_row["last_check_date"] = current_time
        # Update counts and merge info
        new_row = _update_counts_and_merge_info(log_df, new_row, current_time)
        # Convert date strings to datetime
        new_row = _convert_date_strings_to_datetime(new_row, current_time)
        # Remove old entry and update DataFrame
        log_df = _remove_old_entry_and_update_df(log_df, new_row)
        # Save to Excel
        _save_log_df_to_excel(log_df)
        return log_df
    except Exception as e:
        logger.error("Error updating download log: %s", e)
        return log_df


def _extract_email_metadata(msg_data):
    headers = msg_data["payload"].get("headers", [])
    subject = next((h["value"] for h in headers if h["name"] == "Subject"), "No Subject")
    sender = next((h["value"] for h in headers if h["name"] == "From"), "Unknown Sender")
    date_header = next((h["value"] for h in headers if h["name"] == "Date"), "")
    thread_id = msg_data.get("threadId", "")
    return subject, sender, date_header, thread_id


def _monitor_classify_and_thread_check(subject, body, thread_id, log_df):
    classification = classify_email(subject, body)
    if classification not in ["ENQUIRY", "QUOTATION"]:
        return classification, False, f"Not a valid enquiry or quotation email (got {classification})"
    if thread_id:
        thread_entries = log_df[log_df["thread_id"] == thread_id]
        if not thread_entries.empty:
            existing_entry = thread_entries.iloc[0]
            if existing_entry["classification"] == classification:
                return classification, False, "Thread already processed with same classification"
    return classification, True, None


def _extract_pdf_attachments_monitor(msg_data):
    parts = msg_data["payload"].get("parts", [])
    if not parts and msg_data["payload"].get("filename"):
        parts = [msg_data["payload"]]
    pdf_parts = [part for part in parts if part.get("filename", "").lower().endswith(".pdf")]
    return pdf_parts, parts


def _monitor_process_single_pdf_attachment(part, service, msg, log_df, classification, thread_id):
    filename = part.get("filename", "")
    if not filename or "body" not in part or "attachmentId" not in part["body"]:
        return None
    try:
        attachment = (
            service.users()
            .messages()
            .attachments()
            .get(
                userId="me",
                messageId=msg["id"],
                id=part["body"]["attachmentId"],
            )
            .execute()
        )
        file_data = base64.urlsafe_b64decode(attachment["data"])
        is_duplicate, reason = is_file_already_downloaded(log_df, filename, file_data, msg["id"], thread_id)
        if is_duplicate:
            return (filename, None, reason)
        base_name, ext = os.path.splitext(filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_filename = f"{classification.lower()}_{timestamp}_{filename}"
        file_path = SAVE_PATH / final_filename
        counter = 1
        while file_path.exists():
            final_filename = f"{classification.lower()}_{timestamp}_{base_name}_{counter}{ext}"
            file_path = SAVE_PATH / final_filename
            counter += 1
        file_path.write_bytes(file_data)
        file_hash = generate_file_hash(file_data)
        unique_file_id = generate_unique_file_id(filename, file_hash, msg["id"])
        return (filename, final_filename, file_path, file_hash, unique_file_id)
    except Exception as e:
        return (filename, None, str(e))


def _monitor_build_log_row(
    subject,
    sender,
    date_header,
    thread_id,
    msg,
    msg_data,
    successfully_downloaded,
    file_paths,
    file_hashes,
    unique_file_ids,
    classification,
    current_time,
):
    return {
        "subject": subject,
        "first_inbox_msg": date_header,
        "last_check_date": current_time,
        "download_date": current_time,
        "count_download": len(successfully_downloaded),
        "list_name_count": ", ".join(successfully_downloaded),
        "email_id": msg["id"],
        "thread_id": thread_id,
        "sender": sender,
        "attachment_names": ", ".join(successfully_downloaded),
        "file_paths": ", ".join(file_paths),
        "message_hash": generate_message_hash(msg_data),
        "file_hashes": ", ".join(file_hashes),
        "unique_file_ids": ", ".join(unique_file_ids),
        "process_status": "downloaded",
        "classification": classification,
        "res_status": "pending",
        "res_path": "",
        "duplicate_check_date": current_time,
        "duplicate_status": "unique",
    }


def _process_single_email_for_monitor(msg, service, log_df):
    try:
        msg_data = service.users().messages().get(userId="me", id=msg["id"]).execute()
        subject, sender, date_header, thread_id = _extract_email_metadata(msg_data)
        body = extract_email_body(msg_data)
        classification, should_process, skip_reason = _monitor_classify_and_thread_check(
            subject, body, thread_id, log_df
        )
        if not should_process:
            return {
                "skipped": [f"'{subject}' from {sender} - {skip_reason}"],
                "skip_details": [f"  • {subject}: {skip_reason}"],
                "downloaded": [],
                "processed_emails": [],
                "log_df": log_df,
            }
        pdf_parts, parts = _extract_pdf_attachments_monitor(msg_data)
        if not pdf_parts:
            skip_reason = "No PDF attachments found"
            return {
                "skipped": [f"'{subject}' from {sender} - {skip_reason}"],
                "skip_details": [f"  • {subject}: {skip_reason}"],
                "downloaded": [],
                "processed_emails": [],
                "log_df": log_df,
            }
        successfully_downloaded = []
        file_paths = []
        file_hashes = []
        unique_file_ids = []
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        skipped = []
        skip_details = []
        downloaded = []
        for part in parts:
            if part.get("filename", "") in [p.get("filename", "") for p in pdf_parts]:
                result = _monitor_process_single_pdf_attachment(part, service, msg, log_df, classification, thread_id)
                if result is None:
                    continue
                if len(result) == 3 and result[1] is None:
                    skipped.append(result[0])
                    skip_details.append(f"  • {result[0]}: {result[2]}")
                    continue
                filename, final_filename, file_path, file_hash, unique_file_id = result
                successfully_downloaded.append(final_filename)
                file_paths.append(str(file_path))
                file_hashes.append(file_hash)
                unique_file_ids.append(unique_file_id)
                downloaded.append(f"{filename} -> {final_filename}")
        processed_emails = []
        if successfully_downloaded:
            new_row = _monitor_build_log_row(
                subject,
                sender,
                date_header,
                thread_id,
                msg,
                msg_data,
                successfully_downloaded,
                file_paths,
                file_hashes,
                unique_file_ids,
                classification,
                current_time,
            )
            log_df = update_download_log(log_df, new_row)
            processed_emails.append(f"📧 {sender}: {subject}")
        return {
            "skipped": skipped,
            "skip_details": skip_details,
            "downloaded": downloaded,
            "processed_emails": processed_emails,
            "log_df": log_df,
        }
    except Exception:
        return {"skipped": [], "skip_details": [], "downloaded": [], "processed_emails": [], "log_df": log_df}


def monitor_gmail_for_new_attachments_with_logging() -> str:
    """
    Monitor Gmail for new emails with document attachments in the last 24 hours
    with enhanced duplicate prevention.
    """
    try:
        # Initialize logging
        initialize_log_file()
        log_df = load_log_data()
        service = get_gmail_service()
        yesterday = datetime.now() - timedelta(days=1)
        query = f"has:attachment in:inbox -from:me after:{yesterday.strftime('%Y/%m/%d')}"
        logger.info("Searching for emails with query: %s", query)
        results = service.users().messages().list(userId="me", q=query).execute()
        messages = results.get("messages", [])
        if not messages:
            return "📭 No new emails with attachments found in the last 24 hours."
        downloaded = []
        skipped = []
        processed_emails = []
        skip_details = []
        for msg in messages:
            result = _process_single_email_for_monitor(msg, service, log_df)
            downloaded.extend(result["downloaded"])
            skipped.extend(result["skipped"])
            skip_details.extend(result["skip_details"])
            processed_emails.extend(result["processed_emails"])
            log_df = result["log_df"]
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
            summary.append("\n📨 Processed emails:")
            for email in processed_emails:
                summary.append(f"   {email}")
        summary.append(f"\n📊 Log file: {LOG_FILE}")
        summary.append(f"💾 Save path: {SAVE_PATH}")
        return "\n".join(summary) if downloaded or skipped else "📭 No new document attachments found in recent emails."
    except Exception as e:
        error_msg = f"❌ Error monitoring emails: {str(e)}"
        logger.error(error_msg)
        return error_msg


# --- Download helpers ---
def _search_gmail_for_pdf_emails(service, date_limit):
    query = f"has:attachment in:inbox -from:me after:{date_limit}"
    logger.info("Searching for emails with query: %s", query)
    try:
        results = service.users().messages().list(userId="me", q=query).execute()
        return results.get("messages", [])
    except Exception as e:
        logger.error("Error searching emails: %s", str(e))
        return None


def _classify_and_check_thread(msg_data, log_df):
    headers = msg_data["payload"].get("headers", [])
    subject = next((h["value"] for h in headers if h["name"] == "Subject"), "No Subject")
    sender = next((h["value"] for h in headers if h["name"] == "From"), "Unknown Sender")
    date_header = next((h["value"] for h in headers if h["name"] == "Date"), "")
    thread_id = msg_data.get("threadId", "")
    body = extract_email_body(msg_data)
    classification = classify_email(subject, body)
    # Thread check
    if thread_id:
        thread_entries = log_df[log_df["thread_id"] == thread_id]
        if not thread_entries.empty:
            existing_entry = thread_entries.iloc[0]
            if existing_entry["classification"] == classification:
                return (
                    subject,
                    sender,
                    date_header,
                    thread_id,
                    classification,
                    False,
                    "Thread already processed with same classification",
                )
    return subject, sender, date_header, thread_id, classification, True, None


def _extract_pdf_parts(msg_data):
    parts = msg_data["payload"].get("parts", [])
    if not parts:
        parts = [msg_data["payload"]] if msg_data["payload"].get("filename") else []
    pdf_parts = [part for part in parts if part.get("filename", "").lower().endswith(".pdf")]
    return pdf_parts


def _process_pdf_attachment(part, service, msg, log_df, classification, thread_id):
    filename = part.get("filename")
    if not filename:
        return None
    logger.info("Processing attachment: %s", filename)
    if "body" not in part or "attachmentId" not in part["body"]:
        logger.info("Skipping %s: Invalid attachment data", filename)
        return None
    try:
        attachment = (
            service.users()
            .messages()
            .attachments()
            .get(
                userId="me",
                messageId=msg["id"],
                id=part["body"]["attachmentId"],
            )
            .execute()
        )
        file_data = base64.urlsafe_b64decode(attachment["data"])
        is_duplicate, reason = is_file_already_downloaded(log_df, filename, file_data, msg["id"], thread_id)
        if is_duplicate:
            logger.info("Skipping duplicate: %s", reason)
            return (filename, None, reason)
        # Generate unique filename with classification prefix
        base_name, ext = os.path.splitext(filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_filename = f"{classification.lower()}_{timestamp}_{filename}"
        file_path = SAVE_PATH / final_filename
        counter = 1
        while file_path.exists():
            final_filename = f"{classification.lower()}_{timestamp}_{base_name}_{counter}{ext}"
            file_path = SAVE_PATH / final_filename
            counter += 1
        file_path.write_bytes(file_data)
        file_hash = generate_file_hash(file_data)
        unique_file_id = generate_unique_file_id(filename, file_hash, msg["id"])
        logger.info("Successfully downloaded: %s", final_filename)
        return (filename, final_filename, file_path, file_hash, unique_file_id)
    except Exception as e:
        logger.error("Error downloading %s: %s", filename, e)
        return (filename, None, str(e))


def _generate_download_summary(downloaded, skipped, skip_details, processed_emails, LOG_FILE, SAVE_PATH):
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
        summary.append("\n📨 Processed emails:")
        for email in processed_emails:
            summary.append(f"   {email}")
    summary.append(f"\n📊 Log file: {LOG_FILE}")
    summary.append(f"💾 Save path: {SAVE_PATH}")
    return "\n".join(summary)


# --- Helper for processing a single email ---
def _process_single_email_for_download(msg, service, log_df):
    try:
        msg_data = service.users().messages().get(userId="me", id=msg["id"]).execute()
        subject, sender, date_header, thread_id, classification, should_process, skip_reason = (
            _classify_and_check_thread(msg_data, log_df)
        )
        if not should_process:
            return {
                "skipped": [f"'{subject}' - {skip_reason}"],
                "skip_details": [f"  • {subject}: {skip_reason}"],
                "downloaded": [],
                "processed_emails": [],
                "log_df": log_df,
            }
        if classification not in ["ENQUIRY", "QUOTATION"]:
            skip_reason = f"Not a valid enquiry or quotation email (got {classification})"
            return {
                "skipped": [f"'{subject}' from {sender} - {skip_reason}"],
                "skip_details": [f"  • {subject}: {skip_reason}"],
                "downloaded": [],
                "processed_emails": [],
                "log_df": log_df,
            }
        logger.info("Email classified as: %s", classification)
        pdf_parts = _extract_pdf_parts(msg_data)
        if not pdf_parts:
            skip_reason = "No PDF attachments found"
            return {
                "skipped": [f"'{subject}' from {sender} - {skip_reason}"],
                "skip_details": [f"  • {subject}: {skip_reason}"],
                "downloaded": [],
                "processed_emails": [],
                "log_df": log_df,
            }
        successfully_downloaded = []
        file_paths = []
        file_hashes = []
        unique_file_ids = []
        message_hash = generate_message_hash(msg_data)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        skipped = []
        skip_details = []
        downloaded = []
        for part in pdf_parts:
            result = _process_pdf_attachment(part, service, msg, log_df, classification, thread_id)
            if result is None:
                continue
            if len(result) == 3 and result[1] is None:
                # duplicate or error
                skipped.append(result[0])
                skip_details.append(f"  • {result[0]}: {result[2]}")
                continue
            filename, final_filename, file_path, file_hash, unique_file_id = result
            successfully_downloaded.append(final_filename)
            file_paths.append(str(file_path))
            file_hashes.append(file_hash)
            unique_file_ids.append(unique_file_id)
            downloaded.append(f"{filename} -> {final_filename}")
        processed_emails = []
        if successfully_downloaded:
            res_status_list = ["pending"] * len(successfully_downloaded)
            res_path_list = [""] * len(successfully_downloaded)
            new_row = {
                "subject": subject,
                "first_inbox_msg": date_header,
                "last_check_date": current_time,
                "download_date": current_time,
                "count_download": len(successfully_downloaded),
                "list_name_count": ",".join(successfully_downloaded),
                "email_id": msg["id"],
                "thread_id": thread_id,
                "sender": sender,
                "attachment_names": ",".join(successfully_downloaded),
                "file_paths": ",".join(file_paths),
                "message_hash": message_hash,
                "file_hashes": ",".join(file_hashes),
                "unique_file_ids": ",".join(unique_file_ids),
                "process_status": "downloaded",
                "classification": classification,
                "res_status": ",".join(res_status_list),
                "res_path": ",".join(res_path_list),
                "duplicate_check_date": current_time,
                "duplicate_status": "unique",
            }
            log_df = update_download_log(log_df, new_row)
            processed_emails.append(f"📧 {sender}: {subject}")
        return {
            "skipped": skipped,
            "skip_details": skip_details,
            "downloaded": downloaded,
            "processed_emails": processed_emails,
            "log_df": log_df,
        }
    except Exception as e:
        logger.error("Error processing message: %s", e)
        return {"skipped": [], "skip_details": [], "downloaded": [], "processed_emails": [], "log_df": log_df}


# --- Main orchestrator ---
def download_gmail_attachments_with_logging() -> str:
    """Download PDF attachments from business-related Enquiry or Quotation emails only."""
    try:
        logger.info("Starting email attachment download process...")
        initialize_log_file()
        log_df = load_log_data()
        service = get_gmail_service()
        date_limit = (datetime.now() - timedelta(days=1)).strftime("%Y/%m/%d")
        messages = _search_gmail_for_pdf_emails(service, date_limit)
        if messages is None:
            return "❌ Failed to search emails. Please check your connection and try again."
        if not messages:
            return "📭 No new emails with PDF attachments found in the last 30 days."
        downloaded = []
        skipped = []
        processed_emails = []
        skip_details = []
        for msg in messages:
            result = _process_single_email_for_download(msg, service, log_df)
            downloaded.extend(result["downloaded"])
            skipped.extend(result["skipped"])
            skip_details.extend(result["skip_details"])
            processed_emails.extend(result["processed_emails"])
            log_df = result["log_df"]
        return _generate_download_summary(downloaded, skipped, skip_details, processed_emails, LOG_FILE, SAVE_PATH)
    except Exception as e:
        error_msg = f"❌ Error in download process: {str(e)}"
        logger.error(error_msg)
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
        unique_senders = log_df["sender"].nunique() if "sender" in log_df.columns else 0
        total_files = log_df["count_download"].sum() if "count_download" in log_df.columns else 0
        unique_threads = log_df["thread_id"].nunique() if "thread_id" in log_df.columns else 0

        # File hash statistics (if available)
        unique_files_by_content = 0
        if "file_hash" in log_df.columns:
            all_hashes = []
            for _, row in log_df.iterrows():
                if pd.notna(row["file_hash"]) and row["file_hash"]:
                    hashes = str(row["file_hash"]).split(", ")
                    all_hashes.extend(hashes)
            unique_files_by_content = len(set(all_hashes)) if all_hashes else 0

        # Recent downloads (last 7 days)
        if "download_date" in log_df.columns:
            log_df["download_date"] = pd.to_datetime(log_df["download_date"], errors="coerce")
            recent = log_df[log_df["download_date"] > (datetime.now() - timedelta(days=7))]
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
                subject = row.get("subject", "N/A")
                sender = row.get("sender", "N/A")
                files = row.get("list_name_count", "N/A")
                thread_id = row.get("thread_id", "N/A")[:8] + "..." if row.get("thread_id") else "N/A"
                summary.append(f"   • {subject[:40]}... from {sender}")
                summary.append(f"     Files: {files} | Thread: {thread_id}")

        return "\n".join(summary)

    except Exception as e:
        return f"❌ Error reading log: {str(e)}"


# --- Duplicate removal helpers ---
def _remove_duplicates_by_file_hashes(log_df):
    if "file_hashes" in log_df.columns:
        log_df["hash_set"] = log_df["file_hashes"].apply(lambda x: set(str(x).split(",")) if pd.notna(x) else set())
        seen_hashes = set()
        duplicate_indices = []
        for idx, row in log_df.iterrows():
            current_hashes = row["hash_set"]
            if not current_hashes:
                continue
            if current_hashes.issubset(seen_hashes):
                duplicate_indices.append(idx)
            else:
                seen_hashes.update(current_hashes)
        log_df = log_df.drop(duplicate_indices)
        log_df = log_df.drop("hash_set", axis=1)
    return log_df


def _remove_duplicates_by_thread_and_filenames(log_df):
    if "thread_id" in log_df.columns and "original_filenames" in log_df.columns:
        thread_groups = log_df.groupby("thread_id")
        duplicate_indices = []
        for _, group in thread_groups:
            if len(group) <= 1:
                continue
            filename_sets = group["original_filenames"].apply(
                lambda x: set(str(x).split(",")) if pd.notna(x) else set()
            )
            seen_files = set()
            for idx, files in filename_sets.items():
                if not files:
                    continue
                if files.issubset(seen_files):
                    duplicate_indices.append(idx)
                else:
                    seen_files.update(files)
        log_df = log_df.drop(duplicate_indices)
    return log_df


def _remove_duplicates_by_unique_file_ids(log_df):
    if "unique_file_ids" in log_df.columns:
        log_df["id_set"] = log_df["unique_file_ids"].apply(lambda x: set(str(x).split(",")) if pd.notna(x) else set())
        seen_ids = set()
        duplicate_indices = []
        for idx, row in log_df.iterrows():
            current_ids = row["id_set"]
            if not current_ids:
                continue
            if current_ids.issubset(seen_ids):
                duplicate_indices.append(idx)
            else:
                seen_ids.update(current_ids)
        log_df = log_df.drop(duplicate_indices)
        log_df = log_df.drop("id_set", axis=1)
    return log_df


# --- Main orchestrator ---
def clear_duplicate_entries() -> str:
    """Clean up duplicate entries in the log file based on file content."""
    try:
        if not LOG_FILE.exists():
            return "📄 No log file found."
        log_df = pd.read_excel(
            LOG_FILE,
            parse_dates=[
                "first_inbox_msg",
                "last_check_date",
                "download_date",
                "duplicate_check_date",
            ],
        )
        if log_df.empty:
            return "📄 Log file is empty."
        original_count = len(log_df)
        log_df = _remove_duplicates_by_file_hashes(log_df)
        log_df = _remove_duplicates_by_thread_and_filenames(log_df)
        log_df = _remove_duplicates_by_unique_file_ids(log_df)
        cleaned_count = len(log_df)
        removed_count = original_count - cleaned_count
        if removed_count > 0:
            with pd.ExcelWriter(LOG_FILE, engine="openpyxl", datetime_format="YYYY-MM-DD HH:MM:SS") as writer:
                log_df.to_excel(writer, index=False)
            return f"✅ Cleaned log file: Removed {removed_count} duplicate entries. {cleaned_count} entries remain."
        else:
            return "✅ No duplicate entries found in log file."
    except Exception as e:
        return f"❌ Error cleaning log: {str(e)}"


def update_processing_status(file_name: str, new_status: str) -> str:
    """
    Update the processing status for a specific PDF file in the log.

    Args:
        file_name (str): Name of the PDF file to update
        new_status (str): New status to set ('pending', 'processing', 'completed', 'failed')

    Returns:
        str: Message indicating the result of the update
    """
    try:
        if not LOG_FILE.exists():
            return "❌ Log file not found"

        # Read the Excel file
        log_df = pd.read_excel(LOG_FILE)

        # Find rows where this file exists
        for idx, row in log_df.iterrows():
            if pd.isna(row["list_name_count"]) or pd.isna(row["res_status"]):
                continue

            file_names = str(row["list_name_count"]).split(",")
            current_statuses = str(row["res_status"]).split(",")

            # Ensure lists have same length
            if len(file_names) != len(current_statuses):
                logger.warning(f"Mismatch in file names and statuses for row {idx}")
                continue

            # Find the file in the list
            try:
                file_index = file_names.index(file_name.strip())
                # Update the status for this file
                current_statuses[file_index] = new_status
                # Join back into comma-separated string
                log_df.at[idx, "res_status"] = ",".join(current_statuses)

                # Save the updated DataFrame
                with pd.ExcelWriter(LOG_FILE, engine="openpyxl", datetime_format="YYYY-MM-DD HH:MM:SS") as writer:
                    log_df.to_excel(writer, index=False)

                return f"✅ Updated status of '{file_name}' to '{new_status}'"
            except ValueError:
                continue

        return f"❌ File '{file_name}' not found in log"

    except Exception as e:
        logger.error(f"Error updating processing status: {str(e)}")
        return f"❌ Error: {str(e)}"


def get_processing_status(file_name: str = None) -> str:
    """
    Get the processing status of files in the log.

    Args:
        file_name (str, optional): Specific file to check. If None, returns status of all files.

    Returns:
        str: Status information
    """
    try:
        if not LOG_FILE.exists():
            return "❌ Log file not found"

        log_df = pd.read_excel(LOG_FILE)

        summary = []
        summary.append("📊 Processing Status Report")
        summary.append("=" * 40)

        for _idx, row in log_df.iterrows():
            if pd.isna(row["list_name_count"]) or pd.isna(row["res_status"]):
                continue

            file_names = str(row["list_name_count"]).split(",")
            current_statuses = str(row["res_status"]).split(",")

            if file_name:
                # Looking for specific file
                try:
                    file_index = file_names.index(file_name.strip())
                    status = current_statuses[file_index]
                    return f"Status of '{file_name}': {status}"
                except ValueError:
                    continue
            else:
                # Report all files in this row
                summary.append(f"\n📧 Email: {row.get('subject', 'Unknown Subject')}")
                summary.append(f"From: {row.get('sender', 'Unknown Sender')}")
                summary.append("Files:")

                for f_name, status in zip(file_names, current_statuses, strict=False):
                    status_emoji = {
                        "pending": "⏳",
                        "processing": "🔄",
                        "completed": "✅",
                        "failed": "❌",
                    }.get(status.lower(), "❓")

                    summary.append(f"{status_emoji} {f_name}: {status}")

        if file_name:
            return f"❌ File '{file_name}' not found in log"

        return "\n".join(summary)

    except Exception as e:
        logger.error(f"Error getting processing status: {str(e)}")
        return f"❌ Error: {str(e)}"


def _initialize_gmail_credentials():
    """Helper to initialize Gmail credentials and resource service."""
    try:
        credentials = get_gmail_credentials(
            token_file="token.json",
            scopes=SCOPES,
            client_secrets_file="credentials.json",
        )
        build_resource_service(credentials=credentials)
        logger.info("✅ Gmail credentials initialized successfully")
        return {"status": "success", "credentials": credentials}
    except Exception as e:
        error_msg = f"Failed to initialize Gmail credentials: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "error",
            "message": "Gmail initialization failed",
            "error": error_msg,
        }


def _run_monitor_step():
    logger.info("\n1. Monitoring Gmail for new attachments...")
    try:
        monitor_result = monitor_gmail_for_new_attachments_with_logging()
        logger.info("✅ Monitoring completed successfully")
        return ("Monitor", "success", monitor_result)
    except Exception as e:
        logger.error(f"❌ Monitoring failed: {str(e)}")
        return ("Monitor", "error", str(e))


def _run_download_step():
    logger.info("\n2. Downloading relevant PDF attachments...")
    try:
        download_result = download_gmail_attachments_with_logging()
        logger.info("✅ Download completed successfully")
        return ("Download", "success", download_result)
    except Exception as e:
        logger.error(f"❌ Download failed: {str(e)}")
        return ("Download", "error", str(e))


def _run_log_view_step():
    logger.info("\n3. Checking download log...")
    try:
        log_result = view_download_log()
        logger.info("✅ Log check completed successfully")
        return ("Log View", "success", log_result)
    except Exception as e:
        logger.error(f"❌ Log check failed: {str(e)}")
        return ("Log View", "error", str(e))


def _run_cleanup_step():
    logger.info("\n4. Cleaning up duplicate entries...")
    try:
        cleanup_result = clear_duplicate_entries()
        logger.info("✅ Cleanup completed successfully")
        return ("Cleanup", "success", cleanup_result)
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {str(e)}")
        return ("Cleanup", "error", str(e))


def _generate_final_report(results):
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
    return final_output, all_successful


def agent_main():
    """Main function to initialize and run the Gmail processing agent."""
    try:
        logger.info("Initializing Gmail Processing Agent...")
        # Step 1: Initialize Gmail credentials
        cred_result = _initialize_gmail_credentials()
        if cred_result["status"] != "success":
            return cred_result
        # Step 2: Run main steps and collect results
        results = [
            _run_monitor_step(),
            _run_download_step(),
            _run_log_view_step(),
            _run_cleanup_step(),
        ]
        # Step 3: Generate final report
        final_output, all_successful = _generate_final_report(results)
        return {
            "status": "success" if all_successful else "partial_success",
            "output": final_output,
            "error": None if all_successful else "Some steps failed, see output for details",
        }
    except Exception as e:
        error_msg = f"Error in setup: {str(e)}"
        logger.error(f"\n❌ {error_msg}")
        if "API key" in str(e):
            logger.info("\nAPI key error. Please:")
            logger.info("1. Check if GEMINI_API_KEY is set in your .env file")
            logger.info("2. Verify the API key is valid")
            logger.info("3. Generate a new API key if needed")
        return {"status": "error", "message": "Setup failed", "error": error_msg}


if __name__ == "__main__":
    try:
        # Run the main function
        result = agent_main()
        if result["status"] != "success":
            logger.error("\nExecution completed with issues:")
            logger.error(result["error"])
        else:
            logger.info("\nExecution completed successfully!")
    except Exception as e:
        logger.error(f"\nUnexpected error: {str(e)}")
    finally:
        cleanup_resources()
        logger.info("\nExecution finished.")
