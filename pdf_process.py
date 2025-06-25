from langchain_google_genai import GoogleGenerativeAI

import os
from dotenv import load_dotenv
import json
import logging
import warnings
from cryptography.utils import CryptographyDeprecationWarning


# Filter out the cryptography deprecation warning
warnings.filterwarnings('ignore', category=CryptographyDeprecationWarning)

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LangChain configuration
google_api_key = os.getenv('GEMINI_API_KEY')
if not google_api_key:
    logger.warning("GEMINI_API_KEY not found in environment. Document processing will be limited.")
    llm = None
else:
    llm = GoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=google_api_key,
        temperature=0.7,
        top_p=0.98,
        top_k=20,
        max_output_tokens=8192
    )

# Create a PromptTemplate for extraction
EXTRACTION_PROMPT = """You are an expert data extraction assistant. Your task is to extract data from PDF documents and return it in JSON format.

FIELD DEFINITIONS:
{field_definitions}

DOCUMENT CONTENT:
{pdf_text}

INSTRUCTIONS:
1. Extract data for each field defined above
2. Return ONLY a JSON object - no other text or formatting
3. Use null for missing or empty values
4. Use exact field names from the definitions

RESPONSE FORMAT:
Return only a JSON object like this (using actual field names from definitions):
{{
    "actual_field_name1": "value1",
    "actual_field_name2": null,
    "actual_field_name3": ["value1", "value2"]
}}
"""

def format_field_definitions(field_definitions):
    """Format field definitions for the prompt"""
    field_def_text = "\n"
    for i, field in enumerate(field_definitions, 1):
        name = field.get('name', '').lower()
        description = field.get('description', '')
        field_def_text += f"        {i}. {name}: Location: {description}\n"
        field_def_text += "           If not marked, return null\n\n"
    return field_def_text.strip()

def extract_json_from_response(response_text):
    """Extract JSON from response text with improved error handling"""
    try:
        # If already a dict, return as is
        if isinstance(response_text, dict):
            return response_text

        # Convert response to string if needed
        if not isinstance(response_text, str):
            response_text = str(response_text)

        # Clean up the text
        text = response_text.strip()
        
        try:
            # Try direct JSON parsing first
            return json.loads(text)
        except:
            # Find the first { and last }
            start = text.find('{')
            end = text.rfind('}')
            
            if start != -1 and end != -1:
                # Extract just the JSON part
                json_text = text[start:end+1]
                
                # Clean up common issues
                json_text = json_text.replace('\n', ' ')
                json_text = json_text.replace('\r', ' ')
                json_text = ' '.join(json_text.split())  # Normalize whitespace
                json_text = json_text.replace("'", '"')  # Replace single quotes
                
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON: {e}")
                    logger.error(f"Attempted to parse: {json_text}")
                    return {}
            
            logger.error("No JSON object found in response")
            logger.error(f"Response text: {text}")
            return {}

    except Exception as e:
        logger.error(f"Error in extract_json_from_response: {str(e)}")
        logger.error(f"Response text: {response_text}")
        return {}