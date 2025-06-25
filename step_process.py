from extract_pdf_data import process_pdf_image
from pdf_process import format_field_definitions, EXTRACTION_PROMPT, extract_json_from_response
from classification_engine import classify_document_clean
import pandas as pd
import os
import re
import json
from langchain_google_genai import GoogleGenerativeAI

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to backend directory and then to download_email
path_log_fetch_pdf = os.path.join(os.path.dirname(current_dir), "download_email")
path_result_json = os.path.join(os.path.dirname(current_dir), "Agent_AI", "result_json")

# Create result_json directory if it doesn't exist
os.makedirs(path_result_json, exist_ok=True)

def clean_file_path(file_path):
    """Clean and normalize file path"""
    # Remove any leading/trailing whitespace
    file_path = file_path.strip()
    
    # Remove any quotes if present
    file_path = file_path.strip('"').strip("'")
    
    # Replace any double backslashes with single forward slash
    file_path = file_path.replace('\\\\', '/')
    
    # Replace backslash with forward slash
    file_path = file_path.replace('\\', '/')
    
    return file_path

def initialize_llm():
    """Initialize the LLM model with configuration"""
    return GoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv('GEMINI_API_KEY'),
        temperature=0.7,
        top_p=0.98,
        top_k=20,
        max_output_tokens=8192
    )

def get_schema_path(document_type, base_dir):
    """Get the appropriate schema path based on document type"""
    schema_mapping = {
        'Enquiry': 'extruder_enquiry_fields.json',
        'Quotation': 'extruder_quotation_fields.json'
    }
    
    # Clean and normalize document type
    doc_type = document_type.split(',')[0].strip()
    
    if doc_type not in schema_mapping:
        print(f"Warning: Unsupported document type: {doc_type}")
        return None
        
    schema_file = schema_mapping[doc_type]
    schema_path = os.path.join(current_dir, 'schemas', schema_file)
    
    if not os.path.exists(schema_path):
        print(f"Error: Schema file not found: {schema_path}")
        return None
        
    return schema_path

def process_pdf_document(pdf_path, llm):
    """Process a single PDF document and extract information"""
    try:
        # Process PDF to images and get output folder
        output_folder = process_pdf_image(pdf_path)
        if not output_folder:
            raise ValueError("Failed to process PDF to images")
            
        # Classify document
        document_type = classify_document_clean(output_folder)
        if not document_type:
            raise ValueError("Failed to classify document")
            
        # Get schema path
        schema_path = get_schema_path(document_type, os.path.dirname(current_dir))
        if not schema_path:
            raise ValueError(f"No schema available for document type: {document_type}")
            
        print(f"Using schema: {schema_path}")
        
        # Extract text from PDF
        output_text_path = os.path.join(output_folder, 'output_all_pages.md')
        if not os.path.exists(output_text_path):
            raise ValueError(f"PDF text output not found: {output_text_path}")
            
        with open(output_text_path, 'r', encoding='utf-8') as file:
            pdf_text = file.read()
        
        if not pdf_text:
            raise ValueError("No text extracted from PDF")
            
        print(f"Successfully read {len(pdf_text)} characters from file")
        
        # Load and process schema
        try:
            with open(schema_path, 'r') as f:
                schema = json.load(f)
                if not schema or 'fields' not in schema:
                    raise ValueError("Invalid schema format")
                fields = schema['fields']
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in schema file: {str(e)}")
            
        # Format fields and create prompt
        field_def_text = format_field_definitions(fields)
        if not field_def_text:
            raise ValueError("Failed to format field definitions")
            
        prompt = EXTRACTION_PROMPT.format(
            field_definitions=field_def_text,
            pdf_text=pdf_text
        )
        
        # Get LLM response
        response = llm.invoke(prompt)
        if not response:
            raise ValueError("No response from LLM")
            
        extraction_json_data = extract_json_from_response(response)
        if not extraction_json_data:
            raise ValueError("Failed to extract JSON from LLM response")
            
        print(f"Successfully extracted data from document")
        
        # Save results
        output_folder_name = os.path.basename(output_folder)
        result_path = os.path.join(path_result_json, f"{output_folder_name}.json")
        
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        
        with open(result_path, 'w') as f:
            json.dump(extraction_json_data, f, indent=2)
            
        return result_path, 'success'
        
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        return None, f"error: {str(e)}"

def process_file_batch(file_paths):
    """Process a batch of files and update the log"""
    if not file_paths:
        return [], []
    
    # Initialize results lists
    res_paths = []
    res_statuses = []
    
    # Initialize LLM
    try:
        llm = initialize_llm()
    except Exception as e:
        print(f"Error initializing LLM: {str(e)}")
        return [''] * len(file_paths), ['error'] * len(file_paths)
    
    for file_path in file_paths:
        try:
            # Clean and validate file path
            file_path = clean_file_path(file_path)
            if not file_path:
                res_paths.append('')
                res_statuses.append('error')
                continue
                
            # Convert to absolute path if needed
            if not os.path.isabs(file_path):
                file_path = os.path.join(path_log_fetch_pdf, file_path)
            file_path = os.path.normpath(file_path)
            
            if os.path.exists(file_path):
                print(f"Processing: {file_path}")
                result_path, status = process_pdf_document(file_path, llm)
                res_paths.append(result_path if result_path else '')
                res_statuses.append(status)
            else:
                print(f"File not found: {file_path}")
                res_paths.append('')
                res_statuses.append('error')
                
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            res_paths.append('')
            res_statuses.append('error')
    
    return res_paths, res_statuses

def main():
    """Main function to process documents from Excel log"""
    try:
        # Read Excel file
        log_df = pd.read_excel(os.path.join(path_log_fetch_pdf, "email_download_log.xlsx"))
        
        # Process each row
        for index, row in log_df.iterrows():
            if pd.notna(row['file_paths']):
                # Split file paths and clean them
                raw_paths = str(row['file_paths'])
                file_paths = [path.strip() for path in re.split(r',(?![^(]*\))', raw_paths)]
                
                # Process batch of files
                res_paths, res_statuses = process_file_batch(file_paths)
                
                # Ensure the results are lists and have the same length as file_paths
                if not isinstance(res_paths, list):
                    res_paths = [res_paths]
                if not isinstance(res_statuses, list):
                    res_statuses = [res_statuses]
                
                # Convert results to string format for DataFrame storage
                res_paths_str = ','.join(str(path) if path else '' for path in res_paths)
                res_statuses_str = ','.join(str(status) if status else 'error' for status in res_statuses)
                
                # Check if columns exist and add them if they don't
                if 'res_path' not in log_df.columns:
                    log_df['res_path'] = ''
                if 'res_status' not in log_df.columns:
                    log_df['res_status'] = ''

                # Update DataFrame with string values
                log_df.at[index, 'res_path'] = res_paths_str
                log_df.at[index, 'res_status'] = res_statuses_str
        
        # Save updated log
        log_df.to_excel(os.path.join(path_log_fetch_pdf, 'email_download_log.xlsx'), index=False)
        print("Successfully processed all documents and updated the log.")
        
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        raise  # Re-raise the exception for debugging purposes

if __name__ == "__main__":
    main()

            

          







