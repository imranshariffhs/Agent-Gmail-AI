from extract_pdf_data import process_pdf_image
from pdf_process import format_field_definitions, EXTRACTION_PROMPT, extract_json_from_response
from classification_engine import classify_document_clean
import pandas as pd
import os
import re
import json
from langchain_google_genai import GoogleGenerativeAI
import time
from logger import logger

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to backend directory and then to download_email
path_log_fetch_pdf = os.path.join(current_dir, "download_email")
path_result_json = os.path.join(current_dir, "result_json")

# Create result_json directory if it doesn't exist
os.makedirs(path_result_json, exist_ok=True)
# Create download_email directory if it doesn't exist
os.makedirs(path_log_fetch_pdf, exist_ok=True)

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
    try:
        return GoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv('GEMINI_API_KEY'),
            temperature=0.7,
            top_p=0.98,
            top_k=20,
            max_output_tokens=8192
        )
    except Exception as e:
        logger.error("Error initializing LLM: %s", str(e))
        raise

def get_schema_path(document_type, base_dir):
    """Get the appropriate schema path based on document type"""
    schema_mapping = {
        'Enquiry': 'extruder_enquiry_fields.json',
        'Quotation': 'extruder_quotation_fields.json'
    }
    
    # Clean and normalize document type
    doc_type = document_type.split(',')[0].strip()
    
    if doc_type not in schema_mapping:
        logger.warning("Warning: Unsupported document type: %s", doc_type)
        return None
        
    schema_file = schema_mapping[doc_type]
    schema_path = os.path.join(current_dir, 'schemas', schema_file)
    
    if not os.path.exists(schema_path):
        logger.error("Error: Schema file not found: %s", schema_path)
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
            
        logger.info("Using schema: %s", schema_path)
        
        # Extract text from PDF
        output_text_path = os.path.join(output_folder, 'output_all_pages.md')
        if not os.path.exists(output_text_path):
            raise ValueError(f"PDF text output not found: {output_text_path}")
            
        with open(output_text_path, 'r', encoding='utf-8') as file:
            pdf_text = file.read()
        
        if not pdf_text:
            raise ValueError("No text extracted from PDF")
            
        logger.info("Successfully read %d characters from file", len(pdf_text))
        
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
            
        logger.info("Successfully extracted data from document")
        
        # Save results
        output_folder_name = os.path.basename(output_folder)
        result_path = os.path.join(path_result_json, f"{output_folder_name}.json")
        
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        
        with open(result_path, 'w') as f:
            json.dump(extraction_json_data, f, indent=2)
            
        # If we successfully saved the JSON file, return completed status
        return result_path, 'completed'
        
    except Exception as e:
        logger.error("Error processing document: %s", str(e))
        return None, f"error: {str(e)}"

def process_file_batch(file_paths):
    """Process a batch of files and update the log"""
    if not file_paths:
        return [], [], []
    
    # Initialize results lists
    res_paths = []
    res_statuses = []
    markdown_statuses = []
    
    # Initialize LLM
    try:
        llm = initialize_llm()
    except Exception as e:
        logger.error("Error initializing LLM: %s", str(e))
        return [''] * len(file_paths), ['error'] * len(file_paths), ['pending'] * len(file_paths)
    
    for file_path in file_paths:
        try:
            # Clean and validate file path
            file_path = clean_file_path(file_path)
            if not file_path:
                res_paths.append('')
                res_statuses.append('error')
                markdown_statuses.append('pending')
                continue
                
            # Convert to absolute path if needed
            if not os.path.isabs(file_path):
                file_path = os.path.join(path_log_fetch_pdf, file_path)
            file_path = os.path.normpath(file_path)
            
            if os.path.exists(file_path):
                logger.info("Processing: %s", file_path)
                result_path, status = process_pdf_document(file_path, llm)
                
                # Ensure result_path is absolute
                if result_path:
                    result_path = os.path.abspath(result_path)
                    logger.info("Generated result path: %s", result_path)
                    
                    # Check if JSON file exists
                    if os.path.exists(result_path):
                        logger.info("JSON file generated successfully")
                        status = 'completed'  # Ensure status is completed when JSON exists
                    else:
                        logger.warning("No result file found for %s", os.path.basename(file_path))
                        status = 'error'
                
                res_paths.append(result_path if result_path else '')
                res_statuses.append(status if status else 'error')
                markdown_statuses.append('pending')
                
                if result_path:
                    logger.info("Successfully processed file: %s", file_path)
                    logger.info("Result saved to: %s", result_path)
            else:
                logger.info("File not found: %s", file_path)
                res_paths.append('')
                res_statuses.append('error')
                markdown_statuses.append('pending')
                
        except Exception as e:
            logger.error("Error processing file %s: %s", file_path, str(e))
            res_paths.append('')
            res_statuses.append('error')
            markdown_statuses.append('pending')
    
    # Log the results
    logger.info('II'*30)
    logger.info("res_paths: %s", res_paths)
    logger.info("res_statuses: %s", res_statuses)
    logger.info("markdown_statuses: %s", markdown_statuses)
    logger.info('II'*30)
    
    return res_paths, res_statuses, markdown_statuses

def update_excel_file(df, file_path, row_index=None):
    """Helper function to safely update the Excel file"""
    try:
        # Create a backup before saving
        backup_path = file_path.replace('.xlsx', f'_backup_{int(time.time())}.xlsx')
        
        # Ensure all required columns exist
        required_columns = ['res_path', 'res_status', 'markdown_status']
        for col in required_columns:
            if col not in df.columns:
                df[col] = ''
        
        # Create backup
        df.to_excel(backup_path, index=False)
        
        # Save to the main file
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='w', datetime_format='YYYY-MM-DD HH:MM:SS') as writer:
            df.to_excel(writer, index=False)
        
        # Verify the save was successful by reading back the file
        try:
            verification_df = pd.read_excel(file_path)
            if not all(col in verification_df.columns for col in required_columns):
                raise ValueError("Required columns missing after save")
        except Exception as verify_error:
            logger.error("Save verification failed: %s", str(verify_error))
            if os.path.exists(backup_path):
                os.replace(backup_path, file_path)  # Restore from backup
                logger.info("Restored from backup due to verification failure")
            return False
        
        # Remove backup if save was successful and verified
        if os.path.exists(backup_path):
            os.remove(backup_path)
            
        if row_index is not None:
            logger.info("Successfully updated row %d in Excel file", row_index)
            logger.info("Updated fields: res_path, res_status, markdown_status")
        else:
            logger.info("Successfully updated Excel file with all required fields")
            
        return True
    except Exception as e:
        logger.error("Error saving to Excel: %s", str(e))
        if os.path.exists(backup_path):
            logger.info("Backup file preserved at: %s", backup_path)
        return False

def main():
    """Main function to process documents from Excel log"""
    try:
        logger.info("Starting step process")
        logger.info(f"Current directory: {current_dir}")
        logger.info(f"Looking for Excel file in: {path_log_fetch_pdf}")
        
        excel_path = os.path.join(path_log_fetch_pdf, "email_download_log.xlsx")
        
        # Check if Excel file exists
        if not os.path.exists(excel_path):
            logger.error(f"Excel log file not found at: {excel_path}")
            # Try to find the Excel file in the current directory
            alt_excel_path = os.path.join(current_dir, "email_download_log.xlsx")
            if os.path.exists(alt_excel_path):
                excel_path = alt_excel_path
                logger.info(f"Found Excel file in current directory: {excel_path}")
            else:
                return {"status": "error", "message": f"Excel log file not found at: {excel_path} or {alt_excel_path}"}

        # Read Excel file with all columns as string type initially
        try:
            log_df = pd.read_excel(excel_path, dtype=str)
            logger.info(f"Successfully read Excel file with {len(log_df)} rows")
            logger.info(f"Columns found in Excel: {list(log_df.columns)}")
        except Exception as e:
            logger.error(f"Error reading Excel file: {str(e)}")
            return {"status": "error", "message": f"Failed to read Excel file: {str(e)}"}

        # Check if file_paths column exists
        if 'file_paths' not in log_df.columns:
            logger.error("Required column 'file_paths' not found in Excel file")
            return {"status": "error", "message": "Required column 'file_paths' not found"}

        # Check if there are any files to process
        files_to_process = log_df['file_paths'].notna().sum()
        if files_to_process == 0:
            logger.info("No files found to process in the Excel log")
            return {"status": "success", "message": "No files to process"}
        else:
            logger.info(f"Found {files_to_process} rows with files to process")
            # Print the first few file paths for debugging
            sample_paths = log_df.loc[log_df['file_paths'].notna(), 'file_paths'].head()
            logger.info("Sample file paths to process:")
            for idx, path in enumerate(sample_paths):
                logger.info(f"Row {idx}: {path}")
        
        # Convert date columns to datetime
        date_columns = ['first_inbox_msg', 'last_check_date', 'download_date', 'duplicate_check_date']
        for col in date_columns:
            if col in log_df.columns:
                log_df[col] = pd.to_datetime(log_df[col], errors='coerce')
        
        # Convert numeric columns
        numeric_columns = ['count_download', 'list_name_count']
        for col in numeric_columns:
            if col in log_df.columns:
                log_df[col] = pd.to_numeric(log_df[col], errors='coerce').fillna(0).astype(int)
        
        # Create a copy of the DataFrame to track changes
        updated_df = log_df.copy()
        changes_made = False
        
        # Process each row
        for index, row in updated_df.iterrows():
            if pd.notna(row['file_paths']):
                try:
                    # Split file paths and clean them
                    raw_paths = str(row['file_paths'])
                    file_paths = [path.strip() for path in re.split(r',(?![^(]*\))', raw_paths)]
                    
                    # Log the files being processed
                    logger.info(f"Processing files for row {index}: {file_paths}")
                    
                    # Process batch of files
                    res_paths, res_statuses, markdown_statuses = process_file_batch(file_paths)
                    
                    # Ensure results are lists
                    res_paths = [res_paths] if not isinstance(res_paths, list) else res_paths
                    res_statuses = [res_statuses] if not isinstance(res_statuses, list) else res_statuses
                    markdown_statuses = [markdown_statuses] if not isinstance(markdown_statuses, list) else markdown_statuses
                    
                    # Convert results to string format
                    res_paths_str = ','.join(str(path) if path else '' for path in res_paths)
                    res_statuses_str = ','.join(str(status) if status else 'error' for status in res_statuses)
                    markdown_statuses_str = ','.join(str(status) if status else 'pending' for status in markdown_statuses)
                    
                    # Update the DataFrame
                    updated_df.loc[index, 'res_path'] = res_paths_str
                    updated_df.loc[index, 'res_status'] = res_statuses_str
                    updated_df.loc[index, 'markdown_status'] = markdown_statuses_str
                    changes_made = True
                    
                    # Print update information
                    logger.info(f"\nUpdating row {index}:")
                    logger.info(f"Result paths: {res_paths_str}")
                    logger.info(f"Status: {res_statuses_str}")
                    logger.info(f"Markdown status: {markdown_statuses_str}")
                    
                    # Try to save after each update
                    if not update_excel_file(updated_df, excel_path, index):
                        logger.warning(f"Warning: Failed to save changes for row {index}")
                    
                except Exception as row_error:
                    logger.error(f"Error processing row {index}: {str(row_error)}")
                    continue
        
        # Final save if any changes were made
        if changes_made:
            if update_excel_file(updated_df, excel_path):
                logger.info("\nFinal save successful")
                logger.info("Progress saved successfully")
                logger.info(f"Excel file updated at: {excel_path}")
                return {"status": "success", "message": "All documents processed successfully"}
            else:
                return {"status": "error", "message": "Failed to save final updates"}
        else:
            logger.info("No changes were necessary in the log file.")
            return {"status": "success", "message": "No updates required"}
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    main()
