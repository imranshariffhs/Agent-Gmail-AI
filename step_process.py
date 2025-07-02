import json
import os
import re
import time

import pandas as pd
from langchain_google_genai import GoogleGenerativeAI

from classification_engine import classify_document_clean
from extract_pdf_data import process_pdf_image
from logger import logger
from pdf_process import (
    EXTRACTION_PROMPT,
    extract_json_from_response,
    format_field_definitions,
)

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
    file_path = file_path.replace("\\\\", "/")

    # Replace backslash with forward slash
    file_path = file_path.replace("\\", "/")

    return file_path


def initialize_llm():
    """Initialize the LLM model with configuration"""
    try:
        return GoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.7,
            top_p=0.98,
            top_k=20,
            max_output_tokens=8192,
        )
    except Exception as e:
        logger.error("Error initializing LLM: %s", str(e))
        raise


def get_schema_path(document_type, base_dir):
    """Get the appropriate schema path based on document type"""
    schema_mapping = {
        "Enquiry": "extruder_enquiry_fields.json",
        "Quotation": "extruder_quotation_fields.json",
    }

    # Clean and normalize document type
    doc_type = document_type.split(",")[0].strip()

    if doc_type not in schema_mapping:
        logger.warning("Warning: Unsupported document type: %s", doc_type)
        return None

    schema_file = schema_mapping[doc_type]
    schema_path = os.path.join(current_dir, "schemas", schema_file)

    if not os.path.exists(schema_path):
        logger.error("Error: Schema file not found: %s", schema_path)
        return None

    return schema_path


def _load_and_validate_schema(schema_path):
    try:
        with open(schema_path) as f:
            schema = json.load(f)
            if not schema or "fields" not in schema:
                raise ValueError("Invalid schema format")
            return schema["fields"]
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in schema file: {str(e)}") from e


def _extract_pdf_text(output_folder):
    output_text_path = os.path.join(output_folder, "output_all_pages.md")
    if not os.path.exists(output_text_path):
        raise ValueError(f"PDF text output not found: {output_text_path}")
    with open(output_text_path, encoding="utf-8") as file:
        pdf_text = file.read()
    if not pdf_text:
        raise ValueError("No text extracted from PDF")
    logger.info("Successfully read %d characters from file", len(pdf_text))
    return pdf_text


def _construct_prompt(fields, pdf_text):
    field_def_text = format_field_definitions(fields)
    if not field_def_text:
        raise ValueError("Failed to format field definitions")
    return EXTRACTION_PROMPT.format(field_definitions=field_def_text, pdf_text=pdf_text)


def _get_llm_response(prompt, llm):
    response = llm.invoke(prompt)
    if not response:
        raise ValueError("No response from LLM")
    extraction_json_data = extract_json_from_response(response)
    if not extraction_json_data:
        raise ValueError("Failed to extract JSON from LLM response")
    logger.info("Successfully extracted data from document")
    return extraction_json_data


def _save_result_json(output_folder, extraction_json_data):
    output_folder_name = os.path.basename(output_folder)
    result_path = os.path.join(path_result_json, f"{output_folder_name}.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(extraction_json_data, f, indent=2)
    return result_path


def process_pdf_document(pdf_path, llm):
    """Process a single PDF document and extract information"""
    try:
        output_folder = process_pdf_image(pdf_path)
        if not output_folder:
            raise ValueError("Failed to process PDF to images")
        document_type = classify_document_clean(output_folder)
        if not document_type:
            raise ValueError("Failed to classify document")
        schema_path = get_schema_path(document_type, os.path.dirname(current_dir))
        if not schema_path:
            raise ValueError(f"No schema available for document type: {document_type}")
        logger.info("Using schema: %s", schema_path)
        fields = _load_and_validate_schema(schema_path)
        pdf_text = _extract_pdf_text(output_folder)
        prompt = _construct_prompt(fields, pdf_text)
        extraction_json_data = _get_llm_response(prompt, llm)
        result_path = _save_result_json(output_folder, extraction_json_data)
        return result_path, "completed"
    except Exception as e:
        logger.error("Error processing document: %s", str(e))
        return None, f"error: {str(e)}"


def prepare_and_validate_path(file_path):
    file_path = clean_file_path(file_path)
    if not file_path:
        return None
    if not os.path.isabs(file_path):
        abs_path = os.path.join(path_log_fetch_pdf, file_path)
    else:
        abs_path = file_path
    abs_path = os.path.normpath(abs_path)
    if not os.path.exists(abs_path):
        logger.info("File not found: %s", abs_path)
        return None
    return abs_path


def handle_result_path(result_path, abs_path):
    if result_path:
        result_path = os.path.abspath(result_path)
        logger.info("Generated result path: %s", result_path)
        if os.path.exists(result_path):
            logger.info("JSON file generated successfully")
            return result_path, "completed"
        else:
            logger.warning("No result file found for %s", os.path.basename(abs_path))
            return result_path, "error"
    return "", "error"


def _initialize_llm_safe():
    try:
        return initialize_llm(), False
    except Exception as e:
        logger.error("Error initializing LLM: %s", str(e))
        return None, True


def _append_error(res_paths, res_statuses, markdown_statuses):
    res_paths.append("")
    res_statuses.append("error")
    markdown_statuses.append("pending")


def _process_files_loop(file_paths, llm):
    res_paths = []
    res_statuses = []
    markdown_statuses = []
    for file_path in file_paths:
        abs_path = prepare_and_validate_path(file_path)
        if not abs_path:
            _append_error(res_paths, res_statuses, markdown_statuses)
            continue
        try:
            result_path, status = process_pdf_document(abs_path, llm)
            result_path, status = handle_result_path(result_path, abs_path)
            res_paths.append(result_path)
            res_statuses.append(status)
            markdown_statuses.append("pending")
            if result_path and status == "completed":
                logger.info("Successfully processed file: %s", file_path)
                logger.info("Result saved to: %s", result_path)
        except Exception as e:
            logger.error("Error processing file %s: %s", file_path, str(e))
            _append_error(res_paths, res_statuses, markdown_statuses)
    return res_paths, res_statuses, markdown_statuses


def process_file_batch(file_paths):
    """Process a batch of files and update the log"""
    if not file_paths:
        return [], [], []
    llm, llm_error = _initialize_llm_safe()
    if llm_error:
        return ["" for _ in file_paths], ["error" for _ in file_paths], ["pending" for _ in file_paths]
    return _process_files_loop(file_paths, llm)


def update_excel_file(df, file_path, row_index=None):
    """Helper function to safely update the Excel file"""
    try:
        # Create a backup before saving
        backup_path = file_path.replace(".xlsx", f"_backup_{int(time.time())}.xlsx")

        # Ensure all required columns exist
        required_columns = ["res_path", "res_status", "markdown_status"]
        for col in required_columns:
            if col not in df.columns:
                df[col] = ""

        # Create backup
        df.to_excel(backup_path, index=False)

        # Save to the main file
        with pd.ExcelWriter(
            file_path,
            engine="openpyxl",
            mode="w",
            datetime_format="YYYY-MM-DD HH:MM:SS",
        ) as writer:
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


def _find_excel_file():
    excel_path = os.path.join(path_log_fetch_pdf, "email_download_log.xlsx")
    if os.path.exists(excel_path):
        return excel_path
    alt_excel_path = os.path.join(current_dir, "email_download_log.xlsx")
    if os.path.exists(alt_excel_path):
        logger.info(f"Found Excel file in current directory: {alt_excel_path}")
        return alt_excel_path
    logger.error(f"Excel log file not found at: {excel_path} or {alt_excel_path}")
    return None


def _read_and_validate_excel(excel_path):
    try:
        log_df = pd.read_excel(excel_path, dtype=str)
        logger.info(f"Successfully read Excel file with {len(log_df)} rows")
        logger.info(f"Columns found in Excel: {list(log_df.columns)}")
    except Exception as e:
        logger.error(f"Error reading Excel file: {str(e)}")
        return None, f"Failed to read Excel file: {str(e)}"
    if "file_paths" not in log_df.columns:
        logger.error("Required column 'file_paths' not found in Excel file")
        return None, "Required column 'file_paths' not found"
    return log_df, None


def _preprocess_dataframe(log_df):
    date_columns = [
        "first_inbox_msg",
        "last_check_date",
        "download_date",
        "duplicate_check_date",
    ]
    for col in date_columns:
        if col in log_df.columns:
            log_df[col] = pd.to_datetime(log_df[col], errors="coerce")
    numeric_columns = ["count_download", "list_name_count"]
    for col in numeric_columns:
        if col in log_df.columns:
            log_df[col] = pd.to_numeric(log_df[col], errors="coerce").fillna(0).astype(int)
    return log_df


def _extract_and_pad_statuses(updated_df, index, file_paths):
    def _get_and_pad(col_name, default):
        values = []
        if col_name in updated_df.columns and pd.notna(updated_df.loc[index, col_name]):
            values = [s.strip() for s in str(updated_df.loc[index, col_name]).split(",")]
        while len(values) < len(file_paths):
            values.append(default)
        return values

    current_res_statuses = _get_and_pad("res_status", "")
    current_res_paths = _get_and_pad("res_path", "")
    current_markdown_statuses = _get_and_pad("markdown_status", "pending")
    return current_res_statuses, current_res_paths, current_markdown_statuses


def _recombine_results(
    file_paths,
    current_res_statuses,
    current_res_paths,
    current_markdown_statuses,
    res_paths,
    res_statuses,
    markdown_statuses,
):
    final_res_paths = []
    final_res_statuses = []
    final_markdown_statuses = []
    process_counter = 0
    for i in range(len(file_paths)):
        if str(current_res_statuses[i]).lower() == "completed":
            final_res_paths.append(current_res_paths[i])
            final_res_statuses.append(current_res_statuses[i])
            final_markdown_statuses.append(current_markdown_statuses[i])
        else:
            final_res_paths.append(res_paths[process_counter] if process_counter < len(res_paths) else "")
            final_res_statuses.append(res_statuses[process_counter] if process_counter < len(res_statuses) else "error")
            final_markdown_statuses.append(
                markdown_statuses[process_counter] if process_counter < len(markdown_statuses) else "pending"
            )
            process_counter += 1
    return final_res_paths, final_res_statuses, final_markdown_statuses


def _update_row_in_df(updated_df, index, final_res_paths, final_res_statuses, final_markdown_statuses, excel_path):
    res_paths_str = ",".join(str(path) if path else "" for path in final_res_paths)
    res_statuses_str = ",".join(str(status) if status else "error" for status in final_res_statuses)
    markdown_statuses_str = ",".join(str(status) if status else "pending" for status in final_markdown_statuses)
    updated_df.loc[index, "res_path"] = res_paths_str
    updated_df.loc[index, "res_status"] = res_statuses_str
    updated_df.loc[index, "markdown_status"] = markdown_statuses_str
    logger.info(f"\nUpdating row {index}:")
    logger.info(f"Result paths: {res_paths_str}")
    logger.info(f"Status: {res_statuses_str}")
    logger.info(f"Markdown status: {markdown_statuses_str}")
    if not update_excel_file(updated_df, excel_path, index):
        logger.warning(f"Warning: Failed to save changes for row {index}")


def _process_rows(updated_df, excel_path):
    changes_made = False
    for index, row in updated_df.iterrows():
        if pd.notna(row["file_paths"]):
            try:
                raw_paths = str(row["file_paths"])
                file_paths = [path.strip() for path in re.split(r",(?![^(]*\))", raw_paths)]
                current_res_statuses, current_res_paths, current_markdown_statuses = _extract_and_pad_statuses(
                    updated_df, index, file_paths
                )
                files_to_process = []
                for i, status in enumerate(current_res_statuses):
                    if str(status).lower() != "completed":
                        files_to_process.append(file_paths[i])
                if files_to_process:
                    res_paths, res_statuses, markdown_statuses = process_file_batch(files_to_process)
                else:
                    res_paths, res_statuses, markdown_statuses = [], [], []
                final_res_paths, final_res_statuses, final_markdown_statuses = _recombine_results(
                    file_paths,
                    current_res_statuses,
                    current_res_paths,
                    current_markdown_statuses,
                    res_paths,
                    res_statuses,
                    markdown_statuses,
                )
                _update_row_in_df(
                    updated_df, index, final_res_paths, final_res_statuses, final_markdown_statuses, excel_path
                )
                changes_made = True
            except Exception as row_error:
                logger.error(f"Error processing row {index}: {str(row_error)}")
                continue
    return changes_made, updated_df


def _final_save(changes_made, updated_df, excel_path):
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


def main():
    """Main function to process documents from Excel log"""
    try:
        logger.info("Starting step process")
        logger.info(f"Current directory: {current_dir}")
        logger.info(f"Looking for Excel file in: {path_log_fetch_pdf}")
        excel_path = _find_excel_file()
        if not excel_path:
            return {"status": "error", "message": "Excel log file not found"}
        log_df, error = _read_and_validate_excel(excel_path)
        if error:
            return {"status": "error", "message": error}
        files_to_process = log_df["file_paths"].notna().sum()
        if files_to_process == 0:
            logger.info("No files found to process in the Excel log")
            return {"status": "success", "message": "No files to process"}
        else:
            logger.info(f"Found {files_to_process} rows with files to process")
            sample_paths = log_df.loc[log_df["file_paths"].notna(), "file_paths"].head()
            logger.info("Sample file paths to process:")
            for idx, path in enumerate(sample_paths):
                logger.info(f"Row {idx}: {path}")
        log_df = _preprocess_dataframe(log_df)
        updated_df = log_df.copy()
        changes_made, updated_df = _process_rows(updated_df, excel_path)
        return _final_save(changes_made, updated_df, excel_path)
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        return {"status": "error", "message": str(e)}


# if __name__ == "__main__":
#     main()
