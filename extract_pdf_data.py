import base64
import hashlib
import json
import os
import time
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pdf2image import convert_from_path

from logger import logger

# ---------- 1. Configuration ----------

# Load environment variables
load_dotenv()

# Check for API key and provide helpful error message if missing
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    # Try to load from parent directory's .env file
    parent_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(parent_env_path):
        from dotenv import load_dotenv

        load_dotenv(parent_env_path)
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set. "
        "Please create a .env file in the backend directory with: GEMINI_API_KEY=your-api-key-here"
    )

# Set Google API key from Gemini key
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# ---------- 2. Enhanced PDF to Image Conversion ----------


def pdf_to_image(upload_path):
    """
    Convert an uploaded PDF to images.
    - Validates the path.
    - Creates output folder under /image/{filename}/
    - Saves one PNG per page.
    Returns:
    - String: Path to the folder containing the images and markdown file
    """
    try:
        # logger.info("Converting PDF to images: %s", upload_path)

        # Debug: Print current working directory
        cwd = os.getcwd()
        # logger.info("Current working directory: %s", cwd)

        # Normalize paths
        upload_path = os.path.abspath(os.path.normpath(upload_path))
        # logger.info("Normalized upload path: %s", upload_path)

        # Get the base filename without extension
        filename = os.path.splitext(os.path.basename(upload_path))[0]
        # logger.info("Base filename: %s", filename)

        # Create output folder path
        output_folder = os.path.join("image", filename)
        abs_output_folder = os.path.abspath(output_folder)
        # logger.info("Creating output folder at: %s", abs_output_folder)

        # Create output folder
        os.makedirs(abs_output_folder, exist_ok=True)
        # logger.info("Created output folder at: %s", abs_output_folder)

        # logger.info("📄 Converting PDF to images: %s", upload_path)

        # Convert PDF to images
        images = convert_from_path(upload_path, dpi=300)
        # logger.info("Converted %d pages", len(images))

        if not images:
            raise ValueError("❌ No pages found in PDF")

        # Save images
        for i, img in enumerate(images):
            output_path = os.path.join(abs_output_folder, f"page_{i + 1}.png")
            img.save(output_path, "PNG")
            # logger.info("✅ Saved page %d to: %s", i + 1, output_path)

        # Create markdown file
        output_md_path = os.path.join(abs_output_folder, "output_all_pages.md")
        # logger.info("Creating markdown file at: %s", output_md_path)

        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write(f"# PDF Processing Results\n\nProcessing {filename}\n\n")

        logger.info("✅ Created markdown file at: %s", output_md_path)
        logger.info("✅ All files saved in folder: %s", abs_output_folder)

        # Return absolute folder path
        # logger.info("Returning folder path: %s", abs_output_folder)
        return abs_output_folder

    except Exception as e:
        logger.error("Error during conversion: %s", str(e))
        raise


# ---------- 3. Enhanced Gemini Extraction with Retry Logic ----------


def load_image_bytes(image_path):
    """Load image with validation"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if os.path.getsize(image_path) == 0:
        raise ValueError(f"Image file is empty: {image_path}")

    try:
        with open(image_path, "rb") as img_file:
            image_data = img_file.read()
            if not image_data:
                raise ValueError(f"No data read from image: {image_path}")
            return base64.b64encode(image_data).decode("utf-8")
    except Exception as e:
        logger.error("Error loading image %s: %s", image_path, str(e))
        raise


def extract_image_to_markdown(image_path, max_retries=3):
    """Extract with retry logic and content validation"""

    # Enhanced prompt specifically for radio button and checkbox detection
    # enhanced_prompt = """
    # Extract ALL content from this image and format it in clean, structured Markdown.

    # IMPORTANT: Do not skip any text, fields, or form elements. Extract everything visible.

    # CRITICAL: Pay special attention to radio buttons, checkboxes, and form selections:

    # **For CHECKBOXES:**
    # - When a checkbox is CHECKED/FILLED/SELECTED: Format as `select[X] Option Text`
    # - When a checkbox is UNCHECKED/EMPTY: Format as `unselect[] Option Text`
    # - Look carefully for visual indicators like checkmarks, X marks, filled squares, or darker boxes
    # - Include ALL checkbox options (both checked and unchecked) with their proper formatting

    # **For RADIO BUTTONS:**
    # - ONLY extract the option that is SELECTED/FILLED/CHECKED
    # - Look for filled circles (●), dots, or darker/highlighted options
    # - If a radio button is empty/unfilled/unchecked, do NOT include it in the output
    # - Format as: **Question:** Selected Answer Only

    # **For TEXT FIELDS and OTHER CONTENT:**
    # - Extract ALL visible text including headers, labels, instructions
    # - Include any handwritten or filled-in text
    # - Preserve table structures, lists, and formatting
    # - Include field labels even if fields are empty
    # - Extract signatures, dates, and any other visible content

    # **For other form elements:**
    # - Dropdown selections: Include the selected value if visible
    # - Text areas: Include any filled-in text
    # - Preserve the exact text of all options and content

    # Format guidelines:
    # - Use clear headings for sections
    # - For form fields, use format: **Field Name:** Value (or empty if blank)
    # - For checkbox groups, list all options with proper select[X]/unselect[] formatting
    # - Maintain the logical structure and flow of the document
    # - Include page numbers, headers, footers if present

    # EXTRACT EVERYTHING - missing content is worse than including too much.
    # """

    enhanced_prompt = """
You are an expert AI assistant specialized in extracting data from images of manufacturing industry quotation and 
enquiry forms, and converting it into a clean, well-formatted Markdown file.
Input: You will receive an image of a form. This form may contain:
Tables (with or without clearly defined borders)
Checkboxes (checked or unchecked)
Radio buttons (selected or unselected)
Input fields (filled or blank)
Text fields (containing single-line or multi-line text)
Handwritten text (if present, try to interpret it and flag if uncertain)
Varying fonts and font sizes
Noise, shadows, or slight distortions
Task:
Text Extraction: Accurately extract all text elements from the image, using OCR or other appropriate methods. 
Pay close attention to detail to ensure that no information is missed. Correct any OCR errors, especially in
 technical terms, part numbers, or industry-specific vocabulary.
Data Interpretation: Analyze the extracted text to identify the different form elements (tables, checkboxes,
 radio buttons, fields, etc.) and their relationships. Understand the form's structure to correctly interpret 
 the data.
Markdown Conversion: Convert the extracted data into a clean, well-formatted Markdown file. Follow these 
specific formatting
! important follow this - Do not put ```markdown and ``` at starting and ending of markdown data
guidelines:
Headings: Use appropriate Markdown headings ( ##, ###, etc.) to structure the document and clearly 
identify different sections of the form (e.g., "Company Details", "Product Specifications", "Contact Information"). 
Maintain the hierarchy of the form in the headings.

Lists: Use Markdown lists (-, *, 1., etc.) to represent lists of items.
Checkboxes and Radio Buttons: Represent checkboxes and radio buttons using the following format:
Checked: [x]
Unchecked: [ ]
Radio buttons:
- Use (•) for selected option
- Use ( ) for unselected option
Enclose these in bullet points within a section describing the options. For example:
Generated markdown
**Material Options:**
- [x] Steel
- [ ] Aluminum
- [ ] Plastic
Use code with caution.
 
Tables: Convert tables into Markdown tables using | to separate columns and --- to create 
the header row separator. Ensure that the table is properly aligned and readable. If the 
table has no visible borders, infer the structure from the data.

Generated markdown
| Header 1 | Header 2 | Header 3 |
|---|---|---|
| Data 1 | Data 2 | Data 3 |
| Data 4 | Data 5 | Data 6 |
Use code with caution.
 
Input Fields and Text Fields: Represent input fields and text fields with the field label 
followed by the extracted text (if any) or a placeholder (e.g., ______) if the field is blank. 
If the label is missing, infer it from the surrounding context. For multi-line text fields, 
preserve the line breaks in the Markdown output.

Generated markdown
Name: John Doe
Email: john.doe@example.com
Comments:
This is a multi-line comment.
It spans several lines.
 
 
Important Notes: Display important notes or disclaimers in a distinct way, using blockquotes or bold text.
Generated markdown
> **Note:** All prices are in USD and do not include shipping.
 
 
Uncertainty Flagging: If you are uncertain about any extracted data 
(e.g., due to poor image quality or handwriting), add a comment in the Markdown file indicating the uncertainty. 
Use the following format:

Generated markdown
<!-- Possible OCR error: "Part Number: AB1234?" - Please verify. -->
 
 
Original Form Layout: While the focus is on structured data, try to preserve the original form's 
layout as much as possible in the Markdown structure to improve readability.
Cleanliness: Ensure that the final Markdown file is clean, well-formatted, and easy to read. 
Remove any unnecessary characters or formatting.
 
Constraints:
You should not add any information that is not present in the image.
Focus on accuracy and completeness.
Prioritize readability in the Markdown output.
Do not put ```markdown and ``` at starting and ending of markdown data ! important follow this
at last I need to combine multiple md files together , there will be a issue with this approach
Output:
A complete Markdown file containing all the extracted data from the image, 
formatted according to the guidelines above."""

    for attempt in range(max_retries):
        try:
            # logger.info(
            #     "🔄 Attempt %d/%d for %s",
            #     attempt + 1,
            #     max_retries,
            #     os.path.basename(image_path),
            # )

            image_base64 = load_image_bytes(image_path)

            response = llm.invoke(
                [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": enhanced_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                            },
                        ]
                    )
                ]
            )

            content = response.content

            # Validate response content
            if not content or len(content.strip()) < 10:
                raise ValueError(f"Response too short or empty: {len(content) if content else 0} characters")

            # Check for common extraction failures
            if "unable to" in content.lower() or "cannot extract" in content.lower():
                raise ValueError("Extraction failed - model reported inability to process")

            # logger.info("✅ Successfully extracted %d characters", len(content))
            return content

        except Exception as e:
            logger.warning("⚠️ Attempt %d failed: %s", attempt + 1, str(e))
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # Progressive backoff
                # logger.info("⏳ Waiting %d seconds before retry...", wait_time)
                time.sleep(wait_time)
            else:
                logger.error("❌ All attempts failed for %s", image_path)
                # Return a placeholder to prevent complete data loss
                return (
                    f"# Error Processing {os.path.basename(image_path)}\n\n"
                    f"*Failed to extract content after {max_retries} attempts.*\n\n"
                    f"**Error:** {str(e)}\n"
                )


# ---------- 5. Enhanced Main Process with Data Loss Prevention ----------


def save_progress(all_markdown, output_file):
    """
    Save processing progress with enhanced error handling and metadata tracking.

    Args:
        all_markdown (list): List of markdown content to save
        output_file (str): Target file path for saving progress

    Returns:
        tuple: (success: bool, metadata: dict)
            - success: True if save operation was successful
            - metadata: Dictionary containing:
                - timestamp: Save operation timestamp
                - file_size: Size of saved content
                - checksum: Content checksum
                - status: Processing status
    """
    try:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)

        # Generate metadata
        content = "".join(all_markdown)
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "file_size": len(content),
            "checksum": hashlib.md5(content.encode()).hexdigest(),
            "status": "completed",
        }

        # Save content
        with open(output_file, "w", encoding="utf-8") as f:
            f.writelines(all_markdown)

        # Save metadata alongside content
        metadata_file = output_file + ".meta"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        # logger.info("✅ Progress saved successfully to %s", output_file)
        # logger.info("📊 Metadata saved to %s", metadata_file)

        return True, metadata

    except Exception as e:
        error_metadata = {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "status": "failed",
        }
        logger.error("❌ Error saving progress: %s", str(e))
        return False, error_metadata


def convert_wsl_to_windows_path(wsl_path):
    """Convert WSL path to Windows path"""
    if wsl_path.startswith("/mnt/"):
        # Remove /mnt/ and get the drive letter
        path_parts = wsl_path.split("/")
        drive_letter = path_parts[2].upper()
        # Reconstruct the path
        windows_path = drive_letter + ":\\" + "\\".join(path_parts[3:])
        return windows_path
    return wsl_path


def normalize_filename(filename):
    """Normalize filename by removing date prefix and cleaning up the name"""
    try:
        # Remove file extension
        filename = os.path.splitext(filename)[0]

        # Split by underscore
        parts = filename.split("_")

        # If it follows our naming pattern (enquiry_YYYYMMDD_HHMMSS_actualname)
        if len(parts) >= 4 and parts[1].isdigit() and parts[2].isdigit():
            # Join all parts after the date/time
            base_name = "_".join(parts[3:])
        else:
            # If it doesn't follow the pattern, use the whole name
            base_name = filename

        # Clean up the name
        base_name = base_name.lower().strip()

        return base_name
    except Exception as e:
        logger.error("Error normalizing filename %s: %s", filename, str(e))
        return filename


def find_matching_row(df, update_filename, update_path):
    """Find the matching row in DataFrame"""
    try:
        normalized_update = normalize_filename(update_filename)
        # logger.info("Looking for normalized name: '%s'", normalized_update)
        # logger.info("Original filename: '%s'", update_filename)

        for _idx, row in df.iterrows():
            if pd.notna(row["file_paths"]):
                file_paths = [p.strip() for p in str(row["file_paths"]).split(",")]

                # logger.debug("Checking row %d with %d files", _idx, len(file_paths))
                for i, path in enumerate(file_paths):
                    current_filename = os.path.basename(path)
                    normalized_current = normalize_filename(current_filename)

                    # logger.debug("Comparing:")
                    # logger.debug("  Current file: '%s'", current_filename)
                    # logger.debug("  Normalized current: '%s'", normalized_current)
                    # logger.debug("  Target file: '%s'", update_filename)
                    # logger.debug("  Normalized target: '%s'", normalized_update)

                    # Try different matching strategies
                    if (
                        normalized_current == normalized_update  # Exact normalized match
                        or current_filename.lower() == update_filename.lower()  # Exact filename match
                        or normalized_current in normalized_update  # Partial match
                        or normalized_update in normalized_current
                    ):  # Reverse partial match
                        # logger.info("✅ Found match in row %d, position %d", _idx, i)
                        return _idx, i, file_paths

        logger.warning("❌ No match found for file: %s", update_filename)
        logger.warning("Available files in Excel:")
        for _idx, row in df.iterrows():
            if pd.notna(row["file_paths"]):
                paths = str(row["file_paths"]).split(",")
                for path in paths:
                    logger.warning(
                        "  - '%s' (normalized: '%s')",
                        os.path.basename(path.strip()),
                        normalize_filename(os.path.basename(path.strip())),
                    )
        return None, None, None

    except Exception as e:
        logger.error("Error in find_matching_row: %s", str(e))
        return None, None, None


def _load_excel(file_resd_xlsx):
    if not os.path.exists(file_resd_xlsx):
        logger.error("❌ Excel log file not found: %s", file_resd_xlsx)
        return None
    try:
        df = pd.read_excel(file_resd_xlsx, engine="openpyxl", keep_default_na=True)
        # logger.info("📊 Successfully read Excel file with %d rows", len(df))
        return df
    except Exception as excel_read_error:
        logger.error("❌ Error reading Excel file: %s", str(excel_read_error))
        return None


def _prepare_status_and_count_lists(df, idx, total_files):
    current_res_status = str(df.at[idx, "res_status"]) if pd.notna(df.at[idx, "res_status"]) else ""
    res_status = current_res_status.split(",") if current_res_status else []
    res_status = [s.strip().lower() for s in res_status]
    current_count = str(df.at[idx, "count_download"]) if pd.notna(df.at[idx, "count_download"]) else ""
    count_list = current_count.split(",") if current_count else []
    count_list = [c.strip() for c in count_list]
    while len(res_status) < total_files:
        res_status.append("pending")
    while len(count_list) < total_files:
        count_list.append("0")
    return res_status, count_list


def _balance_positions(total_files, res_status, position):
    completed_positions = []
    pending_positions = []
    for i in range(total_files):
        if i == position or (i < len(res_status) and res_status[i].lower() == "completed"):
            completed_positions.append(i)
        else:
            pending_positions.append(i)
    while len(completed_positions) > len(pending_positions):
        if completed_positions[-1] != position:
            completed_positions.pop()
        elif len(completed_positions) > 1:
            completed_positions.pop(-2)
        else:
            break
    while len(completed_positions) < len(pending_positions):
        pending_positions.pop()
    return completed_positions, pending_positions


def _update_dataframe_columns(df, idx, file_paths, new_status, new_count, update_row_index, position, total_files):
    file_paths[position] = update_row_index
    file_hashes = str(df.at[idx, "file_hash"]).split(",") if pd.notna(df.at[idx, "file_hash"]) else [""] * total_files
    file_hashes = [h.strip() for h in file_hashes]
    while len(file_hashes) < total_files:
        file_hashes.append("")
    file_hashes[position] = update_row_index
    df.at[idx, "file_paths"] = ",".join(file_paths)
    df.at[idx, "res_status"] = ",".join(new_status)
    df.at[idx, "count_download"] = ",".join(new_count)
    df.at[idx, "file_hash"] = ",".join(file_hashes)


def _update_row_values(df, idx, position, update_row_index):
    try:
        file_paths = [p.strip() for p in str(df.at[idx, "file_paths"]).split(",")]
        total_files = len(file_paths)
        res_status, count_list = _prepare_status_and_count_lists(df, idx, total_files)
        # logger.info("Before update:")
        # logger.info(f"Status: {res_status}")
        # logger.info(f"Counts: {count_list}")
        new_status = ["pending"] * total_files
        new_count = ["0"] * total_files
        new_status[position] = "completed"
        new_count[position] = "1"
        completed_positions, pending_positions = _balance_positions(total_files, res_status, position)
        for pos in completed_positions:
            new_status[pos] = "completed"
            new_count[pos] = "1"
        for pos in pending_positions:
            new_status[pos] = "pending"
            new_count[pos] = "0"
        _update_dataframe_columns(df, idx, file_paths, new_status, new_count, update_row_index, position, total_files)
        final_completed = sum(1 for s in new_status if s == "completed")
        final_pending = sum(1 for s in new_status if s == "pending")
        final_count_1 = sum(1 for c in new_count if c == "1")
        final_count_0 = sum(1 for c in new_count if c == "0")
        # logger.info("\nFinal state:")
        # logger.info(f"Completed positions: {completed_positions}")
        # logger.info(f"Pending positions: {pending_positions}")
        # logger.info(f"Status - Completed: {final_completed}, Pending: {final_pending}")
        # logger.info(f"Counts - Ones: {final_count_1}, Zeros: {final_count_0}")
        # logger.info(f"Status list: {new_status}")
        # logger.info(f"Count list: {new_count}")
        if final_completed == final_count_1 and final_pending == final_count_0:
            logger.info("✅ Perfect balance achieved")
        else:
            logger.error("❌ Balance check failed!")
            logger.error(f"Completed count mismatch: Status={final_completed}, Count={final_count_1}")
            logger.error(f"Pending count mismatch: Status={final_pending}, Count={final_count_0}")
        return True
    except Exception as update_error:
        logger.error("❌ Error updating values: %s", str(update_error))
        return False


def _save_excel(df, file_resd_xlsx):
    try:
        with pd.ExcelWriter(file_resd_xlsx, engine="openpyxl", mode="w") as writer:
            df.to_excel(writer, index=False)
        # logger.info("💾 Successfully saved updates to Excel file")
        return True
    except Exception as e:
        logger.error("❌ Error saving Excel file: %s", str(e))
        return False


def update_excel_log(output_file):
    """Update Excel log with processing status"""
    try:
        update_row_index = output_file.split("/output_all_pages.md")[0] + ".pdf"
        update_row_index = convert_wsl_to_windows_path(update_row_index)
        update_row_index = os.path.normpath(update_row_index)
        file_resd_xlsx = "download_email/email_download_log.xlsx"
        df = _load_excel(file_resd_xlsx)
        if df is None:
            return False
        update_filename = os.path.basename(update_row_index)
        # logger.info("Processing file: %s", update_filename)
        idx, position, file_paths = find_matching_row(df, update_filename, update_row_index)
        if idx is not None and position is not None:
            if not _update_row_values(df, idx, position, update_row_index):
                return False
            if not _save_excel(df, file_resd_xlsx):
                return False
            return True
        else:
            logger.warning("⚠️ No matching file found in Excel log for: %s", update_row_index)
            return False
    except Exception as e:
        logger.error("❌ Error updating Excel log: %s", str(e))
        return False


def _convert_pdf_and_get_folder(pdf_path):
    try:
        image_folder = pdf_to_image(pdf_path)
        abs_image_folder = os.path.abspath(image_folder)
        # logger.info("Working with image folder: %s", abs_image_folder)
        return abs_image_folder
    except Exception as e:
        logger.error("❌ Failed to convert PDF to images: %s", str(e))
        return None


def _get_image_files(abs_image_folder):
    if not os.path.exists(abs_image_folder):
        logger.error("❌ Image folder not found: %s", abs_image_folder)
        return []
    image_files = [f for f in sorted(os.listdir(abs_image_folder)) if f.endswith(".png") and f.startswith("page_")]
    if not image_files:
        logger.error("❌ No page images found in %s", abs_image_folder)
    return image_files


def _process_pages(image_files, abs_image_folder, temp_output_file):
    processed_count = 0
    failed_count = 0
    all_markdown = []
    for i, filename in enumerate(image_files, 1):
        image_path = os.path.join(abs_image_folder, filename)
        # logger.info("\n🔍 Processing %d/%d: %s", i, len(image_files), filename)
        try:
            if not os.path.exists(image_path):
                logger.warning("⚠️ Image file missing: %s", filename)
                failed_count += 1
                continue
            if os.path.getsize(image_path) == 0:
                logger.warning("⚠️ Image file is empty: %s", filename)
                failed_count += 1
                continue
            markdown_text = extract_image_to_markdown(image_path)
            if not markdown_text or len(markdown_text.strip()) < 5:
                logger.warning("⚠️ Warning: Very little content extracted from %s", filename)
            page_content = f"## {filename}\n\n{markdown_text}\n\n---\n\n"
            all_markdown.append(page_content)
            processed_count += 1
            # logger.info("✅ Completed: %s (%d characters)", filename, len(markdown_text))
            if i % 3 == 0:
                save_progress(all_markdown, temp_output_file)
                # logger.info("💾 Progress saved (processed %d pages)", processed_count)
        except KeyboardInterrupt:
            logger.warning("\n⚠️ Process interrupted by user")
            logger.warning("💾 Saving progress for %d completed pages...", processed_count)
            break
        except Exception as e:
            logger.error("❌ Error processing %s: %s", filename, str(e))
            failed_count += 1
            error_content = f"## {filename}\n\n*Error processing this page: {str(e)}*\n\n---\n\n"
            all_markdown.append(error_content)
            continue
    return all_markdown, processed_count, failed_count


def _final_save_and_summary(
    all_markdown, output_file, temp_output_file, pdf_path, image_files, processed_count, failed_count
):
    if all_markdown:
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                summary = (
                    f"# PDF Extraction Summary\n"
                    f"**File:** {pdf_path}\n"
                    f"**Total Pages:** {len(image_files)}\n"
                    f"**Successfully Processed:** {processed_count}\n"
                    f"**Failed:** {failed_count}\n"
                    f"**Extraction Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"---\n"
                )
                f.write(summary)
                f.writelines(all_markdown)
            # logger.info("\n✅ Final output saved to '%s'", output_file)
            # logger.info(
            #     "📊 Summary: %d successful, %d failed out of %d total pages",
            #     processed_count,
            #     failed_count,
            #     len(image_files),
            # )
            if not update_excel_log(output_file):
                logger.warning("⚠️ Failed to update Excel log, but PDF processing completed successfully")
            if os.path.exists(temp_output_file):
                os.remove(temp_output_file)
            return os.path.dirname(output_file)
        except Exception as e:
            logger.error("❌ Error saving final output: %s", str(e))
            return None
    else:
        logger.error("❌ No content was successfully extracted")
        return None


def process_pdf_image(pdf_path):
    """Process PDF with comprehensive error handling and data preservation"""
    if not os.environ.get("GOOGLE_API_KEY"):
        logger.error("❌ GOOGLE_API_KEY not set in environment variables")
        return None
    abs_image_folder = _convert_pdf_and_get_folder(pdf_path)
    if not abs_image_folder:
        return None
    image_files = _get_image_files(abs_image_folder)
    if not image_files:
        return None
    # logger.info("📋 Found %d pages to process", len(image_files))
    output_file = os.path.join(abs_image_folder, "output_all_pages.md")
    temp_output_file = os.path.join(abs_image_folder, "temp_output_all_pages.md")
    all_markdown, processed_count, failed_count = _process_pages(image_files, abs_image_folder, temp_output_file)
    return _final_save_and_summary(
        all_markdown, output_file, temp_output_file, pdf_path, image_files, processed_count, failed_count
    )


# ---------- 6. Main Execution ----------

# if __name__ == "__main__":
#     # Get the current directory
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     # Construct path to the uploads directory
#     pdf_path = os.path.join(
#         os.path.dirname(current_dir),
#         'uploads',
#         '1f702d07-27b0-45a1-9ff5-9deba4e1e2de_Enquiry_form_-_Gulf_Additives_Revised_11112024.pdf'
#     )
#     # Normalize the path
#     pdf_path = os.path.normpath(pdf_path)
#     print(f"PDF path: {pdf_path}")
#     process_pdf_image(pdf_path)
