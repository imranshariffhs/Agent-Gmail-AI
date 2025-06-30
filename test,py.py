import os
import shutil
import pandas as pd
import re

# File and directory paths
input_excel = "download_email/email_download_log.xlsx"
output_excel = "download_email/email_download_log.xlsx"
destination_dir = "processed_files"

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Load the Excel file
try:
    df = pd.read_excel(input_excel)
    print(f"📄 Loaded Excel file with {len(df)} rows and {len(df.columns)} columns")
    print(f"📋 Available columns: {list(df.columns)}")
except Exception as e:
    print(f"❌ Error loading Excel file: {e}")
    exit(1)

# Ensure required columns exist
if "res_status" not in df.columns:
    df["res_status"] = ""
    print("✨ Added 'res_status' column")

if "process_status" not in df.columns:
    df["process_status"] = ""
    print("✨ Added 'process_status' column")

# Handle the comma-separated res_status values
def process_res_status_row(row_index):
    """Process a single row and update its res_status"""
    row = df.loc[row_index]
    
    # Get the res_status value and split by comma
    res_status_str = str(row.get("res_status", "")).strip()
    if not res_status_str or res_status_str.lower() in ["nan", ""]:
        print(f"Row {row_index}: No res_status found, skipping")
        return False
    
    # Split the res_status by comma and strip whitespace
    status_list = [s.strip().lower() for s in res_status_str.split(",") if s.strip()]
    
    # Check if there are any pending statuses
    if "pending" not in status_list:
        print(f"Row {row_index}: No pending status found (current: {status_list}), skipping")
        return False
    
    # Get file paths - handle paths that may contain commas
    file_list_str = str(row.get("file_paths", "")).strip()
    
    if not file_list_str or file_list_str.lower() in ["nan", ""]:
        print(f"Row {row_index}: No file_paths found, skipping")
        return False
    
    # Parse file paths using a more robust approach
    file_paths = parse_file_paths(file_list_str)
    
    if not file_paths:
        print(f"Row {row_index}: No valid file paths found, skipping")
        print(f"  Raw file_paths string: {file_list_str}")
        return False
    
    print(f"Row {row_index}: Found {len(status_list)} statuses and {len(file_paths)} files")
    print(f"  Statuses: {status_list}")
    print(f"  Subject: {row.get('subject', 'N/A')}")
    print(f"  Parsed file paths:")
    for i, fp in enumerate(file_paths):
        print(f"    {i+1}. {os.path.basename(fp)}")
    
    # Find the first pending status and try to process it
    for i, status in enumerate(status_list):
        if status == "pending":
            print(f"  Processing pending item {i+1}/{len(status_list)}")
            
            # Try to move the corresponding file (if exists)
            if i < len(file_paths):
                file_path = file_paths[i].strip()
                abs_path = os.path.abspath(file_path)
                
                print(f"  Checking file: {os.path.basename(abs_path)}")
                print(f"  Full path: {abs_path}")
                
                if os.path.isfile(abs_path):
                    try:
                        dest_path = os.path.join(destination_dir, os.path.basename(abs_path))
                        
                        # If destination file already exists, add a suffix
                        counter = 1
                        base_dest_path = dest_path
                        while os.path.exists(dest_path):
                            name, ext = os.path.splitext(base_dest_path)
                            dest_path = f"{name}_{counter}{ext}"
                            counter += 1
                        
                        shutil.move(abs_path, dest_path)
                        print(f"✔️ Row {row_index}: Moved file {i+1}: {os.path.basename(abs_path)} → {os.path.basename(dest_path)}")
                        
                        # Update the status for this specific file
                        status_list[i] = "processed"
                        
                        # Update the res_status in the dataframe
                        updated_status = ",".join(status_list)
                        df.at[row_index, "res_status"] = updated_status
                        
                        # Update process_status if all files are now processed
                        if "pending" not in status_list:
                            df.at[row_index, "process_status"] = "completed"
                            print(f"  ✅ All files in row {row_index} are now processed - marked as completed")
                        else:
                            df.at[row_index, "process_status"] = "partial"
                            print(f"  ⏳ Row {row_index} still has pending files - marked as partial")
                        
                        return True
                        
                    except Exception as e:
                        print(f"❌ Row {row_index}: Error moving file {i+1} ({os.path.basename(abs_path)}): {e}")
                        return False
                else:
                    print(f"⚠️ Row {row_index}: File {i+1} not found: {abs_path}")
                    # Mark as file_not_found to avoid infinite loop
                    status_list[i] = "file_not_found"
                    updated_status = ",".join(status_list)
                    df.at[row_index, "res_status"] = updated_status
                    
                    # Update process_status
                    if "pending" not in status_list:
                        df.at[row_index, "process_status"] = "completed_with_missing"
                    else:
                        df.at[row_index, "process_status"] = "partial"
                    
                    return True
            else:
                print(f"⚠️ Row {row_index}: No corresponding file path for pending status {i+1}")
                # Mark this status as no_file_path to avoid infinite loop
                status_list[i] = "no_file_path"
                updated_status = ",".join(status_list)
                df.at[row_index, "res_status"] = updated_status
                return True
    
    return False

def parse_file_paths(file_list_str):
    """Parse comma-separated file paths that may contain commas in filenames"""
    file_paths = []
    
    # Method 1: Use regex to find full paths with file extensions
    # Look for patterns like: /path/to/file.ext
    pattern = r'(/[^,]*?\.(?:pdf|doc|docx|txt|xlsx|xls|png|jpg|jpeg|gif|zip|rar))'
    matches = re.findall(pattern, file_list_str, re.IGNORECASE)
    
    if matches:
        file_paths = matches
        return file_paths
    
    # Method 2: Split by looking for file extension + comma + path separator
    pattern = r'(.*?\.(?:pdf|doc|docx|txt|xlsx|xls|png|jpg|jpeg|gif|zip|rar)),(?=/)'
    matches = re.finditer(pattern, file_list_str, re.IGNORECASE)
    last_end = 0
    
    for match in matches:
        file_paths.append(match.group(1).strip())
        last_end = match.end() - 1  # Don't include the comma
    
    # Add the last file path (after the last comma)
    remaining = file_list_str[last_end:].lstrip(',').strip()
    if remaining and re.search(r'\.(?:pdf|doc|docx|txt|xlsx|xls|png|jpg|jpeg|gif|zip|rar)$', remaining, re.IGNORECASE):
        file_paths.append(remaining)
    
    # Method 3: Fallback - try to reconstruct paths
    if not file_paths:
        parts = file_list_str.split(',')
        current_path = ""
        
        for part in parts:
            part = part.strip()
            if current_path:
                current_path += "," + part
            else:
                current_path = part
            
            # If this part ends with a file extension, it's likely a complete path
            if re.search(r'\.(?:pdf|doc|docx|txt|xlsx|xls|png|jpg|jpeg|gif|zip|rar)$', current_path, re.IGNORECASE):
                file_paths.append(current_path)
                current_path = ""
        
        # Add any remaining part if it looks like a file
        if current_path and re.search(r'\.(?:pdf|doc|docx|txt|xlsx|xls|png|jpg|jpeg|gif|zip|rar)$', current_path, re.IGNORECASE):
            file_paths.append(current_path)
    
    return file_paths

# Debug: Show initial status
print("📊 Initial data overview:")
print(f"📄 Total rows: {len(df)}")

# Show res_status column values
print("\n📊 res_status column values:")
for idx, row in df.iterrows():
    res_status = str(row.get("res_status", "")).strip()
    subject = str(row.get("subject", "N/A"))[:50] + "..." if len(str(row.get("subject", "N/A"))) > 50 else str(row.get("subject", "N/A"))
    
    if res_status and res_status.lower() != "nan":
        print(f"Row {idx}: {res_status} | Subject: {subject}")

print()

# Process rows one by one
processed_count = 0
total_pending = 0

# Count total pending statuses
for idx, row in df.iterrows():
    res_status_str = str(row.get("res_status", "")).strip()
    if res_status_str and res_status_str.lower() != "nan":
        status_list = [s.strip().lower() for s in res_status_str.split(",")]
        total_pending += status_list.count("pending")

print(f"🔍 Total pending statuses found: {total_pending}")
print(f"🚀 Starting to process rows one by one...\n")

# Process each row that has pending status
for idx, row in df.iterrows():
    res_status_str = str(row.get("res_status", "")).strip()
    if res_status_str and res_status_str.lower() != "nan" and "pending" in res_status_str.lower():
        print(f"🔄 Processing row {idx}...")
        success = process_res_status_row(idx)
        if success:
            processed_count += 1
            print(f"✅ Successfully processed 1 item from row {idx}")
            
            # Save after each successful processing
            try:
                df.to_excel(output_excel, index=False)
                print(f"💾 Excel file updated after processing row {idx}\n")
            except Exception as e:
                print(f"❌ Error saving Excel file: {e}")
            
            # Break after processing one item (as requested)
            break
        else:
            print(f"❌ Failed to process row {idx}\n")

print(f"\n📈 Summary:")
print(f"✅ Total items processed in this run: {processed_count}")
print(f"📄 Excel file saved to: {output_excel}")

# Show final status
print("\n📊 Final res_status values:")
for idx, row in df.iterrows():
    res_status = str(row.get("res_status", "")).strip()
    subject = str(row.get("subject", "N/A"))[:50] + "..." if len(str(row.get("subject", "N/A"))) > 50 else str(row.get("subject", "N/A"))
    
    if res_status and res_status.lower() != "nan":
        print(f"Row {idx}: {res_status} | Subject: {subject}")

print(f"\n🏁 Script completed!")