import os
import re
import time
import threading
import pandas as pd
import pdfplumber
from openai import OpenAI
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===================== Configuration =====================
# API configuration
API_KEY = "YOUR_API_KEY"                                 # Replace with your API key
BASE_URL = ""   # API base URL
MODEL_NAME = ""                   # Model name
# Path configuration
FOLDER_PATH = "./pdf_files"                              # PDF literature folder
RESULT_EXCEL = "./secondary_screening_results.xlsx"      # Output Excel file
# Runtime configuration
MAX_WORKERS = 5          # Number of concurrent threads
API_RETRY = 3            # API retry attempts
API_TIMEOUT = 30         # API timeout (seconds)
API_INTERVAL = 0.5       # Interval between API calls in a single thread (seconds)
# ==========================================================

# Unified logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Thread lock: ensure safe writing to result list in multi-threading
result_lock = threading.Lock()


def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF file (using pdfplumber)
    """
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from PDF {pdf_path}: {e}")
        return ""


def read_pdfs_from_folder(folder_path):
    """
    Read all PDF files from folder and extract text
    """
    pdf_files = []

    # Check if folder exists
    if not os.path.exists(folder_path):
        logger.error(f"Folder does not exist: {folder_path}")
        return pdf_files

    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            logger.info(f"Reading file: {filename}")

            text = extract_text_from_pdf(file_path)

            pdf_files.append({
                'filename': filename,
                'file_path': file_path,
                'text': text,
                'text_length': len(text)
            })

    logger.info(f"\n✅ Total {len(pdf_files)} PDF literatures read")
    return pdf_files


def call_api(prompt):
    """
    Call API with retry mechanism
    """
    for attempt in range(API_RETRY):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional medical literature screening assistant, you need to strictly screen literatures according to PICOS principles and inclusion/exclusion criteria. Please respond in English only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=4000,  # Increase token limit to allow longer reasons
                timeout=API_TIMEOUT
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"API call failed (Retry {attempt + 1}): {e}")
            time.sleep(2 ** attempt)  # Exponential backoff retry
    return "API call failed"


def screen_single_literature(pdf_file):
    """
    Screen a single literature (for multi-threaded calls)
    """
    filename = pdf_file['filename']
    literature_text = pdf_file['text']

    # Check if text is empty
    if not literature_text or len(literature_text.strip()) == 0:
        logger.warning(f"Literature {filename} has empty text, skip screening")
        return {
            'File Name': filename,
            'Screening Result': "Unscreenable",
            'Screening Reason': "PDF text extraction failed or empty"
        }

    # Adjust text length limit (increase to allow more content)
    max_text_length = 15000  # Increase text truncation length
    if len(literature_text) > max_text_length:
        literature_text = literature_text[:max_text_length] + "...[Text truncated due to length limit]"

    # Build screening prompt
    prompt = f"""
Please strictly judge whether the literature meets the inclusion criteria according to the full text and PICOS principles:

Literature File Name: {filename}
Full Literature Text:
{literature_text}

Inclusion Criteria:

Extra prompts：

Please respond strictly in the following format, only keep the screening result and reason, no additional content:
Screening Result: [Eligible/Not Eligible]
Reason: [Detailed explanation of the reasons for eligibility/ineligibility, covering judgment basis for P, I, S dimensions]
    """

    # Call API
    api_result = call_api(prompt)
    time.sleep(API_INTERVAL)  # Interval in single thread to avoid rate limiting

    # Parse results
    try:
        result_match = re.search(r'Screening Result:\s*(Eligible|Not Eligible)', api_result)
        reason_match = re.search(r'Reason:\s*(.+)', api_result, re.DOTALL)

        final_result = result_match.group(1).strip() if result_match else "Parsing Failed"
        final_reason = re.sub(r'\s+', ' ', reason_match.group(1).strip()) if reason_match else "Unable to parse"
    except:
        final_result = "Parsing Failed"
        final_reason = f"API returned abnormal content: {api_result[:200]}..."

    # Organize results (remove File Path and Text Length columns)
    result_data = {
        'File Name': filename,
        'Screening Result': final_result,
        'Screening Reason': final_reason  # Remove word limit
    }

    return result_data


def save_to_excel(results, filename):
    """
    Save results to Excel file
    """
    try:
        df = pd.DataFrame(results)
        # Clean special characters to avoid Excel writing errors
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace('\n', ' ').str.replace('\r', '')
        df.to_excel(filename, index=False)
        logger.info(f"Final results saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save Excel file: {e}")


def main():
    # 1. Read all PDF literatures
    pdf_files = read_pdfs_from_folder(FOLDER_PATH)
    if not pdf_files:
        logger.error("❌ No PDF literatures read, program exited")
        return

    # 2. Initialize result list
    results = []

    # 3. Multi-threaded concurrent screening
    logger.info(f"\n🚀 Starting secondary screening with {MAX_WORKERS} threads...")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="ScreenThread") as executor:
        # Submit all tasks
        future_to_pdf = {executor.submit(screen_single_literature, pdf_file): pdf_file for pdf_file in pdf_files}

        # Get results one by one
        for idx, future in enumerate(as_completed(future_to_pdf), 1):
            pdf_file = future_to_pdf[future]
            try:
                result = future.result()
                # Lock to write result list (thread-safe)
                with result_lock:
                    results.append(result)
                    logger.info(f"Secondary screening completed: {pdf_file['filename']} | Result: {result['Screening Result']}")

            except Exception as e:
                logger.error(f"Error screening literature {pdf_file['filename']}: {e}")
                # Record failure information (remove File Path and Text Length columns)
                with result_lock:
                    results.append({
                        'File Name': pdf_file['filename'],
                        'Screening Result': "Screening Failed",
                        'Screening Reason': f"Error during screening: {str(e)}"  # Remove word limit
                    })

    # 4. Save final results only
    save_to_excel(results, RESULT_EXCEL)

    # 5. Statistical results
    total_time = time.time() - start_time
    passed_count = len([r for r in results if r['Screening Result'] == "Eligible"])
    failed_count = len([r for r in results if r['Screening Result'] == "Not Eligible"])
    unfilterable_count = len([r for r in results if r['Screening Result'] in ["Unscreenable", "Parsing Failed", "Screening Failed"]])

    logger.info(f"\n========== Secondary Screening Completed ==========")
    logger.info(f"Total PDF literatures: {len(pdf_files)}")
    logger.info(f"Eligible: {passed_count}")
    logger.info(f"Not eligible: {failed_count}")
    logger.info(f"Unscreenable/Failed: {unfilterable_count}")
    logger.info(f"Total time: {total_time:.1f} seconds (average {total_time / len(pdf_files):.2f} seconds/article)")
    logger.info(f"Final result file: {RESULT_EXCEL}")


if __name__ == "__main__":
    main()