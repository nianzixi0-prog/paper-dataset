import os
import re
import time
import threading
import pandas as pd
from openai import OpenAI
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===================== Configuration =====================
# API configuration
API_KEY = "YOUR_API_KEY"                                 # Replace with your API key
BASE_URL = ""   # API base URL
MODEL_NAME = ""                   # Model name
# Path configuration
FOLDER_PATH = "./literature_files"                       # Folder containing .txt files
RESULT_EXCEL = "./screening_results.xlsx"               # Output Excel file
# Runtime configuration
MAX_WORKERS = 8          # Number of concurrent threads
API_RETRY = 3            # API retry attempts
API_TIMEOUT = 30         # API timeout (seconds)
API_INTERVAL = 0.5       # Interval between API calls in a single thread (seconds)
# ==========================================================

# Unified logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Thread lock for safe writing to result list
result_lock = threading.Lock()


def read_single_literature_file(file_path):
    """
    Read a single txt literature file and parse literature list
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return []

    # Regex to match numbered literature entries (compatible with multiple formats)
    pattern = r'(\d+)\.\s+(.*?)(?=\n\s*\d+\.\s+|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)

    literature_list = []
    for idx, (lit_id, content_text) in enumerate(matches):
        lit_id = lit_id.strip()
        content_text = content_text.strip()
        lines = [line.strip() for line in content_text.split('\n') if line.strip()]

        title = lines[0] if lines else ""
        abstract = ' '.join(lines[1:]).strip() if len(lines) > 1 else ""

        literature_list.append({
            'id': lit_id,
            'title': title,
            'abstract': abstract,
            'full_content': content_text,
            'source_file': os.path.basename(file_path),
            'global_idx': idx + 1  # Global sequence number
        })
    return literature_list


def read_literature_from_folder(folder_path):
    """
    Traverse folder and read all txt literature files
    """
    all_literatures = []
    if not os.path.exists(folder_path):
        logger.error(f"Folder does not exist: {folder_path}")
        return all_literatures

    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            logger.info(f"Reading file: {file_path}")
            literatures = read_single_literature_file(file_path)
            all_literatures.extend(literatures)

    logger.info(f"\n✅ Total {len(all_literatures)} literatures read")
    return all_literatures


def is_garbled(text):
    """
    Optimized garbled text detection
    """
    if not text or len(text) < 5:
        return False

    # Count Chinese characters
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    chinese_ratio = len(chinese_chars) / len(text) if text else 0

    # Count garbled characters (non-Chinese, non-digit, non-common punctuation)
    garbage_chars = re.findall(r'[^\u4e00-\u9fff0-9a-zA-Z\s，。！？；：""''（）《》、·]', text)
    garbage_ratio = len(garbage_chars) / len(text) if text else 0

    return chinese_ratio < 0.1 and garbage_ratio > 0.3


def call_api(prompt):
    """
    Call API with retry mechanism
    """
    for attempt in range(API_RETRY):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a professional medical literature screening assistant, strictly screening according to PICOS principles, and only respond in English."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=4000,
                timeout=API_TIMEOUT
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"API call failed (Retry {attempt + 1}): {e}")
            time.sleep(2 ** attempt)
    return "API call failed"


def screen_single_literature(literature):
    """
    Screen a single literature (for multi-threaded calls)
    """
    lit_id = literature['id']
    title = literature['title']
    abstract = literature['abstract']

    # Build screening content
    if is_garbled(abstract):
        screening_content = f"Literature Title: {title}\nNote: Abstract is garbled, judge only based on title."
    else:
        abstract_trunc = abstract[:3000] if len(abstract) > 3000 else abstract
        screening_content = f"Literature Title: {title}\nLiterature Abstract: {abstract_trunc}"

    prompt = f"""
Please strictly screen the literature according to the following PICOS principles and output only in the specified format:

Inclusion Criteria

Extra prompts

Literature to be screened:
{screening_content}

Output Format (strictly follow, no word limit for reasons):
Initial Screening Result: [Eligible/Not Eligible]
Reason: [Provide core judgment basis according to P, I, S principles]
    """

    # Call API
    api_result = call_api(prompt)
    time.sleep(API_INTERVAL)

    # Parse result
    try:
        result_match = re.search(r'Initial Screening Result\s*[:：]\s*(Eligible|Not Eligible)', api_result)
        reason_match = re.search(r'Reason\s*[:：]\s*(.+)', api_result, re.DOTALL)

        final_result = result_match.group(1).strip() if result_match else "Parsing Failed"
        final_reason = re.sub(r'\s+', ' ', reason_match.group(1).strip()) if reason_match else "Unable to parse"
    except:
        final_result = "Parsing Failed"
        final_reason = f"API returned abnormal content: {api_result[:200]}..."

    result_data = {
        'Literature ID': lit_id,
        'Title': title,
        'Initial Screening Result': final_result,
        'Initial Screening Reason': final_reason
    }

    return result_data


def save_to_excel(results, filename):
    """
    Save results to Excel file
    """
    try:
        df = pd.DataFrame(results)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace('\n', ' ').str.replace('\r', '')
        df.to_excel(filename, index=False)
        logger.info(f"Final results saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save Excel file: {e}")


def main():
    # 1. Read all literatures
    literature_list = read_literature_from_folder(FOLDER_PATH)
    if not literature_list:
        logger.error("❌ No literature read, program exited")
        return

    # 2. Initialize result list
    results = []

    # 3. Multi-threaded concurrent screening
    logger.info(f"\n🚀 Starting initial screening with {MAX_WORKERS} threads...")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="ScreenThread") as executor:
        future_to_lit = {executor.submit(screen_single_literature, lit): lit for lit in literature_list}

        for idx, future in enumerate(as_completed(future_to_lit), 1):
            lit = future_to_lit[future]
            try:
                result = future.result()
                with result_lock:
                    results.append(result)
                    logger.info(f"Screening completed: {lit['id']} | Result: {result['Initial Screening Result']}")

            except Exception as e:
                logger.error(f"Error screening literature {lit['id']}: {e}")
                with result_lock:
                    results.append({
                        'Literature ID': lit['id'],
                        'Title': lit['title'],
                        'Initial Screening Result': "Screening Failed",
                        'Initial Screening Reason': f"Error during screening: {str(e)}"
                    })

    # 4. Save final results
    save_to_excel(results, RESULT_EXCEL)

    # 5. Statistical results
    total_time = time.time() - start_time
    passed_count = len([r for r in results if r['Initial Screening Result'] == "Eligible"])
    failed_count = len([r for r in results if r['Initial Screening Result'] == "Not Eligible"])
    unfilterable_count = len([r for r in results if r['Initial Screening Result'] in ["Parsing Failed", "Screening Failed"]])

    logger.info(f"\n========== Initial Screening Completed ==========")
    logger.info(f"Total literatures: {len(literature_list)}")
    logger.info(f"Eligible: {passed_count}")
    logger.info(f"Not eligible: {failed_count}")
    logger.info(f"Unscreenable/Failed: {unfilterable_count}")
    logger.info(f"Total time: {total_time:.1f} seconds (average {total_time / len(literature_list):.2f} seconds/article)")
    logger.info(f"Final result file: {RESULT_EXCEL}")


if __name__ == "__main__":
    main()