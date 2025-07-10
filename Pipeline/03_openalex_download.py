import concurrent.futures
import json
import logging
import os
import re
import threading
import time

import pandas as pd
import psutil
import requests
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

# Set your email here
my_email = "xlct43@durham.ac.uk"


# --- Utility: Setup logger ---
def setup_logger(
    name, log_file, level=logging.INFO, fmt="%(asctime)s - %(levelname)s - %(message)s"
):
    """Set up a logger that writes to log_file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Remove any existing handlers (if re‑using the same name)
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(log_file)
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


# Global logger variables (will be re‑assigned for each painter)
fetch_logger = None
excel_logger = None
download_logger = None
cpu_logger = None

# Global list to store download time data (reset for each painter)
download_time_data = []


# --- Utility functions ---
def convert_to_str(value):
    """Convert dictionaries or lists to a JSON string; otherwise, return the value."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def safe_json_parse(cell_value):
    """
    Return *cell_value* if it is already a dict / list.
    Otherwise try ``json.loads`` and return the result.
    """
    if isinstance(cell_value, (dict, list)):
        return cell_value
    try:
        return json.loads(cell_value)
    except Exception:
        return None


def sanitize_filename(name):
    """Remove characters not allowed in file names."""
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()


# --- Core functions ---
def fetch_works(painter_name):
    """
    Fetch all works from OpenAlex that meet the following criteria:
      - language is English (language:en)
      - the work is open access (is_oa:true)
      - the work has one of the specified topic IDs in its topics list
      - the work's searchable metadata mentions the painter_name
    """
    base_url = "https://api.openalex.org/works"
    topics_ids = (
        "T14092|T14191|T12372|T14469|T12680|T14366|T13922|T12444|"
        "T13133|T12179|T13342|T12632|T14002|T14322"
    )
    filter_str = f"language:en,is_oa:true,topics.id:{topics_ids}"
    cursor = "*"
    per_page = 200
    works = []
    page_count = 0
    headers = {"User-Agent": f"MyPythonClient (mailto:{my_email})"}

    fetch_logger.info(f"Starting query for works mentioning '{painter_name}'...")
    while cursor:
        page_count += 1
        fetch_logger.info(f"Querying page {page_count}...")
        params = {
            "filter": filter_str,
            "search": painter_name,
            "per_page": per_page,
            "cursor": cursor,
            "mailto": my_email,
        }
        while True:
            response = requests.get(base_url, params=params, headers=headers)
            if response.status_code == 429:
                fetch_logger.warning("Rate limit exceeded. Sleeping for 60 seconds...")
                time.sleep(60)
            else:
                break
        if response.status_code != 200:
            fetch_logger.error(f"Error: Received status code {response.status_code}")
            fetch_logger.error(f"Response content: {response.text}")
            break
        data = response.json()
        results = data.get("results", [])
        works.extend(results)
        fetch_logger.info(f"Page {page_count}: Retrieved {len(results)} works.")
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            fetch_logger.info("No further pages found.")
            break
        time.sleep(1)
    fetch_logger.info(
        "Query complete. Total pages: %d. Total works: %d",
        page_count,
        len(works),
    )
    return works


def get_candidate_links(row):
    """
    Given a row (from the Filtered DataFrame), extract candidate download links.
    """
    candidates = []
    best_oa = safe_json_parse(row.get("best_oa_location"))
    if best_oa and isinstance(best_oa, dict):
        if best_oa.get("pdf_url"):
            candidates.append(best_oa["pdf_url"])
        elif best_oa.get("landing_page_url"):
            candidates.append(best_oa["landing_page_url"])
    oa = safe_json_parse(row.get("open_access"))
    if oa and isinstance(oa, dict) and oa.get("oa_url"):
        candidates.append(oa["oa_url"])
    primary = safe_json_parse(row.get("primary_location"))
    if primary and isinstance(primary, dict):
        if primary.get("pdf_url"):
            candidates.append(primary["pdf_url"])
        elif primary.get("landing_page_url"):
            candidates.append(primary["landing_page_url"])
    locs = safe_json_parse(row.get("locations"))
    if locs and isinstance(locs, list):
        for loc in locs:
            if isinstance(loc, dict):
                if loc.get("pdf_url"):
                    candidates.append(loc["pdf_url"])
                elif loc.get("landing_page_url"):
                    candidates.append(loc["landing_page_url"])
    return list(dict.fromkeys(candidates))


def get_best_and_backup_links(row):
    """
    Return (best_link, backup_link) for one *row* of **df_filtered**.

    Strategy:
      • first candidate that contains “.pdf”  → best_link
      • next candidate (if any)              → backup_link
    """
    candidates = get_candidate_links(row)
    best_link = ""
    backup_link = ""
    for link in candidates:
        if ".pdf" in link.lower():
            best_link = link
            break
    if not best_link and candidates:
        best_link = candidates[0]
    for link in candidates:
        if link != best_link:
            backup_link = link
            break
    return best_link, backup_link


def create_excel_file(painter_name, works):
    """
    Build three DataFrames from *works* then write them to one workbook.

    Sheets
    ------
    • Main          – every column in the JSON returned by OpenAlex
    • Filtered      – subset of useful columns
    • Downloadable  – Title, Relevance Score, Best/Back-up link, OpenAlexID, Type
    """
    # If no works were fetched, create an empty DataFrame with the expected columns.
    if works:
        df_main = pd.DataFrame(works)
    else:
        expected_columns = [
            "title",
            "relevance_score",
            "id",
            "doi",
            "primary_location",
            "type",
            "open_access",
            "locations",
            "best_oa_location",
        ]
        df_main = pd.DataFrame(columns=expected_columns)

    filtered_columns = [
        "title",
        "relevance_score",
        "id",
        "doi",
        "primary_location",
        "type",
        "open_access",
        "locations",
        "best_oa_location",
    ]
    df_filtered = df_main[filtered_columns].copy()

    downloadable_data = []
    for _, row in df_filtered.iterrows():
        title = row.get("title", "")
        relevance = row.get("relevance_score", 0)
        openalex_id = row.get("id", "")
        work_type = row.get("type", "")
        best_link, backup_link = get_best_and_backup_links(row)
        downloadable_data.append(
            {
                "Title": title,
                "Relevance Score": relevance,
                "Best Link": best_link,
                "Back Up Link": backup_link,
                "OpenAlexID": openalex_id,
                "Type": work_type,
            }
        )
    df_downloadable = pd.DataFrame(downloadable_data)

    filename = os.path.join("ExcelFiles", f"{painter_name.lower()}_works.xlsx")
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        df_main.to_excel(writer, index=False, sheet_name="Main")
        df_filtered.to_excel(writer, index=False, sheet_name="Filtered")
        df_downloadable.to_excel(writer, index=False, sheet_name="Downloadable")

    wb = load_workbook(filename)
    ws_main = wb["Main"]
    for col in ws_main.columns:
        col_letter = get_column_letter(col[0].column)
        max_length = max(
            (len(str(cell.value)) for cell in col if cell.value is not None), default=0
        )
        ws_main.column_dimensions[col_letter].width = max_length + 2
    for sheet in ["Filtered", "Downloadable"]:
        ws = wb[sheet]
        for col in ws.columns:
            col_letter = get_column_letter(col[0].column)
            ws.column_dimensions[col_letter].width = 35
    wb.save(filename)
    excel_logger.info(
        "Excel file %s created with sheets: Main, Filtered, Downloadable.",
        filename,
    )
    return filename


def download_pdf(
    index, title, best_link, backup_link, openalex_id, work_type, directory
):
    """
    Download one PDF.

    Order of attempts
    -----------------
    1. *best_link*
    2. *backup_link* (if the first fails)

    On success the file is saved as
    ``{index}-{sanitized_title}.pdf`` in *directory* and timing is logged.
    """
    sanitized_title = sanitize_filename(title)
    pdf_filename = f"{index}-{sanitized_title}.pdf"
    filepath = os.path.join(directory, pdf_filename)
    start_time = time.perf_counter()
    for link in [best_link, backup_link]:
        if link and isinstance(link, str) and link.strip():
            try:
                download_logger.info(f"Row {index}: Attempting download from: {link}")
                response = requests.get(link, stream=True, timeout=20)
                if response.status_code == 200:
                    with open(filepath, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    end_time = time.perf_counter()
                    elapsed_sec = round(end_time - start_time, 3)
                    download_logger.info(
                        "Row %d: downloaded %s in %.3f s",
                        index,
                        filepath,
                        elapsed_sec,
                    )
                    download_time_data.append(
                        {
                            "PDF Name": pdf_filename,
                            "Time": elapsed_sec,
                            "Type": work_type,
                        }
                    )
                    return True
                else:
                    download_logger.warning(
                        "Row %d: status %s on link %s",
                        index,
                        response.status_code,
                        link,
                    )
            except Exception as e:
                download_logger.error(
                    f"Row {index}: Error downloading from {link}: {e}"
                )
    download_logger.error(
        f"Row {index}: All download attempts failed (ID: {openalex_id})."
    )
    return False


def monitor_cpu(stop_event, interval=1):
    """
    Monitors and logs CPU usage every `interval` seconds until stop_event is set.
    """
    cpu_usages = []
    while not stop_event.is_set():
        usage = psutil.cpu_percent(interval=interval)
        cpu_usages.append(usage)
        cpu_logger.info(f"Current CPU usage: {usage}%")
    avg = sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0
    cpu_logger.info(f"Average CPU usage during download: {avg:.2f}%")


def append_download_time_sheet(excel_file, painter_name, total_runtime):
    """
    Appends a new sheet "Download Times" to the existing Excel file.
    The sheet contains a table (PDF Name, Time, Type) sorted by Time descending,
    with a final row showing the total program runtime.
    """
    if download_time_data:
        df_time = pd.DataFrame(download_time_data, columns=["PDF Name", "Time", "Type"])
        df_time = df_time.sort_values(by="Time", ascending=False).reset_index(drop=True)
        total_row = pd.DataFrame(
            {
                "PDF Name": ["Total Program Runtime"],
                "Time": [f"{total_runtime:.3f} seconds"],
                "Type": [""],
            }
        )
        df_time = pd.concat([df_time, total_row], ignore_index=True)

        wb = load_workbook(excel_file)
        if "Download Times" in wb.sheetnames:
            ws_del = wb["Download Times"]
            wb.remove(ws_del)
        ws = wb.create_sheet(title="Download Times")

        headers = list(df_time.columns)
        ws.append(headers)
        for row in df_time.itertuples(index=False, name=None):
            ws.append(list(row))

        for col in ws.columns:
            col_letter = get_column_letter(col[0].column)
            max_length = max(
                (len(str(cell.value)) for cell in col if cell.value is not None),
                default=0,
            )
            ws.column_dimensions[col_letter].width = max_length + 2
        wb.save(excel_file)
        download_logger.info(
            f"Download time data appended as a new sheet in '{excel_file}'."
        )
        print(f"Download time data appended as a new sheet in '{excel_file}'.")
    else:
        download_logger.info("No download time data to append.")


def download_all_pdfs(excel_file, painter_name, max_workers):
    """
    Read sheet “Downloadable”, create the painter’s PDF dir and download PDFs
    in parallel (relevance score > 1).  CPU usage is monitored in a thread.
    A “Download Times” sheet is appended after completion.
    """
    df = pd.read_excel(excel_file, sheet_name="Downloadable", engine="openpyxl")
    # Remove duplicates based on the Title column
    df = df.drop_duplicates(subset=["Title"], keep="first")

    pdf_dir = os.path.join("PDFs", painter_name.lower())
    os.makedirs(pdf_dir, exist_ok=True)

    stop_event = threading.Event()
    cpu_thread = threading.Thread(target=monitor_cpu, args=(stop_event,), daemon=True)
    cpu_thread.start()

    start_time = time.perf_counter()
    download_logger.info("Starting parallel PDF download tasks...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, row in enumerate(df.itertuples(index=False)):
            relevance = row[1]
            if relevance <= 1:
                download_logger.info(
                    f"Row {i}: Skipping due to low relevance score ({relevance})."
                )
                continue
            title = row[0]
            best_link = row[2]
            backup_link = row[3]
            openalex_id = row[4]
            work_type = row[5]
            download_logger.info(f"Submitting download task for row {i}: {title}")
            futures.append(
                executor.submit(
                    download_pdf,
                    i,
                    title,
                    best_link,
                    backup_link,
                    openalex_id,
                    work_type,
                    pdf_dir,
                )
            )
        download_logger.info("Waiting for all download tasks to complete...")
        concurrent.futures.wait(futures)
    stop_event.set()
    cpu_thread.join()
    end_time = time.perf_counter()
    total_runtime = round(end_time - start_time, 3)
    download_logger.info(f"All downloads attempted in {total_runtime:.2f} seconds.")
    print(f"All downloads attempted in {total_runtime:.2f} seconds.")

    # Append the download time sheet to the Excel file.
    append_download_time_sheet(excel_file, painter_name, total_runtime)


# --- Painter processing ---
def process_painter(painter, max_workers):
    """
    End-to-end routine for a single *painter*:
      1. configure loggers
      2. fetch works (OpenAlex)
      3. write Excel workbook
      4. download PDFs
      5. log total runtime
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    global fetch_logger, excel_logger, download_logger, cpu_logger, download_time_data
    # Set up individual log files (stored in respective directories)
    fetch_logger = setup_logger(
        "fetch", os.path.join("fetch-logs", f"{painter}-{timestamp}.log")
    )
    excel_logger = setup_logger(
        "excel", os.path.join("excel-logs", f"{painter}-{timestamp}.log")
    )
    download_logger = setup_logger(
        "download", os.path.join("download-logs", f"{painter}-{timestamp}.log")
    )
    cpu_logger = setup_logger(
        "cpu", os.path.join("cpu-logs", f"{painter}-{timestamp}.log")
    )

    # Reset download time data for this painter
    download_time_data = []

    overall_start = time.perf_counter()
    print(f"Processing painter: {painter}")
    fetch_logger.info(f"Processing painter: {painter}")

    works = fetch_works(painter)
    print(f"Fetched {len(works)} works for {painter}")
    if not works:
        fetch_logger.info(
            f"No works fetched for {painter}. Skipping further processing."
        )
        print(f"No works fetched for {painter}. Skipping further processing.")
        return

    excel_file = create_excel_file(painter, works)
    print(f"Excel file created: {excel_file}")

    download_all_pdfs(excel_file, painter, max_workers)

    overall_end = time.perf_counter()
    runtime = round(overall_end - overall_start, 3)
    download_logger.info(f"Total program runtime for {painter}: {runtime:.2f} seconds.")
    print(f"Total runtime for {painter}: {runtime:.2f} seconds.")


# --- Main execution ---
def main():
    # Create required directories if they don't exist.
    for dirname in [
        "ExcelFiles",
        "PDFs",
        "excel-logs",
        "cpu-logs",
        "download-logs",
        "fetch-logs",
    ]:
        os.makedirs(dirname, exist_ok=True)

    # Read painters.xlsx to generate the list of painters.
    # The Excel file should have columns: "Artist" and "Query String"
    df_painters = pd.read_excel("painters.xlsx")

    # Specify which rows of the Excel file to process.
    # start_row=1 → first data row (below headers); end_row is inclusive.
    start_row = 345  # For example, process starting with row 126
    end_row = 451  # For example, process through row 150.

    # Convert to zero-based indices.
    start_index = start_row - 1
    # Since iloc slicing is exclusive at the end, we use end_row as is.
    df_subset = df_painters.iloc[start_index:end_row]

    # Convert the subset DataFrame to a list of dictionaries.
    painters = df_subset.to_dict(orient="records")

    # Configure max_workers (you can adjust this value)
    max_workers = 5

    # Process each painter one after the other.
    for idx, painter_info in enumerate(painters, start=start_row):
        query_string = painter_info["Query String"]
        print(f"Processing row: {idx}")
        # Call process_painter with the query string and number of workers.
        process_painter(query_string, max_workers)


if __name__ == "__main__":
    # main()
    print("Get logging files in directory first!")
