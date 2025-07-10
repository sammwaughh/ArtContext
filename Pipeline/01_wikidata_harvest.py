import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from pathlib import Path

import pandas as pd
import requests
from openpyxl.utils import get_column_letter
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util import Retry

# ---------------- Constants ----------------
LIMIT = 50_000  # max paintings to fetch
CHUNK_SIZE = 1_000  # rows per SPARQL page
SITELINKS_THRESHOLD = 1  # min sitelinks (notability proxy)
WORKERS = 8  # threads for aggregated queries
SUBCHUNK_SIZE = 200  # ids per aggregated SPARQL query           <─ NEW
OUT_FILE = Path("paintings.xlsx")

# ---------------- Logging ----------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "wikidata_harvest.log"),
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

print("Harvesting painting metadata…")  # brief console notice

total_limit = LIMIT
chunk_size = CHUNK_SIZE
sitelinks_threshold = SITELINKS_THRESHOLD


# ---------------- Helpers ----------------
def chunked(iterable, size):
    """Yield successive *size*-element tuples from *iterable*."""
    it = iter(iterable)
    while chunk := tuple(islice(it, size)):
        yield chunk


# ---------------- session with retry ----------------
def make_session():
    retries = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503),
    )
    adapter = HTTPAdapter(max_retries=retries)
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "ArtContext/0.1 (contact: your@email)",
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded",
        }
    )
    s.mount("https://", adapter)
    return s


SESSION = make_session()


# ---------------- Function to query Wikidata with retries ----------------
def query_wikidata(query):
    url = "https://query.wikidata.org/sparql"
    try:
        logging.info("Sending query to Wikidata...")
        # Use POST so the query is sent in the body, not in the URL.
        response = SESSION.post(url, data={"query": query})
        response.raise_for_status()
        logging.info("Query successful.")
        return response.json()
    except requests.exceptions.HTTPError as e:
        logging.error("HTTP error encountered: %s", e)
        raise


# ---------------- Function: Get Basic Records ----------------
def get_basic_records(offset: int, chunk_size: int, threshold: int):
    basic_query = f"""
        SELECT ?painting ?paintingLabel ?creator ?creatorLabel
               ?inception ?wikipedia_url ?linkCount
        WHERE {{
          ?painting wdt:P31 wd:Q3305213.
          ?painting wikibase:sitelinks ?linkCount.
          FILTER(?linkCount >= {threshold}).
          OPTIONAL {{ ?painting wdt:P170 ?creator. }}
          OPTIONAL {{ ?painting wdt:P571 ?inception. }}
          OPTIONAL {{
            ?paintingArticle schema:about ?painting ;
                             schema:inLanguage "en" ;
                             schema:isPartOf <https://en.wikipedia.org/> .
            BIND(?paintingArticle AS ?wikipedia_url)
          }}
          SERVICE wikibase:label {{
            bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en".
          }}
        }}
        ORDER BY DESC(?linkCount)
        LIMIT {chunk_size} OFFSET {offset}
        """
    data = query_wikidata(basic_query)
    bindings = data.get("results", {}).get("bindings", [])
    basic_records = {}
    painting_ids = []
    for item in bindings:
        pid = item.get("painting", {}).get("value", "")
        if pid:
            basic_records[pid] = {
                "Painting ID": pid,
                "Title": item.get("paintingLabel", {}).get("value", ""),
                "Creator ID": item.get("creator", {}).get("value", ""),
                "Creator": item.get("creatorLabel", {}).get("value", ""),
                "Inception": item.get("inception", {}).get("value", ""),
                "Wikipedia URL": item.get("wikipedia_url", {}).get("value", ""),
                "Link Count": int(item.get("linkCount", {}).get("value", 0)),
            }
            painting_ids.append(pid)
    return basic_records, painting_ids


# ---------------- Function: Get Aggregated Fields in Subchunks ----------------
def get_aggregated_fields(painting_ids, workers):
    agg_records = {}
    subchunk_size = SUBCHUNK_SIZE  # NEW

    def one_subquery(sub_ids):
        agg_query = f"""
             SELECT ?painting
                   (GROUP_CONCAT(DISTINCT ?depictsLabel;
                                 separator=", ") AS ?depictsAggregated)
                   (GROUP_CONCAT(DISTINCT ?movementLabel;
                                 separator=", ") AS ?movements)
                   (GROUP_CONCAT(DISTINCT ?movement; separator=", ")
                                 AS ?movementIDs)
            WHERE {{
              VALUES ?painting {{ {" ".join(sub_ids)} }}
              OPTIONAL {{
                  ?painting wdt:P180 ?depicts.
                  ?depicts rdfs:label ?depictsLabel.
                  FILTER(LANG(?depictsLabel) = "en")
              }}
              OPTIONAL {{
                  ?painting wdt:P135 ?movement.
                  ?movement rdfs:label ?movementLabel.
                  FILTER(LANG(?movementLabel) = "en")
              }}
            }}
            """
        sub_agg_data = query_wikidata(agg_query)
        sub_bindings = sub_agg_data.get("results", {}).get("bindings", [])
        sub_agg_records = {}
        for item in sub_bindings:
            pid = item.get("painting", {}).get("value", "")
            if pid:
                sub_agg_records[pid] = {
                    "Depicts": item.get("depictsAggregated", {}).get("value", ""),
                    "Movements": item.get("movements", {}).get("value", ""),
                    "Movement IDs": item.get("movementIDs", {}).get("value", ""),
                }
        return sub_agg_records

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(one_subquery, chunk)
            for chunk in chunked(painting_ids, subchunk_size)
        ]
        for f in as_completed(futures):
            agg_records.update(f.result())
    return agg_records


# ---------------- Function: Merge Basic and Aggregated Records ----------------
def merge_records(basic_records, agg_records):
    merged = []
    for pid, basic in basic_records.items():
        if pid in agg_records:
            basic.update(agg_records[pid])
        else:
            basic["Depicts"] = ""
            basic["Movements"] = ""
            basic["Movement IDs"] = ""
        merged.append(basic)
    return merged


# ---------------- Main Loop ----------------
logging.info(f"Starting retrieval of metadata for up to {total_limit} paintings...")
logging.info(f"Chunk size: {chunk_size}, Sitelinks threshold: >= {sitelinks_threshold}")

all_results: list[dict] = []
offset = 0

pbar = tqdm(total=LIMIT, unit=" rows")  # bar length = fixed limit
try:
    while len(all_results) < LIMIT:  # args.* removed
        logging.info(f"Querying basic records with OFFSET {offset} ...")
        basic_records, painting_ids = get_basic_records(
            offset, chunk_size, sitelinks_threshold
        )
        if not basic_records:
            logging.info("No more basic records returned; ending pagination.")
            break
        logging.info(f"Retrieved {len(basic_records)} unique basic records.")

        agg_records = get_aggregated_fields(painting_ids, WORKERS)
        merged_records = merge_records(basic_records, agg_records)

        # Append new records, avoiding duplicates.
        existing_ids = {r["Painting ID"] for r in all_results}
        new_records = []
        for record in merged_records:
            if record["Painting ID"] not in existing_ids:
                new_records.append(record)
        all_results.extend(new_records)

        offset += chunk_size
        if len(all_results) >= total_limit:
            logging.info("Reached the total desired number of records.")
            break
        time.sleep(1)  # Throttle between chunks
        pbar.update(len(new_records))
except KeyboardInterrupt:
    logging.info("Process interrupted by user.")
finally:
    pbar.close()

logging.info(f"Collected {len(all_results)} rows from the queries.")

#
# Vectorised post-processing in one DataFrame.
#

df = pd.DataFrame(all_results)

# File Name  (extract Q-ID at tail of URL)
df["File Name"] = df["Painting ID"].str.extract(r"([^/]+)$")[0].fillna("") + "_0.png"

# Year from first four chars of Inception
df["Year"] = pd.to_numeric(df["Inception"].str.slice(0, 4), errors="coerce")

# Re-order once the new columns exist
# ---------------- Create DataFrame and Reorder Columns ----------------
# Final column order (kept short for E501)
final_order = [
    "Title",
    "File Name",
    "Creator",
    "Movements",
    "Depicts",
    "Year",
    "Wikipedia URL",
    "Link Count",
    "Painting ID",
    "Creator ID",
    "Movement IDs",
]

for col in final_order:
    if col not in df.columns:
        df[col] = ""
df = df[final_order]

df["Link Count"] = pd.to_numeric(df["Link Count"], errors="coerce")
df.sort_values(by="Link Count", ascending=False, inplace=True)

logging.info("DataFrame created. Number of unique records: %s", len(df))

# ---------------- Write DataFrame to Excel ----------------
output_filename = OUT_FILE
logging.info(f"Writing data to Excel file '{output_filename}'...")

with pd.ExcelWriter(output_filename, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="Paintings")
    worksheet = writer.sheets["Paintings"]
    # Adjust column widths based on maximum content length, capped at 60.
    for i, column in enumerate(df.columns, 1):
        max_length = max(df[column].astype(str).map(len).max(), len(column))
        adjusted_width = max_length + 2  # Extra space for readability
        if adjusted_width > 40:
            adjusted_width = 40
        worksheet.column_dimensions[get_column_letter(i)].width = adjusted_width

logging.info("Excel file saved to '%s'.", output_filename)
print("Done.")  # brief console notice
