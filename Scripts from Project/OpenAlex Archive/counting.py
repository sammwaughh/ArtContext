import time

import pandas as pd
import requests


def extract_topic_code(topic_url):
    """
    Given a full topic URL (e.g., "https://openalex.org/T14092"),
    extract the topic code (e.g., "T14092").
    """
    return topic_url.rstrip("/").split("/")[-1]


def get_works_count(topic_code, english_only=False, oa_only=False):
    """
    Query OpenAlex for works that have the given topic code.
    Parameters:
      - english_only: if True, adds language filter (English).
      - oa_only: if True, adds open access filter (is_oa:true).
    Returns the count of works (from meta.count).
    """
    base_url = "https://api.openalex.org/works"
    # Build the filter string: start with the topic
    filter_str = f"topics.id:{topic_code}"
    if oa_only:
        filter_str += ",is_oa:true"
    if english_only:
        filter_str += ",language:en"

    # Request only 1 item per page (we only need meta.count)
    query_url = f"{base_url}?filter={filter_str}&per_page=1"

    response = requests.get(query_url)
    if response.status_code == 200:
        data = response.json()
        return data.get("meta", {}).get("count", 0)
    else:
        print(
            f"Error {response.status_code} when querying topic {topic_code} with filter '{filter_str}'"
        )
        return None


def main(input_excel="topics_art_filtered.xlsx", output_excel="topic_counts.xlsx"):
    # Read the input Excel file (with columns "Topic Name" and "Topic ID")
    df = pd.read_excel(input_excel, engine="openpyxl")

    # Prepare lists to store results
    topic_names = []
    topic_ids_full = []
    total_counts = []
    english_counts = []
    oa_counts = []
    oa_english_counts = []

    # Process each topic
    for idx, row in df.iterrows():
        topic_name = row["Topic Name"]
        topic_id_full = row["Topic ID"]
        topic_code = extract_topic_code(topic_id_full)

        print(f"Processing '{topic_name}' ({topic_code})...")
        total_count = get_works_count(topic_code, english_only=False, oa_only=False)
        english_count = get_works_count(topic_code, english_only=True, oa_only=False)
        oa_count = get_works_count(topic_code, english_only=False, oa_only=True)
        oa_english_count = get_works_count(topic_code, english_only=True, oa_only=True)

        topic_names.append(topic_name)
        topic_ids_full.append(topic_id_full)
        total_counts.append(total_count)
        english_counts.append(english_count)
        oa_counts.append(oa_count)
        oa_english_counts.append(oa_english_count)

        # Pause briefly to be polite to the API
        time.sleep(1)

    # Create a result DataFrame with the counts
    result_df = pd.DataFrame(
        {
            "Topic Name": topic_names,
            "Topic ID": topic_ids_full,
            "Total Works Count": total_counts,
            "English Works Count": english_counts,
            "Open Access Works Count": oa_counts,
            "Open Access English Works Count": oa_english_counts,
        }
    )

    # Save the results to a new Excel file
    result_df.to_excel(output_excel, index=False, engine="openpyxl")
    print(f"Results saved to '{output_excel}'.")


if __name__ == "__main__":
    main()
