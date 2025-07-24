import csv
import time

import requests

API_BASE_URL = "https://api.unpaywall.org/v2/search"
EMAIL = (
    "xlct43@durham.ac.uk"  # Replace with the email you use for Unpaywall API requests
)


def search_unpaywall_by_title(
    keywords,
    email,
    is_oa=None,
    max_pages=5,
    output_csv="unpaywall_results.csv",
    wait_s=1.0,
):
    """
    Query the Unpaywall search endpoint with the given keywords, fetch results,
    and save them to a CSV file.

    Args:
        keywords (str): Space-separated keywords (e.g. "Titian paintings")
                        or a more complex query using quoted text, OR, etc.
        email (str)   : Your email parameter required by Unpaywall.
        is_oa (bool)  : If True, only retrieve OA articles;
                        if False, only non-OA; if None, any article.
        max_pages (int): Max pages to retrieve (50 results per page).
        output_csv (str): Path for CSV output.
        wait_s (float): Seconds to wait between page requests to avoid flooding the API.
    """
    # Prepare CSV output
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Write header (customize fields you wish to store)
        writer.writerow(
            [
                "doi",
                "title",
                "publisher",
                "year",
                "oa_status",
                "best_oa_location_url",
                "journal_name",
            ]
        )

        # For each page, call the Unpaywall search endpoint
        for page_num in range(1, max_pages + 1):
            # Build query parameters
            params = {"query": keywords, "email": email, "page": page_num}  # required
            if is_oa is True:
                params["is_oa"] = "true"
            elif is_oa is False:
                params["is_oa"] = "false"

            print(f"[INFO] Fetching page {page_num} with params={params}")

            # Make the request
            response = requests.get(API_BASE_URL, params=params)
            response.raise_for_status()  # will raise exception if the request failed
            data = response.json()

            # data structure:
            # {
            #   "results": [
            #       {
            #         "response": { ... DOI Object ...},
            #         "score": <float>,
            #         "snippet": <str>
            #       }, ...
            #   ]
            # }
            results = data.get("results", [])
            if not results:
                print(f"[INFO] No more articles found. Stopping at page {page_num}.")
                break

            # Process each match
            for result_item in results:
                doi_obj = result_item.get("response", {})

                # Basic fields from the DOI object
                doi = doi_obj.get("doi")
                title = doi_obj.get("title")
                publisher = doi_obj.get("publisher")
                year = doi_obj.get("year")
                oa_status = doi_obj.get("oa_status")
                journal_name = doi_obj.get("journal_name")

                # best_oa_location is usually the recommended OA link
                best_oa_loc = doi_obj.get("best_oa_location") or {}
                best_oa_url = best_oa_loc.get("url_for_pdf") or best_oa_loc.get("url")

                # Write to CSV
                writer.writerow(
                    [doi, title, publisher, year, oa_status, best_oa_url, journal_name]
                )

            print(f"[INFO] Fetched {len(results)} results from page {page_num}.")

            # Be polite: small pause to avoid hitting rate limits
            time.sleep(wait_s)

    print(f"[DONE] Results saved to '{output_csv}'.")


if __name__ == "__main__":
    # Example usage: fetch articles that have the words "Titian" AND "Paintings" in the title
    # Keep in mind:
    #  - " " (space) => AND
    #  - Use quotes to match exact phrases, e.g. "\"heavy metal\" AND toxicity"
    #  - Use OR for broader matches, e.g. "Titian OR paintings"
    #  - Prefix with "-" to exclude words, e.g. "paintings -digital"
    query_keywords = "Titian paintings"

    # Adjust parameters as needed
    search_unpaywall_by_title(
        keywords=query_keywords,
        email=EMAIL,
        is_oa=None,  # or True/False
        max_pages=3,
        output_csv="titian_paintings2.csv",
        wait_s=1.0,
    )
