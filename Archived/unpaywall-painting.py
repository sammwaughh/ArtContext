import requests
import pandas as pd
import time
import re

def find_and_save_papers_about_bacchus_and_ariadne(
    query="Bacchus and Ariadne art history",
    email="xlct43@durham.ac.uk",
    output_excel="bacchus_ariadne_papers.xlsx"
):
    """
    Searches for papers about 'Bacchus and Ariadne' in an Art History context,
    checks their availability via the Unpaywall API (based on their DOIs),
    and saves the results in an Excel file.

    Now handles edge cases when 'best_oa_location' is None or missing.
    """

    crossref_url = "https://api.crossref.org/works"
    params = {
        "query": query,
        "rows": 20,
        "mailto": email
    }
    try:
        crossref_response = requests.get(crossref_url, params=params, timeout=30)
        crossref_response.raise_for_status()
        print("[DEBUG] Crossref API call succeeded. Status:", crossref_response.status_code)
    except requests.exceptions.RequestException as e:
        print("[ERROR] Crossref API call failed:", e)
        return

    crossref_data = crossref_response.json()
    items = crossref_data.get("message", {}).get("items", [])
    print(f"[DEBUG] Number of items retrieved from Crossref: {len(items)}")

    all_papers = []

    for idx, item in enumerate(items, start=1):
        title_list = item.get("title", ["No Title"])
        title = title_list[0] if title_list else "No Title"
        doi = item.get("DOI", None)

        authors_list = []
        if "author" in item:
            for author in item["author"]:
                given = author.get("given", "")
                family = author.get("family", "")
                full_name = (given + " " + family).strip()
                authors_list.append(full_name)
        authors_str = ", ".join(authors_list) if authors_list else "Unknown"

        # Derive publication date
        pub_date = "Unknown"
        if "published-print" in item and "date-parts" in item["published-print"]:
            date_parts = item["published-print"]["date-parts"]
            if date_parts and isinstance(date_parts, list) and len(date_parts) > 0:
                pub_date = "-".join(str(part) for part in date_parts[0])
        elif "published-online" in item and "date-parts" in item["published-online"]:
            date_parts = item["published-online"]["date-parts"]
            if date_parts and isinstance(date_parts, list) and len(date_parts) > 0:
                pub_date = "-".join(str(part) for part in date_parts[0])
        else:
            pub_date = item.get("created", {}).get("date-time", "Unknown")

        # If no DOI, skip unpaywall
        if not doi:
            all_papers.append({
                "Paper Name": title,
                "Authors": authors_str,
                "Date of Publication": pub_date,
                "Is Available for Download": False,
                "DOI": "None",
                "Download URL": ""
            })
            continue

        # Query Unpaywall for OA details
        unpaywall_url = f"https://api.unpaywall.org/v2/{doi}"
        params_unpaywall = {"email": email}
        time.sleep(0.5)  # Polite delay

        try:
            unpaywall_resp = requests.get(unpaywall_url, params=params_unpaywall, timeout=30)
            if unpaywall_resp.status_code == 200:
                unpaywall_json = unpaywall_resp.json()
                is_oa = unpaywall_json.get("is_oa", False)

                # Safely handle best_oa_location
                best_oa = unpaywall_json.get("best_oa_location") or {}
                # If best_oa_location is None, this will become an empty dict

                pdf_link = best_oa.get("pdf_url", "")
                # If pdf_url doesn't exist, this is just ""

            else:
                print(f"[WARN] Unpaywall returned status {unpaywall_resp.status_code}, defaulting to non-OA.")
                is_oa = False
                pdf_link = ""
        except requests.exceptions.RequestException as e:
            print("[ERROR] Unpaywall lookup failed:", e)
            is_oa = False
            pdf_link = ""

        all_papers.append({
            "Paper Name": title,
            "Authors": authors_str,
            "Date of Publication": pub_date,
            "Is Available for Download": is_oa,
            "DOI": doi,
            "Download URL": pdf_link
        })

    # Save results
    df = pd.DataFrame(all_papers)
    df.to_excel(output_excel, index=False)
    print(f"[INFO] Saved {len(df)} records to '{output_excel}'.")

    print("\n[DEBUG] Converting results to DataFrame...")
    df = pd.DataFrame(all_papers)
    print(f"[DEBUG] DataFrame created with {df.shape[0]} rows and {df.shape[1]} columns.")

    print("[DEBUG] Saving DataFrame to Excel:", output_excel)
    df.to_excel(output_excel, index=False)
    print("[INFO] Saved DataFrame to", output_excel)

if __name__ == "__main__":
    find_and_save_papers_about_bacchus_and_ariadne(
        query="Bacchus and Ariadne art history",
        email="xlct43@durham.ac.uk",
        output_excel="bacchus_ariadne_papers2.xlsx"
    )