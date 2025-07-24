import time

import pandas as pd
import requests

BASE_URL = "https://api.openalex.org/concepts"


def get_concept_by_name(name):
    params = {"search": name, "per_page": 5}
    r = requests.get(BASE_URL, params=params)
    r.raise_for_status()
    for concept in r.json()["results"]:
        if concept["display_name"].lower() == name.lower():
            return concept
    return None


def fetch_concepts(filter_str):
    results = []
    cursor = "*"
    while True:
        params = {"filter": filter_str, "per_page": 200, "cursor": cursor}
        r = requests.get(BASE_URL, params=params)
        r.raise_for_status()
        data = r.json()
        results.extend(data.get("results", []))
        next_cursor = data.get("meta", {}).get("next_cursor")
        if not next_cursor:
            break
        cursor = next_cursor
        time.sleep(1)
    return results


def get_children(parent):
    child_level = parent["level"] + 1
    filter_str = f"ancestors.id:{parent['id']},level:{child_level}"
    return fetch_concepts(filter_str)


def main():
    fields = ["Art History", "Visual Arts", "Aesthetics"]
    with pd.ExcelWriter("openalex_concepts.xlsx", engine="openpyxl") as writer:
        for field in fields:
            print(f"Processing field: {field}")
            field_concept = get_concept_by_name(field)
            if not field_concept:
                print(f"Field '{field}' not found. Skipping.")
                continue
            rows = []
            subfields = get_children(field_concept)
            for sub in subfields:
                rows.append(
                    {
                        "Level": "Sub-field",
                        "Name": sub["display_name"],
                        "ID": sub["id"],
                        "Works Count": sub["works_count"],
                        "OpenAlex URL": sub["id"],
                        "Parent": field_concept["display_name"],
                    }
                )
                topics = get_children(sub)
                for topic in topics:
                    rows.append(
                        {
                            "Level": "Topic",
                            "Name": topic["display_name"],
                            "ID": topic["id"],
                            "Works Count": topic["works_count"],
                            "OpenAlex URL": topic["id"],
                            "Parent": sub["display_name"],
                        }
                    )
            df = pd.DataFrame(rows)
            sheet_name = field if len(field) <= 31 else field[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print("Data written to openalex_concepts.xlsx")


if __name__ == "__main__":
    main()
