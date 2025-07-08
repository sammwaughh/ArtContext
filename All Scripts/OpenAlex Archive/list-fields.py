import requests
import pandas as pd

# OpenAlex API base URL
OPENALEX_BASE_URL = "https://api.openalex.org"

# The Concept ID for "Art" domain in OpenAlex
ART_DOMAIN_ID = "C142362112"

# Function to fetch fields under the "Art" domain
def get_fields_under_art():
    url = f"{OPENALEX_BASE_URL}/concepts?filter=level:1,ancestors.id:{ART_DOMAIN_ID}"
    
    response = requests.get(url)

    if response.status_code != 200:
        print("Error fetching data from OpenAlex:", response.status_code)
        return None

    data = response.json()

    # Extract relevant fields
    fields_data = []
    for field in data.get("results", []):
        fields_data.append({
            "Field Name": field.get("display_name", "Unknown"),
            "Field ID": field.get("id", "N/A"),
            "Works Count": field.get("works_count", 0),
            "OpenAlex URL": field.get("id", "N/A")
        })

    return fields_data

# Function to save results to an Excel file
def save_to_excel(fields_data, filename="art_domain_fields.xlsx"):
    df = pd.DataFrame(fields_data)

    # Create a nicely formatted Excel file
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Art Domain Fields", index=False)

    print(f"Saved results to {filename}")

# Main function to fetch data and store it in Excel
def main():
    print("Fetching all fields under the 'Art' domain...")
    fields_data = get_fields_under_art()

    if fields_data:
        save_to_excel(fields_data)
        print("Process complete. Check the Excel file for results.")
    else:
        print("No data retrieved.")

if __name__ == "__main__":
    main()
