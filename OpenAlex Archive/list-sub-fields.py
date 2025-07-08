import requests
import pandas as pd

# OpenAlex API base URL
OPENALEX_BASE_URL = "https://api.openalex.org"

# Field IDs for Art History, Visual Arts, and Aesthetics
FIELDS = {
    "Art History": "C52119013",
    "Visual Arts": "C153349607",
    "Aesthetics": "C107038049"
}

# Function to fetch subfields under a given field ID
def get_subfields(field_name, field_id):
    url = f"{OPENALEX_BASE_URL}/concepts?filter=level:2,ancestors.id:{field_id}"
    
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Error fetching data for {field_name}: {response.status_code}")
        return []

    data = response.json()

    # Extract relevant subfield details
    subfields_data = []
    for subfield in data.get("results", []):
        subfields_data.append({
            "Subfield Name": subfield.get("display_name", "Unknown"),
            "Subfield ID": subfield.get("id", "N/A"),
            "Works Count": subfield.get("works_count", 0),
            "OpenAlex URL": subfield.get("id", "N/A")
        })

    return subfields_data

# Function to save results to an Excel file with distinct sheets
def save_to_excel(subfields_dict, filename="art_subfields.xlsx"):
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        for field_name, subfields in subfields_dict.items():
            df = pd.DataFrame(subfields)
            df.to_excel(writer, sheet_name=field_name, index=False)

    print(f"Saved results to {filename}")

# Main function to fetch subfields for each field and store them
def main():
    print("Fetching subfields under Art History, Visual Arts, and Aesthetics...")

    subfields_dict = {}
    for field_name, field_id in FIELDS.items():
        subfields_dict[field_name] = get_subfields(field_name, field_id)

    if any(subfields_dict.values()):  # Ensure there's at least one valid entry
        save_to_excel(subfields_dict)
        print("Process complete. Check the Excel file for results.")
    else:
        print("No data retrieved.")

if __name__ == "__main__":
    main()
