import ast

import pandas as pd


def parse_entity(entity):
    """
    Given an entity cell (expected to be a dict or its string representation)
    with keys 'display_name' and 'id', return a tuple: (display_name, id).
    If parsing fails, return (entity, "").
    """
    if pd.isna(entity):
        return ("", "")
    if isinstance(entity, dict):
        return (entity.get("display_name", ""), entity.get("id", ""))
    try:
        # Try converting a string representation to a dict.
        entity_dict = ast.literal_eval(entity)
        if isinstance(entity_dict, dict):
            return (entity_dict.get("display_name", ""), entity_dict.get("id", ""))
    except Exception:
        pass
    # Fallback: return the original entity as the name.
    return (entity, "")


def clean_topics(
    input_file="openalex_topics_formatted.xlsx",
    output_file="openalex_topics_cleaned.xlsx",
):
    # Read the formatted Excel file
    df = pd.read_excel(input_file, engine="openpyxl")

    # Parse the subfield, field, and domain columns into name and ID parts
    df["Subfield Name"], df["Subfield ID"] = zip(*df["subfield"].apply(parse_entity))
    df["Field Name"], df["Field ID"] = zip(*df["field"].apply(parse_entity))
    df["Domain Name"], df["Domain ID"] = zip(*df["domain"].apply(parse_entity))

    # Build a new DataFrame with the desired columns:
    # Topic Name, Topic ID, Subfield Name, Subfield ID, Field Name, Field ID, Domain Name, Domain ID,
    # Works Count, Cited By Count, Description, Keywords
    df_new = df[
        [
            "display_name",
            "id",
            "Subfield Name",
            "Subfield ID",
            "Field Name",
            "Field ID",
            "Domain Name",
            "Domain ID",
            "works_count",
            "cited_by_count",
            "description",
            "keywords",
        ]
    ].copy()

    # Rename the columns to the final names
    df_new.columns = [
        "Topic Name",
        "Topic ID",
        "Subfield Name",
        "Subfield ID",
        "Field Name",
        "Field ID",
        "Domain Name",
        "Domain ID",
        "Works Count",
        "Cited By Count",
        "Description",
        "Keywords",
    ]

    # Write the cleaned data to a new Excel file
    df_new.to_excel(output_file, index=False, engine="openpyxl")
    print(f"Cleaned file saved as {output_file}")


if __name__ == "__main__":
    clean_topics()
