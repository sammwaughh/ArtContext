"""
Data loader module for loading paintings and painters Excel files.
Also provides a helper to obtain the query string from painters.xlsx.
"""

import pandas as pd


def load_paintings(file_path):
    """
    Load paintings.xlsx into a DataFrame.
    Expected columns include: "Title", "File Name", "Creator", "Movements",
    "Depicts", "Year", "Wikipedia URL", "Link Count", "Painting ID",
    "Creator ID", "Movement IDs".
    """
    try:
        return pd.read_excel(file_path)
    except Exception as e:
        print(f"Error loading paintings.xlsx: {e}")
        return None


def load_painters(file_path):
    """
    Load painters.xlsx into a DataFrame.
    Expected columns: "Artist" and "Query String".
    """
    try:
        return pd.read_excel(file_path)
    except Exception as e:
        print(f"Error loading painters.xlsx: {e}")
        return None


def get_query_string(creator, painters_df):
    """
    Given a creator name from paintings.xlsx, search painters_df (from painters.xlsx)
    for a row where "Artist" matches (case-insensitive). Return the corresponding
    "Query String" if found; otherwise, return None.
    """
    norm_creator = creator.strip().lower()
    for _, row in painters_df.iterrows():
        artist = str(row.get("Artist", "")).strip().lower()
        if artist == norm_creator:
            return row.get("Query String", "").strip()
    return None
