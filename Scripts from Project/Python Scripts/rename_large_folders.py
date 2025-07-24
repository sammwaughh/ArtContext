import os

import pandas as pd

# Read painters.xlsx
df = pd.read_excel("painters.xlsx")

# Base directory containing the folders
base_dir = "Large Works"

# Iterate over painters.xlsx rows
for idx, row in df.iterrows():
    # Get the painter's name from the "Query Search" column
    painter_name = str(row["Query String"]).lower()
    # Determine the prefix (row number minus 1; idx is 0-indexed)
    prefix = idx
    # Construct the expected current folder name (e.g., "large_bearden")
    old_folder_name = f"large_{painter_name}"
    old_folder_path = os.path.join(base_dir, old_folder_name)

    if os.path.isdir(old_folder_path):
        # Construct the new folder name with the prefix (e.g., "72_large_bearden")
        new_folder_name = f"{prefix}_{old_folder_name}"
        new_folder_path = os.path.join(base_dir, new_folder_name)
        os.rename(old_folder_path, new_folder_path)
        print(f"Renamed '{old_folder_name}' to '{new_folder_name}'")
    else:
        print(f"Folder not found: '{old_folder_name}'")
