import os
import pandas as pd
from openpyxl import load_workbook

# Read painters.xlsx and build a mapping: lowercase painter name -> new painter name with prefix.
painters_df = pd.read_excel("painters.xlsx")
painter_mapping = {}
for idx, row in painters_df.iterrows():
    # Use the "Query String" column and strip extra spaces.
    painter_name = str(row["Query String"]).strip()
    # idx is already 0-indexed, so it satisfies "row number minus 1"
    painter_mapping[painter_name.lower()] = f"{idx}_{painter_name}"

# Read grouped_ranked_large_works.xlsx
df = pd.read_excel("grouped_ranked_large_works.xlsx")

# Function to rename a painter using the mapping; if not found, assign a default high prefix.
def get_new_painter(painter):
    key = str(painter).lower().strip()
    if key in painter_mapping:
        return painter_mapping[key]
    else:
        # Default prefix for painters not found in painters.xlsx.
        return f"9999_{painter.strip()}"

# Rename painters using the mapping.
df["painter"] = df["painter"].apply(get_new_painter)

# Extract numeric prefix from the renamed painter.
def extract_prefix(painter_str):
    prefix_part = painter_str.split('_')[0].strip()
    try:
        return int(prefix_part)
    except ValueError:
        return 9999  # Fallback if conversion fails.

df["painter_prefix"] = df["painter"].apply(extract_prefix)

# Sort by the numeric prefix (ensuring order 1,2,3,...,10,11,...) and then by relevance score (descending)
df_sorted = df.sort_values(by=["painter_prefix", "relevance score"], ascending=[True, False])

# Within each painter group, rank the works by relevance score (highest gets rank 1).
df_sorted["rank"] = (
    df_sorted.groupby("painter")["relevance score"]
    .rank(method="dense", ascending=False)
    .fillna(0)
    .astype(int)
)

# Drop the temporary prefix column.
df_sorted.drop(columns=["painter_prefix"], inplace=True)

# Save the result to large_works.xlsx.
output_file = "large_works.xlsx"
df_sorted.to_excel(output_file, index=False)

# Adjust column widths for readability using openpyxl.
wb = load_workbook(output_file)
ws = wb.active

for col in ws.columns:
    col_letter = col[0].column_letter
    max_length = max((len(str(cell.value)) if cell.value is not None else 0) for cell in col)
    ws.column_dimensions[col_letter].width = max_length + 2

wb.save(output_file)
