import glob
import os

import pandas as pd
from openpyxl import load_workbook

folder = "ExcelFiles"
excel_files = glob.glob(os.path.join(folder, "*.xlsx"))
all_large_rows = []

for count, file_path in enumerate(excel_files, start=1):
    file_name = os.path.basename(file_path)
    print(f"Processing file {count}/{len(excel_files)}: {file_name}")

    try:
        df = pd.read_excel(file_path, sheet_name="For Markdown")
    except Exception as e:
        print(f"  Error reading '{file_name}': {e}")
        continue

    # Filter rows with file size > 10000
    large_df = df[df["file size"] > 10000].copy()
    if large_df.empty:
        continue

    # Extract painter name (everything before the underscore)
    painter = file_name.split("_")[0]
    large_df["painter"] = painter

    # Select required columns
    large_df = large_df[["full name", "relevance score", "file size", "painter"]]

    # Convert file size to MB (divide by 1000 and round to 1 decimal place)
    large_df["file size"] = (large_df["file size"] / 1000).round(1)

    all_large_rows.append(large_df)

# Concatenate all large rows into one DataFrame
result_df = (
    pd.concat(all_large_rows, ignore_index=True)
    if all_large_rows
    else pd.DataFrame(columns=["full name", "relevance score", "file size", "painter"])
)

output_file = "large_works.xlsx"
result_df.to_excel(output_file, index=False)

# Read the main sheet from large_works.xlsx
df = pd.read_excel("large_works.xlsx")

# Sort by painter (alphabetically) then by relevance score (descending)
df_sorted = df.sort_values(by=["painter", "relevance score"], ascending=[True, False])

# Group by painter and rank the works by relevance score.
# Use fillna(0) to handle missing scores before converting to int.
df_sorted["rank"] = (
    df_sorted.groupby("painter")["relevance score"]
    .rank(method="dense", ascending=False)
    .fillna(0)
    .astype(int)
)

# Optionally, write the grouped and ranked data to a new Excel file.
output_file = "grouped_ranked_large_works.xlsx"
df_sorted.to_excel(output_file, index=False)
print(f"Grouped and ranked data saved to {output_file}")

# Format columns for readability using openpyxl
wb = load_workbook(output_file)
ws = wb.active

for col in ws.columns:
    col_letter = col[0].column_letter
    max_length = max(
        (len(str(cell.value)) if cell.value is not None else 0) for cell in col
    )
    ws.column_dimensions[col_letter].width = max_length + 2

wb.save(output_file)
