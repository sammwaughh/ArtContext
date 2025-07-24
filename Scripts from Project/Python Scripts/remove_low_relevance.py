import os

import pandas as pd

excel_file = "large_works.xlsx"
df = pd.read_excel(excel_file)

# Remove files with relevance score < 5
for _, row in df.iterrows():
    if row["relevance score"] < 5:
        painter = row["painter"]
        if "_" in painter:
            number, painter_name = painter.split("_", 1)
            folder_name = f"{number}_large_{painter_name}  "
            # Need to add and remove "  " to deal with alternative names
        else:
            folder_name = painter
        folder_path = os.path.join("Large Works", folder_name)
        file_path = os.path.join(folder_path, str(row["full name"]))
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed: {file_path}")
        else:
            print(f"File not found: {file_path}")

# Create a filtered DataFrame with relevance score >= 5
filtered_df = df[df["relevance score"] >= 5]

# Write the filtered DataFrame as a new sheet in the Excel file
with pd.ExcelWriter(
    excel_file, engine="openpyxl", mode="a", if_sheet_exists="replace"
) as writer:
    filtered_df.to_excel(writer, sheet_name="Filtered Works", index=False)

print("New sheet 'Filtered Works' added to", excel_file)
