import os
import pandas as pd
import shutil

# Read large_works.xlsx
df = pd.read_excel("large_works.xlsx")

# Get unique painters from the DataFrame
painters = df["painter"].dropna().unique()

for painter in painters:
    print(f"Processing painter: {painter}")
    
    # Filter rows for the current painter (case-insensitive match)
    df_painter = df[df["painter"].str.lower() == painter.lower()]
    
    if df_painter.empty:
        print(f"No works found for painter: {painter}")
        continue

    # Define source directory for the painter's PDFs and destination directory
    source_dir = os.path.join("PDFs", painter)
    dest_dir = os.path.join("Large Works", f"large_{painter}")
    os.makedirs(dest_dir, exist_ok=True)

    # Copy each PDF from the source directory to the destination directory
    for _, row in df_painter.iterrows():
        file_name = row["full name"]
        if not file_name.lower().endswith(".pdf"):
            file_name += ".pdf"
        src_file = os.path.join(source_dir, file_name)
        dest_file = os.path.join(dest_dir, file_name)
        if os.path.exists(src_file):
            shutil.copy2(src_file, dest_file)
        
print("Done.")
