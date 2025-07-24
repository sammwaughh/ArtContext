#!/usr/bin/env python3
"""
Module for writing results to an Excel file using openpyxl.
Uses pandas with the openpyxl engine and adjusts column widths.
The output file will contain only the PaintingID, Title, and VisualDescription,
where the VisualDescription is an approximately 80–100 word text.
"""

import pandas as pd
from openpyxl import load_workbook


def write_excel(output_file, rows):
    """
    Write a list of dictionaries (rows) to an Excel file.
    Expected dictionary keys: "PaintingID", "Title", "VisualDescription".
    Adjusts column widths for clear viewing.
    """
    headers = ["PaintingID", "Title", "VisualDescription"]
    df = pd.DataFrame(rows, columns=headers)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Results", index=False)

    wb = load_workbook(output_file)
    ws = wb["Results"]

    ws.column_dimensions["A"].width = 15  # PaintingID
    ws.column_dimensions["B"].width = 30  # Title
    ws.column_dimensions["C"].width = 150  # VisualDescription

    wb.save(output_file)
