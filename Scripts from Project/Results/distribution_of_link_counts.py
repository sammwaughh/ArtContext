#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd

EXCEL_FILE = "paintings_metadata.xlsx"
NROWS = 300
HIST_BINS = 30
OUTPUT_PNG = "linkcount_histogram.png"


def main():
    # 1. Read column headers only
    df_head = pd.read_excel(EXCEL_FILE, nrows=0)
    print("Available columns:")
    for col in df_head.columns:
        print(" -", col)
    print("\nUsing only 'Title' and 'Link Count' from first {} rows.\n".format(NROWS))

    # 2. Load first NROWS rows, only Title and Link Count
    df = pd.read_excel(EXCEL_FILE, usecols=["Title", "Link Count"], nrows=NROWS)

    # Optional: report basic stats
    print("Link Count stats (first {} paintings):".format(NROWS))
    print(df["Link Count"].describe())

    # 3. Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(df["Link Count"], bins=HIST_BINS, edgecolor="black")
    plt.xlabel("Link Count")
    plt.ylabel("Number of Paintings")
    plt.title("Distribution of Link Count (first {} paintings)".format(NROWS))
    plt.tight_layout()

    # 4. Save to PNG
    plt.savefig(OUTPUT_PNG)
    print("\nSaved histogram to '{}'.".format(OUTPUT_PNG))


if __name__ == "__main__":
    main()
