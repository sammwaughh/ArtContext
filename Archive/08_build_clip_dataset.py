import time

import nltk
import pandas as pd


def find_matching_paintings(paintings_range, painter_name):
    # Derive candidate strings (after stripping whitespace and lowercasing)
    full_name = painter_name.strip().lower()
    first_name = painter_name.split()[0].strip().lower()
    last_name = painter_name.split()[-1].strip().lower()

    # Build exact equality masks.
    mask_full = paintings_range["Creator"].str.strip().str.lower() == full_name
    mask_first = paintings_range["Creator"].str.strip().str.lower() == first_name
    mask_last = paintings_range["Creator"].str.strip().str.lower() == last_name

    combined_mask = mask_full | mask_first | mask_last
    count = combined_mask.sum()
    if count > 0:
        print(f"Found {count} paintings matching variations of '{painter_name}'.")
        return paintings_range[combined_mask]
    else:
        print(f"No paintings found for variations of '{painter_name}'.")
        return pd.DataFrame()


def main():
    nltk.download("punkt")
    nltk.download("punkt_tab")

    start_time = time.perf_counter()

    # Load painters.xlsx and paintings.xlsx.
    painters_df = pd.read_excel("painters.xlsx")
    paintings_df = pd.read_excel("paintings.xlsx")

    # Optionally, restrict to a specific inclusive range of paintings.
    painting_start_index = 0
    painting_end_index = 40000  # Process rows 0 through 40000 inclusive.
    paintings_range = paintings_df.iloc[painting_start_index : painting_end_index + 1]

    # Optionally, restrict to a specific inclusive range of painters.
    painter_start_index = 0
    painter_end_index = 454  # Process painters rows 0 through 454 inclusive.
    painters_subset = painters_df.iloc[painter_start_index : painter_end_index + 1]

    # List to store dataset records.
    dataset_rows = []

    # Iterate over each painter in the subset.
    for _, painter_row in painters_subset.iterrows():
        painter_name = painter_row["Artist"]
        print(f"\nProcessing painter: {painter_name}")
        matching_paintings = find_matching_paintings(paintings_range, painter_name)
        if matching_paintings.empty:
            continue

        for _, row in matching_paintings.iterrows():
            title = row["Title"]
            file_name = row["File Name"]

            # Ensure sentence columns are strings then join them.
            caption = " ".join(
                [str(row["Sentence 1"]), str(row["Sentence 2"]), str(row["Sentence 3"])]
            ).strip()

            dataset_rows.append(
                {"Title": title, "File Name": file_name, "Caption": caption}
            )

    # Create a DataFrame from the records and save as CSV.
    dataset_df = pd.DataFrame(dataset_rows)
    output_csv = "CLIP_dataset_labels.csv"
    dataset_df.to_csv(output_csv, index=False)

    elapsed_time = time.perf_counter() - start_time
    print(f"\nProcessed {len(dataset_df)} paintings in {elapsed_time:.1f} seconds.")
    print(f"Dataset saved to {output_csv}")


if __name__ == "__main__":
    main()
