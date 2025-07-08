#!/usr/bin/env python3
"""
Main script to process paintings from paintings.xlsx.
It extracts candidate sentences from markdown articles that mention the painting name
(or its coreferent mentions via a sliding window), then builds an expanded query based on the painting's title and depicted content.
It extracts an approximately 100-word visual description using a hybrid ranking approach.
The results are written to an Excel file.
"""

import os
import re
import nltk
import pandas as pd

from data_loader import load_paintings, load_painters, get_query_string
from text_processing import (
    preprocess_markdown,
    resolve_coreferences,
    filter_non_heading_lines,
    split_into_sentences,
    build_expanded_query_sentence,
    gather_sentences_with_sliding_window,
)
from sentence_ranking import extract_visual_description
from excel_writer import write_excel

# --- Configuration ---
START_ROW = 6
END_ROW = 6                # Process first 5 paintings (for example)
WINDOW_SIZE = 10             # Sliding window ±10 sentences
TARGET_WORD_COUNT = 100      # Desired word count for the visual description
MAX_MARKDOWN_FILES = 3      # Maximum number of markdown files to read per painting
# ----------------------

def normalize_string(s):
    """Lowercase and collapse whitespace."""
    return re.sub(r"\s+", " ", s).strip()

def get_painter_directory(base_dir, query_str):
    """Locate the painter's folder in 'Small Markdown' matching query_str (case-insensitive)."""
    small_md_dir = os.path.join(base_dir, "Small Markdown")
    norm_query = normalize_string(query_str)
    for folder in os.listdir(small_md_dir):
        folder_path = os.path.join(small_md_dir, folder)
        if os.path.isdir(folder_path) and normalize_string(folder) == norm_query:
            return folder_path
    return None

def get_works_metadata_file(base_dir, query_str):
    """Locate the works metadata file named '{query_str}_works.xlsx' (case-insensitive)."""
    works_meta_dir = os.path.join(base_dir, "Works Metadata")
    norm_query = normalize_string(query_str)
    for file in os.listdir(works_meta_dir):
        if file.lower().endswith("_works.xlsx"):
            file_query = file.lower().replace("_works.xlsx", "").strip()
            if file_query == norm_query:
                return os.path.join(works_meta_dir, file)
    return None

def get_article_directory(painter_dir, article_full_name):
    """
    Compare subdirectory names (after stripping and lowercasing)
    to the article_full_name and return the matching folder path.
    """
    target = article_full_name.strip().lower()
    for folder in os.listdir(painter_dir):
        folder_path = os.path.join(painter_dir, folder)
        if os.path.isdir(folder_path) and folder.strip().lower() == target:
            return folder_path
    return None

def process_painting(painting, base_dir, painters_df):
    """
    Process one painting:
      1. Determine the correct query string.
      2. Locate the painter folder and works metadata.
      3. For each article, open markdown files; preprocess the text, resolve coreferences,
         filter out non-content lines, split into sentences, and gather a sliding window
         (±WINDOW_SIZE) around sentences that mention the painting.
         Only process up to MAX_MARKDOWN_FILES markdown files per painting.
      4. Build an expanded query from the painting title and depicted content.
      5. Extract an approximately 100-word visual description using our ranking method.
      6. Return a dict with the painting ID, title, and visual description.
    """
    painting_id = painting["Painting ID"]
    title = painting["Title"]
    creator = painting["Creator"]
    depicts = painting["Depicts"]

    print(f"Processing Painting {painting_id}: '{title}' by {creator}")

    query_str = get_query_string(creator, painters_df) or creator
    painter_dir = get_painter_directory(base_dir, query_str)
    if painter_dir is None:
        print(f"Folder for '{query_str}' not found in Small Markdown.")
        return None
    print(f"Opening directory for artist '{query_str}': {painter_dir}")

    works_metadata_file = get_works_metadata_file(base_dir, query_str)
    if works_metadata_file is None:
        print(f"Works metadata file for '{query_str}' not found.")
        return None

    try:
        works_df = pd.read_excel(works_metadata_file, sheet_name="For Markdown")
    except Exception as e:
        print(f"Error loading works metadata for '{query_str}': {e}")
        return None

    col_map = {col.strip().lower(): col for col in works_df.columns}
    if "full name" not in col_map:
        print(f"'Full Name' column missing in works metadata for '{query_str}'.")
        return None

    raw_article_names = works_df[col_map["full name"]].tolist()
    article_names = [re.sub(r"\.pdf$", "", name.strip(), flags=re.IGNORECASE)
                     for name in raw_article_names]

    candidate_sentences = []
    articles_read = 0
    md_files_read = 0  # Counter for markdown files processed
    nltk.download("punkt", quiet=True)

    for article_name in article_names:
        if md_files_read >= MAX_MARKDOWN_FILES:
            break
        art_dir = get_article_directory(painter_dir, article_name)
        if art_dir is None:
            continue
        articles_read += 1
        for file in os.listdir(art_dir):
            if md_files_read >= MAX_MARKDOWN_FILES:
                break
            if file.lower().endswith(".md"):
                md_file = os.path.join(art_dir, file)
                print(f"Opening markdown file: {file}")
                try:
                    with open(md_file, "r", encoding="utf-8") as f:
                        md_content = f.read()
                except Exception as e:
                    print(f"Error reading {md_file}: {e}")
                    continue

                raw_text = preprocess_markdown(md_content)
                if title.lower() not in raw_text.lower():
                    continue

                resolved_text = resolve_coreferences(raw_text, title)
                filtered_text = filter_non_heading_lines(resolved_text)
                sentences = split_into_sentences(filtered_text)
                windowed_sents = gather_sentences_with_sliding_window(sentences, title, window=WINDOW_SIZE)
                candidate_sentences.extend(windowed_sents)
                md_files_read += 1

    print(f"Processed {articles_read} articles and {md_files_read} markdown files for painting '{title}'.")
    if not candidate_sentences:
        print(f"No candidate sentences found for painting '{title}'.")
        return None

    query_sentence = build_expanded_query_sentence(title, depicts)
    visual_description = extract_visual_description(query_sentence, candidate_sentences, target_word_count=TARGET_WORD_COUNT)

    return {
        "PaintingID": painting_id,
        "Title": title,
        "VisualDescription": visual_description,
    }

def main():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    paintings_file = os.path.join(base_dir, "paintings.xlsx")
    paintings_df = load_paintings(paintings_file)
    if paintings_df is None:
        return

    painters_file = os.path.join(base_dir, "painters.xlsx")
    painters_df = load_painters(painters_file)

    subset = paintings_df.iloc[START_ROW:END_ROW + 1]
    results = []
    for _, painting in subset.iterrows():
        row_data = process_painting(painting, base_dir, painters_df)
        if row_data:
            results.append(row_data)
        else:
            results.append({
                "PaintingID": painting.get("Painting ID", ""),
                "Title": painting.get("Title", ""),
                "VisualDescription": "",
            })

    output_file = os.path.join(base_dir, "visual_descriptions.xlsx")
    write_excel(output_file, results)
    print(f"Results written to {output_file}")

if __name__ == "__main__":
    main()
