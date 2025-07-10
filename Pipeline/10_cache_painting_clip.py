import os
import re
import time

import nltk
import pandas as pd
import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# We no longer need to import peft or safetensors here
# from peft import get_peft_model, LoraConfig
# from safetensors.torch import load_file as safe_load


def find_matching_paintings(paintings_range, painter_name):
    mask_exact = paintings_range["Creator"].str.lower() == painter_name.lower()
    if mask_exact.sum() > 0:
        print(
            f"Found {mask_exact.sum()} paintings with exact match for '{painter_name}'."
        )
        return paintings_range[mask_exact]

    first_name = painter_name.split()[0]
    mask_first = (
        paintings_range["Creator"]
        .str.lower()
        .str.contains(first_name.lower(), na=False)
    )
    if mask_first.sum() > 0:
        print(
            f"No exact match found. Found {mask_first.sum()} "
            f"paintings with first name '{first_name}'."
        )
        return paintings_range[mask_first]

    last_name = painter_name.split()[-1]
    mask_last = (
        paintings_range["Creator"].str.lower().str.contains(last_name.lower(), na=False)
    )
    if mask_last.sum() > 0:
        print(
            "No exact or first name match found. "
            f"Found {mask_last.sum()} paintings with last name '{last_name}'."
        )
        return paintings_range[mask_last]

    print(f"No paintings found for variations of '{painter_name}'.")
    return pd.DataFrame()


def extract_sentences_from_markdown(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    # Remove code blocks and inline code
    content = re.sub(r"```[\s\S]*?```", "", content)
    content = re.sub(r"`[^`]*`", "", content)
    # Replace Markdown links with just the text
    content = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", content)
    # Remove image links and remaining markdown symbols
    content = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", content)
    content = re.sub(r"[#>*_~\-]", "", content)
    sentences = nltk.sent_tokenize(content)
    return [s.strip() for s in sentences if len(s.split()) > 3]


def build_context_sentences(sentences):
    candidate_contexts = []
    for i, sentence in enumerate(sentences):
        context = []
        if i > 0:
            context.append(sentences[i - 1])
        context.append(sentence)
        if i < len(sentences) - 1:
            context.append(sentences[i + 1])
        candidate_contexts.append(" ".join(context))
    return candidate_contexts


def encode_texts(texts, model, processor, batch_size, device):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
        batch_texts = texts[i : i + batch_size]
        inputs = processor(
            text=batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(device)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        all_embeddings.append(text_features.cpu())
    return torch.cat(all_embeddings)


def process_artist(artist_dir_name, root_dir, model, processor, device):
    artist_dir = os.path.join(root_dir, str(artist_dir_name))
    print(f"\nProcessing artist directory: {artist_dir}")
    candidate_contexts_all = []
    candidate_sentences_all = []
    total_md_files = 0

    if not os.path.exists(artist_dir):
        print(f"Artist directory {artist_dir} does not exist.")
        return None

    paper_dirs = [
        d for d in os.listdir(artist_dir) if os.path.isdir(os.path.join(artist_dir, d))
    ]
    for paper_dir in tqdm(paper_dirs, desc="Processing papers", leave=False):
        paper_path = os.path.join(artist_dir, paper_dir)
        md_files = [f for f in os.listdir(paper_path) if f.endswith(".md")]
        if not md_files:
            continue
        total_md_files += len(md_files)
        md_file_path = os.path.join(paper_path, md_files[0])
        sentences = extract_sentences_from_markdown(md_file_path)
        if not sentences:
            continue
        contexts = build_context_sentences(sentences)
        candidate_contexts_all.extend(contexts)
        candidate_sentences_all.extend(sentences)

    print(f"Total markdown files processed: {total_md_files}")
    print(f"Candidate sentences collected: {len(candidate_sentences_all)}")
    print(f"Candidate contexts built: {len(candidate_contexts_all)}")
    if not candidate_contexts_all:
        print("No candidate sentences found.")
        return None

    print("Encoding candidate contexts in batches...")
    candidate_embeddings = encode_texts(
        candidate_contexts_all, model, processor, batch_size=64, device=device
    )
    print("Candidate embeddings computed for artist directory:", artist_dir)
    return {
        "candidate_contexts": candidate_contexts_all,
        "candidate_sentences": candidate_sentences_all,
        "candidate_embeddings": candidate_embeddings,
    }


def main():
    nltk.download("punkt")
    nltk.download("punkt_tab")

    start_time = time.perf_counter()

    # Load painters data from Excel
    painters_df = pd.read_excel("painters.xlsx")
    root_dir = "Markdown Academic Papers"
    cache_dir = "cache_clip_finetuned_embeddings"
    os.makedirs(cache_dir, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    print("Loading fine-tuned CLIP model and processor using Hugging Face...")

    # --- Load the fine-tuned model ---
    ft_model_path = os.path.join("..", "FineTune", "clip-finetuned-lora")
    ft_model = CLIPModel.from_pretrained(ft_model_path)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    ft_model.to(device)

    painter_start_index = 0  # First 20 artists inclusive: indices 0 to 20
    painter_end_index = 20
    painters_subset = painters_df.iloc[painter_start_index : painter_end_index + 1]

    total_processed = 0

    for _, row in painters_subset.iterrows():
        artist = row["Artist"]
        artist_dir_name = row["Query String"]
        # Save file as {artist}_finetuned_clip_embedding_cache.pt
        cache_file = os.path.join(
            cache_dir, f"{artist}_finetuned_clip_embedding_cache.pt"
        )
        if os.path.exists(cache_file):
            print(f"Cache for artist {artist} already exists. Skipping.")
            continue
        print(
            f"\nProcessing artist: {artist} with directory mapping: {artist_dir_name}"
        )
        data = process_artist(artist_dir_name, root_dir, ft_model, processor, device)
        if data is None:
            print(f"No data for artist {artist}.")
            continue
        print(
            "Candidate embeddings computed. Now saving candidate embeddings to cache..."
        )
        torch.save(data, cache_file)
        print(f"Saved cache for artist {artist} to {cache_file}")
        total_processed += 1

    elapsed_time = time.perf_counter() - start_time
    elapsed_rounded = round(elapsed_time, 1)
    rate = (
        round(elapsed_time / total_processed, 1)
        if total_processed > 0
        else float("inf")
    )

    print("\nFinished processing painters in the specified range.")
    print(
        f"Processed {total_processed} painters in {elapsed_rounded} s "
        f"({rate} s per painter)."
    )


if __name__ == "__main__":
    main()
