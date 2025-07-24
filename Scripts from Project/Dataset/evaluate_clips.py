import os
import sys

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from safetensors.torch import load_file as safe_load
from transformers import CLIPModel, CLIPProcessor


def load_painting_info(paintings_xlsx, row_index):
    paintings_df = pd.read_excel(paintings_xlsx)
    if row_index < 0 or row_index >= len(paintings_df):
        raise IndexError("Row index out of range.")
    return paintings_df.iloc[row_index]


def get_artist_cache_file(painters_xlsx, creator, cache_dir, suffix):
    painters_df = pd.read_excel(painters_xlsx)
    creator_tokens = creator.lower().split()
    found_artist = None
    for artist in painters_df["Artist"]:
        lower_artist = artist.lower()
        if len(creator_tokens) == 1:
            if creator_tokens[0] in lower_artist:
                found_artist = artist
                break
        else:
            # Match using the first and last token.
            if creator_tokens[0] in lower_artist and creator_tokens[-1] in lower_artist:
                found_artist = artist
                break
    if found_artist is None:
        artist_name = creator.strip()
    else:
        artist_name = found_artist.strip()
    return os.path.join(cache_dir, f"{artist_name}_{suffix}")


def load_cached_candidate_embeddings(cache_file):
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Cache file not found: {cache_file}")
    data = torch.load(cache_file, map_location="cpu")
    return data  # Expected keys: candidate_contexts, candidate_sentences, candidate_embeddings


def load_image(image_dir, image_filename):
    image_path = os.path.join(image_dir, image_filename)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def compute_image_embedding(image, model, processor, device):
    inputs = processor(
        images=image, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.squeeze(0)


def compute_contrastive_scores(candidate_embeddings, image_embedding):
    candidate_embeddings = candidate_embeddings.to(image_embedding.device)
    scores = candidate_embeddings @ image_embedding
    return scores


def print_top_candidates(title, candidate_sentences, scores, top_k=3):
    top_scores, top_indices = torch.topk(scores, k=top_k)
    print(f"\n{title}")
    for rank, (score, idx) in enumerate(zip(top_scores, top_indices), start=1):
        sentence = candidate_sentences[idx]
        print(f"{rank}. (Score: {score.item():.4f}) {sentence}")


def main():
    # File and directory settings.
    paintings_xlsx = "paintings.xlsx"
    painters_xlsx = "painters.xlsx"
    base_cache_dir = "cache_clip_embeddings"
    ft_cache_dir = "cache_clip_finetuned_embeddings"
    image_dir = "Images"

    # Get painting row index from command-line argument.
    if len(sys.argv) < 2:
        print("Usage: python evaluate_finetuned_clips.py <row_index>")
        sys.exit(1)
    try:
        row_index = int(sys.argv[1])
    except ValueError:
        print("Row index must be an integer.")
        sys.exit(1)

    # Load painting info.
    painting_row = load_painting_info(paintings_xlsx, row_index)
    creator = painting_row["Creator"]
    image_filename = painting_row["File Name"]

    print(f"Evaluating painting from creator: {creator}, image file: {image_filename}")

    # Load cached candidate embeddings for the base model.
    base_cache_file = get_artist_cache_file(
        painters_xlsx, creator, base_cache_dir, "clip_embedding_cache.pt"
    )
    print(f"Using cached candidate embeddings (Base): {base_cache_file}")
    base_cached_data = load_cached_candidate_embeddings(base_cache_file)
    candidate_embeddings_base = base_cached_data["candidate_embeddings"]
    candidate_sentences = base_cached_data[
        "candidate_sentences"
    ]  # Assume same sentences across caches.

    # Load cached candidate embeddings for the fine-tuned model.
    ft_cache_file = get_artist_cache_file(
        painters_xlsx, creator, ft_cache_dir, "finetuned_clip_embedding_cache.pt"
    )
    print(f"Using cached candidate embeddings (Fine-Tuned): {ft_cache_file}")
    ft_cached_data = load_cached_candidate_embeddings(ft_cache_file)
    candidate_embeddings_ft = ft_cached_data["candidate_embeddings"]

    # Load the image.
    image = load_image(image_dir, image_filename)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # ----- Base Model Evaluation -----
    base_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    base_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    base_model.to(device)

    base_image_embedding = compute_image_embedding(
        image, base_model, base_processor, device
    )
    base_scores = compute_contrastive_scores(
        candidate_embeddings_base, base_image_embedding
    )
    print_top_candidates(
        "Top Candidate Sentences for Base Model:",
        candidate_sentences,
        base_scores,
        top_k=3,
    )

    # ----- Fine-Tuned Model Evaluation -----
    # Load the fine-tuned model using the same procedure as in training.
    ft_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    lora_config = LoraConfig(
        r=32,  # Updated to training value
        lora_alpha=48,  # Updated to training value
        target_modules=["text_projection"],
        lora_dropout=0.1,
        bias="none",
    )
    ft_model = get_peft_model(ft_model, lora_config)
    ft_model_path = os.path.join("..", "FineTune", "clip-finetuned-lora")
    adapter_weights_path = os.path.join(ft_model_path, "adapter_model.safetensors")
    state_dict = safe_load(adapter_weights_path)
    ft_model.load_state_dict(state_dict, strict=False)
    ft_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    ft_model.to(device)

    ft_image_embedding = compute_image_embedding(image, ft_model, ft_processor, device)
    ft_scores = compute_contrastive_scores(candidate_embeddings_ft, ft_image_embedding)
    print_top_candidates(
        "Top Candidate Sentences for Fine-Tuned Model:",
        candidate_sentences,
        ft_scores,
        top_k=3,
    )


if __name__ == "__main__":
    main()
