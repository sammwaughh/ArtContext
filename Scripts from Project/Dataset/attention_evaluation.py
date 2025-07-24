import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism warning

import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from safetensors.torch import load_file as safe_load
from transformers import CLIPModel, CLIPProcessor


# --- Utility functions ---
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
            # Use first and last token for matching.
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


# --- Functions to compute embeddings ---
def compute_text_embedding(model, processor, text, device):
    text_inputs = processor(
        text=text, return_tensors="pt", truncation=True, padding=True
    )
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.squeeze(0)


def compute_image_embedding(image, model, processor, device):
    image_inputs = processor(
        images=image, return_tensors="pt", padding=True, truncation=True
    )
    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.squeeze(0)


# --- Functions to extract attention ---
def get_text_attention(model, processor, text, device):
    text_inputs = processor(text=text, return_tensors="pt", truncation=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    outputs = model.text_model(**text_inputs, output_attentions=True)
    attn = outputs.attentions[-1].mean(dim=1).squeeze(0)  # shape: [seq_len, seq_len]
    token_scores = attn.sum(dim=0).detach().cpu().numpy()
    tokens = processor.tokenizer.convert_ids_to_tokens(text_inputs["input_ids"][0])
    # Remove the first token (often the CLS or start token) from visualization.
    tokens = tokens[1:]
    token_scores = token_scores[1:]
    return tokens, token_scores


def get_image_attention(model, processor, image, device):
    image_inputs = processor(images=image, return_tensors="pt")
    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
    outputs = model.vision_model(**image_inputs, output_attentions=True)
    attn = (
        outputs.attentions[-1].mean(dim=1).squeeze(0)
    )  # shape: [num_tokens, num_tokens]
    patch_attn = attn[0, 1:].detach().cpu().numpy()
    side = int(np.sqrt(patch_attn.shape[0]))
    attn_map = patch_attn.reshape(side, side)
    return attn_map


# --- Visualization ---
def plot_attention_figure(
    image,
    tokens,
    token_scores,
    attn_map,
    model_type,
    timestamp,
    painting_title,
    cos_sim,
    input_text,
):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    axs[0].bar(range(len(tokens)), token_scores)
    axs[0].set_xticks(range(len(tokens)))
    axs[0].set_xticklabels(tokens, rotation=45, ha="right")
    axs[0].set_title(f"{model_type} Model - Text Attention")
    axs[0].set_ylabel("Attention Score")
    axs[1].imshow(image)
    im = axs[1].imshow(
        attn_map, cmap="jet", alpha=0.5, extent=(0, image.size[0], image.size[1], 0)
    )
    axs[1].set_title(f"{model_type} Model - Image Attention")
    axs[1].axis("off")
    fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    fig.suptitle(f"Cosine Similarity: {cos_sim:.4f}", fontsize=16)
    # Adjust layout to make space at the bottom and add the full text.
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])
    fig.text(0.5, 0.03, input_text, wrap=True, ha="center", fontsize=10)
    sanitized_title = painting_title.replace(" ", "_")
    model_name = model_type.lower().replace(" ", "_").replace("-", "")
    filename = f"{sanitized_title}_{model_name}_{timestamp}.png"
    filepath = os.path.join("attention-maps", filename)
    plt.savefig(filepath)
    plt.close(fig)
    print(f"Saved {model_type} attention plot as {filepath}")


# --- Main program ---
def main():
    paintings_xlsx = "paintings.xlsx"
    painters_xlsx = "painters.xlsx"
    base_cache_dir = "cache_clip_embeddings"
    ft_cache_dir = "cache_clip_finetuned_embeddings"
    image_dir = "Images"

    if len(sys.argv) < 2:
        print("Usage: python attention_visualization.py <row_index>")
        sys.exit(1)
    try:
        row_index = int(sys.argv[1])
    except ValueError:
        print("Row index must be an integer.")
        sys.exit(1)

    painting_row = load_painting_info(paintings_xlsx, row_index)
    creator = painting_row["Creator"]
    image_filename = painting_row["File Name"]
    painting_title = painting_row["Title"]
    print(
        f"Selected painting: Creator={creator}, File Name={image_filename}, Title={painting_title}"
    )
    image = load_image(image_dir, image_filename)

    base_cache_file = get_artist_cache_file(
        painters_xlsx, creator, base_cache_dir, "clip_embedding_cache.pt"
    )
    ft_cache_file = get_artist_cache_file(
        painters_xlsx, creator, ft_cache_dir, "finetuned_clip_embedding_cache.pt"
    )
    candidate_data_base = load_cached_candidate_embeddings(base_cache_file)
    candidate_data_ft = load_cached_candidate_embeddings(ft_cache_file)

    text_input_base = (
        "To illustrate, consider Rembrandt's painting known as The Jewish Bride. "
        "The couple represented is far from physically 11 attractive; yet there is great beauty in them. "
        "To appreciate this beauty, we need to be sensitive to the work's subject matter, "
        "namely human love, and to the particular way in which it is handled by Rembrandt."
    )
    text_input_ft = text_input_base

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # ----- Base CLIP Model -----
    base_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32", attn_implementation="eager"
    )
    base_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    base_model.config.output_attentions = True
    base_model.to(device)
    tokens_base, token_scores_base = get_text_attention(
        base_model, base_processor, text_input_base, device
    )
    attn_map_base = get_image_attention(base_model, base_processor, image, device)

    # Compute cosine similarity for Base model.
    base_text_embedding = compute_text_embedding(
        base_model, base_processor, text_input_base, device
    )
    base_image_embedding = compute_image_embedding(
        image, base_model, base_processor, device
    )
    cos_sim_base = torch.dot(base_text_embedding, base_image_embedding).item()
    print("Cosine similarity (Base):", cos_sim_base)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_attention_figure(
        image,
        tokens_base,
        token_scores_base,
        attn_map_base,
        "Base",
        timestamp,
        painting_title,
        cos_sim_base,
        text_input_base,
    )

    # ----- Fine-Tuned CLIP Model -----
    # Load the base CLIP model and then wrap it with the LoRA adapter.
    ft_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32", attn_implementation="eager"
    )
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
    # Merge adapter weights into the base model for inference.
    ft_model.merge_and_unload()
    ft_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    ft_model.config.output_attentions = True
    ft_model.to(device)
    tokens_ft, token_scores_ft = get_text_attention(
        ft_model, ft_processor, text_input_ft, device
    )
    attn_map_ft = get_image_attention(ft_model, ft_processor, image, device)

    # Compute cosine similarity for Fine-Tuned model.
    ft_text_embedding = compute_text_embedding(
        ft_model, ft_processor, text_input_ft, device
    )
    ft_image_embedding = compute_image_embedding(image, ft_model, ft_processor, device)
    cos_sim_ft = torch.dot(ft_text_embedding, ft_image_embedding).item()
    print("Cosine similarity (Fine-Tuned):", cos_sim_ft)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_attention_figure(
        image,
        tokens_ft,
        token_scores_ft,
        attn_map_ft,
        "Fine-Tuned",
        timestamp,
        painting_title,
        cos_sim_ft,
        text_input_ft,
    )


if __name__ == "__main__":
    main()
