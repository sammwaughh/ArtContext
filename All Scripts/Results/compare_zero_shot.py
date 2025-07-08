#!/usr/bin/env python3
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
from peft import PeftModel

# ──────────── USER CONFIGURATION ────────────
METADATA_PATH = Path("paintings_metadata.xlsx")
ROW_INDICES   = [0, 275, 12052]  # three 0-based row numbers
IMAGE_FILES   = [
    "Q12418_0.png",    # Mona Lisa
    "Q1206860_0.png",  # Bacchus and Ariadne
    "Q20810246_0.png", # A Lady Playing the Guitar
]
CACHE_LEO     = Path("Leonardo da Vinci_clip_embedding_cache.pt")
CACHE_TITIAN  = Path("Titian_clip_embedding_cache.pt")
CACHE_VERMEER = Path("Johannes Vermeer_clip_embedding_cache.pt")
LORA_DIR      = Path("clip_finetuned_lora_best")
IMAGE_DIR     = Path(".")
OUTPUT_PATH   = Path("zero_shot_results.xlsx")
# ──────────────────────────────────────────

def load_text_cache(cache_path: Path):
    """
    Load a .pt cache file containing keys:
      - 'candidate_embeddings': Tensor (N, 512)
      - 'candidate_sentences': list of N strings
    """
    data = torch.load(cache_path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError(f"{cache_path!r} did not load as a dict")
    for key in ("candidate_embeddings", "candidate_sentences"):
        if key not in data:
            raise ValueError(f"{cache_path!r} missing key '{key}'")
    return data["candidate_embeddings"], data["candidate_sentences"]

def evaluate_zero_shot(model, processor, img_path, text_embs, sentences, device):
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        img_feats = model.get_image_features(**{k: v for k, v in inputs.items()}).squeeze(0)
    img_feats = F.normalize(img_feats, dim=-1)
    txt_feats = F.normalize(text_embs.to(device), dim=-1)
    sims = (txt_feats @ img_feats).cpu()
    best = sims.argmax().item()
    return sentences[best], sims[best].item()

def main():
    # 1) Device selection (prefer MPS, then CUDA, else CPU)
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    # 2) Load metadata
    df = pd.read_excel(METADATA_PATH)
    for col in ("Title", "File Name", "Creator"):
        if col not in df.columns:
            raise ValueError(f"Metadata missing required column '{col}'")
    subset = df.iloc[ROW_INDICES].reset_index(drop=True)

    # 3) Load text caches explicitly with the correct keys
    leo_embs,     leo_sents     = load_text_cache(CACHE_LEO)
    titian_embs,  titian_sents  = load_text_cache(CACHE_TITIAN)
    vermeer_embs, vermeer_sents = load_text_cache(CACHE_VERMEER)
    caches = {
        "leonardo da vinci": (leo_embs,     leo_sents),
        "titian":            (titian_embs,  titian_sents),
        "johannes vermeer":  (vermeer_embs, vermeer_sents),
    }

    # 4) Prepare CLIP models & processor
    MODEL_ID  = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    vanilla   = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
    base_clip = CLIPModel.from_pretrained(MODEL_ID)
    mint      = PeftModel.from_pretrained(base_clip, LORA_DIR).to(device).eval()

    # 5) Loop over paintings, run inference
    records = []
    for (row_idx, fname) in zip(ROW_INDICES, IMAGE_FILES):
        row      = df.iloc[row_idx]
        title    = row["Title"]
        creator  = row["Creator"].lower()
        img_path = IMAGE_DIR / fname
        if creator not in caches:
            raise KeyError(f"No cache provided for creator '{creator}'")
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")

        text_embs, sentences = caches[creator]
        v_sent, v_score = evaluate_zero_shot(vanilla, processor, img_path, text_embs, sentences, device)
        m_sent, m_score = evaluate_zero_shot(mint,    processor, img_path, text_embs, sentences, device)

        records.append({
            "Title":            title,
            "File Name":        fname,
            "Creator":          row["Creator"],
            "Vanilla Sentence": v_sent,
            "Vanilla Score":    v_score,
            "Mint Sentence":    m_sent,
            "Mint Score":       m_score,
        })

    # 6) Save results
    pd.DataFrame(records).to_excel(OUTPUT_PATH, index=False)
    print(f"Results written to {OUTPUT_PATH.resolve()}")

if __name__ == "__main__":
    main()
