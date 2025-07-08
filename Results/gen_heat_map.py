#!/usr/bin/env python3
"""
MintCLIP heat-map visualisation
───────────────────────────────
Edit TEXT_PROMPT, IMG_PATH, LORA_DIR as you like and run:

    $ python mintclip_heatmap.py

Requires: torch, transformers, peft, pillow, matplotlib, scipy
"""

from __future__ import annotations
import pathlib, numpy as np, torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from peft import PeftModel
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter          # for blob-size control


# ── user-tweakable parameters ──────────────────────────────────────────────── #
TEXT_PROMPT = (
    "The Milkmaid (1660) by Johannes Vermeer. Style: Baroque, Dutch Golden Age "
    "painting. Depicts: big chest, drapery, heating pad. Although this work does "
    "not break free from the inherent teasing hints of maids, it gives workers a "
    "sacred posture in this theme, demonstrating recognition and respect for working women."
)
IMG_PATH = pathlib.Path("Q167605_0.png")            # any RGB image
MODEL_ID = "openai/clip-vit-base-patch32"
LORA_DIR = pathlib.Path("clip_finetuned_lora_best") # MintCLIP LoRA
OUT_PATH = pathlib.Path("heatmap.png")

SIGMA = 8        # ↑ bigger = larger / blurrier blobs
GAMMA = 0.7      # <1 enlarges faint regions, >1 sharpens
ALPHA = 0.45     # heat-map opacity
# ───────────────────────────────────────────────────────────────────────────── #


def load_mintclip(model_id: str, lora_dir: pathlib.Path,
                  device: torch.device) -> CLIPModel:
    """Base CLIP + LoRA adapters → MintCLIP."""
    base = CLIPModel.from_pretrained(model_id)
    return PeftModel.from_pretrained(base, lora_dir).to(device).eval()


def saliency_map(model: CLIPModel, proc: CLIPProcessor,
                 image: Image.Image, text: str,
                 device: torch.device) -> np.ndarray:
    """Gradient-based saliency wrt similarity(image, text)."""
    inputs = proc(text=[text], images=image,
                  return_tensors="pt", padding=True).to(device)
    inputs.pixel_values.requires_grad_(True)

    img_feat = torch.nn.functional.normalize(
        model.get_image_features(pixel_values=inputs.pixel_values), dim=-1)
    txt_feat = torch.nn.functional.normalize(
        model.get_text_features(input_ids=inputs.input_ids,
                                attention_mask=inputs.attention_mask), dim=-1)
    sim = (img_feat * txt_feat).sum()
    sim.backward()

    # → NumPy early, then blur / gamma
    grad = (
        inputs.pixel_values.grad[0]     # tensor (3,H,W)
        .abs()
        .mean(0)                        # (H,W)
        .detach()
        .cpu()
        .numpy()                        # NumPy array
    )
    grad = gaussian_filter(grad, sigma=SIGMA)       # blob size
    grad -= grad.min()
    grad /= grad.max() + 1e-8
    grad = grad ** GAMMA                            # blob spread
    return grad                                     # 0-1 heat-map


def overlay_heatmap(image: Image.Image, heat: np.ndarray,
                    out_path: pathlib.Path) -> None:
    """Save image + semi-transparent heat-map."""
    heat_img = Image.fromarray((heat * 255).astype("uint8"))
    heat_img = heat_img.resize(image.size, resample=Image.BICUBIC)
    heat     = np.asarray(heat_img) / 255.0

    plt.figure(figsize=(image.width / 100, image.height / 100), dpi=100)
    plt.imshow(image)
    plt.imshow(heat, cmap="jet", alpha=ALPHA)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model = load_mintclip(MODEL_ID, LORA_DIR, device)

    img = Image.open(IMG_PATH).convert("RGB")
    heat = saliency_map(model, processor, img, TEXT_PROMPT, device)
    overlay_heatmap(img, heat, OUT_PATH)
    print(f"✅  Saved visualisation → {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
