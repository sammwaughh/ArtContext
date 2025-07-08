#!/usr/bin/env python3
"""
MintCLIP heat-map visualisation with evidence switches
"""
from __future__ import annotations
import pathlib, numpy as np, torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from peft import PeftModel
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ── user-tweakable ───────────────────────────────────── #
TEXT_PROMPT = (
    "The Renaissance masters neither created, nor opened, but merely demonstrated the symbolic nature of the myth of female nudity."
)
IMG_PATH = pathlib.Path("Q151047_0.png")
MODEL_ID = "openai/clip-vit-base-patch32"
LORA_DIR = pathlib.Path("clip_finetuned_lora_best")
OUT_PATH = pathlib.Path("heatmap.png")

SIGMA  = 8      # blob size
GAMMA  = 0.7    # blob spread
ALPHA  = 0.45   # overlay opacity
POS_ONLY = True # True → highlight positive evidence, False → use abs
INVERT  = True # True → visualise *negative* evidence instead
# ─────────────────────────────────────────────────────── #

def load_mintclip(model_id, lora_dir, device):
    base = CLIPModel.from_pretrained(model_id)
    return PeftModel.from_pretrained(base, lora_dir).to(device).eval()

def saliency_map(model, proc, image, text, device):
    inputs = proc(text=[text], images=image,
                  return_tensors="pt", padding=True).to(device)
    inputs.pixel_values.requires_grad_(True)

    img_feat = torch.nn.functional.normalize(
        model.get_image_features(pixel_values=inputs.pixel_values), dim=-1)
    txt_feat = torch.nn.functional.normalize(
        model.get_text_features(input_ids=inputs.input_ids,
                                attention_mask=inputs.attention_mask), dim=-1)
    (img_feat * txt_feat).sum().backward()

    grads = inputs.pixel_values.grad[0]                # (3,H,W)
    if POS_ONLY:
        grads = torch.relu(grads)                      # keep positives
    else:
        grads = grads.abs()                            # magnitude of both

    grad = grads.mean(0).detach().cpu().numpy()        # (H,W) → NumPy
    grad = gaussian_filter(grad, SIGMA)
    grad -= grad.min()
    grad /= grad.max() + 1e-8
    grad **= GAMMA
    if INVERT:
        grad = 1.0 - grad                              # show opposing pixels
    return grad

def overlay_heatmap(image, heat, out_path):
    heat_img = Image.fromarray((heat * 255).astype("uint8")).resize(
        image.size, resample=Image.BICUBIC)
    heat = np.asarray(heat_img) / 255.0

    plt.figure(figsize=(image.width / 100, image.height / 100), dpi=100)
    plt.imshow(image)
    plt.imshow(heat, cmap="jet", alpha=ALPHA)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model = load_mintclip(MODEL_ID, LORA_DIR, device)

    img = Image.open(IMG_PATH).convert("RGB")
    heat = saliency_map(model, processor, img, TEXT_PROMPT, device)
    overlay_heatmap(img, heat, OUT_PATH)
    print(f"✅  Saved → {OUT_PATH.resolve()}")

if __name__ == "__main__":
    main()
