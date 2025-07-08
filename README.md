# ArtContext — PaintingCLIP Pipeline  
*End-to-end generation of a fine-tuned CLIP model for art-historical images*

---

## 1 What this repository does

ArtContext starts with nothing but public data (Wikidata + OpenAlex) and
finishes with **PaintingCLIP** – a CLIP ViT-B/32 model adapted to
paintings via LoRA fine-tuning.  
The **`Pipeline/`** directory now contains every script that is actually
executed:

| # | Script | Purpose (one-line) |
|---|--------|--------------------|
| 01 | `01_wikidata_harvest.py` | Harvest master list of paintings from Wikidata → `paintings.xlsx`. |
| 02 | `02_painter_list.py` | Write 200-row `painters.xlsx` (artist ↔ query string). |
| 03 | `03_openalex_download.py` | Query OpenAlex, save PDFs + `<painter>_works.xlsx`. |
| 04 | `04_prepare_for_md.py` | Add *For Markdown* sheet that points to each downloaded PDF. |
| 05 | `05_pdf_to_markdown.py` | Convert selected PDFs (<10 MB) to Markdown with Marker. |
| 06 | `06_cache_sbert.py` | Embed 3-sentence contexts with SBERT → `cache/`. |
| 07 | `07_top3_sentences.py` | Pick top-3 descriptive sentences per painting; write to `paintings.xlsx`. |
| 08 | `08_build_clip_dataset.py` | Build `CLIP_dataset_labels.csv` (image + caption pairs). |
| 09 | `09_finetune_clip_lora.py` | Train LoRA adapters → **PaintingCLIP** in `clip_finetuned_lora_best/`. |
| 10 | `10_cache_painting_clip.py` | Re-encode contexts with PaintingCLIP text encoder. |
| 11 | `11_eval_top10.py` | Retrieve top-10 zero-shot sentences (vanilla vs PaintingCLIP). |

Intermediate folders are created on-the-fly:

```
Dataset/Images/            # painting PNGs (store yourself)
PDFs/                      # raw OA PDFs
ExcelFiles/                # per-painter metadata
Small_Markdown/            # Marker output
cache/                     # SBERT embeddings
cache_clip_embeddings/     # vanilla CLIP text embeddings
cache_clip_finetuned_embeddings/   # PaintingCLIP embeddings
clip_finetuned_lora_best/  # final LoRA adapters
Results/                   # evaluation artefacts
```

---

## 2 One-time setup

### 2.1 Install system packages

| Requirement | Why | macOS | Ubuntu |
|-------------|-----|-------|--------|
| **Marker**  | PDF → Markdown | `pip install marker` | same |
| **Graphviz** (optional) | pretty diagrams in notebooks | `brew install graphviz` | `sudo apt install graphviz` |

### 2.2 Create Python environment

```bash
git clone https://github.com/<you>/ArtContext.git
cd ArtContext

# venv example (Conda works equally well)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip wheel
pip install -r requirements.txt
```

*GPU:*  
– Apple Silicon users get Metal/MPS automatically.  
– CUDA users must install the matching NVIDIA toolkit **before**
`pip install torch`.

### 2.3 Download NLTK data

```bash
python - <<'PY'
import nltk, ssl
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download("punkt", quiet=True, raise_on_error=True)
PY
```

---

## 3 Running the full pipeline

> The whole process (esp. OpenAlex downloads) can take many hours and
> several gigabytes of storage.  Every stage is restart-safe; you can
> stop and continue later.

```bash
cd Pipeline

# 1 Metadata
python 01_wikidata_harvest.py        # creates paintings.xlsx  (~2 min)
python 02_painter_list.py            # creates painters.xlsx   (<1 s)

# 2 OpenAlex crawl   (edit constants at top of 03_* to limit rows)
python 03_openalex_download.py       # large download  (hours)

# 3 Organise PDFs & convert to Markdown
python 04_prepare_for_md.py
python 05_pdf_to_markdown.py         # needs 'marker_single' CLI

# 4 Sentence caches
python 06_cache_sbert.py             # SBERT   (30-60 min)

# 5 Pick sentences & build CLIP dataset
python 07_top3_sentences.py
python 08_build_clip_dataset.py

# 6 Fine-tune CLIP (≈3 h on M1 Pro, faster on high-end GPU)
python 09_finetune_clip_lora.py

# 7 PaintingCLIP cache & evaluation
python 10_cache_painting_clip.py
python 11_eval_top10.py
```

After stage 11 you will have:

* `clip_finetuned_lora_best/` – PaintingCLIP adapters  
* `vanilla_clip.xlsx` & `painting_clip.xlsx` – zero-shot top-10 sentences  
  ready for manual “relevant / irrelevant” labelling.

To visualise macro Precision-Recall curves copy
`Results/precision_recall_curves.py` into `Pipeline/` and run it once
the `Label` columns are filled.

---

## 4 Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `HTTP 429` from OpenAlex | Rate-limit | 03 script already sleeps and retries; just wait. |
| `marker_single: command not found` | Marker not in PATH | `pip install marker` then reopen terminal. |
| `torch.mps not available` | Old macOS / PyTorch | Upgrade to PyTorch ≥ 2.2 and macOS ≥ 12.3. |
| Excel columns show “####” | cell too narrow | double-click the column edge or use OpenPyXL to auto-size again. |

---

## 5 Contributing guidelines

1. Use **absolute paths via `Path(__file__).resolve().parent`** to keep
   the scripts relocatable.  
2. Prefer `logging` over `print`; respect the existing log-file layout.  
3. In comments / docs always say **PaintingCLIP** (legacy name
   *MintCLIP* is being phased out).  
4. Pull-requests should update `requirements.txt` and this README when
   new dependencies or steps are added.

Happy hacking 🖼️🤖!
