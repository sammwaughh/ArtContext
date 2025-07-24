# ArtContext – PaintingCLIP Pipeline  
**Version 0.1 (last generated 2025-07-08)**  

This document is the single source of truth describing every production
script located in [`Pipeline/`](./).  
It explains the order in which the programs are intended to be executed,
the data that flows between them, and important implementation details
(function names, constants, environment variables, side-effects, …).

Throughout the text **PaintingCLIP** designates the *fine-tuned* CLIP
model produced at stage 9.  Earlier comments or variable names that
still mention *MintCLIP* are synonymous and will be updated over time.

---

## Table of contents
1.  End-to-end diagram  
2.  Prerequisites & directory conventions  
3.  Stage-by-stage walkthrough  
4.  Detailed per-file API notes  
5.  Development guidelines  

---

## 1 End-to-end diagram

```
Wikidata  ─┐
           │      +──► 02_painter_list.py  ──► 03_openalex_download.py
           │      │                           │
           │      │                           └──► PDFs/ & ExcelFiles/
01_wikidata_harvest.py                        │
        │                                     │
        │                                     ▼
        │                          04_prepare_for_md.py
        │                                     │
        ▼                                     ▼
paintings.xlsx                    (adds sheet “For Markdown”)
                                              │
                                              ▼
                                  05_pdf_to_markdown.py ──► Small_Markdown/
                                              │
                                              ▼
                                  06_cache_sbert.py  ──► cache/
                                              │
                                              ▼
                                  07_top3_sentences.py
                                              │
                                              ▼
                                  08_build_clip_dataset.py ──► CLIP_dataset_labels.csv
                                              │
                                              ▼
                                  09_finetune_clip_lora.py ──► clip_finetuned_lora_best/
                                              │
                                              ├──► loss_curves.png
                                              │
                                              ▼
                                  10_cache_painting_clip.py ──► cache_clip_finetuned_embeddings/
                                              │
                                              ▼
                                  11_eval_top10.py ──► vanilla_clip.xlsx / painting_clip.xlsx
```

---

## 2 Prerequisites & directory conventions

* Python ≥ 3.10  
* `requirements.txt` in the project root lists all pip dependencies.  
* Folder layout expected **relative to the repository root**:
  ```
  Dataset/
      Images/                # original painting PNGs
  PDFs/                      # raw PDFs downloaded from OpenAlex
  ExcelFiles/                # per-painter metadata workbooks
  Small_Markdown/            # Marker output
  cache/                     # SBERT text caches
  cache_clip_embeddings/     # Vanilla-CLIP text caches
  cache_clip_finetuned_embeddings/   # PaintingCLIP caches
  Results/                   # evaluation artefacts (optional)
  ```
* All scripts auto-create missing directories where they write.

---

## 3 Stage-by-stage walkthrough (high-level)

| Stage | Script | In → Out | Purpose |
|-------|--------|----------|---------|
| 1 | 01_wikidata_harvest.py | Wikidata SPARQL → `paintings.xlsx` | Collect master painting list and metadata. |
| 2 | 02_painter_list.py | – → `painters.xlsx` | Define the 200 painters we search for in OpenAlex. |
| 3 | 03_openalex_download.py | `painters.xlsx` → PDFs + `ExcelFiles/*.xlsx` | Query OpenAlex, save metadata + PDFs. |
| 4 | 04_prepare_for_md.py | PDFs + `ExcelFiles` → updated `ExcelFiles` | Match downloaded PDFs back to metadata, produce “For Markdown” sheet. |
| 5 | 05_pdf_to_markdown.py | PDFs → `Small_Markdown/` | Convert selected PDFs (< 10 MB) to Markdown using *Marker*. |
| 6 | 06_cache_sbert.py | Markdown → `cache/*.pt` | Encode 3-sentence contexts with SBERT. |
| 7 | 07_top3_sentences.py | SBERT caches + `paintings.xlsx` → filled sentence columns (same file) | Select top-3 descriptive sentences per painting. |
| 8 | 08_build_clip_dataset.py | `paintings.xlsx` → `CLIP_dataset_labels.csv` | Assemble image–caption training pairs (≤ 77 tokens). |
| 9 | 09_finetune_clip_lora.py | images + CSV → `clip_finetuned_lora_best/` | Train LoRA adapters to create **PaintingCLIP**. |
| 10 | 10_cache_painting_clip.py | PaintingCLIP + Markdown → `cache_clip_finetuned_embeddings` | Re-encode contexts with PaintingCLIP text encoder. |
| 11 | 11_eval_top10.py | PaintingCLIP caches + images → Excel workbooks | Retrieve top-10 zero-shot sentences; prepares manual eval. |

---

## 4 Detailed per-file notes

### 01_wikidata_harvest.py
* **Main function:** `list_paintings(limit=...)`.  
* Queries the Wikidata endpoint in chunks of 1 000 IDs.  
* Calls `get_aggregated_fields()` twice to fetch **Depicts** and **Movements** in separate queries (avoids HTTP 414).  
* Output columns (excerpt):  
  `QID | Title | Creator | Year | Depicts | Movements | Movement IDs | Link Count | File Name`  
* Writes to `paintings.xlsx` and auto-sizes columns with `openpyxl.utils.get_column_letter`.

### 02_painter_list.py
* Single function `write_painter_sheet(output="painters.xlsx")`.  
* Hard-coded list of 200 `{"Artist": str, "Query String": str}`.  
* Very small; no external network calls.

### 03_openalex_download.py
* Central constants at top: `TOPIC_IDS`, `MAX_PAGES`, `MAX_WORKERS`.  
* **`fetch_works(painter)`**  
  * GET `https://api.openalex.org/works` with filters: `is_oa:true`, `language:en`, `topics.id=<14 ids>`, `search=painter`.  
  * Handles 429 throttling (sleep 60 s and retry).  
* **`create_excel_file(painter, works)`** – sheets  
  1. Main (full JSON)  
  2. Filtered (subset)  
  3. Downloadable (`best_link`, `backup_link`, `relevance`)  
* **`download_all_pdfs()`**  
  * ThreadPoolExecutor; monitors CPU via separate `monitor_cpu()` thread.  
  * Saves per-file time and appends “Download Times” sheet.  
* Log files per painter: `fetch-logs/`, `excel-logs/`, `download-logs/`, `cpu-logs/`.

### 04_prepare_for_md.py
* Opens every `<qs>_works.xlsx` and sheet **Downloadable**.  
* Builds a Pandas DF with columns: `FileName`, `Size_kB`, `Relevance`, `Type`.  
* Writes new sheet **For Markdown**; duplicates removed (`drop_duplicates("Title")`).  
* Large (> 10 MB) files are left flagged for optional removal.

### 05_pdf_to_markdown.py
* Reads `painters.xlsx` rows (`START_ROW … END_ROW`).  
* For each painter: selects rows in “For Markdown” sheet where `Size_kB < 10 000`.  
* Invokes  
  ```bash
  marker_single "<pdf>" --output_format=markdown --output_dir="Small_Markdown/<qs>/<slug>"
  ```  
  through `subprocess.run`, 1-hour timeout.  
* ThreadPoolExecutor (default 5) for concurrency.

### 06_cache_sbert.py
* Model constant: `MODEL_NAME = "paraphrase-MiniLM-L6-v2"`.  
* Traverses every `.md`, cleans markdown (regex + `nltk.sent_tokenize`).  
* Builds overlapping 3-sentence contexts; retains sentences with ≥ 4 tokens.  
* Encodes in mini-batches (`BATCH_SIZE = 64`) on GPU (`"mps"` if available, else CUDA/CPU).  
* Writes  
  `cache/<qs>_cache.pt` with keys `candidate_contexts`, `candidate_sentences`, `candidate_embeddings`.  

### 07_top3_sentences.py
* Loads `paintings.xlsx` and `painters.xlsx`.  
* For each painting row slice (`start_idx … end_idx` constants):  
  1. Find SBERT cache (`<qs>_cache.pt`).  
  2. Build text query:  
     `"Describe the visual content of '<Title>' by <Creator>."`  
  3. Cosine similarity vs embeddings (chunked to avoid OOM).  
  4. Pick top-3 sentences, write them back into columns “Sentence 1-3” (keeps formats via `openpyxl`).  
* Progress bar via `tqdm`; script prints total runtime.

### 08_build_clip_dataset.py
* Reads `paintings.xlsx`, selects rows where “Sentence 1” is non-empty.  
* Concatenates `Sentence 1 « » Sentence 2 « » Sentence 3`, then trims to ≤ 77 whitespace tokens.  
* Writes `CLIP_dataset_labels.csv` with columns:  
  `Title`, `File_Name` (image file), `Caption`.

### 09_finetune_clip_lora.py
* CLI flags (argparse): `--epochs`, `--batch-size`, `--lora-r`, `--lora-alpha`, …  
  Defaults: 20 epochs, batch 16, r = 16, α = 32.  
* Data loader uses `CLIPProcessor` for image-text tokenisation.  
* Loss: built-in InfoNCE (`model(..., return_loss=True)`).  
* Validation every epoch; keeps best epoch by **lowest bi-directional InfoNCE**.  
* Saves adapters to  
  `clip_finetuned_lora/` (latest) and `clip_finetuned_lora_best/` (best).  
* Produces two PNGs: `train_loss_over_epochs.png` and `val_nce_over_epochs.png`.

### 10_cache_painting_clip.py
* Loads PaintingCLIP adapter from `clip_finetuned_lora_best/`.  
* Re-uses logic of 06_cache_sbert but encoder = PaintingCLIP text tower.  
* Outputs `cache_clip_finetuned_embeddings/<qs>_finetuned_clip_embedding_cache.pt`.

### 11_eval_top10.py
* Loads three items for each painter:  
  1. Vanilla-CLIP text cache (`cache_clip_embeddings`)  
  2. PaintingCLIP text cache (`cache_clip_finetuned_embeddings`)  
  3. List of images from `FineTune/fine_tune_dataset.xlsx` (or the same CSV).  
* For every painting:  
  * Compute image embedding with base CLIP and with PaintingCLIP.  
  * Dot product against corresponding cache → top-10 sentences.  
* Writes two Excel workbooks:  
  `vanilla_clip.xlsx` and `painting_clip.xlsx` with columns  
  `File_Name | Title | Creator | SentenceRank | Sentence | Score | Label`.  
  Colour rules, column widths auto-sized.

---

## 5 Development guidelines

1. **Imports**  
   Always use absolute OS-agnostic imports (`from pathlib import Path`).  
2. **GPU selection**  
   `device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")`  
3. **Logging**  
   Prefer the built-in `logging` module to stray `print` statements.  
4. **Configuration vs code**  
   Each script has a *“Constants”* block near the top.  When adding new
   flags move them to argparse so that CI can vary them without editing
   the file.  
5. **Naming**  
   Use **PaintingCLIP** in comments, doc-strings and output filenames
   going forward.  A follow-up PR will rename variables such as
   `mint_clip_cache_dir` to `painting_clip_cache_dir`.

---

*End of Documentation.md*