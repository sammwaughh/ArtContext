#!/usr/bin/env python3
"""
Build macro-averaged Precision–Recall curves for Vanilla-CLIP and Mint-CLIP
using the workbooks produced by *top_ten_sentences.py*.

The script automatically keeps **only** those paintings that have
exactly 10 non-empty labels (0 or 1) – perfect if you have annotated
only the first few paintings so far.

Workbook layout expected (one file per model):

File_Name | Title | Creator | SentenceRank | Sentence | Score | Label
----------|-------|---------|--------------|----------|-------|------
Q123.png  | …     | …       | 1-10         | …        | 0.87  | 0/1

* `File_Name` is used as the painting identifier.
* `Label` must be 0 or 1; other entries / blanks are ignored.
"""

from __future__ import annotations
import pathlib
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

# ───────────── configuration ─────────────
VANILLA_PATH = pathlib.Path("vanilla_clip.xlsx")
MINT_PATH    = pathlib.Path("mint_clip.xlsx")
OUTPUT_PNG   = pathlib.Path("avg_precision_recall.png")

VANILLA_COLOUR = "#f3e5ab"   # vanilla ice-cream
MINT_COLOUR    = "#98ff98"   # mint green
RECALL_GRID    = np.linspace(0, 1, 101)   # 0 … 1   (macro-interp grid)
# ─────────────────────────────────────────


# ╭──────────────── helpers ───────────────╮
def load_valid_paintings(
    workbook: pathlib.Path,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Return {painting_id : (scores, labels)} *only* for paintings that have
    exactly 10 non-NaN labels.

    Rows whose `Label` is not 0/1 are dropped.
    """
    df = pd.read_excel(workbook, engine="openpyxl")

    for col in ("File_Name", "Score", "Label"):
        if col not in df.columns:
            raise ValueError(f"{workbook}: missing required column {col!r}")

    # Keep only rows whose Label is 0 or 1
    df = df[pd.to_numeric(df["Label"], errors="coerce").isin([0, 1])]

    groups: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for pid, block in df.groupby("File_Name"):
        if len(block) == 10:                       # need full set of 10
            block = block.sort_values("Score", ascending=False)
            groups[pid] = (
                block["Score"].to_numpy(),
                block["Label"].astype(int).to_numpy(),
            )
    return groups


def macro_average(curves: List[Tuple[np.ndarray, np.ndarray]]
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate each PR curve on RECALL_GRID and macro-average."""
    interp_precisions = []
    for prec, rec in curves:
        # precision must be monotone non-increasing wrt recall – enforce
        interp = np.maximum.accumulate(np.interp(RECALL_GRID, rec[::-1], prec[::-1]))[::-1]
        interp_precisions.append(interp)
    return RECALL_GRID, np.mean(interp_precisions, axis=0)


def pr_curve_for(workbook: pathlib.Path
                 ) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute macro-averaged PR curve for one workbook.
    Returns (recall, precision, n_paintings_used).
    """
    per_painting = load_valid_paintings(workbook)
    curves = []
    for scores, labels in per_painting.values():
        prec, rec, _ = precision_recall_curve(labels, scores)
        curves.append((prec, rec))
    if not curves:
        raise RuntimeError(f"{workbook}: no painting has 10 valid labels.")
    r, p = macro_average(curves)
    return r, p, len(curves)
# ╰─────────────────────────────────────────╯


def main() -> None:
    rec_v, prec_v, n_v = pr_curve_for(VANILLA_PATH)
    rec_m, prec_m, n_m = pr_curve_for(MINT_PATH)

    # — Plot —
    plt.figure(figsize=(6, 5))
    plt.plot(rec_v, prec_v, label=f"CLIP (n={n_v})", color=VANILLA_COLOUR, linewidth=2)
    plt.plot(rec_m, prec_m, label=f"PaintingCLIP (n={n_m})",    color=MINT_COLOUR,    linewidth=2)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Macro-averaged Precision-Recall")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300)
    plt.close()

    print(f"Saved PR-curve plot → {OUTPUT_PNG.resolve()}")
    print(f"Paintings used: CLIP={n_v}, PaintingCLIP={n_m}")

if __name__ == "__main__":
    main()
