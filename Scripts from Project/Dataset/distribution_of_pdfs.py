#!/usr/bin/env python3
import os
import subprocess

import matplotlib.pyplot as plt

BASE_DIR = "All Works"


def get_pdf_counts(base_dir):
    counts = {}
    for artist in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, artist)
        if not os.path.isdir(path):
            continue
        # count PDFs via bash find
        cmd = f"find \"{path}\" -maxdepth 1 -type f -name '*.pdf' | wc -l"
        n = int(subprocess.check_output(cmd, shell=True).strip())
        counts[artist] = n
    return counts


def plot_histogram(counts, out_png):
    plt.figure()
    plt.hist(list(counts.values()), bins=30)
    plt.xlabel("Number of PDFs per artist")
    plt.ylabel("Number of artists")
    plt.title("Distribution of PDF counts across artists")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_top_bottom_bars(counts, out_png, top_n=20):
    # sort by count
    items = sorted(counts.items(), key=lambda x: x[1])
    bottom = items[:top_n]
    top = items[-top_n:]
    labels = [a for a, _ in bottom] + [a for a, _ in top]
    values = [n for _, n in bottom] + [n for _, n in top]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=90)
    plt.ylabel("PDF count")
    plt.title(f"Bottom {top_n} vs Top {top_n} artists by PDF count")

    # annotate counts above bars
    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,  # x position: center of bar
            h + 1,  # y position: a bit above the bar
            f"{int(h)}",  # label
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main():
    counts = get_pdf_counts(BASE_DIR)
    os.makedirs("figures", exist_ok=True)
    plot_histogram(counts, "figures/pdf_count_histogram.png")
    plot_top_bottom_bars(counts, "figures/pdf_count_top_bottom.png")


if __name__ == "__main__":
    main()
