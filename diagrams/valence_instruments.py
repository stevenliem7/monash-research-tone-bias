#!/usr/bin/env python3
"""Grouped bars for Table II, valence instruments on Gemini's replies.

Protocol (must match Table II):
  * zero-shot rows  — 229-clip human-GT pilot set (not CV)
  * LDA / RBF SVM   — StratifiedGroupKFold by question on 854 clips
  * Partial / full FT — nested CV OOF, also question-grouped, 854 clips

Balanced accuracy is the unweighted mean of the three class recalls.
Sized as a two-column figure* (~7 in wide) with 7–8 pt text.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent

METRICS = ["Acc.", "Bal. acc.", "Macro-F1", "Pos. F1", "Neu. F1", "Neg. F1"]
COLORS = ["#1B4F72", "#2F6F4E", "#8C3A3A", "#5B8C6A", "#C8C4BA", "#C47B4A"]

# Question-grouped CV for supervised rows; zero-shot on 229-clip pilot.
SCORES = pd.DataFrame(
    {
        "emotion2vec+\nlarge\nzero-shot*": [
            0.533,
            0.396,
            0.367,
            0.551,
            0.551,
            0.000,
        ],
        "EIV, averaged\nsoftmax\nzero-shot*": [
            0.402,
            0.389,
            0.302,
            0.087,
            0.551,
            0.267,
        ],
        "emotion2vec+\nlarge +\nshrinkage LDA": [
            0.474,
            0.490,
            0.442,
            0.464,
            0.539,
            0.323,
        ],
        "WavLM-large\n+ RBF SVM": [
            0.647,
            0.584,
            0.589,
            0.554,
            0.721,
            0.490,
        ],
        "Partial\nfine-tuned\nWavLM": [
            0.691,
            0.653,
            0.646,
            0.629,
            0.752,
            0.557,
        ],
        "Full\nfine-tuned\nWavLM": [
            0.659,
            0.577,
            0.585,
            0.610,
            0.726,
            0.420,
        ],
    },
    index=METRICS,
)


def main() -> None:
    instruments = list(SCORES.columns)
    x = np.arange(len(instruments))
    width = 0.13

    # Two-column / figure* width so 7–8 pt text stays readable in print.
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, (metric, color) in enumerate(zip(METRICS, COLORS)):
        offset = (i - (len(METRICS) - 1) / 2) * width
        values = SCORES.loc[metric].to_numpy()
        # Exact zeros get a faint stub so "0.000" is not read as missing data.
        plot_heights = np.where(values > 0, values, 0.018)
        bars = ax.bar(
            x + offset,
            plot_heights,
            width=width * 0.92,
            color=color,
            label=metric,
            edgecolor="white",
            linewidth=0.35,
            zorder=2,
        )
        for bar, value in zip(bars, values):
            if value <= 0:
                bar.set_alpha(0.45)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.014,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                rotation=90,
                zorder=3,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(instruments, fontsize=7)
    ax.set_ylabel("Score", fontsize=8)
    ax.set_xlabel("Instrument", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylim(0, 1.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        frameon=False,
        ncol=6,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        fontsize=7,
        handlelength=1.1,
        columnspacing=0.9,
    )
    fig.tight_layout(pad=0.4)

    png = OUT_DIR / "fig_valence_instruments.png"
    pdf = OUT_DIR / "fig_valence_instruments.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
