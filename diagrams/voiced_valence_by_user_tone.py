#!/usr/bin/env python3
"""Stacked bar chart of voiced valence by the tone stated in the user turn.

Counts are the 854 human-labelled clips (both corpora). Percentages in the
paper table are these counts rounded to one decimal place.

Sized for a single-column figure (~3.5 in wide) with 7–8 pt text.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent

# Clip counts, not rounded percentages. Angry's printed row sums to 99.9
# because 20.6 + 71.2 + 8.1 lost a tenth in rounding.
COUNTS = pd.DataFrame(
    {
        "positive": [75, 37, 33, 30, 28, 39, 242],
        "neutral": [6, 44, 114, 132, 72, 140, 508],
        "negative": [0, 1, 13, 58, 11, 21, 104],
    },
    index=["Happy", "Neutral", "Angry", "Disgust", "Fearful", "Sad", "All clips"],
)
N = COUNTS.sum(axis=1)
SHARES = COUNTS.div(N, axis=0) * 100

COLORS = {
    "positive": "#2F6F4E",
    "neutral": "#C8C4BA",
    "negative": "#8C3A3A",
}


def main() -> None:
    tones = list(COUNTS.index[:-1])
    positions = list(range(len(tones))) + [len(tones) + 0.55]
    labels = tones + ["All clips"]

    # ~3.5 in column width; text sized so it stays ~7–8 pt in print.
    fig, ax = plt.subplots(figsize=(3.5, 4.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    lefts = {label: 0.0 for label in labels}
    for valence in ("positive", "neutral", "negative"):
        widths = [float(SHARES.loc[label, valence]) for label in labels]
        ax.barh(
            positions,
            widths,
            left=[lefts[label] for label in labels],
            height=0.68,
            color=COLORS[valence],
            label=valence.capitalize(),
            edgecolor="white",
            linewidth=0.35,
        )
        for y, label, width in zip(positions, labels, widths):
            # Keep Happy's 7.4% neutral share; skip only tiny slivers.
            if width < 6.5:
                continue
            ax.text(
                lefts[label] + width / 2,
                y,
                f"{width:.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if valence != "neutral" else "#333333",
            )
        for label, width in zip(labels, widths):
            lefts[label] += width

    ax.set_yticks(positions)
    ax.set_yticklabels(
        [f"{label} (n={int(N.loc[label])})" for label in labels],
        fontsize=7.5,
    )
    ax.set_xlabel("Share of clips (%)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.55, positions[-1] + 0.55)
    ax.invert_yaxis()
    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        fontsize=7.5,
        handlelength=1.2,
        columnspacing=1.0,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.35)

    png = OUT_DIR / "fig1_voiced_valence_by_user_tone.png"
    pdf = OUT_DIR / "fig1_voiced_valence_by_user_tone.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
