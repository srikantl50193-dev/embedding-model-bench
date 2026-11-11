"""Draw the README's charts from the committed results.

Run: uv run --group docs python scripts/make_charts.py

Colours, mark sizes and spacing follow the project's chart rules: two categorical slots that pass
the colour-blind checks, thin marks with a surface ring, recessive hairline grids, text in text
colours (never the series colour), direct labels where they help, and a legend for two series.
"""

import json
import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "charts"

SURFACE = "#fcfcfb"
GRID = "#e7e6e2"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#a3a29d"
ONE_MILLION = 1_000_000
BLUE = "#2a78d6"  # categorical slot 1
ORANGE = "#eb6834"  # categorical slot 2

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "text.color": TEXT,
        "axes.labelcolor": TEXT_2,
        "axes.edgecolor": GRID,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
    }
)


def save(fig: plt.Figure, name: str) -> None:
    """Write the SVG, plus a PNG to CHART_PREVIEW_DIR if set, for checking the layout by eye."""
    fig.savefig(OUT / f"{name}.svg")
    if preview := os.environ.get("CHART_PREVIEW_DIR"):
        fig.savefig(Path(preview) / f"{name}.png", dpi=110)


def load(path: str) -> dict:
    return json.loads((RESULTS / path).read_text(encoding="utf-8"))


def hits(scores: dict, k: int = 5) -> int:
    return sum(r is not None and r <= k for r in scores["first_relevant_ranks"])


def style(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def dot(ax: plt.Axes, x: float, y: float, color: str, label: str | None = None) -> None:
    ax.scatter([x], [y], s=60, color=color, edgecolors=SURFACE, linewidths=2, zorder=3, label=label)


def latency_chart() -> None:
    """Hit@5 on tuning and fresh held-out, against projected query latency at 1M chunks."""
    cost = {
        Path(p).stem: json.loads(Path(p).read_text()) for p in (RESULTS / "cost").glob("*.json")
    }

    def p95(slug: str) -> float:
        return next(pt for pt in cost[slug]["points"] if pt["num_vectors"] == ONE_MILLION)[
            "projected_query_p95_ms_mps"
        ]

    held = {r["config"]["name"]: r for r in load("holdout/result.json")["sets"][0]["rows"]}
    tuning_ft = load("finetune/all.json")["tuning"]
    quant_small = load("quantization/bge-small-en-v1.5-384.json")["schemes"]
    quant_base = load("quantization/bge-base-en-v1.5-768.json")["schemes"]
    mat = {v["dim"]: v for v in load("matryoshka/bge-base-en-v1.5.json")["variants"]}
    rows = [
        # label, latency, tuning hits, held-out name or None, label offset (points)
        (
            "bge-small (baseline)",
            p95("bge-small-en-v1.5-384-float32"),
            hits(load("models/bge-small-en-v1.5.json")["tuning"]),
            "bge-small 384d float32 (baseline)",
            (-8, 6),
        ),
        # Same latency as the baseline: drawn 12% to the right so the two don't overlap.
        (
            "bge-small, fine-tuned",
            1.12 * p95("bge-small-en-v1.5-384-float32"),
            hits(tuning_ft),
            "bge-small 384d float32, fine-tuned",
            (8, 6),
        ),
        (
            "bge-small binary + rerank",
            p95("bge-small-en-v1.5-384-binary_rescore_float"),
            hits(quant_small[4]["tuning"]),
            None,
            (8, -12),
        ),
        (
            "bge-base binary + rerank",
            p95("bge-base-en-v1.5-768-binary_rescore_float"),
            hits(quant_base[4]["tuning"]),
            "bge-base 768d binary + float32 rescoring (recommended)",
            (8, 6),
        ),
        (
            "bge-base 256d",
            p95("bge-base-en-v1.5-256-float32"),
            hits(mat[256]["tuning"]),
            "bge-base 256d float32",
            (8, 6),
        ),
        (
            "bge-base",
            p95("bge-base-en-v1.5-768-float32"),
            hits(load("models/bge-base-en-v1.5.json")["tuning"]),
            "bge-base 768d float32",
            (-8, 6),
        ),
        (
            "bge-base int8",
            p95("bge-base-en-v1.5-768-int8"),
            hits(quant_base[1]["tuning"]),
            None,
            (-8, -14),
        ),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    style(ax)
    first_t = first_h = True
    for label, x, t, held_name, offset in rows:
        y_t = 100 * t / 31
        if held_name:
            y_h = 100 * hits(held[held_name]["scores"]) / 30
            ax.plot([x, x], [y_t, y_h], color=MUTED, linewidth=1, zorder=2)
            dot(ax, x, y_h, ORANGE, "Fresh held-out (n=30)" if first_h else None)
            first_h = False
        dot(ax, x, y_t, BLUE, "Tuning (n=31)" if first_t else None)
        first_t = False
        ha = "left" if offset[0] > 0 else "right"
        ax.annotate(
            label,
            (x, max(y_t, y_h) if held_name else y_t),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.5,
            color=TEXT_2,
            ha=ha,
        )
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 40, 80, 160], ["10", "20", "40", "80", "160"])
    ax.set_xlim(8, 220)
    ax.set_ylim(55, 92)
    ax.set_xlabel(
        "Projected query p95 at 1M chunks, ms (encode on Apple GPU + exact search, log scale)"
    )
    ax.set_ylabel("hit@5, %")
    ax.set_title(
        "Quality against query latency: tuning lead didn't hold on held-out",
        loc="left",
        fontsize=11.5,
        color=TEXT,
        pad=12,
    )
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: not labels[i].startswith("Tuning"))
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        frameon=False,
        loc="lower right",
        labelcolor=TEXT_2,
    )
    fig.text(
        0.01,
        0.01,
        "One question = 3.2 points (tuning) / 3.3 points (held-out); gaps under "
        "~7 points are likely noise.\nFine-tuned bge-small has the baseline's latency; it is "
        "drawn slightly right so the two don't overlap.",
        fontsize=8,
        color=TEXT_2,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "recall_vs_latency")
    plt.close(fig)


def storage_chart() -> None:
    """bge-base quality against memory per vector: fewer dimensions or fewer bits."""
    mat = sorted(load("matryoshka/bge-base-en-v1.5.json")["variants"], key=lambda v: v["dim"])
    quant = {s["scheme"]: s for s in load("quantization/bge-base-en-v1.5-768.json")["schemes"]}
    q_rows = [
        (3072, quant["float32"]),
        (768, quant["int8"]),
        (96, quant["binary, rescored with float32 vectors"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharex=True)
    panels = [
        (axes[0], "Tuning hit@5, % (n=31)", lambda s: 100 * hits(s["tuning"]) / 31, (40, 82)),
        (axes[1], "SciFact nDCG@10 (n=300)", lambda s: s["scifact"]["ndcg_at_10"], (0.45, 0.76)),
    ]
    for ax, label, value, ylim in panels:
        style(ax)
        xs = [v["dim"] * 4 for v in mat]
        ax.plot(xs, [value(v) for v in mat], color=BLUE, linewidth=2, zorder=2)
        for x, v in zip(xs, mat, strict=True):
            dot(ax, x, value(v), BLUE, "Fewer dimensions (float32)" if v is mat[0] else None)
        ax.plot(
            [x for x, _ in q_rows],
            [value(s) for _, s in q_rows],
            color=ORANGE,
            linewidth=2,
            zorder=2,
        )
        for i, (x, s) in enumerate(q_rows):
            dot(ax, x, value(s), ORANGE, "Fewer bits (768d)" if i == 0 else None)
        ax.set_xscale("log", base=2)
        ax.set_xticks([96, 256, 768, 3072], ["96 B", "256 B", "768 B", "3 KB"])
        ax.set_xlim(70, 4500)
        ax.set_ylim(*ylim)
        ax.set_title(label, loc="left", fontsize=10, color=TEXT)
        ax.set_xlabel("RAM per vector (log scale)")
        x, s = q_rows[2]
        ax.annotate(
            "binary + rerank",
            (x, value(s)),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8.5,
            color=TEXT_2,
        )
    axes[0].legend(frameon=False, loc="lower right", labelcolor=TEXT_2, fontsize=8.5)
    fig.suptitle(
        "bge-base: binary + rerank keeps quality at 1/32 of the memory; short vectors don't",
        x=0.01,
        ha="left",
        fontsize=11.5,
        color=TEXT,
    )
    fig.tight_layout()
    save(fig, "recall_vs_storage")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    latency_chart()
    storage_chart()
    print(f"Wrote {sorted(p.name for p in OUT.glob('*.svg'))}")
