"""
Five-Category Decomposition of RAG Evaluation Results

For each question × model pair, classifies the outcome into one of five
mutually exclusive categories based on pretrained vs RAG-informed answers:

  1. RAG-SUFFICIENT   — Pretrained wrong, RAG-informed correct
                        (RAG rescued the model)
  2. LLM-KNOWLEDGE    — Pretrained correct, RAG-informed correct
                        (model already knew it, RAG didn't hurt)
  3. LLM-RESCUED      — Pretrained correct, RAG-informed wrong
                        but hybrid fell back to pretrained and got it right
                        (hybrid safety net worked)
  4. RAG-DEGRADED     — Pretrained correct, RAG-informed wrong
                        (RAG actively hurt — the concerning case)
  5. ALL-FAILED       — Pretrained wrong, RAG-informed wrong
                        (neither approach works)

Uses the detailed per-question JSON files from the evaluation pipeline.
Outputs summary JSON + stacked bar chart + heatmap.

Usage:
    python five_category_decomposition.py
"""

import json
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    # Font
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.title_fontsize": 9,
    # Spines
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    # Ticks
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    # Grid
    "axes.grid": False,
    # Figure
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    # Legend
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.85",
    "legend.borderpad": 0.5,
})

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCORES_DIR = Path(__file__).resolve().parent / "scores" / "rag_full_logprob"
OUTPUT_DIR = Path(__file__).resolve().parent / "scores" / "decomposition"
FIGURES_DIR = Path(__file__).resolve().parent / "figures" / "final"

BROKEN_MODELS = {
    "huggingface:deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "huggingface:microsoft/Phi-3-mini-4k-instruct",
    "huggingface:microsoft/Phi-3-medium-4k-instruct",
    "huggingface:Qwen/Qwen3-8B",
    "claude-3-7-sonnet-20250219",
}

MODEL_PARAMS_B = {
    "meta-llama/Llama-3.2-1B-Instruct": 1,
    "Qwen/Qwen2.5-7B-Instruct": 7,
    "mistralai/Mistral-7B-Instruct-v0.2": 7,
    "meta-llama/Llama-3.1-8B-Instruct": 8,
    "google/gemma-2-9b-it": 9,
    "Qwen/Qwen2.5-14B-Instruct": 14,
    "google/gemma-3-27b-it": 27,
    "Qwen/Qwen2.5-32B-Instruct": 32,
    "CohereLabs/aya-expanse-32b": 32,
    "Qwen/Qwen3-32B": 32,
    "meta-llama/Llama-3.3-70B-Instruct": 70,
    "Qwen/Qwen2.5-72B-Instruct": 72,
    "mistralai/Mixtral-8x22B-Instruct-v0.1": 141,
    "deepseek-ai/DeepSeek-R1": 671,
}

CATEGORIES = [
    "RAG-SUFFICIENT",
    "LLM-KNOWLEDGE",
    "RAG-DEGRADED",
    "ALL-FAILED",
]

CATEGORY_COLORS = {
    "RAG-SUFFICIENT": "#2E7D32",   # dark green
    "LLM-KNOWLEDGE":  "#1565C0",   # dark blue
    "RAG-DEGRADED":   "#C62828",   # dark red
    "ALL-FAILED":     "#757575",   # medium grey
}

# Clearer legend labels that show what each bar segment actually means
CATEGORY_DISPLAY = {
    "RAG-SUFFICIENT": "RAG added value (pretrained wrong, RAG correct)",
    "LLM-KNOWLEDGE":  "Both correct (pretrained correct, RAG correct)",
    "RAG-DEGRADED":   "RAG hurt (pretrained correct, RAG wrong)",
    "ALL-FAILED":     "Both wrong (pretrained wrong, RAG wrong)",
}

CATEGORY_DESCRIPTIONS = {
    "RAG-SUFFICIENT": "Pretrained wrong -> RAG-informed correct (RAG rescued)",
    "LLM-KNOWLEDGE": "Both correct (model already knew)",
    "RAG-DEGRADED": "Pretrained correct -> RAG-informed wrong (RAG hurt)",
    "ALL-FAILED": "Both wrong (neither approach works)",
}

# Representative models spanning the full parameter range
REPRESENTATIVE_MODELS = [
    "huggingface:meta-llama/Llama-3.2-1B-Instruct",
    "huggingface:Qwen/Qwen2.5-7B-Instruct",
    "huggingface:Qwen/Qwen2.5-14B-Instruct",
    "huggingface:Qwen/Qwen2.5-32B-Instruct",
    "huggingface:meta-llama/Llama-3.3-70B-Instruct",
    "huggingface:deepseek-ai/DeepSeek-R1",
]

# Qwen 2.5 family for parametric scaling plot
QWEN_FAMILY = [
    "huggingface:Qwen/Qwen2.5-7B-Instruct",
    "huggingface:Qwen/Qwen2.5-14B-Instruct",
    "huggingface:Qwen/Qwen2.5-32B-Instruct",
    "huggingface:Qwen/Qwen2.5-72B-Instruct",
]


# ---------------------------------------------------------------------------
# Data loading (matches plot_final.py logic)
# ---------------------------------------------------------------------------
def short_name(llm: str) -> str:
    """Extract short model name from full LLM string."""
    name = llm.replace("huggingface:", "")
    # Take last part after /
    if "/" in name:
        name = name.split("/")[-1]
    # Remove common suffixes for brevity
    for suffix in ["-Instruct", "-it", "-Instruct-v0.1", "-Instruct-v0.2"]:
        name = name.replace(suffix, "")
    return name


def get_params(llm: str) -> int:
    """Get parameter count in billions for a model."""
    name = llm.replace("huggingface:", "")
    return MODEL_PARAMS_B.get(name, 0)


def load_detailed_results() -> list[dict]:
    """Load all detailed evaluation results, filtering broken models.

    Returns list of result dicts, each containing Question Details.
    Deduplicates by (LLM, category) keeping highest pretrained accuracy.
    """
    pattern = str(SCORES_DIR / "eval_*_detailed.json")
    files = glob.glob(pattern)

    if not files:
        print(f"ERROR: No detailed JSON files found in {SCORES_DIR}")
        sys.exit(1)

    all_results = []
    for fpath in sorted(files):
        with open(fpath) as f:
            data = json.load(f)
        for r in data.get("Results", []):
            llm = r.get("LLM", "")
            if llm in BROKEN_MODELS:
                continue
            if not r.get("Question Details"):
                continue
            all_results.append(r)

    # Deduplicate: keep highest pretrained accuracy per (LLM, category)
    best = {}
    for r in all_results:
        key = (r["LLM"], r["Quiz Category"])
        existing = best.get(key)
        if existing is None or r["Pretrained Accuracy (%)"] > existing["Pretrained Accuracy (%)"]:
            best[key] = r

    results = list(best.values())
    print(f"Loaded {len(results)} model × category results "
          f"from {len(files)} detailed files")
    return results


# ---------------------------------------------------------------------------
# Five-category classification
# ---------------------------------------------------------------------------
def classify_question(q: dict) -> str:
    """Classify a single question into one of the five categories.

    Uses pretrained_correct and informed_correct (RAG-informed mode),
    which is the most realistic comparison: the model sees retrieved
    context and decides how to use it.
    """
    pre = q.get("pretrained_correct", False)
    inf = q.get("informed_correct", False)

    if not pre and inf:
        return "RAG-SUFFICIENT"
    elif pre and inf:
        return "LLM-KNOWLEDGE"
    elif pre and not inf:
        return "RAG-DEGRADED"
    else:  # not pre and not inf
        return "ALL-FAILED"


def decompose_all(results: list[dict]) -> dict:
    """Run five-category decomposition across all results.

    Returns nested dict:
        by_model_category[llm][category] = {
            "counts": {cat: int},
            "pcts": {cat: float},
            "total": int,
        }
    Also returns aggregated stats.
    """
    by_model_category = defaultdict(lambda: defaultdict(dict))
    by_category_agg = defaultdict(lambda: defaultdict(int))
    by_model_agg = defaultdict(lambda: defaultdict(int))
    grand_total = defaultdict(int)

    for r in results:
        llm = r["LLM"]
        quiz_cat = r["Quiz Category"]
        questions = r["Question Details"]

        counts = defaultdict(int)
        for q in questions:
            cat = classify_question(q)
            counts[cat] += 1

        total = len(questions)
        pcts = {c: round(100 * counts[c] / total, 1) if total else 0
                for c in CATEGORIES}

        by_model_category[llm][quiz_cat] = {
            "counts": dict(counts),
            "pcts": pcts,
            "total": total,
        }

        # Aggregate by category
        for c in CATEGORIES:
            by_category_agg[quiz_cat][c] += counts[c]
            by_model_agg[llm][c] += counts[c]
            grand_total[c] += counts[c]

    return {
        "by_model_category": {k: dict(v) for k, v in by_model_category.items()},
        "by_category_agg": dict(by_category_agg),
        "by_model_agg": dict(by_model_agg),
        "grand_total": dict(grand_total),
    }


# ---------------------------------------------------------------------------
# Analysis: cross with model size
# ---------------------------------------------------------------------------
def size_analysis(decomp: dict) -> dict:
    """Analyse whether smaller models are more prone to RAG-DEGRADED."""
    rows = []
    for llm, cats in decomp["by_model_agg"].items():
        params = get_params(llm)
        if params == 0:
            continue
        total = sum(cats.values())
        if total == 0:
            continue
        rows.append({
            "llm": short_name(llm),
            "params_b": params,
            "rag_sufficient_pct": round(100 * cats.get("RAG-SUFFICIENT", 0) / total, 1),
            "llm_knowledge_pct": round(100 * cats.get("LLM-KNOWLEDGE", 0) / total, 1),
            "rag_degraded_pct": round(100 * cats.get("RAG-DEGRADED", 0) / total, 1),
            "all_failed_pct": round(100 * cats.get("ALL-FAILED", 0) / total, 1),
        })

    rows.sort(key=lambda r: r["params_b"])

    # Spearman correlation: size vs RAG-DEGRADED %
    correlation = None
    if len(rows) >= 5:
        from scipy import stats
        sizes = [r["params_b"] for r in rows]
        degraded = [r["rag_degraded_pct"] for r in rows]
        rho, p = stats.spearmanr(np.log10(sizes), degraded)
        correlation = {
            "spearman_rho": round(rho, 3),
            "p_value": round(p, 4),
            "interpretation": (
                "Smaller models are MORE prone to RAG degradation"
                if rho < 0 and p < 0.05
                else "Larger models are MORE prone to RAG degradation"
                if rho > 0 and p < 0.05
                else "No significant relationship between size and RAG degradation"
            ),
        }

    return {"by_model_size": rows, "size_vs_degraded_correlation": correlation}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_stacked_bars(decomp: dict, quiz_category: str = None):
    """Stacked bar chart: category % per model, sorted by param count."""
    # Use per-model aggregated data (across all quiz categories)
    # or filter to a specific quiz category
    if quiz_category:
        data = {}
        for llm, cats in decomp["by_model_category"].items():
            if quiz_category in cats:
                data[llm] = cats[quiz_category]
        suffix = f"_{quiz_category.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
        title_suffix = f" — {quiz_category}"
    else:
        # Aggregate across all categories
        data = {}
        for llm, cats_agg in decomp["by_model_agg"].items():
            total = sum(cats_agg.values())
            if total == 0:
                continue
            data[llm] = {
                "pcts": {c: round(100 * cats_agg[c] / total, 1) for c in CATEGORIES},
                "total": total,
            }
        suffix = "_all"
        title_suffix = " — All Categories"

    if not data:
        print(f"No data for quiz_category={quiz_category}")
        return

    # Sort by parameter count
    models = sorted(data.keys(), key=lambda m: get_params(m))
    # Short name only on tick — param size as secondary line in smaller font
    tick_labels = [short_name(m) for m in models]
    param_labels = [f"{get_params(m)}B" for m in models]

    fig, ax = plt.subplots(figsize=(18, 7))

    bottoms = np.zeros(len(models))
    x = np.arange(len(models))

    for cat in CATEGORIES:
        vals = [data[m]["pcts"].get(cat, 0) for m in models]
        ax.bar(x, vals, bottom=bottoms,
               color=CATEGORY_COLORS[cat],
               label=CATEGORY_DISPLAY[cat], width=0.65)
        # Percentage labels for segments > 9%
        for i, v in enumerate(vals):
            if v > 9:
                ax.text(i, bottoms[i] + v / 2, f"{v:.0f}%",
                        ha="center", va="center", fontsize=7.5,
                        fontweight="bold", color="white")
        bottoms += np.array(vals)

    # Compact RAG-correct annotation above each bar — one line only
    for i, m in enumerate(models):
        rag_correct = (data[m]["pcts"].get("RAG-SUFFICIENT", 0)
                       + data[m]["pcts"].get("LLM-KNOWLEDGE", 0))
        ax.text(i, bottoms[i] + 1.2, f"{rag_correct:.0f}%",
                ha="center", va="bottom", fontsize=7.5, color="#1565C0",
                fontweight="bold")

    # Two-row x-axis: model name + param count below
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=8.5)
    # Add param count as a second row using ax.text at fixed y offset
    for i, pl in enumerate(param_labels):
        ax.text(i, -6.5, pl, ha="center", va="top", fontsize=7.5,
                color="#555555", transform=ax.get_xaxis_transform()
                if False else ax.transData)

    ax.set_ylabel("Questions (%)")
    ax.set_title(f"Five-Category Outcome Decomposition{title_suffix}", pad=10)
    ax.set_ylim(-8, 118)
    ax.set_xlim(-0.6, len(models) - 0.4)

    # RAG correct label in top-left so it doesn't crowd the bars
    ax.text(0.01, 1.01, "Blue number above bar = RAG correct %",
            transform=ax.transAxes, fontsize=7.5, color="#1565C0",
            va="bottom", ha="left")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=2, frameon=True, fontsize=8.5)

    plt.subplots_adjust(bottom=0.26)

    outpath = FIGURES_DIR / f"decomposition_stacked{suffix}.png"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Saved: {outpath}")


def plot_heatmap(decomp: dict, quiz_category: str = None):
    """Heatmap: models × categories with colour intensity = %."""
    if quiz_category:
        data = {}
        for llm, cats in decomp["by_model_category"].items():
            if quiz_category in cats:
                data[llm] = cats[quiz_category]["pcts"]
        suffix = f"_{quiz_category.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
        title_suffix = f" — {quiz_category}"
    else:
        data = {}
        for llm, cats_agg in decomp["by_model_agg"].items():
            total = sum(cats_agg.values())
            if total == 0:
                continue
            data[llm] = {c: round(100 * cats_agg[c] / total, 1) for c in CATEGORIES}
        suffix = "_all"
        title_suffix = " — All Categories"

    if not data:
        return

    models = sorted(data.keys(), key=lambda m: get_params(m))
    labels = [f"{short_name(m)} ({get_params(m)}B)" for m in models]

    matrix = np.array([[data[m].get(c, 0) for c in CATEGORIES] for m in models])

    fig, ax = plt.subplots(figsize=(8, max(6, len(models) * 0.45)))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=60)

    ax.set_xticks(range(len(CATEGORIES)))
    ax.set_xticklabels(CATEGORIES, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(CATEGORIES)):
            val = matrix[i, j]
            color = "white" if val > 35 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_title(f"Five-Category Decomposition{title_suffix}", fontsize=13)
    plt.colorbar(im, ax=ax, label="Questions (%)", shrink=0.8)
    plt.tight_layout()

    outpath = FIGURES_DIR / f"decomposition_heatmap{suffix}.png"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


# ---------------------------------------------------------------------------
# New figures
# ---------------------------------------------------------------------------

def plot_average_across_models(decomp: dict):
    """Stacked bar: one bar per quiz category, averaged across all models.

    Shows the typical outcome distribution for each domain — easier to read
    than 14 individual model bars.
    """
    quiz_cats = ["General Knowledge", "Science", "C. elegans", "C. elegans (Corpus)"]
    short_labels = ["General\nKnowledge", "Science", "C. elegans\n(General)", "C. elegans\n(Corpus)"]

    # Average pcts across models for each quiz category
    avg_pcts = {}
    for qc in quiz_cats:
        cat_pcts = {c: [] for c in CATEGORIES}
        for llm, cats in decomp["by_model_category"].items():
            if qc in cats:
                pcts = cats[qc]["pcts"]
                for c in CATEGORIES:
                    cat_pcts[c].append(pcts.get(c, 0))
        avg_pcts[qc] = {c: np.mean(v) if v else 0 for c, v in cat_pcts.items()}

    fig, ax = plt.subplots(figsize=(8, 6))
    bottoms = np.zeros(len(quiz_cats))

    for cat in CATEGORIES:
        vals = [avg_pcts[qc][cat] for qc in quiz_cats]
        ax.bar(short_labels, vals, bottom=bottoms,
               color=CATEGORY_COLORS[cat],
               label=CATEGORY_DISPLAY[cat], width=0.55)
        for i, v in enumerate(vals):
            if v > 6:
                ax.text(i, bottoms[i] + v / 2, f"{v:.0f}%",
                        ha="center", va="center", fontsize=8,
                        fontweight="bold", color="white")
        bottoms += np.array(vals)

    # Annotation: RAG correct only where RAG actually fires (C. elegans categories)
    # For GK/Science, retrieval threshold means no corpus chunks returned — LLM answers alone
    RAG_ACTIVE_CATS = {"C. elegans", "C. elegans (Corpus)"}
    for i, qc in enumerate(quiz_cats):
        if qc in RAG_ACTIVE_CATS:
            rag_correct = avg_pcts[qc]["RAG-SUFFICIENT"] + avg_pcts[qc]["LLM-KNOWLEDGE"]
            ax.text(i, bottoms[i] + 1.5, f"RAG correct:\n{rag_correct:.0f}%",
                    ha="center", va="bottom", fontsize=7.5, color="#1565C0",
                    fontweight="bold")
        else:
            ax.text(i, bottoms[i] + 1.5, "No retrieval\napplied",
                    ha="center", va="bottom", fontsize=7.5, color="#757575",
                    fontstyle="italic")

    ax.set_ylabel("Questions (%)")
    ax.set_title("Outcome Decomposition by Domain (averaged across all models)\n"
                 "Note: RAG retrieval only applied to C. elegans categories")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=2, frameon=True)
    # Footnote clarifying GK/Science behaviour
    fig.text(0.5, -0.04,
             "* For General Knowledge and Science, no corpus chunks are retrieved (below similarity threshold).\n"
             "  'Both correct' reflects LLM parametric knowledge only — RAG plays no role.",
             ha="center", fontsize=7.5, color="#555555", style="italic")
    plt.tight_layout()

    outpath = FIGURES_DIR / "decomposition_avg_by_domain.png"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


def plot_representative_models(decomp: dict, quiz_category: str = "C. elegans (Corpus)"):
    """Stacked bar for ~6 representative models spanning the parameter range.

    Cleaner than showing all 14 — picks one from each size tier.
    """
    available = [m for m in REPRESENTATIVE_MODELS
                 if m in decomp["by_model_category"]
                 and quiz_category in decomp["by_model_category"][m]]

    if not available:
        print(f"  No representative models found for {quiz_category}")
        return

    models = sorted(available, key=lambda m: get_params(m))
    labels = [f"{short_name(m)}\n({get_params(m)}B)" for m in models]

    fig, ax = plt.subplots(figsize=(9, 6))
    bottoms = np.zeros(len(models))

    for cat in CATEGORIES:
        vals = [decomp["by_model_category"][m][quiz_category]["pcts"].get(cat, 0)
                for m in models]
        ax.bar(labels, vals, bottom=bottoms,
               color=CATEGORY_COLORS[cat],
               label=CATEGORY_DISPLAY[cat], width=0.6)
        for i, v in enumerate(vals):
            if v > 7:
                ax.text(i, bottoms[i] + v / 2, f"{v:.0f}%",
                        ha="center", va="center", fontsize=8,
                        fontweight="bold", color="white")
        bottoms += np.array(vals)

    for i, m in enumerate(models):
        pcts = decomp["by_model_category"][m][quiz_category]["pcts"]
        rag_correct = pcts.get("RAG-SUFFICIENT", 0) + pcts.get("LLM-KNOWLEDGE", 0)
        ax.text(i, bottoms[i] + 1.5, f"RAG correct:\n{rag_correct:.0f}%",
                ha="center", va="bottom", fontsize=7, color="#1565C0",
                fontweight="bold")

    ax.set_ylabel("Questions (%)")
    ax.set_title(f"Outcome Decomposition — {quiz_category}\n(representative models)")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=2, frameon=True)
    plt.tight_layout()

    safe = quiz_category.lower().replace(" ", "_").replace("(", "").replace(")", "")
    outpath = FIGURES_DIR / f"decomposition_representative_{safe}.png"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


def plot_qwen_scaling(decomp: dict, quiz_category: str = "C. elegans (Corpus)"):
    """Stacked bar for Qwen 2.5 family only — shows parametric scaling effect."""
    available = [m for m in QWEN_FAMILY
                 if m in decomp["by_model_category"]
                 and quiz_category in decomp["by_model_category"][m]]

    if not available:
        print(f"  No Qwen models found for {quiz_category}")
        return

    models = sorted(available, key=lambda m: get_params(m))
    labels = [f"{short_name(m)}\n({get_params(m)}B)" for m in models]

    fig, ax = plt.subplots(figsize=(8, 6))
    bottoms = np.zeros(len(models))

    for cat in CATEGORIES:
        vals = [decomp["by_model_category"][m][quiz_category]["pcts"].get(cat, 0)
                for m in models]
        ax.bar(labels, vals, bottom=bottoms,
               color=CATEGORY_COLORS[cat],
               label=CATEGORY_DISPLAY[cat], width=0.55)
        for i, v in enumerate(vals):
            if v > 7:
                ax.text(i, bottoms[i] + v / 2, f"{v:.0f}%",
                        ha="center", va="center", fontsize=9,
                        fontweight="bold", color="white")
        bottoms += np.array(vals)

    for i, m in enumerate(models):
        pcts = decomp["by_model_category"][m][quiz_category]["pcts"]
        rag_correct = pcts.get("RAG-SUFFICIENT", 0) + pcts.get("LLM-KNOWLEDGE", 0)
        ax.text(i, bottoms[i] + 1.5, f"RAG correct:\n{rag_correct:.0f}%",
                ha="center", va="bottom", fontsize=8, color="#1565C0",
                fontweight="bold")

    ax.set_ylabel("Questions (%)")
    ax.set_title(f"Outcome Decomposition — Qwen 2.5 Scaling\n{quiz_category}")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=2, frameon=True)
    plt.tight_layout()

    safe = quiz_category.lower().replace(" ", "_").replace("(", "").replace(")", "")
    outpath = FIGURES_DIR / f"decomposition_qwen_scaling_{safe}.png"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


def plot_qwen_rescue_rate(decomp: dict, quiz_category: str = "C. elegans (Corpus)"):
    """Line plot: RAG rescue rate across Qwen 2.5 model sizes.

    Rescue rate = RAG-SUFFICIENT / (RAG-SUFFICIENT + ALL-FAILED) × 100
    i.e. of questions the LLM got wrong, what % did RAG recover?
    Normalises for model capability — shows RAG's actual contribution
    independently of how much the LLM already knew.
    """
    available = [m for m in QWEN_FAMILY
                 if m in decomp["by_model_category"]
                 and quiz_category in decomp["by_model_category"][m]]
    if not available:
        return

    models = sorted(available, key=lambda m: get_params(m))
    params = [get_params(m) for m in models]
    labels = [f"{short_name(m)}\n({get_params(m)}B)" for m in models]

    rescue_rates, pretrained_accs, rag_total_accs = [], [], []
    for m in models:
        pcts = decomp["by_model_category"][m][quiz_category]["pcts"]
        suf  = pcts.get("RAG-SUFFICIENT", 0)
        fail = pcts.get("ALL-FAILED", 0)
        rescue = 100 * suf / (suf + fail) if (suf + fail) > 0 else 0
        rescue_rates.append(rescue)
        pretrained_accs.append(pcts.get("LLM-KNOWLEDGE", 0) + pcts.get("RAG-DEGRADED", 0))
        rag_total_accs.append(pcts.get("RAG-SUFFICIENT", 0) + pcts.get("LLM-KNOWLEDGE", 0))

    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(x, rescue_rates, "o-", color="#2E7D32", linewidth=2.5,
            markersize=9, label="RAG rescue rate\n(% of LLM errors recovered by RAG)")
    ax.plot(x, pretrained_accs, "s--", color="#455A64", linewidth=1.8,
            markersize=7, alpha=0.7, label="Pretrained accuracy")
    ax.plot(x, rag_total_accs, "^--", color="#1565C0", linewidth=1.8,
            markersize=7, alpha=0.7, label="RAG total accuracy")

    for i, (rr, pre, rag) in enumerate(zip(rescue_rates, pretrained_accs, rag_total_accs)):
        ax.annotate(f"{rr:.0f}%", (i, rr), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8.5,
                    color="#2E7D32", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate (%)")
    ax.set_title(f"RAG Rescue Rate — Qwen 2.5 Scaling\n{quiz_category}")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    plt.tight_layout()

    safe = quiz_category.lower().replace(" ", "_").replace("(", "").replace(")", "")
    outpath = FIGURES_DIR / f"decomposition_qwen_rescue_rate_{safe}.png"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


def plot_qwen_dual_axis(decomp: dict, quiz_category: str = "C. elegans (Corpus)"):
    """Stacked decomposition bars + pretrained accuracy line on secondary axis.

    Makes the ceiling effect visible: as the pretrained line rises,
    the green RAG-SUFFICIENT bar shrinks because fewer questions are left
    for RAG to rescue — not because RAG is less effective.
    """
    available = [m for m in QWEN_FAMILY
                 if m in decomp["by_model_category"]
                 and quiz_category in decomp["by_model_category"][m]]
    if not available:
        return

    models = sorted(available, key=lambda m: get_params(m))
    labels = [f"{short_name(m)}\n({get_params(m)}B)" for m in models]
    x = np.arange(len(models))

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()

    # Stacked bars on ax1
    bottoms = np.zeros(len(models))
    pretrained_line = []
    for cat in CATEGORIES:
        vals = [decomp["by_model_category"][m][quiz_category]["pcts"].get(cat, 0)
                for m in models]
        ax1.bar(x, vals, bottom=bottoms, color=CATEGORY_COLORS[cat],
                label=CATEGORY_DISPLAY[cat], width=0.55, zorder=2)
        for i, v in enumerate(vals):
            if v > 7:
                ax1.text(i, bottoms[i] + v / 2, f"{v:.0f}%",
                         ha="center", va="center", fontsize=8.5,
                         fontweight="bold", color="white", zorder=3)
        bottoms += np.array(vals)

    # Pretrained accuracy line on ax2
    for m in models:
        pcts = decomp["by_model_category"][m][quiz_category]["pcts"]
        pretrained_line.append(pcts.get("LLM-KNOWLEDGE", 0) + pcts.get("RAG-DEGRADED", 0))

    ax2.plot(x, pretrained_line, "D--", color="#F57F17", linewidth=2.5,
             markersize=9, zorder=5, label="Pretrained accuracy")
    ax2.set_ylabel("Pretrained Accuracy (%)", color="#F57F17")
    ax2.tick_params(axis="y", labelcolor="#F57F17")
    ax2.set_ylim(0, 105)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#F57F17")
    ax2.spines["right"].set_linewidth(0.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Questions (%)")
    ax1.set_title(f"Outcome Decomposition + Pretrained Accuracy — Qwen 2.5 Scaling\n"
                  f"{quiz_category}\n"
                  "As pretrained accuracy rises, RAG has fewer errors to rescue")
    ax1.set_ylim(0, 115)

    # Combined legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2,
               loc="upper center", bbox_to_anchor=(0.5, -0.13),
               ncol=2, frameon=True)

    plt.subplots_adjust(bottom=0.28)
    safe = quiz_category.lower().replace(" ", "_").replace("(", "").replace(")", "")
    outpath = FIGURES_DIR / f"decomposition_qwen_dual_axis_{safe}.png"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
def print_summary(decomp: dict, size_info: dict):
    """Print human-readable summary to console."""
    gt = decomp["grand_total"]
    total = sum(gt.values())

    print(f"\n{'=' * 60}")
    print("FIVE-CATEGORY DECOMPOSITION — GRAND TOTALS")
    print(f"{'=' * 60}")
    print(f"Total question × model pairs: {total}\n")

    for cat in CATEGORIES:
        n = gt.get(cat, 0)
        pct = 100 * n / total if total else 0
        print(f"  {cat:20s}  {n:5d}  ({pct:5.1f}%)  — {CATEGORY_DESCRIPTIONS[cat]}")

    # Per quiz category
    print(f"\n{'=' * 60}")
    print("BY QUIZ CATEGORY")
    print(f"{'=' * 60}")
    for quiz_cat, counts in sorted(decomp["by_category_agg"].items()):
        cat_total = sum(counts.values())
        print(f"\n  {quiz_cat} (n={cat_total}):")
        for cat in CATEGORIES:
            n = counts.get(cat, 0)
            pct = 100 * n / cat_total if cat_total else 0
            print(f"    {cat:20s}  {n:4d}  ({pct:5.1f}%)")

    # Size correlation
    if size_info.get("size_vs_degraded_correlation"):
        corr = size_info["size_vs_degraded_correlation"]
        print(f"\n{'=' * 60}")
        print("MODEL SIZE vs RAG-DEGRADED CORRELATION")
        print(f"{'=' * 60}")
        print(f"  Spearman rho = {corr['spearman_rho']}")
        print(f"  p-value      = {corr['p_value']}")
        print(f"  {corr['interpretation']}")

    # Per model table
    print(f"\n{'=' * 60}")
    print("PER MODEL (sorted by size)")
    print(f"{'=' * 60}")
    print(f"  {'Model':<30s} {'Params':>6s}  {'RAG-SUF':>7s}  {'LLM-KN':>7s}  "
          f"{'DEGRAD':>7s}  {'FAILED':>7s}")
    print(f"  {'-' * 30} {'-' * 6}  {'-' * 7}  {'-' * 7}  {'-' * 7}  {'-' * 7}")
    for row in size_info["by_model_size"]:
        print(f"  {row['llm']:<30s} {row['params_b']:>5d}B  "
              f"{row['rag_sufficient_pct']:>6.1f}%  {row['llm_knowledge_pct']:>6.1f}%  "
              f"{row['rag_degraded_pct']:>6.1f}%  {row['all_failed_pct']:>6.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Load data
    results = load_detailed_results()

    # Decompose
    decomp = decompose_all(results)
    size_info = size_analysis(decomp)

    # Save JSON
    output = {
        "description": "Five-category decomposition of RAG evaluation results",
        "categories": CATEGORY_DESCRIPTIONS,
        "grand_total": decomp["grand_total"],
        "by_category": decomp["by_category_agg"],
        "by_model_size": size_info["by_model_size"],
        "size_vs_degraded_correlation": size_info["size_vs_degraded_correlation"],
        "by_model_category": {
            llm: {
                cat: info
                for cat, info in cats.items()
            }
            for llm, cats in decomp["by_model_category"].items()
        },
    }

    outpath = OUTPUT_DIR / "five_category_decomposition.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {outpath}")

    # Print summary
    print_summary(decomp, size_info)

    # Generate plots
    print("\nGenerating figures...")

    # 1. Full stacked bars — corpus category only (all models, too cluttered for others)
    plot_stacked_bars(decomp, quiz_category="C. elegans (Corpus)")

    # 2. Average across models — one bar per domain
    plot_average_across_models(decomp)

    # 3. Representative models — 6 spanning parameter range
    plot_representative_models(decomp, quiz_category="C. elegans (Corpus)")

    # 4. Qwen 2.5 scaling — same family, parametric size change
    plot_qwen_scaling(decomp, quiz_category="C. elegans (Corpus)")

    # 5. Qwen rescue rate line plot — normalises for ceiling effect
    plot_qwen_rescue_rate(decomp, quiz_category="C. elegans (Corpus)")

    # 6. Qwen dual-axis — stacked bars + pretrained accuracy line
    plot_qwen_dual_axis(decomp, quiz_category="C. elegans (Corpus)")

    print("\nDone.")


if __name__ == "__main__":
    main()
