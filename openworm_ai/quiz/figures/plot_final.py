"""
Final dissertation & presentation figures — clean, readable, no clutter.
Filters broken models, deduplicates, produces 7 key figures + scatter plots.
"""

import json
import glob
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mtick
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────
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

# Nice short names and company colors
SHORT_NAMES = {
    "meta-llama/Llama-3.2-1B-Instruct": "Llama-1B",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "mistralai/Mistral-7B-Instruct-v0.2": "Mistral-7B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-8B",
    "google/gemma-2-9b-it": "Gemma-2-9B",
    "Qwen/Qwen2.5-14B-Instruct": "Qwen2.5-14B",
    "google/gemma-3-27b-it": "Gemma-3-27B",
    "Qwen/Qwen2.5-32B-Instruct": "Qwen2.5-32B",
    "CohereLabs/aya-expanse-32b": "Aya-32B",
    "Qwen/Qwen3-32B": "Qwen3-32B",
    "meta-llama/Llama-3.3-70B-Instruct": "Llama-70B",
    "Qwen/Qwen2.5-72B-Instruct": "Qwen2.5-72B",
    "mistralai/Mixtral-8x22B-Instruct-v0.1": "Mixtral-8x22B",
    "deepseek-ai/DeepSeek-R1": "DeepSeek-R1",
}

COMPANY_COLORS = {
    "Llama-1B": "#00A8E8", "Llama-8B": "#00A8E8", "Llama-70B": "#00A8E8",
    "Qwen2.5-7B": "#FF6A00", "Qwen2.5-14B": "#FF6A00", "Qwen2.5-32B": "#FF6A00",
    "Qwen2.5-72B": "#FF6A00", "Qwen3-32B": "#FF6A00",
    "Gemma-2-9B": "#EA4335", "Gemma-3-27B": "#EA4335",
    "Mistral-7B": "#F2A900", "Mixtral-8x22B": "#F2A900",
    "Aya-32B": "#39594D",
    "DeepSeek-R1": "#7F39FB",
}

COMPANY_LABELS = {
    "#00A8E8": "Meta (Llama)", "#FF6A00": "Alibaba (Qwen)",
    "#EA4335": "Google (Gemma)", "#F2A900": "Mistral AI",
    "#39594D": "Cohere (Aya)", "#7F39FB": "DeepSeek",
}

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

MODE_COLORS = {
    "Pretrained":    "#455A64",   # blue-grey
    "RAG Only":      "#2E7D32",   # dark green
    "Hybrid":        "#E65100",   # dark orange
    "RAG-Informed":  "#1565C0",   # dark blue
}


def _short(llm):
    m = llm.replace("huggingface:", "")
    return SHORT_NAMES.get(m, m.split("/")[-1])


def _params(llm):
    m = llm.replace("huggingface:", "")
    return MODEL_PARAMS_B.get(m)


def load_and_clean(score_dir):
    """Load all eval JSONs, filter broken, deduplicate (keep best pretrained)."""
    all_results = []
    for f in sorted(glob.glob(f"{score_dir}/eval_*[!d].json")):
        if "detailed" in f:
            continue
        data = json.load(open(f))
        for r in data.get("Results", []):
            if r["LLM"] not in BROKEN_MODELS:
                all_results.append(r)

    # Deduplicate: keep entry with highest pretrained accuracy per LLM × category
    best = {}
    for r in all_results:
        key = (r["LLM"], r.get("Quiz Category", ""))
        pre = r.get("Pretrained Accuracy (%)", 0)
        if key not in best or pre > best[key].get("Pretrained Accuracy (%)", 0):
            best[key] = r

    return list(best.values())


def load_detailed(score_dir):
    """Load detailed results for calibration."""
    all_results = []
    for f in sorted(glob.glob(f"{score_dir}/eval_*_detailed.json")):
        data = json.load(open(f))
        for r in data.get("Results", []):
            if r["LLM"] not in BROKEN_MODELS and r.get("Question Details"):
                all_results.append(r)
    return all_results


# ═══════════════════════════════════════════════════════════════════════
# Figure 1: 4-mode bar chart for a single category
# ═══════════════════════════════════════════════════════════════════════

def plot_4mode_bars(results, output_dir, category):
    """Clean grouped bar chart for one category."""
    subset = [r for r in results if r.get("Quiz Category") == category]
    if not subset:
        return

    # Sort by informed accuracy descending
    subset.sort(key=lambda r: r.get("Informed Accuracy (%)", 0), reverse=True)

    llms = [_short(r["LLM"]) for r in subset]
    pre = [r.get("Pretrained Accuracy (%)", 0) for r in subset]
    rag = [r.get("RAG Accuracy Overall (%)", 0) for r in subset]
    hyb = [r.get("Hybrid Accuracy (%)", 0) for r in subset]
    inf = [r.get("Informed Accuracy (%)", 0) for r in subset]

    x = np.arange(len(llms))
    width = 0.19

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - 1.5*width, pre, width, label="Pretrained", color=MODE_COLORS["Pretrained"], alpha=0.85)
    ax.bar(x - 0.5*width, rag, width, label="RAG Only", color=MODE_COLORS["RAG Only"], alpha=0.85)
    ax.bar(x + 0.5*width, hyb, width, label="Hybrid", color=MODE_COLORS["Hybrid"], alpha=0.85)
    ax.bar(x + 1.5*width, inf, width, label="RAG-Informed", color=MODE_COLORS["RAG-Informed"], alpha=0.85)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"4-Mode RAG Evaluation — {category}")
    ax.set_xticks(x)
    ax.set_xticklabels(llms, rotation=40, ha="right")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4)
    ax.set_ylim(0, 105)

    # Add delta annotations on top of informed bars for corpus
    if "Corpus" in category:
        for i, (p, inf_val) in enumerate(zip(pre, inf)):
            delta = inf_val - p
            if delta > 0:
                ax.text(x[i] + 1.5*width, inf_val + 1, f"+{delta:.0f}",
                       ha="center", va="bottom", fontsize=7, fontweight="bold", color="#2E7D32")

    plt.subplots_adjust(bottom=0.25)
    safe = category.lower().replace(" ", "_").replace("(", "").replace(")", "")
    path = output_dir / f"bars_{safe}.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 2: RAG delta — corpus-only horizontal bars
# ═══════════════════════════════════════════════════════════════════════

def plot_rag_delta_corpus(results, output_dir):
    """Horizontal bar: informed - pretrained delta on corpus questions only."""
    subset = [r for r in results if r.get("Quiz Category") == "C. elegans (Corpus)"]
    if not subset:
        return

    items = [(r, r.get("Informed vs Pretrained Delta (pp)", 0)) for r in subset]
    items.sort(key=lambda x: x[1])

    llms = [_short(r["LLM"]) for r, _ in items]
    deltas = [d for _, d in items]
    pre_scores = [r.get("Pretrained Accuracy (%)", 0) for r, _ in items]
    inf_scores = [r.get("Informed Accuracy (%)", 0) for r, _ in items]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#C62828" if d < 0 else "#2E7D32" for d in deltas]
    bars = ax.barh(llms, deltas, color=colors, alpha=0.85, height=0.7)
    ax.axvline(x=0, color="black", linewidth=0.8)

    for i, (d, pre, inf_val) in enumerate(zip(deltas, pre_scores, inf_scores)):
        label = f"+{d:.0f}pp ({pre:.0f}% -> {inf_val:.0f}%)"
        ax.text(d + (0.5 if d >= 0 else -0.5), i, label,
                va="center", ha="left" if d >= 0 else "right",
                fontsize=8, fontweight="bold")

    ax.set_xlabel("Accuracy Change (pp): RAG-Informed vs Pretrained")
    ax.set_title("RAG Impact on C. elegans Corpus Questions")
    ax.grid(axis="x", alpha=0.2, linewidth=0.5, color='0.7')

    plt.tight_layout()
    path = output_dir / "rag_delta_corpus.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 3: Scaling law — RAG delta vs model size + ceiling effect
# ═══════════════════════════════════════════════════════════════════════

def plot_scaling_law(results, output_dir, category="C. elegans (Corpus)"):
    """Two-panel figure:
      Left  — RAG delta (pp) vs model params (log scale). Does larger = less benefit?
      Right — RAG delta vs pretrained accuracy. Ceiling effect: already-good models gain less.
    Points coloured by model family, sized by params, with trend lines + 95% CI bands.
    """
    subset = [r for r in results if r.get("Quiz Category") == category]
    if not subset:
        return

    # Build (params, pretrained, delta, name, color) per model
    points = []
    for r in subset:
        p = _params(r["LLM"])
        if p is None:
            continue
        pre   = r.get("Pretrained Accuracy (%)", 0)
        inf   = r.get("Informed Accuracy (%)", 0)
        delta = inf - pre
        name  = _short(r["LLM"])
        color = COMPANY_COLORS.get(name, "#888888")
        points.append((p, pre, delta, name, color))

    if not points:
        return

    points.sort(key=lambda x: x[0])  # sort by params
    params_arr = np.array([p[0] for p in points])
    pre_arr    = np.array([p[1] for p in points])
    delta_arr  = np.array([p[2] for p in points])
    names      = [p[3] for p in points]
    colors     = [p[4] for p in points]

    # Marker size proportional to log(params) for visual weight
    sizes = [max(60, 18 * np.log10(p)) for p in params_arr]

    # ── Bootstrap trend + 95% CI ──────────────────────────────────────
    def trend_and_ci(xs_log, ys, n_boot=2000, x_range=None):
        z = np.polyfit(xs_log, ys, 1)
        poly = np.poly1d(z)
        x_fit = x_range if x_range is not None else np.linspace(xs_log.min(), xs_log.max(), 100)
        rng = np.random.default_rng(42)
        boot_lines = []
        for _ in range(n_boot):
            idx = rng.choice(len(xs_log), size=len(xs_log), replace=True)
            zb = np.polyfit(xs_log[idx], ys[idx], 1)
            boot_lines.append(np.poly1d(zb)(x_fit))
        boot_arr = np.array(boot_lines)
        return poly, x_fit, np.percentile(boot_arr, 2.5, axis=0), np.percentile(boot_arr, 97.5, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Panel 1: delta vs params ──────────────────────────────────────
    log_params = np.log10(params_arr)
    poly1, x_fit1, ci_lo1, ci_hi1 = trend_and_ci(log_params, delta_arr)

    ax1.axhline(0, color="0.4", linewidth=1, linestyle="--", zorder=1)
    ax1.axhspan(0, delta_arr.max() + 5, alpha=0.04, color="#2E7D32", zorder=0)
    ax1.axhspan(delta_arr.min() - 2, 0, alpha=0.04, color="#C62828", zorder=0)

    # CI band + trend
    ax1.fill_between(10**x_fit1, ci_lo1, ci_hi1, alpha=0.15, color="#1565C0", zorder=2)
    ax1.plot(10**x_fit1, poly1(x_fit1), color="#1565C0", linewidth=2,
             linestyle="-", zorder=3, label="Trend (log-linear)")

    ax1.scatter(params_arr, delta_arr, c=colors, s=sizes, zorder=5,
                edgecolors="white", linewidth=0.8)

    # Label each point — alternate above/below to reduce overlap
    for i, (xi, yi, name) in enumerate(zip(params_arr, delta_arr, names)):
        va = "bottom" if i % 2 == 0 else "top"
        dy = 3 if va == "bottom" else -3
        ax1.annotate(name, (xi, yi), textcoords="offset points",
                     xytext=(0, dy), ha="center", va=va,
                     fontsize=7, color="0.25")

    ax1.set_xscale("log")
    ax1.set_xlabel("Model Parameters (Billions)")
    ax1.set_ylabel("RAG-Informed vs Pretrained (pp)")
    ax1.set_title("RAG Benefit vs Model Scale\n(C. elegans Corpus questions)")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_xlim(0.7, 1000)
    ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"{int(x)}B" if x >= 1 else f"{x:.1f}B"
    ))

    # Spearman rho annotation
    from scipy.stats import spearmanr
    rho, pval = spearmanr(log_params, delta_arr)
    sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
    ax1.text(0.03, 0.04, f"Spearman rho = {rho:.2f}  {sig}",
             transform=ax1.transAxes, fontsize=8, color="0.35",
             va="bottom", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="0.8", alpha=0.9))

    # ── Panel 2: delta vs pretrained accuracy (ceiling effect) ────────
    poly2, x_fit2, ci_lo2, ci_hi2 = trend_and_ci(pre_arr, delta_arr,
                                                   x_range=np.linspace(pre_arr.min(), pre_arr.max(), 100))

    ax2.axhline(0, color="0.4", linewidth=1, linestyle="--", zorder=1)
    ax2.axhspan(0, delta_arr.max() + 5, alpha=0.04, color="#2E7D32", zorder=0)
    ax2.axhspan(delta_arr.min() - 2, 0, alpha=0.04, color="#C62828", zorder=0)

    ax2.fill_between(x_fit2, ci_lo2, ci_hi2, alpha=0.15, color="#E65100", zorder=2)
    ax2.plot(x_fit2, poly2(x_fit2), color="#E65100", linewidth=2,
             linestyle="-", zorder=3, label="Trend (linear)")

    ax2.scatter(pre_arr, delta_arr, c=colors, s=sizes, zorder=5,
                edgecolors="white", linewidth=0.8)

    for i, (xi, yi, name) in enumerate(zip(pre_arr, delta_arr, names)):
        va = "bottom" if i % 2 == 0 else "top"
        dy = 3 if va == "bottom" else -3
        ax2.annotate(name, (xi, yi), textcoords="offset points",
                     xytext=(0, dy), ha="center", va=va,
                     fontsize=7, color="0.25")

    ax2.set_xlabel("Pretrained Accuracy (%)")
    ax2.set_ylabel("RAG Delta (pp)")
    ax2.set_title("Ceiling Effect: RAG Benefit\nvs Baseline Model Accuracy")
    ax2.legend(fontsize=8, loc="upper right")

    rho2, pval2 = spearmanr(pre_arr, delta_arr)
    sig2 = "***" if pval2 < 0.001 else ("**" if pval2 < 0.01 else ("*" if pval2 < 0.05 else "ns"))
    ax2.text(0.03, 0.04, f"Spearman rho = {rho2:.2f}  {sig2}",
             transform=ax2.transAxes, fontsize=8, color="0.35",
             va="bottom", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="0.8", alpha=0.9))

    # ── Shared company colour legend ──────────────────────────────────
    seen = {}
    for name, color in zip(names, colors):
        company = COMPANY_LABELS.get(color, color)
        if company not in seen:
            seen[company] = color
    legend_handles = [
        matplotlib.patches.Patch(facecolor=c, label=label, edgecolor="white")
        for label, c in seen.items()
    ]
    fig.legend(handles=legend_handles, title="Model family",
               loc="lower center", ncol=len(seen),
               bbox_to_anchor=(0.5, -0.04), fontsize=8,
               title_fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.14)
    path = output_dir / "scaling_law.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 4: Domain gap — accuracy vs category (line plot, clean)
# ═══════════════════════════════════════════════════════════════════════

REPRESENTATIVE_MODELS = [
    "huggingface:meta-llama/Llama-3.2-1B-Instruct",
    "huggingface:Qwen/Qwen2.5-7B-Instruct",
    "huggingface:Qwen/Qwen2.5-14B-Instruct",
    "huggingface:Qwen/Qwen2.5-32B-Instruct",
    "huggingface:meta-llama/Llama-3.3-70B-Instruct",
    "huggingface:deepseek-ai/DeepSeek-R1",
]


# Sequential palette for representative plot: light -> dark = small -> large
SIZE_PALETTE = ["#B3CDE3", "#6BAED6", "#2171B5", "#08519C", "#74C476", "#006D2C"]


def _plot_domain_gap_inner(complete, sorted_llms, output_dir, filename, title,
                           figsize=(10, 6), use_size_palette=False):
    """Shared rendering logic for domain gap plots."""
    cat_order = ["General Knowledge", "Science", "C. elegans", "C. elegans (Corpus)"]
    cat_labels = ["General\nKnowledge", "Science", "C. elegans\n(General)", "C. elegans\n(Corpus)"]

    fig, ax = plt.subplots(figsize=figsize)

    for i, llm in enumerate(sorted_llms):
        name = _short(llm)
        if use_size_palette:
            color = SIZE_PALETTE[i % len(SIZE_PALETTE)]
            lw, alpha = 2.8, 0.95
        else:
            color = COMPANY_COLORS.get(name, "#888888")
            params = _params(llm)
            lw = 2.5 if params and params >= 32 else 1.8
            alpha = 0.95 if params and params >= 32 else 0.75

        scores = [complete[llm][c] for c in cat_order]
        params = _params(llm)
        label = f"{name}  ({params}B)" if params else name

        ax.plot(cat_labels, scores, "o-", label=label,
                color=color, linewidth=lw, alpha=alpha, markersize=8)

        # Score label at the corpus endpoint
        ax.annotate(f"{scores[-1]:.0f}%", xy=(3, scores[-1]),
                    xytext=(8, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=8.5,
                    color=color, fontweight="bold")

    ax.set_ylabel("Pretrained Accuracy (%)")
    ax.set_title(title)
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.2, 3.7)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5, color="0.7")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8.5,
              title="Model  (params)", title_fontsize=8.5)

    # Shade the specialist zone
    ax.axvspan(1.5, 3.7, alpha=0.04, color="#1565C0", zorder=0)
    ax.text(2.5, 2, "C. elegans specialist domain", ha="center", va="bottom",
            fontsize=8, color="#1565C0", alpha=0.55, style="italic")

    plt.tight_layout()
    path = output_dir / filename
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_domain_gap(results, output_dir):
    """Two domain gap figures: all models (full) + 6 representative models (clean)."""
    cat_order = ["General Knowledge", "Science", "C. elegans", "C. elegans (Corpus)"]

    by_llm = defaultdict(dict)
    for r in results:
        cat = r.get("Quiz Category", "")
        if cat in cat_order:
            by_llm[r["LLM"]][cat] = r.get("Pretrained Accuracy (%)", 0)

    complete = {llm: cats for llm, cats in by_llm.items()
                if all(c in cats for c in cat_order)}

    if not complete:
        print("  Not enough complete models for domain gap plot")
        return

    sorted_all = sorted(complete.keys(),
                        key=lambda l: complete[l]["General Knowledge"], reverse=True)

    # Full version — all models
    _plot_domain_gap_inner(
        complete, sorted_all, output_dir,
        filename="domain_gap_all.png",
        title="The Domain Gap: All Models Across Knowledge Domains",
        figsize=(12, 6),
    )

    # Representative version — 6 models, sequential size palette (light->dark = small->large)
    rep_llms = [llm for llm in REPRESENTATIVE_MODELS if llm in complete]
    rep_sorted = sorted(rep_llms, key=lambda l: _params(l) or 0)
    _plot_domain_gap_inner(
        complete, rep_sorted, output_dir,
        filename="domain_gap_representative.png",
        title="The Domain Gap: Representative Models (1B - 671B)",
        figsize=(10, 6),
        use_size_palette=True,
    )


# ═══════════════════════════════════════════════════════════════════════
# Figure 5: Error breakdown (clean — only working models)
# ═══════════════════════════════════════════════════════════════════════

def plot_error_breakdown(results, output_dir):
    """Side-by-side stacked bar: pretrained vs informed error types."""
    error_types = ["correct", "wrong_answer", "format_error", "parse_failure", "no_response"]
    colors = {"correct": "#2E7D32", "wrong_answer": "#C62828", "format_error": "#E65100",
              "parse_failure": "#6A1E99", "no_response": "#757575"}

    # Aggregate per LLM across all categories
    llm_errors = defaultdict(lambda: {"pretrained": defaultdict(int), "informed": defaultdict(int)})
    for r in results:
        name = _short(r["LLM"])
        for etype in error_types:
            llm_errors[name]["pretrained"][etype] += r.get("Pretrained Errors", {}).get(etype, 0)
            llm_errors[name]["informed"][etype] += r.get("Informed Errors", {}).get(etype, 0)

    if not llm_errors:
        return

    # Sort by total correct (pretrained)
    sorted_llms = sorted(llm_errors.keys(),
                        key=lambda l: llm_errors[l]["pretrained"]["correct"], reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    for ax, mode, title in [(axes[0], "pretrained", "Pretrained"),
                             (axes[1], "informed", "RAG-Informed")]:
        bottom = np.zeros(len(sorted_llms))
        for etype in error_types:
            vals = [llm_errors[llm][mode][etype] for llm in sorted_llms]
            label = etype.replace("_", " ").title()
            ax.barh(sorted_llms, vals, left=bottom, label=label,
                   color=colors[etype], alpha=0.85)
            bottom += np.array(vals, dtype=float)

        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Question Count")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="x", alpha=0.2, linewidth=0.5, color='0.7')

    fig.suptitle("Error Type Breakdown: Pretrained vs RAG-Informed", fontsize=14, y=1.01)
    plt.tight_layout()
    path = output_dir / "error_breakdown.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 6: Calibration curve
# ═══════════════════════════════════════════════════════════════════════

def plot_calibration(detailed_results, output_dir):
    """Similarity score vs RAG accuracy — proves retrieval quality matters."""
    all_questions = []
    for r in detailed_results:
        for q in r.get("Question Details", []):
            if q.get("rag_available"):
                all_questions.append(q)

    if len(all_questions) < 20:
        print("  Not enough RAG questions for calibration — skipping")
        return

    bins = np.arange(0.5, 1.01, 0.05)
    bin_accs, bin_counts, bin_centers = [], [], []

    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        in_bin = [q for q in all_questions if lo <= q.get("best_similarity_score", 0) < hi]
        if len(in_bin) >= 10:
            acc = sum(1 for q in in_bin if q.get("rag_correct")) / len(in_bin) * 100
            bin_accs.append(acc)
            bin_counts.append(len(in_bin))
            bin_centers.append((lo + hi) / 2)

    if len(bin_centers) < 3:
        print("  Not enough bins for calibration — skipping")
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(bin_centers, bin_accs, "o-", color=MODE_COLORS["RAG-Informed"], linewidth=2.5,
            markersize=10, label="RAG Accuracy", zorder=5)

    ax1.set_xlabel("Retrieval Similarity Score")
    ax1.set_ylabel("RAG Accuracy (%)", color=MODE_COLORS["RAG-Informed"])
    ax1.set_title("Retrieval Quality vs RAG Accuracy")
    ax1.set_ylim(0, 100)

    ax2 = ax1.twinx()
    ax2.bar(bin_centers, bin_counts, width=0.04, alpha=0.2, color="gray", label="N questions")
    ax2.set_ylabel("Questions in Bin", color="gray")
    ax2.spines["top"].set_visible(False)

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.grid(True, alpha=0.2, linewidth=0.5, color='0.7')

    plt.tight_layout()
    path = output_dir / "calibration.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 7: Scatter — pretrained vs informed accuracy per model
# ═══════════════════════════════════════════════════════════════════════

def plot_pretrained_vs_informed_scatter(results, output_dir, category=None):
    """Scatter: x = pretrained, y = informed, diagonal = no change. Points above = RAG helps."""
    if category:
        data = [r for r in results if r.get("Quiz Category") == category]
        title = f"Pretrained vs RAG-Informed — {category}"
    else:
        data = results
        title = "Pretrained vs RAG-Informed (All Categories)"

    fig, ax = plt.subplots(figsize=(8, 8))

    xs, ys, names = [], [], []
    for r in data:
        pre = r.get("Pretrained Accuracy (%)", 0)
        inf = r.get("Informed Accuracy (%)", 0)
        if pre == 0 and inf == 0:
            continue
        xs.append(pre)
        ys.append(inf)
        name = _short(r["LLM"])
        names.append(name)

    # Color by company
    colors = [COMPANY_COLORS.get(n, "#888888") for n in names]
    ax.scatter(xs, ys, c=colors, s=80, alpha=0.85, zorder=5, edgecolors="white", linewidth=0.5)

    # Labels
    for xi, yi, name in zip(xs, ys, names):
        ax.annotate(name, (xi, yi), textcoords="offset points",
                   xytext=(5, 4), fontsize=7, alpha=0.8)

    # Diagonal
    lims = [0, 105]
    ax.plot(lims, lims, "k--", alpha=0.3, label="No change")
    ax.fill_between(lims, lims, [105, 105], alpha=0.05, color="green", label="RAG helps")
    ax.fill_between(lims, [0, 0], lims, alpha=0.05, color="red", label="RAG hurts")

    ax.set_xlabel("Pretrained Accuracy (%)")
    ax.set_ylabel("RAG-Informed Accuracy (%)")
    ax.set_title(title)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.2, linewidth=0.5, color='0.7')

    plt.tight_layout()
    safe = category.lower().replace(" ", "_").replace("(", "").replace(")", "") if category else "all"
    path = output_dir / f"scatter_pre_vs_inf_{safe}.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    score_dir = "openworm_ai/quiz/scores/rag_full_logprob"
    output_dir = Path("openworm_ai/quiz/figures/final")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and cleaning results...")
    results = load_and_clean(score_dir)
    print(f"  {len(results)} clean LLM × category entries")
    print(f"  Models: {len(set(r['LLM'] for r in results))}")
    print(f"  Categories: {sorted(set(r.get('Quiz Category','') for r in results))}")

    print(f"\nGenerating figures in {output_dir}/\n")

    # 1. 4-mode bars — key categories
    plot_4mode_bars(results, output_dir, "C. elegans (Corpus)")
    plot_4mode_bars(results, output_dir, "C. elegans")

    # 2. RAG delta — corpus only
    plot_rag_delta_corpus(results, output_dir)

    # 3. Scaling law — RAG delta vs params + ceiling effect
    plot_scaling_law(results, output_dir, category="C. elegans (Corpus)")

    # 4. Domain gap — full (all models) + representative (6 models)
    plot_domain_gap(results, output_dir)

    # 5. Error breakdown
    plot_error_breakdown(results, output_dir)

    # 6. Calibration
    detailed = load_detailed(score_dir)
    if detailed:
        plot_calibration(detailed, output_dir)

    # 7. Scatter: pretrained vs informed
    plot_pretrained_vs_informed_scatter(results, output_dir, "C. elegans (Corpus)")
    plot_pretrained_vs_informed_scatter(results, output_dir, "C. elegans")
    plot_pretrained_vs_informed_scatter(results, output_dir)

    print(f"\nDone! All figures in {output_dir}/")


if __name__ == "__main__":
    main()
