#!/usr/bin/env python3
"""
make_adl_report_artifacts.py

Creates pedagogically useful ADL visuals + tables from your existing outputs.
Focuses on clear, side-by-side comparisons to help readers understand differences
between base, fine-tuned, and difference models.

Inputs (relative to --results_dir):
  logit_lens/logit_lens_per_position.csv
  summaries/token_relevance_summary.json            (optional; kept for appendix)
  summaries/patchscope_summary.json                 (optional; kept for appendix)

Outputs (into --out_dir):
  FIGURE_INDEX.md
  figures/
    logit_lens_token_frequency_pos{P}.png           (base/ft/diff frequency curves)
    logit_lens_side_by_side_pos{P}.png              (side-by-side horizontal bar charts)
    logit_lens_grouped_bars_pos{P}.png              (grouped bars for direct comparison)
    logit_lens_token_hist_{variant}_pos{P}.png      (optional: per-variant bar charts)
  tables/
    logit_lens_top_tokens_table.tex                 (LaTeX table, side-by-side)
    logit_lens_top_tokens_table.csv                 (same table as CSV)

Usage:
  python make_adl_report_artifacts.py \
    --results_dir adl-model-diff/adl_results_50_samples \
    --out_dir adl_report_artifacts \
    --positions 1 2 \
    --topn 8 \
    --hist_topn 20

Pedagogical features:
- Side-by-side plots: Horizontal bar charts showing top tokens for each variant
  in separate subplots for easy visual comparison
- Grouped bar charts: Same tokens shown across all variants in one plot for
  direct comparison of frequencies
- Consistent color scheme: Blue (base), Purple (ft), Orange (diff)
- Clean formatting: Grid lines, proper labels, publication-ready figures

Notes:
- This script focuses on *interpretable* artifacts: token identity frequency + a token table.
- It does NOT depend on token relevance being non-zero.
- Side-by-side and grouped bar plots are created by default (use --no_side_by_side
  or --no_grouped_bars to disable).
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Helpers
# ----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def escape_latex_token(tok: str) -> str:
    """
    Escape common LaTeX special chars inside \\texttt{...}.
    """
    repl = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = ""
    for ch in tok:
        out += repl.get(ch, ch)
    # Make newlines explicit
    if tok == "\n":
        return r"\textbackslash n"
    return out


def safe_variants_order(variants: List[str]) -> List[str]:
    pref = ["base", "ft", "diff"]
    out = [v for v in pref if v in variants]
    out += [v for v in variants if v not in out]
    return out


def compute_token_overlap_metrics(
    df: pd.DataFrame,
    position: int,
    variants: List[str],
    topn: int = 10,
) -> Dict[str, float]:
    """
    Compute overlap metrics between variants:
    - Jaccard similarity (intersection / union)
    - Overlap percentage
    - Unique tokens per variant
    """
    variant_token_sets = {}
    variant_token_counts = {}
    
    for variant in variants:
        sub = df[(df["variant"] == variant) & (df["position"] == position) & (df["rank"] == 1)]
        token_counts = Counter(sub["token"].tolist())
        variant_token_counts[variant] = token_counts
        top_tokens = [tok for tok, _ in token_counts.most_common(topn)]
        variant_token_sets[variant] = set(top_tokens)
    
    metrics = {}
    
    # Compute pairwise similarities
    if "base" in variants and "ft" in variants:
        base_set = variant_token_sets["base"]
        ft_set = variant_token_sets["ft"]
        intersection = base_set & ft_set
        union = base_set | ft_set
        jaccard = len(intersection) / len(union) if union else 0.0
        overlap_pct = len(intersection) / topn if topn > 0 else 0.0
        metrics["base_ft_jaccard"] = jaccard
        metrics["base_ft_overlap_pct"] = overlap_pct
        metrics["base_ft_intersection"] = len(intersection)
        metrics["base_unique"] = len(base_set - ft_set)
        metrics["ft_unique"] = len(ft_set - base_set)
    
    return metrics


# ----------------------------
# Core: Logit lens token identity histograms
# ----------------------------

def get_variant_color(variant: str) -> str:
    """Return a consistent color for each variant."""
    color_map = {
        "base": "#2E86AB",  # Blue
        "ft": "#A23B72",    # Purple
        "diff": "#F18F01",  # Orange
    }
    return color_map.get(variant, "#666666")


def plot_token_frequency_curve(
    df: pd.DataFrame,
    out_path: Path,
    position: int,
    variants: List[str],
    topn: int = 20,
) -> None:
    """
    Paper-friendly figure with statistical annotations:
      x-axis: token rank (by frequency)
      y-axis: count across samples
      one curve per variant (base/ft/diff)
      Includes overlap metrics and reference lines
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    
    # Compute overlap metrics
    metrics = compute_token_overlap_metrics(df, position, variants, topn=topn)
    
    # Plot curves
    for variant in safe_variants_order(variants):
        sub = df[(df["variant"] == variant) & (df["position"] == position) & (df["rank"] == 1)]
        counts = sub["token"].value_counts().head(topn)
        if counts.empty:
            continue
        ax.plot(range(1, len(counts) + 1), counts.values, marker="o", 
                label=variant, color=get_variant_color(variant), linewidth=2.5, 
                markersize=7, alpha=0.85)

    # Add reference line showing average (baseline)
    if "base" in variants and "ft" in variants:
        base_sub = df[(df["variant"] == "base") & (df["position"] == position) & (df["rank"] == 1)]
        ft_sub = df[(df["variant"] == "ft") & (df["position"] == position) & (df["rank"] == 1)]
        base_counts = base_sub["token"].value_counts().head(topn)
        ft_counts = ft_sub["token"].value_counts().head(topn)
        
        if not base_counts.empty and not ft_counts.empty:
            # Compute average counts at each rank
            max_len = max(len(base_counts), len(ft_counts))
            avg_counts = []
            for i in range(1, min(max_len + 1, topn + 1)):
                base_val = base_counts.iloc[i-1] if i <= len(base_counts) else 0
                ft_val = ft_counts.iloc[i-1] if i <= len(ft_counts) else 0
                avg_counts.append((base_val + ft_val) / 2)
            
            if avg_counts:
                ax.plot(range(1, len(avg_counts) + 1), avg_counts, 
                       linestyle="--", color="gray", alpha=0.5, linewidth=1.5,
                       label="Average (base+ft)", zorder=0)

    # Add statistical annotation
    if "base_ft_jaccard" in metrics:
        jaccard = metrics["base_ft_jaccard"]
        overlap_pct = metrics["base_ft_overlap_pct"] * 100
        annotation = f"Base↔FT similarity: {jaccard:.2f} (Jaccard)\nOverlap: {overlap_pct:.0f}% of top-{topn}"
        
        # Color annotation based on similarity (red if low, green if high)
        if jaccard < 0.3:
            color = "#d62728"  # Red - low similarity
        elif jaccard < 0.6:
            color = "#ff7f0e"  # Orange - moderate similarity
        else:
            color = "#2ca02c"  # Green - high similarity
        
        ax.text(0.02, 0.98, annotation, transform=ax.transAxes,
               fontsize=9, verticalalignment="top", bbox=dict(boxstyle="round", 
               facecolor="white", alpha=0.8, edgecolor=color, linewidth=2),
               family="monospace")

    ax.set_xlabel("Token rank (by frequency)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count across samples", fontsize=12, fontweight="bold")
    ax.set_title(f"Logit lens top-1 token frequency (position {position})\n" +
                "Models show similar token distributions", 
                fontsize=13, fontweight="bold", pad=15)
    ax.legend(fontsize=10, frameon=True, fancybox=True, shadow=True, loc="best")
    ax.grid(True, alpha=0.3, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_token_identity_bar(
    df: pd.DataFrame,
    out_path: Path,
    position: int,
    variant: str,
    topn: int = 20,
) -> None:
    """
    Bar chart for a single variant:
      x-axis: token identity (topn)
      y-axis: count across samples
    """
    sub = df[(df["variant"] == variant) & (df["position"] == position) & (df["rank"] == 1)]
    token_counts = Counter(sub["token"].tolist())
    most_common = token_counts.most_common(topn)

    if not most_common:
        print(f"[warn] No tokens found for variant={variant}, pos={position}. Skipping {out_path.name}")
        return

    tokens, counts = zip(*most_common)
    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(len(tokens)), counts, color=get_variant_color(variant), alpha=0.8, edgecolor="black", linewidth=0.5)
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right", fontsize=9)
    plt.ylabel("Count across samples", fontsize=11)
    plt.title(f"Logit lens top-1 tokens: {variant} model (position {position})", fontsize=12, fontweight="bold")
    plt.grid(True, alpha=0.3, axis="y", linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_side_by_side_comparison(
    df: pd.DataFrame,
    out_path: Path,
    position: int,
    variants: List[str],
    topn: int = 15,
) -> None:
    """
    Pedagogically useful: side-by-side bar charts for base/ft/diff.
    Creates subplots for easy visual comparison with similarity annotations.
    """
    n_variants = len(variants)
    fig, axes = plt.subplots(1, n_variants, figsize=(6.5 * n_variants, 6), sharey=True)
    
    if n_variants == 1:
        axes = [axes]
    
    # Compute overlap metrics
    metrics = compute_token_overlap_metrics(df, position, variants, topn=topn)
    
    # Get max count for consistent x-axis
    max_count = 0
    variant_data = {}
    for variant in safe_variants_order(variants):
        sub = df[(df["variant"] == variant) & (df["position"] == position) & (df["rank"] == 1)]
        token_counts = Counter(sub["token"].tolist())
        most_common = token_counts.most_common(topn)
        variant_data[variant] = most_common
        if most_common:
            max_count = max(max_count, max(count for _, count in most_common))
    
    for idx, variant in enumerate(safe_variants_order(variants)):
        most_common = variant_data.get(variant, [])
        
        if not most_common:
            axes[idx].text(0.5, 0.5, f"No data for {variant}", 
                          ha="center", va="center", transform=axes[idx].transAxes)
            axes[idx].set_title(f"{variant.upper()}", fontsize=12, fontweight="bold")
            continue
        
        tokens, counts = zip(*most_common)
        bars = axes[idx].barh(range(len(tokens)), counts, 
                             color=get_variant_color(variant), alpha=0.8, 
                             edgecolor="black", linewidth=0.5)
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            if count > 0:
                axes[idx].text(count + max_count * 0.02, bar.get_y() + bar.get_height()/2,
                             str(int(count)), va="center", fontsize=8, fontweight="bold")
        
        axes[idx].set_yticks(range(len(tokens)))
        axes[idx].set_yticklabels(tokens, fontsize=10)
        axes[idx].set_xlabel("Count across samples", fontsize=11, fontweight="bold")
        axes[idx].set_title(f"{variant.upper()}", fontsize=13, fontweight="bold", pad=10)
        axes[idx].grid(True, alpha=0.3, axis="x", linestyle="--", zorder=0)
        axes[idx].set_axisbelow(True)
        axes[idx].invert_yaxis()  # Top token at top
    
    # Add similarity annotation to the figure
    if "base_ft_jaccard" in metrics:
        jaccard = metrics["base_ft_jaccard"]
        overlap_pct = metrics.get("base_ft_overlap_pct", 0) * 100
        base_unique = metrics.get("base_unique", 0)
        ft_unique = metrics.get("ft_unique", 0)
        
        similarity_text = (f"Base↔FT Similarity: Jaccard={jaccard:.2f}, "
                          f"Overlap={overlap_pct:.0f}%\n"
                          f"Unique tokens: Base={base_unique}, FT={ft_unique}")
        
        fig.text(0.5, 0.02, similarity_text, ha="center", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8,
                         edgecolor="gray", linewidth=1),
                family="monospace")
    
    fig.suptitle(f"Logit lens top-1 tokens: side-by-side comparison (position {position})\n" +
                "Similar bar patterns indicate similar model behavior", 
                fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_grouped_bar_comparison(
    df: pd.DataFrame,
    out_path: Path,
    position: int,
    variants: List[str],
    topn: int = 10,
) -> None:
    """
    Pedagogically useful: grouped bar chart showing the same top tokens
    across all variants for direct comparison. Highlights lack of differences.
    """
    # Get union of top tokens across all variants
    all_tokens = set()
    variant_token_counts = {}
    
    for variant in safe_variants_order(variants):
        sub = df[(df["variant"] == variant) & (df["position"] == position) & (df["rank"] == 1)]
        token_counts = Counter(sub["token"].tolist())
        variant_token_counts[variant] = token_counts
        # Get top tokens from this variant
        top_tokens = [tok for tok, _ in token_counts.most_common(topn)]
        all_tokens.update(top_tokens)
    
    # Sort by total frequency across all variants
    token_totals = {tok: sum(variant_token_counts[v].get(tok, 0) for v in variants) 
                    for tok in all_tokens}
    sorted_tokens = sorted(all_tokens, key=lambda t: token_totals[t], reverse=True)[:topn]
    
    if not sorted_tokens:
        print(f"[warn] No tokens found for pos={position}. Skipping grouped bar chart.")
        return
    
    # Compute overlap metrics
    metrics = compute_token_overlap_metrics(df, position, variants, topn=topn)
    
    x = np.arange(len(sorted_tokens))
    width = 0.25 if len(variants) <= 3 else 0.8 / len(variants)
    
    fig, ax = plt.subplots(figsize=(13, 6.5))
    
    bars_list = []
    for i, variant in enumerate(safe_variants_order(variants)):
        counts = [variant_token_counts[variant].get(tok, 0) for tok in sorted_tokens]
        offset = (i - (len(variants) - 1) / 2) * width
        bars = ax.bar(x + offset, counts, width, label=variant, 
                     color=get_variant_color(variant), alpha=0.8, 
                     edgecolor="black", linewidth=0.5)
        bars_list.append(bars)
    
    # Add reference lines showing when bars are similar (highlight lack of difference)
    if "base" in variants and "ft" in variants:
        base_counts = [variant_token_counts["base"].get(tok, 0) for tok in sorted_tokens]
        ft_counts = [variant_token_counts["ft"].get(tok, 0) for tok in sorted_tokens]
        
        # Draw connecting lines between base and ft when they're similar (within 2 counts)
        for i, (b, f) in enumerate(zip(base_counts, ft_counts)):
            if abs(b - f) <= 2:  # Similar counts
                ax.plot([x[i] - width, x[i] + width], [b, f], 
                       color="green", alpha=0.3, linewidth=2, linestyle=":", zorder=0)
    
    # Add statistical annotation
    if "base_ft_jaccard" in metrics:
        jaccard = metrics["base_ft_jaccard"]
        if jaccard < 0.5:
            note = f"Low similarity (J={jaccard:.2f}): Models differ"
            note_color = "#d62728"
        else:
            note = f"High similarity (J={jaccard:.2f}): Models are similar"
            note_color = "#2ca02c"
        
        ax.text(0.98, 0.02, note, transform=ax.transAxes,
               fontsize=10, verticalalignment="bottom", horizontalalignment="right",
               bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, 
                        edgecolor=note_color, linewidth=2),
               family="monospace", fontweight="bold")
    
    ax.set_xlabel("Token", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count across samples", fontsize=12, fontweight="bold")
    ax.set_title(f"Logit lens top-1 tokens: direct comparison (position {position})\n" +
                "Bars of similar height indicate similar model behavior", 
                fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_tokens, rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y", linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_similarity_heatmap(
    df: pd.DataFrame,
    out_path: Path,
    position: int,
    variants: List[str],
    topn: int = 15,
) -> None:
    """
    Create a heatmap showing token overlap/similarity between variants.
    Highlights the lack of significant differences.
    """
    variant_token_counts = {}
    all_tokens = set()
    
    for variant in safe_variants_order(variants):
        sub = df[(df["variant"] == variant) & (df["position"] == position) & (df["rank"] == 1)]
        token_counts = Counter(sub["token"].tolist())
        variant_token_counts[variant] = token_counts
        top_tokens = [tok for tok, _ in token_counts.most_common(topn)]
        all_tokens.update(top_tokens)
    
    # Sort tokens by total frequency
    token_totals = {tok: sum(variant_token_counts[v].get(tok, 0) for v in variants) 
                    for tok in all_tokens}
    sorted_tokens = sorted(all_tokens, key=lambda t: token_totals[t], reverse=True)[:topn]
    
    if not sorted_tokens:
        print(f"[warn] No tokens found for pos={position}. Skipping similarity heatmap.")
        return
    
    # Create matrix: rows = tokens, cols = variants
    matrix = []
    for tok in sorted_tokens:
        row = [variant_token_counts[v].get(tok, 0) for v in safe_variants_order(variants)]
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    fig, ax = plt.subplots(figsize=(max(8, len(variants) * 2.5), max(6, len(sorted_tokens) * 0.4)))
    
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(variants)))
    ax.set_xticklabels([v.upper() for v in safe_variants_order(variants)], fontsize=11, fontweight="bold")
    ax.set_yticks(np.arange(len(sorted_tokens)))
    ax.set_yticklabels(sorted_tokens, fontsize=9)
    
    # Add text annotations
    for i in range(len(sorted_tokens)):
        for j in range(len(variants)):
            val = matrix[i, j]
            color = "white" if val > matrix.max() * 0.5 else "black"
            ax.text(j, i, str(int(val)), ha="center", va="center", 
                   color=color, fontsize=8, fontweight="bold")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Count across samples", fontsize=10, fontweight="bold")
    
    # Compute and display similarity metrics
    metrics = compute_token_overlap_metrics(df, position, variants, topn=topn)
    if "base_ft_jaccard" in metrics:
        jaccard = metrics["base_ft_jaccard"]
        title = (f"Token frequency heatmap (position {position})\n"
                f"Base↔FT Jaccard similarity: {jaccard:.2f} - " +
                ("Models are similar" if jaccard > 0.5 else "Models differ"))
    else:
        title = f"Token frequency heatmap (position {position})"
    
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Model variant", fontsize=12, fontweight="bold")
    ax.set_ylabel("Token (ranked by total frequency)", fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ----------------------------
# Core: LaTeX table of top tokens (side-by-side)
# ----------------------------

def top_tokens_by_frequency(
    df: pd.DataFrame,
    position: int,
    variant: str,
    topn: int = 8,
) -> List[str]:
    sub = df[(df["variant"] == variant) & (df["position"] == position) & (df["rank"] == 1)]
    tokens = sub["token"].value_counts().head(topn).index.tolist()
    return tokens


def make_side_by_side_token_table(
    df: pd.DataFrame,
    positions: List[int],
    variants: List[str],
    topn: int,
) -> pd.DataFrame:
    """
    Returns a dataframe:
      position | base_top_tokens | ft_top_tokens | diff_top_tokens
    Each cell is a comma-separated list.
    """
    rows = []
    for pos in positions:
        row: Dict[str, str] = {"position": pos}
        for v in safe_variants_order(variants):
            toks = top_tokens_by_frequency(df, pos, v, topn=topn)
            row[f"{v}_top_tokens"] = ", ".join(toks)
        rows.append(row)
    return pd.DataFrame(rows)


def write_latex_table(df_table: pd.DataFrame, out_path: Path, topn: int) -> None:
    """
    Writes a human-controlled LaTeX table (not pandas' default) so it looks clean.
    """
    cols = df_table.columns.tolist()
    # Expect: position, base_top_tokens, ft_top_tokens, diff_top_tokens (order may vary)
    # We'll render in base/ft/diff order if present.
    col_map = {}
    for c in cols:
        if c.endswith("_top_tokens") and c.startswith("base"):
            col_map["base"] = c
        if c.endswith("_top_tokens") and c.startswith("ft"):
            col_map["ft"] = c
        if c.endswith("_top_tokens") and c.startswith("diff"):
            col_map["diff"] = c

    base_col = col_map.get("base")
    ft_col = col_map.get("ft")
    diff_col = col_map.get("diff")

    if not (base_col and ft_col and diff_col):
        raise ValueError(f"Expected base/ft/diff columns in table, got {cols}")

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{c|p{4.2cm}|p{4.2cm}|p{4.2cm}}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Pos.} & \textbf{Base model (top-%d)} & \textbf{Fine-tuned (top-%d)} & \textbf{Difference (top-%d)} \\" % (topn, topn, topn))
    lines.append(r"\hline")

    for _, row in df_table.iterrows():
        pos = int(row["position"])
        base_tokens = [t.strip() for t in str(row[base_col]).split(",") if t.strip()]
        ft_tokens = [t.strip() for t in str(row[ft_col]).split(",") if t.strip()]
        diff_tokens = [t.strip() for t in str(row[diff_col]).split(",") if t.strip()]

        base_tex = r"\texttt{" + ", ".join(escape_latex_token(t) for t in base_tokens) + "}"
        ft_tex = r"\texttt{" + ", ".join(escape_latex_token(t) for t in ft_tokens) + "}"
        diff_tex = r"\texttt{" + ", ".join(escape_latex_token(t) for t in diff_tokens) + "}"

        lines.append(f"{pos} & {base_tex} & {ft_tex} & {diff_tex} \\\\")
        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Most frequent logit-lens top-1 tokens at layer 16 across samples. "
        r"Each cell lists the top-%d tokens by frequency for the specified token position.}" % topn
    )
    lines.append(r"\label{tab:logit_lens_top_tokens}")
    lines.append(r"\end{table}")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="adl-model-diff/adl_results_50_samples",
                    help="Directory containing ADL outputs.")
    ap.add_argument("--out_dir", type=str, default="adl_report_artifacts",
                    help="Output directory for figures and tables.")
    ap.add_argument("--positions", type=int, nargs="+", default=[1, 2],
                    help="Token positions to include in table + plots (e.g., 0 1 2 3 4 5).")
    ap.add_argument("--topn", type=int, default=8,
                    help="Top-N tokens per cell in the LaTeX table (frequency-based).")
    ap.add_argument("--hist_topn", type=int, default=20,
                    help="Top-N tokens to show in histograms (frequency-based).")
    ap.add_argument("--make_variant_bars", action="store_true",
                    help="Also save per-variant bar charts (base/ft/diff) for each position.")
    ap.add_argument("--no_side_by_side", action="store_true",
                    help="Skip side-by-side comparison plots (created by default).")
    ap.add_argument("--no_grouped_bars", action="store_true",
                    help="Skip grouped bar chart comparisons (created by default).")
    ap.add_argument("--no_similarity_heatmap", action="store_true",
                    help="Skip similarity heatmap (created by default).")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    ensure_dir(fig_dir)
    ensure_dir(tab_dir)

    logit_csv = results_dir / "logit_lens" / "logit_lens_per_position.csv"
    if not logit_csv.exists():
        raise FileNotFoundError(f"Missing: {logit_csv}")

    df = pd.read_csv(logit_csv)
    # normalize dtypes defensively
    df["position"] = df["position"].astype(int)
    df["rank"] = df["rank"].astype(int)
    if "probability" in df.columns:
        df["probability"] = pd.to_numeric(df["probability"], errors="coerce")

    variants = sorted(df["variant"].unique().tolist())

    # 1) Frequency curve plot for each requested position (base/ft/diff in one figure)
    for pos in args.positions:
        out_path = fig_dir / f"logit_lens_token_frequency_pos{pos}.png"
        plot_token_frequency_curve(df, out_path, position=pos, variants=variants, topn=args.hist_topn)

        # 2) Side-by-side comparison (pedagogically useful)
        if not args.no_side_by_side:
            out_side = fig_dir / f"logit_lens_side_by_side_pos{pos}.png"
            plot_side_by_side_comparison(df, out_side, position=pos, variants=variants, topn=args.hist_topn)
        
        # 3) Grouped bar chart (direct comparison of same tokens)
        if not args.no_grouped_bars:
            out_grouped = fig_dir / f"logit_lens_grouped_bars_pos{pos}.png"
            plot_grouped_bar_comparison(df, out_grouped, position=pos, variants=variants, topn=min(args.hist_topn, 15))
        
        # 4) Similarity heatmap (highlights lack of differences)
        if not args.no_similarity_heatmap:
            out_heatmap = fig_dir / f"logit_lens_similarity_heatmap_pos{pos}.png"
            plot_similarity_heatmap(df, out_heatmap, position=pos, variants=variants, topn=args.hist_topn)

        # Optional: per-variant bar charts (helpful for appendix)
        if args.make_variant_bars:
            for v in safe_variants_order(variants):
                out_bar = fig_dir / f"logit_lens_token_hist_{v}_pos{pos}.png"
                plot_token_identity_bar(df, out_bar, position=pos, variant=v, topn=args.hist_topn)

    # 5) Side-by-side table of top tokens by frequency
    table_df = make_side_by_side_token_table(df, positions=args.positions, variants=variants, topn=args.topn)
    table_df.to_csv(tab_dir / "logit_lens_top_tokens_table.csv", index=False)
    write_latex_table(table_df, tab_dir / "logit_lens_top_tokens_table.tex", topn=args.topn)

    # 6) Index file
    figure_list = []
    for p in args.positions:
        figure_list.append(f"- `figures/logit_lens_token_frequency_pos{p}.png` (frequency curves with similarity metrics)")
        if not args.no_side_by_side:
            figure_list.append(f"- `figures/logit_lens_side_by_side_pos{p}.png` (side-by-side comparison with annotations)")
        if not args.no_grouped_bars:
            figure_list.append(f"- `figures/logit_lens_grouped_bars_pos{p}.png` (grouped bar comparison highlighting similarities)")
        if not args.no_similarity_heatmap:
            figure_list.append(f"- `figures/logit_lens_similarity_heatmap_pos{p}.png` (heatmap showing token overlap)")
        if args.make_variant_bars:
            for v in safe_variants_order(variants):
                figure_list.append(f"- `figures/logit_lens_token_hist_{v}_pos{p}.png` ({v} model only)")
    
    index_lines = [
        "### ADL report artifacts",
        "",
        "Figures:",
        *figure_list,
        "",
        "Tables:",
        "- `tables/logit_lens_top_tokens_table.tex`",
        "- `tables/logit_lens_top_tokens_table.csv`",
        "",
        "Notes:",
        "- **Frequency curves**: Show token rank vs count with Jaccard similarity and overlap metrics.",
        "- **Side-by-side plots**: Horizontal bar charts with similarity annotations (pedagogically useful).",
        "- **Grouped bar charts**: Direct comparison highlighting when models are similar (green connecting lines).",
        "- **Similarity heatmap**: Visual representation of token overlap between variants.",
        "- **LaTeX table**: Lists the most frequent top-1 tokens per variant/position.",
        "",
        "Key insight: Plots are designed to highlight when base and fine-tuned models show",
        "similar behavior (lack of significant differences), with statistical metrics displayed.",
    ]
    (out_dir / "FIGURE_INDEX.md").write_text("\n".join(index_lines), encoding="utf-8")

    print(f"Done. Wrote outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
