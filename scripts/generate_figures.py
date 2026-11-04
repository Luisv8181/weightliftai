"""Generate publication-grade figures for WeightLiftAI Experiments 001 and 002.
Produces:
1. fig1_interlingua_hourglass.png - Cross-lingual transfer & item similarity across layers (Gemma 3 vs Qwen 2.5)
2. fig2_lexical_shortcut_trap.png - 2x2 counterfactual activations & probe failure rates
3. fig3_dialect_geometry_gap.png - Dominican idioms vs Pan-Hispanic Spanish item-level cosine distribution
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("C:/Users/luisv/Projects/weightliftai")
FIG_DIR = ROOT / "docs/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Editorial styling: dark modern aesthetics matching our visual identity
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
        "axes.edgecolor": "#334155",
        "axes.linewidth": 1.2,
        "grid.color": "#1e293b",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "text.color": "#f8fafc",
        "axes.labelcolor": "#cbd5e1",
        "xtick.color": "#94a3b8",
        "ytick.color": "#94a3b8",
        "figure.facecolor": "#0b0f19",
        "axes.facecolor": "#111827",
    }
)

# ---------------------------------------------------------------------------
# FIGURE 1: The Interlingua Hourglass
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)
fig.suptitle(
    "The Interlingua Hourglass: Cross-Lingual Emotion Representations Across Model Depths",
    fontsize=16,
    fontweight="bold",
    y=0.98,
    color="#f8fafc",
)

# Gemma 3 1B IT (SAE Features)
gemma_layers = [7, 13, 17, 22]
gemma_en_esn = [0.925, 1.000, 0.950, 0.850]
gemma_en_esdo = [0.800, 0.925, 0.825, 0.700]
gemma_esdo_en = [0.700, 0.775, 0.775, 0.550]

ax1.plot(gemma_layers, gemma_en_esn, marker="o", markersize=8, color="#2dd4bf", linewidth=2.5, label="EN → Pan-Hispanic ES (Transfer)")
ax1.plot(gemma_layers, gemma_en_esdo, marker="s", markersize=8, color="#38bdf8", linewidth=2.5, label="EN → Dominican Idioms (Transfer)")
ax1.plot(gemma_layers, gemma_esdo_en, marker="^", markersize=8, color="#f472b6", linewidth=2.2, linestyle="--", label="Dominican Idioms → EN (Transfer)")

ax1.axvspan(11.5, 14.5, color="#2dd4bf", alpha=0.08, label="Layer 13 Concept Hub")
ax1.axhline(0.25, color="#64748b", linestyle=":", linewidth=1.2, label="Chance Level (0.25)")
ax1.set_title("Gemma 3 1B IT (Layer-wise SAEs)", fontsize=13, fontweight="bold", pad=12)
ax1.set_xlabel("Transformer Layer Depth", fontsize=11, labelpad=8)
ax1.set_ylabel("Zero-Shot 4-Way Classification Accuracy", fontsize=11, labelpad=8)
ax1.set_xticks(gemma_layers)
ax1.set_ylim(0.20, 1.05)
ax1.grid(True)
ax1.legend(loc="lower left", framealpha=0.9, facecolor="#1e293b", edgecolor="#334155", fontsize=9.5)

# Annotate the peak and drop
ax1.annotate(
    "Peak Convergence\n(100% Transfer)",
    xy=(13, 1.000),
    xytext=(13.3, 1.02),
    arrowprops=dict(arrowstyle="->", color="#2dd4bf", lw=1.5),
    fontsize=9.5,
    fontweight="bold",
    color="#2dd4bf",
)
ax1.annotate(
    "Late Token-Space Divergence\n(Accuracy Drops 15–22%)",
    xy=(22, 0.700),
    xytext=(17.5, 0.60),
    arrowprops=dict(arrowstyle="->", color="#f472b6", lw=1.5),
    fontsize=9.5,
    color="#f472b6",
)

# Qwen 2.5 0.5B Instruct (Dense Residual Stream)
qwen_layers = [0, 6, 12, 18, 23]
qwen_en_es = [0.425, 0.775, 0.675, 0.825, 0.550]
qwen_es_en = [0.550, 0.800, 0.975, 0.950, 0.625]

ax2.plot(qwen_layers, qwen_en_es, marker="o", markersize=8, color="#fbbf24", linewidth=2.5, label="EN → ES (Dense Residual)")
ax2.plot(qwen_layers, qwen_es_en, marker="D", markersize=8, color="#a78bfa", linewidth=2.5, label="ES → EN (Dense Residual)")

ax2.axvspan(11.5, 18.5, color="#fbbf24", alpha=0.08, label="Layers 12–18 Shared Hub")
ax2.axhline(0.25, color="#64748b", linestyle=":", linewidth=1.2, label="Chance Level (0.25)")
ax2.set_title("Qwen 2.5 0.5B Instruct (Dense Residual Stream)", fontsize=13, fontweight="bold", pad=12)
ax2.set_xlabel("Transformer Layer Depth", fontsize=11, labelpad=8)
ax2.set_ylabel("Zero-Shot 4-Way Classification Accuracy", fontsize=11, labelpad=8)
ax2.set_xticks(qwen_layers)
ax2.set_ylim(0.20, 1.05)
ax2.grid(True)
ax2.legend(loc="lower left", framealpha=0.9, facecolor="#1e293b", edgecolor="#334155", fontsize=9.5)

ax2.annotate(
    "Peak Alignment\n(97.5% Transfer)",
    xy=(12, 0.975),
    xytext=(12.3, 0.99),
    arrowprops=dict(arrowstyle="->", color="#a78bfa", lw=1.5),
    fontsize=9.5,
    fontweight="bold",
    color="#a78bfa",
)
ax2.annotate(
    "Final Output Layer Collapse\n(Drops to 55–62%)",
    xy=(23, 0.550),
    xytext=(18.5, 0.45),
    arrowprops=dict(arrowstyle="->", color="#fbbf24", lw=1.5),
    fontsize=9.5,
    color="#fbbf24",
)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig1_path = FIG_DIR / "fig1_interlingua_hourglass.png"
fig.savefig(fig1_path, facecolor=fig.get_facecolor(), dpi=300)
plt.close(fig)
print(f"Generated {fig1_path}")


# ---------------------------------------------------------------------------
# FIGURE 2: The Lexical Shortcut Trap & Probe Blind Spot
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300, gridspec_kw={"width_ratios": [1.4, 1.0]})
fig.suptitle(
    "The Lexical Shortcut Trap: De-confounding Apparent 'Grief' Features in Gemma 3 1B",
    fontsize=16,
    fontweight="bold",
    y=0.98,
    color="#f8fafc",
)

# Panel A: Feature Activations on the 2x2 Counterfactual Set
features = ["#293\n(people one cares for)", "#1886\n(death / mortality)", "#4284\n('still / todavía')", "#602\n(facial states)", "#701\n(sorrow / eternity)"]
x = np.arange(len(features))
width = 0.20

# Data from experiment_002_summary.json
exp_grief = [375.35, 245.24, 405.10, 86.15, 73.41]
imp_grief = [81.86, 42.99, 50.61, 110.95, 69.41]
lex_trap = [406.20, 163.19, 415.81, 0.00, 0.00]
neutral = [5.00, 0.00, 0.00, 0.00, 0.00]

rects1 = ax1.bar(x - 1.5 * width, exp_grief, width, label="Explicit Grief (Family + Death + 'Still')", color="#c084fc")
rects2 = ax1.bar(x - 0.5 * width, imp_grief, width, label="Implicit Grief (ZERO keyword cues)", color="#38bdf8")
rects3 = ax1.bar(x + 0.5 * width, lex_trap, width, label="Lexical Trap (Morticians/Historians, ZERO grief)", color="#f87171")
rects4 = ax1.bar(x + 1.5 * width, neutral, width, label="Baseline Neutral", color="#64748b")

ax1.set_title("Layer-13 SAE Feature Activations Across Counterfactual Conditions", fontsize=12.5, fontweight="bold", pad=12)
ax1.set_ylabel("Mean Maximum SAE Activation", fontsize=11, labelpad=8)
ax1.set_xticks(x)
ax1.set_xticklabels(features, fontsize=10)
ax1.grid(True, axis="y")
ax1.legend(loc="upper right", framealpha=0.9, facecolor="#1e293b", edgecolor="#334155", fontsize=9)

# Draw highlight boxes around lexical shortcuts vs candidate true grief
ax1.axvspan(-0.4, 2.4, color="#f87171", alpha=0.08)
ax1.text(1.0, 430, "Lexical Shortcuts (#293, #1886, #4284)\nFire equally on neutral morticians/genealogists!", ha="center", fontsize=9.5, color="#fca5a5", fontweight="bold")

ax1.axvspan(2.6, 4.4, color="#38bdf8", alpha=0.08)
ax1.text(3.5, 140, "Candidate Invariant (#602, #701)\nFire on implicit grief; zero on traps", ha="center", fontsize=9.5, color="#7dd3fc", fontweight="bold")

# Panel B: Grief Probe Failure on Held-Out Split B
categories = ["Explicit Grief\nRecall", "Implicit Grief\nRecall", "Lexical Trap\nFalse Positives", "Baseline Neutral\nFalse Positives"]
rates_unclamped = [1.00, 0.10, 1.00, 0.00]
rates_clamped = [1.00, 0.00, 0.90, 0.00]

x_p = np.arange(len(categories))
w_p = 0.35

ax2.bar(x_p - w_p / 2, rates_unclamped, w_p, label="Standard Probe (Unclamped)", color="#ef4444")
ax2.bar(x_p + w_p / 2, rates_clamped, w_p, label="Clamped (#1886, #4284, #293 zeroed)", color="#f97316")

for i, (u, c) in enumerate(zip(rates_unclamped, rates_clamped)):
    ax2.text(i - w_p / 2, u + 0.02, f"{int(u*100)}%", ha="center", fontsize=9, fontweight="bold", color="#fca5a5")
    ax2.text(i + w_p / 2, c + 0.02, f"{int(c*100)}%", ha="center", fontsize=9, fontweight="bold", color="#fed7aa")

ax2.set_title("Grief Probe Detection on Held-Out Split B", fontsize=12.5, fontweight="bold", pad=12)
ax2.set_ylabel("Rate (0.0 to 1.0)", fontsize=11, labelpad=8)
ax2.set_xticks(x_p)
ax2.set_xticklabels(categories, fontsize=9.5)
ax2.set_ylim(0, 1.15)
ax2.grid(True, axis="y")
ax2.legend(loc="upper right", framealpha=0.9, facecolor="#1e293b", edgecolor="#334155", fontsize=9)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig2_path = FIG_DIR / "fig2_lexical_shortcut_trap.png"
fig.savefig(fig2_path, facecolor=fig.get_facecolor(), dpi=300)
plt.close(fig)
print(f"Generated {fig2_path}")


# ---------------------------------------------------------------------------
# FIGURE 3: Dialect Geometry Gap (Standard Spanish vs Dominican Idioms)
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300, gridspec_kw={"width_ratios": [1.1, 1.0]})
fig.suptitle(
    "Representational Geometry of Dialect: Pan-Hispanic Spanish vs. Dominican Idioms of Distress",
    fontsize=16,
    fontweight="bold",
    y=0.98,
    color="#f8fafc",
)

# Panel A: Item-level Cosine Across Layers
layers = [7, 13, 17, 22]
cos_en_esn = [0.693, 0.743, 0.625, 0.391]
cos_en_esdo = [0.554, 0.616, 0.477, 0.314]
cos_esn_esdo = [0.690, 0.740, 0.650, 0.680]

ax1.plot(layers, cos_en_esn, marker="o", markersize=8, color="#2dd4bf", linewidth=2.5, label="English ~ Pan-Hispanic Spanish (Mean Item Cosine)")
ax1.plot(layers, cos_en_esdo, marker="s", markersize=8, color="#f43f5e", linewidth=2.5, label="English ~ Dominican Idioms (Mean Item Cosine)")
ax1.plot(layers, cos_esn_esdo, marker="D", markersize=8, color="#a855f7", linewidth=2.2, linestyle="--", label="Pan-Hispanic ~ Dominican Idioms")

# Shaded gap
ax1.fill_between(layers, cos_en_esn, cos_en_esdo, color="#f43f5e", alpha=0.15, label="Dialect Geometry Gap (Δ ≈ 0.13–0.15)")

ax1.set_title("Matched-Item Feature Cosine Similarity Across Depth", fontsize=12.5, fontweight="bold", pad=12)
ax1.set_xlabel("Transformer Layer Depth", fontsize=11, labelpad=8)
ax1.set_ylabel("Item-Level Cosine Similarity (Average of 40 Triplet Pairs)", fontsize=11, labelpad=8)
ax1.set_xticks(layers)
ax1.set_ylim(0.20, 0.85)
ax1.grid(True)
ax1.legend(loc="lower left", framealpha=0.9, facecolor="#1e293b", edgecolor="#334155", fontsize=9.5)

# Panel B: Asymmetric Cross-Dialect Transfer at Layer 13
transfer_pairs = [
    "EN → Pan-Hispanic ES",
    "EN → Dominican ES",
    "Pan-Hispanic → Dominican",
    "Dominican → Pan-Hispanic",
    "Dominican → English",
]
transfer_scores = [1.000, 0.925, 0.950, 0.825, 0.775]
colors = ["#2dd4bf", "#38bdf8", "#818cf8", "#f472b6", "#fb7185"]

bars = ax2.barh(transfer_pairs, transfer_scores, color=colors, height=0.55)
for b in bars:
    w = b.get_width()
    ax2.text(w + 0.015, b.get_y() + b.get_height() / 2, f"{w:.3f}", va="center", fontsize=9.5, fontweight="bold", color="#f8fafc")

ax2.axvline(0.25, color="#64748b", linestyle=":", linewidth=1.2, label="Chance Level (0.25)")
ax2.set_xlim(0, 1.15)
ax2.set_title("Layer 13 Zero-Shot Transfer Between Dialects", fontsize=12.5, fontweight="bold", pad=12)
ax2.set_xlabel("Classification Accuracy", fontsize=11, labelpad=8)
ax2.grid(True, axis="x")
ax2.invert_yaxis()
ax2.legend(loc="lower right", framealpha=0.9, facecolor="#1e293b", edgecolor="#334155", fontsize=9)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig3_path = FIG_DIR / "fig3_dialect_geometry_gap.png"
fig.savefig(fig3_path, facecolor=fig.get_facecolor(), dpi=300)
plt.close(fig)
print(f"Generated {fig3_path}")
