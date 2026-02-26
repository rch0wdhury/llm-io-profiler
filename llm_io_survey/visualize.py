"""
Visualization module for LLM I/O bottleneck analysis.

Generates publication-quality matplotlib figures corresponding to
the key analyses in the survey paper.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyArrowPatch
from typing import Sequence

from .model_profile import ModelProfile, KNOWN_MODELS
from .hardware import GPUProfile, GPU_PROFILES, get_gpu


# ── Style defaults ───────────────────────────────────────────────────

COLORS = {
    "W": "#e74c3c",   # red - weight I/O
    "K": "#f39c12",   # amber - KV cache I/O
    "A": "#27ae60",   # green - activation I/O
    "sys": "#8e44ad", # purple - system-level
    "ridge": "#2c3e50",
    "gpu": ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12"],
}

def _apply_style(fig, ax):
    """Apply consistent styling."""
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()


# ── 1. Roofline Plot ─────────────────────────────────────────────────

def plot_roofline(
    gpu: GPUProfile | str = "H100",
    models: dict[str, tuple[float, float]] | None = None,
    batch_sizes: list[int] | None = None,
    save_path: str | None = None,
    figsize: tuple = (10, 6),
):
    """
    Plot roofline model for a GPU, optionally with model operating points.

    Parameters
    ----------
    gpu : GPU name or GPUProfile
    models : dict mapping label -> (arithmetic_intensity, attainable_tflops)
             If None, plots default prefill/decode points.
    batch_sizes : List of batch sizes to mark on the decode line.
    save_path : If set, saves figure to this path.
    """
    if isinstance(gpu, str):
        gpu = get_gpu(gpu)

    fig, ax = plt.subplots(figsize=figsize)

    # Roofline envelope
    intensities = np.logspace(-1, 4, 1000)
    # Attainable TFLOP/s = min(peak_compute, BW_TB/s * I)
    att_tflops = np.minimum(gpu.fp16_tflops, gpu.hbm_bandwidth_tbps * intensities)

    ax.loglog(intensities, att_tflops, "k-", linewidth=2.5, zorder=5)

    # Ridge point
    ridge = gpu.ridge_point
    ax.axvline(ridge, color=COLORS["ridge"], linestyle=":", alpha=0.6, linewidth=1.5)
    ax.annotate(f"$I^* = {ridge:.0f}$", xy=(ridge, gpu.fp16_tflops * 0.7),
                fontsize=10, color=COLORS["ridge"], ha="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    # Regions
    ax.fill_between(intensities, att_tflops, alpha=0.03, color="blue",
                     where=intensities < ridge)
    ax.text(2, gpu.fp16_tflops * 0.05, "Memory-bound", fontsize=11,
            color="#2980b9", alpha=0.7, fontstyle="italic")
    ax.text(ridge * 2, gpu.fp16_tflops * 0.7, "Compute-bound", fontsize=11,
            color="#c0392b", alpha=0.7, fontstyle="italic")

    # Default operating points
    if batch_sizes is None:
        batch_sizes = [1, 8, 32, 128, 256]

    if models is None:
        # Plot decode points at various batch sizes
        for i, B in enumerate(batch_sizes):
            I = B  # arithmetic intensity ≈ B for decode
            perf = min(gpu.fp16_tflops, gpu.hbm_bandwidth_tbps * I)
            marker = "o" if B < ridge else "s"
            ax.plot(I, perf, marker, markersize=8 + i,
                    color=COLORS["gpu"][i % len(COLORS["gpu"])],
                    label=f"Decode B={B}", zorder=10)
        # Prefill point
        I_prefill = 2048
        perf_prefill = min(gpu.fp16_tflops, gpu.hbm_bandwidth_tbps * I_prefill)
        ax.plot(I_prefill, perf_prefill, "D", markersize=12, color="#2ecc71",
                label="Prefill (N=2048)", zorder=10)
    else:
        for i, (label, (I, perf)) in enumerate(models.items()):
            ax.plot(I, perf, "o", markersize=10,
                    color=COLORS["gpu"][i % len(COLORS["gpu"])],
                    label=label, zorder=10)

    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=12)
    ax.set_ylabel("Attainable Throughput (TFLOP/s)", fontsize=12)
    ax.set_title(f"Roofline Model — {gpu.name} (FP16/BF16)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0.5, 5000)
    ax.set_ylim(0.1, gpu.fp16_tflops * 2)
    _apply_style(fig, ax)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, ax


# ── 2. Crossover Analysis ───────────────────────────────────────────

def plot_crossover(
    model: ModelProfile | str,
    gpu: GPUProfile | str = "H100",
    context_lengths: list[int] | None = None,
    weight_bits: int = 16,
    kv_bits: int = 16,
    max_batch: int = 256,
    save_path: str | None = None,
    figsize: tuple = (10, 6),
):
    """
    Plot the W vs K crossover: the batch size where KV I/O overtakes weight I/O.

    Shows weight I/O (constant horizontal line) and KV I/O curves for
    multiple context lengths, with crossover points marked.
    """
    from .model_profile import get_model
    if isinstance(model, str):
        model = get_model(model)
    if isinstance(gpu, str):
        gpu = get_gpu(gpu)
    if context_lengths is None:
        context_lengths = [512, 1024, 4096, 16384, 65536]

    fig, ax = plt.subplots(figsize=figsize)
    batch_sizes = np.arange(1, max_batch + 1)

    # Weight I/O (constant w.r.t. batch size)
    w_gb = model.weight_io_bytes(weight_bits) / 1e9
    ax.axhline(w_gb, color=COLORS["W"], linewidth=2.5, linestyle="--",
               label=f"Weight I/O $\\mathcal{{W}}$ = {w_gb:.1f} GB", zorder=5)

    # KV I/O for each context length
    cmap = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(context_lengths)))
    for i, s in enumerate(context_lengths):
        kv_gb = [model.kv_io_per_step_bytes(s, int(B), kv_bits) / 1e9
                 for B in batch_sizes]
        label_s = f"{s//1024}K" if s >= 1024 else str(s)
        ax.plot(batch_sizes, kv_gb, color=cmap[i], linewidth=1.8,
                label=f"KV I/O $\\mathcal{{K}}$ @ {label_s} ctx")

        # Mark crossover
        cross_b = model.crossover_batch_size(s, weight_bits, kv_bits)
        if 1 <= cross_b <= max_batch:
            cross_kv = model.kv_io_per_step_bytes(s, max(1, int(cross_b)), kv_bits) / 1e9
            ax.plot(cross_b, cross_kv, "x", color=cmap[i], markersize=10,
                    markeredgewidth=2.5, zorder=10)
            ax.annotate(f"B={cross_b:.0f}", xy=(cross_b, cross_kv),
                        xytext=(cross_b + 5, cross_kv * 1.15),
                        fontsize=8, color=cmap[i])

    # Max batch annotation
    b_max = model.max_batch_size(gpu, context_lengths[2], 1, weight_bits, kv_bits)
    if 1 < b_max < max_batch:
        ax.axvline(b_max, color="#95a5a6", linestyle=":", alpha=0.7)
        ax.annotate(f"$B_{{max}}$={b_max}\n(HBM limit @ {context_lengths[2]//1024}K)",
                    xy=(b_max, w_gb * 0.3), fontsize=8, color="#7f8c8d", ha="center")

    ax.set_xlabel("Batch Size $B$", fontsize=12)
    ax.set_ylabel("I/O per Decode Step (GB)", fontsize=12)
    ax.set_title(f"W↔K Crossover — {model.name} ({gpu.name}, "
                 f"W{weight_bits}/KV{kv_bits})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(1, max_batch)
    _apply_style(fig, ax)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, ax


# ── 3. KV Cache Scaling ─────────────────────────────────────────────

def plot_kv_scaling(
    models: list[str] | None = None,
    kv_bits: int = 16,
    save_path: str | None = None,
    figsize: tuple = (12, 5),
):
    """
    Reproduce Figure 3 from the paper: KV cache growth vs context length
    for multiple architectures, plus KV as % of weight footprint.
    """
    from .model_profile import get_model

    if models is None:
        models = ["llama-2-7b", "llama-3-8b", "llama-3-70b", "phi-3-mini", "deepseek-v3"]

    ctx_lengths = np.array([1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    colors = plt.cm.Set1(np.linspace(0, 0.8, len(models)))

    for i, mname in enumerate(models):
        m = get_model(mname)
        kv_gb = [m.kv_cache_bytes(int(s), 1, kv_bits) / 1e9 for s in ctx_lengths]
        w_gb = m.weight_footprint_bytes(16) / 1e9
        kv_pct = [100 * kv / w_gb if w_gb > 0 else 0 for kv in kv_gb]

        ax1.semilogy(ctx_lengths / 1024, kv_gb, "o-", color=colors[i],
                     linewidth=1.8, markersize=4, label=f"{m.name} ({m.attention_type})")
        ax2.semilogy(ctx_lengths / 1024, kv_pct, "s-", color=colors[i],
                     linewidth=1.8, markersize=4, label=f"{m.name}")

    # H100 capacity line
    ax1.axhline(80, color="#e74c3c", linestyle="--", alpha=0.5, linewidth=1)
    ax1.text(ctx_lengths[-1] / 1024, 85, "H100 HBM (80 GB)", fontsize=8,
             color="#e74c3c", ha="right")

    ax2.axhline(100, color="#e74c3c", linestyle="--", alpha=0.5, linewidth=1)
    ax2.text(ctx_lengths[-1] / 1024, 115, "KV = Weights", fontsize=8,
             color="#e74c3c", ha="right")

    ax1.set_xlabel("Context Length (K tokens)", fontsize=11)
    ax1.set_ylabel("KV Cache per Sequence (GB)", fontsize=11)
    ax1.set_title("(a) KV Cache Size (B=1, FP16)", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8)

    ax2.set_xlabel("Context Length (K tokens)", fontsize=11)
    ax2.set_ylabel("KV / Weights (%)", fontsize=11)
    ax2.set_title("(b) KV as % of Weight Footprint", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8)

    for a in (ax1, ax2):
        _apply_style(fig, a)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, (ax1, ax2)


# ── 4. Waterfall / Composability Chart ───────────────────────────────

def plot_waterfall(
    model: ModelProfile | str = "llama-3-70b",
    gpu: GPUProfile | str = "H100",
    seq_len: int = 4096,
    batch_size: int = 32,
    save_path: str | None = None,
    figsize: tuple = (12, 6),
):
    """
    Waterfall chart showing cumulative I/O reduction as optimizations stack.
    Corresponds to Table 7 / Section 8 of the paper.
    """
    from .model_profile import get_model
    if isinstance(model, str):
        model = get_model(model)
    if isinstance(gpu, str):
        gpu = get_gpu(gpu)

    bw = gpu.hbm_bandwidth_sustained_tbps * 1e3  # GB/s

    # Compute each configuration
    configs = []

    # 1. FP16 baseline
    w = model.weight_io_bytes(16) / 1e9
    k = model.kv_io_per_step_bytes(seq_len, batch_size, 16) / 1e9
    configs.append(("FP16 Baseline", w, k, "W"))

    # 2. INT4 weights
    w4 = model.weight_io_bytes(4) / 1e9
    k16 = k  # KV still FP16
    configs.append(("+ INT4 Weights", w4, k16, "K"))

    # 3. INT4 KV cache
    k4 = model.kv_io_per_step_bytes(seq_len, batch_size, 4) / 1e9
    configs.append(("+ INT4 KV Cache", w4, k4, "W"))

    # 4. 2:4 sparsity (halves weights)
    w4s = w4 / 2
    configs.append(("+ 2:4 Sparsity", w4s, k4, "W"))

    # 5. Speculative decoding (amortize W by ~4x)
    w_spec = w4s / 4  # ~4 tokens accepted per verification
    configs.append(("+ Spec. Decoding\n(γ=5, α=0.8)", w_spec, k4, "K"))

    labels = [c[0] for c in configs]
    ws = [c[1] for c in configs]
    ks = [c[2] for c in configs]
    totals = [c[1] + c[2] for c in configs]
    dominants = [c[3] for c in configs]

    x = np.arange(len(configs))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=figsize)

    # Stacked bars
    bars_w = ax1.bar(x - width/2, ws, width, label="Weight I/O $\\mathcal{W}$",
                     color=COLORS["W"], alpha=0.85, edgecolor="white")
    bars_k = ax1.bar(x + width/2, ks, width, label="KV Cache I/O $\\mathcal{K}$",
                     color=COLORS["K"], alpha=0.85, edgecolor="white")

    # Total line
    ax1.plot(x, totals, "ko-", linewidth=2, markersize=7, label="Total I/O", zorder=10)

    # TPOT on secondary axis
    ax2 = ax1.twinx()
    tpots = [(t * 1000) / bw for t in totals]  # ms = GB / (GB/s) * 1000
    ax2.plot(x, tpots, "s--", color="#8e44ad", linewidth=1.5, markersize=6,
             label="Est. TPOT (ms)", zorder=9)
    ax2.set_ylabel("Estimated TPOT (ms)", fontsize=11, color="#8e44ad")
    ax2.tick_params(axis="y", labelcolor="#8e44ad")

    # Dominant flow annotations
    for i, dom in enumerate(dominants):
        color = COLORS["W"] if dom == "W" else COLORS["K"]
        ax1.annotate(f"{'W' if dom == 'W' else 'K'}-bound",
                     xy=(i, totals[i] + 3), fontsize=9, ha="center",
                     color=color, fontweight="bold")

    # Reduction annotation
    reduction = totals[0] / totals[-1]
    ax1.annotate(f"{reduction:.0f}× total\nreduction",
                 xy=(len(configs) - 1, totals[-1]),
                 xytext=(len(configs) - 1.5, totals[0] * 0.5),
                 fontsize=10, fontweight="bold", color="#2c3e50",
                 arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.5))

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9, ha="center")
    ax1.set_ylabel("I/O per Decode Step (GB)", fontsize=11)
    ax1.set_title(f"I/O Waterfall — {model.name} ({gpu.name}, B={batch_size}, "
                  f"ctx={seq_len})", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)
    ax2.legend(loc="center right", fontsize=9)
    _apply_style(fig, ax1)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, ax1


# ── 5. Hardware Scaling (Ridge Point Over Generations) ───────────────

def plot_hw_scaling(save_path: str | None = None, figsize: tuple = (9, 5)):
    """
    Reproduce Figure 4: GPU compute vs bandwidth scaling across generations.
    """
    gens = ["V100\n(2017)", "A100\n(2020)", "H100\n(2022)", "B200\n(2024)"]
    keys = ["V100", "A100", "H100", "B200"]
    bws = [GPU_PROFILES[k].hbm_bandwidth_tbps for k in keys]
    computes = [GPU_PROFILES[k].fp16_tflops / 1000 for k in keys]  # PFLOP/s
    ridges = [GPU_PROFILES[k].ridge_point for k in keys]

    fig, ax1 = plt.subplots(figsize=figsize)
    x = np.arange(len(gens))
    width = 0.3

    ax1.bar(x - width/2, bws, width, label="HBM BW (TB/s)", color="#3498db", alpha=0.8)
    ax1.bar(x + width/2, computes, width, label="Compute (PFLOP/s)", color="#e74c3c", alpha=0.8)
    ax1.set_ylabel("TB/s or PFLOP/s", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(gens, fontsize=10)
    ax1.legend(loc="upper left", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(x, ridges, "ko-", linewidth=2, markersize=8, label="Ridge Point", zorder=10)
    for i, r in enumerate(ridges):
        ax2.annotate(f"{r:.0f}", xy=(i, r), xytext=(i + 0.15, r + 10),
                     fontsize=10, fontweight="bold")
    ax2.set_ylabel("Ridge Point (FLOP/byte)", fontsize=11)
    ax2.legend(loc="center right", fontsize=9)

    ax1.set_title("GPU Compute vs. Bandwidth Scaling", fontsize=13, fontweight="bold")
    _apply_style(fig, ax1)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, ax1


# ── 6. TPOT Heatmap ─────────────────────────────────────────────────

def plot_tpot_heatmap(
    model: ModelProfile | str = "llama-3-70b",
    gpu: GPUProfile | str = "H100",
    weight_bits_options: list[int] | None = None,
    kv_bits_options: list[int] | None = None,
    seq_len: int = 4096,
    batch_size: int = 1,
    save_path: str | None = None,
    figsize: tuple = (8, 6),
):
    """
    Heatmap of estimated TPOT across weight and KV quantization settings.
    """
    from .model_profile import get_model
    if isinstance(model, str):
        model = get_model(model)
    if isinstance(gpu, str):
        gpu = get_gpu(gpu)
    if weight_bits_options is None:
        weight_bits_options = [16, 8, 4, 2]
    if kv_bits_options is None:
        kv_bits_options = [16, 8, 4, 2]

    tpot_matrix = np.zeros((len(weight_bits_options), len(kv_bits_options)))
    for i, wb in enumerate(weight_bits_options):
        for j, kb in enumerate(kv_bits_options):
            tpot_matrix[i, j] = model.tpot_ms(seq_len, batch_size, gpu, wb, kb)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(tpot_matrix, cmap="RdYlGn_r", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, label="TPOT (ms)")

    ax.set_xticks(range(len(kv_bits_options)))
    ax.set_xticklabels([f"KV{b}" for b in kv_bits_options])
    ax.set_yticks(range(len(weight_bits_options)))
    ax.set_yticklabels([f"W{b}" for b in weight_bits_options])

    for i in range(len(weight_bits_options)):
        for j in range(len(kv_bits_options)):
            ax.text(j, i, f"{tpot_matrix[i,j]:.1f}",
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color="white" if tpot_matrix[i,j] > tpot_matrix.mean() else "black")

    ax.set_xlabel("KV Cache Precision", fontsize=11)
    ax.set_ylabel("Weight Precision", fontsize=11)
    ax.set_title(f"TPOT (ms) — {model.name} ({gpu.name}, B={batch_size}, "
                 f"ctx={seq_len})", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, ax


# ── 7. Multi-model comparison bar chart ──────────────────────────────

def plot_model_comparison(
    models: list[str] | None = None,
    gpu: GPUProfile | str = "H100",
    seq_len: int = 4096,
    batch_size: int = 1,
    weight_bits: int = 16,
    kv_bits: int = 16,
    save_path: str | None = None,
    figsize: tuple = (14, 6),
):
    """
    Side-by-side bar chart comparing W, K, and total I/O across models.
    """
    from .model_profile import get_model
    if isinstance(gpu, str):
        gpu = get_gpu(gpu)
    if models is None:
        models = ["llama-3-8b", "llama-3-70b", "mistral-7b", "mixtral-8x7b",
                   "qwen-2.5-72b", "deepseek-v3"]

    profiles = [get_model(m) for m in models]
    names = [p.name for p in profiles]

    ws = [p.weight_io_bytes(weight_bits) / 1e9 for p in profiles]
    ks = [p.kv_io_per_step_bytes(seq_len, batch_size, kv_bits) / 1e9 for p in profiles]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width/2, ws, width, label="$\\mathcal{W}$ (Weight I/O)",
           color=COLORS["W"], alpha=0.85)
    ax.bar(x + width/2, ks, width, label="$\\mathcal{K}$ (KV Cache I/O)",
           color=COLORS["K"], alpha=0.85)

    # Dominant flow markers
    for i in range(len(names)):
        dom = "W" if ws[i] > ks[i] else "K"
        ax.text(i, max(ws[i], ks[i]) + 1, f"{dom}-bound",
                ha="center", fontsize=8, fontweight="bold",
                color=COLORS["W"] if ws[i] > ks[i] else COLORS["K"])

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("I/O per Decode Step (GB)", fontsize=11)
    ax.set_title(f"Model I/O Comparison ({gpu.name}, B={batch_size}, "
                 f"ctx={seq_len}, W{weight_bits}/KV{kv_bits})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    _apply_style(fig, ax)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, ax
