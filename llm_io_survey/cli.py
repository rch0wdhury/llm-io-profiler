#!/usr/bin/env python3
"""
llm-io-analyzer: Command-line tool for LLM inference I/O bottleneck analysis.

Usage:
    # Analyze a known model
    python -m llm_io_survey.cli analyze llama-3-70b

    # Analyze from HuggingFace config.json
    python -m llm_io_survey.cli analyze --config path/to/config.json

    # Plot crossover analysis
    python -m llm_io_survey.cli crossover llama-3-70b --save crossover.png

    # Generate all figures from the paper
    python -m llm_io_survey.cli figures --outdir figures/

    # Compare multiple models
    python -m llm_io_survey.cli compare llama-3-8b llama-3-70b mixtral-8x7b
"""

import argparse
import json
import sys
from pathlib import Path


def cmd_analyze(args):
    """Analyze a model's I/O profile."""
    from llm_io_survey import profile_model, from_hf_config, get_gpu

    if args.config:
        override_name = args.model if args.model != "llama-3-70b" else None
        model = from_hf_config(args.config, name=override_name)
    else:
        model = profile_model(args.model)

    gpu = get_gpu(args.gpu)
    s = model.summary(
        gpu=gpu,
        seq_len=args.ctx,
        batch_size=args.batch,
        weight_bits=args.wbits,
        kv_bits=args.kvbits,
    )

    print(f"\n{'='*60}")
    print(f"  I/O Profile: {s['model']}")
    print(f"  GPU: {s['gpu']}  |  Batch: {args.batch}  |  Context: {args.ctx}")
    print(f"  Precision: W{args.wbits} / KV{args.kvbits}")
    print(f"{'='*60}")
    print(f"  Architecture     : {s['attention_type']}" +
          (f" + MoE ({model.num_experts_per_tok}/{model.num_experts})" if s['is_moe'] else ""))
    print(f"  Total params     : {s['total_params_B']:.1f}B")
    print(f"  Active params/tok: {s['active_params_B']:.1f}B")
    print(f"  Weight footprint : {s['weight_footprint_gb']:.1f} GB")
    print(f"{'─'*60}")
    print(f"  Weight I/O (𝒲)   : {s['weight_io_per_step_gb']:.2f} GB / step")
    print(f"  KV Cache I/O (𝒦) : {s['kv_io_per_step_gb']:.2f} GB / step")
    print(f"  Total decode I/O : {s['total_decode_io_gb']:.2f} GB / step")
    print(f"  Dominant flow    : {s['dominant_flow']}")
    print(f"{'─'*60}")
    print(f"  Arith. intensity : {s['arithmetic_intensity']:.1f} FLOP/byte"
          f"  (ridge: {s['ridge_point']:.0f})")
    print(f"  Est. TPOT        : {s['tpot_ms']:.1f} ms")
    print(f"  W↔K crossover    : B ≈ {s['crossover_batch_size']:.1f}")
    max_b = s['max_batch_size']
    max_b_str = f"{max_b}" if max_b > 0 else "0 (needs multi-GPU)"
    print(f"  Max batch (HBM)  : {max_b_str}")
    print(f"  KV / token / layer: {s['kv_per_token_per_layer_bytes']:.0f} bytes")
    print(f"  KV total ({args.ctx} ctx): {s['kv_cache_total_gb']:.2f} GB")
    print(f"{'='*60}\n")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(s, f, indent=2)
        print(f"  → JSON saved to {args.json}")


def cmd_crossover(args):
    """Plot the W vs K crossover analysis."""
    from llm_io_survey import plot_crossover, profile_model, from_hf_config

    if args.config:
        override_name = args.model if args.model != "llama-3-70b" else None
        model = from_hf_config(args.config, name=override_name)
    else:
        model = profile_model(args.model)

    fig, ax = plot_crossover(
        model, gpu=args.gpu,
        weight_bits=args.wbits, kv_bits=args.kvbits,
        max_batch=args.max_batch,
        save_path=args.save,
    )
    if not args.save:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        print(f"  → Saved to {args.save}")


def cmd_compare(args):
    """Compare I/O profiles of multiple models."""
    from llm_io_survey import profile_model, get_gpu

    gpu = get_gpu(args.gpu)

    print(f"\n{'Model':<25} {'Attn':<5} {'Params':<8} {'W(GB)':<8} "
          f"{'K(GB)':<8} {'Total':<8} {'TPOT':<8} {'Dom.':<12} {'B_cross':<8}")
    print("─" * 100)
    for name in args.models:
        m = profile_model(name)
        s = m.summary(gpu, args.ctx, args.batch, args.wbits, args.kvbits)
        print(f"{s['model']:<25} {s['attention_type']:<5} "
              f"{s['total_params_B']:<8.1f} {s['weight_io_per_step_gb']:<8.2f} "
              f"{s['kv_io_per_step_gb']:<8.2f} {s['total_decode_io_gb']:<8.2f} "
              f"{s['tpot_ms']:<8.1f} {s['dominant_flow']:<12} "
              f"{s['crossover_batch_size']:<8.1f}")
    print()

    if args.save:
        from llm_io_survey import plot_model_comparison
        fig, ax = plot_model_comparison(
            args.models, gpu=args.gpu,
            seq_len=args.ctx, batch_size=args.batch,
            weight_bits=args.wbits, kv_bits=args.kvbits,
            save_path=args.save,
        )
        print(f"  → Chart saved to {args.save}")


def cmd_figures(args):
    """Generate all paper figures."""
    from llm_io_survey import (
        plot_roofline, plot_crossover, plot_kv_scaling,
        plot_waterfall, plot_hw_scaling, plot_tpot_heatmap,
        plot_model_comparison,
    )
    import matplotlib
    matplotlib.use("Agg")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Generating paper figures...")

    print("  [1/7] Roofline model...")
    plot_roofline("H100", save_path=str(outdir / "roofline.png"))

    print("  [2/7] W↔K crossover (Llama-3 70B)...")
    plot_crossover("llama-3-70b", "H100", save_path=str(outdir / "crossover_70b.png"))

    print("  [3/7] KV cache scaling...")
    plot_kv_scaling(save_path=str(outdir / "kv_scaling.png"))

    print("  [4/7] I/O waterfall...")
    plot_waterfall("llama-3-70b", "H100", save_path=str(outdir / "waterfall.png"))

    print("  [5/7] Hardware scaling...")
    plot_hw_scaling(save_path=str(outdir / "hw_scaling.png"))

    print("  [6/7] TPOT heatmap...")
    plot_tpot_heatmap("llama-3-70b", "H100", save_path=str(outdir / "tpot_heatmap.png"))

    print("  [7/7] Model comparison...")
    plot_model_comparison(save_path=str(outdir / "model_comparison.png"))

    print(f"\n✓ All figures saved to {outdir}/")


def main():
    parser = argparse.ArgumentParser(
        prog="llm-io-analyzer",
        description="I/O Bottleneck Analysis Toolkit for LLM Inference",
    )
    sub = parser.add_subparsers(dest="command")

    # ── analyze ──
    p_analyze = sub.add_parser("analyze", help="Analyze a model's I/O profile")
    p_analyze.add_argument("model", nargs="?", default="llama-3-70b",
                           help="Model name (e.g., llama-3-70b) or used with --config")
    p_analyze.add_argument("--config", type=str, help="Path to HuggingFace config.json")
    p_analyze.add_argument("--gpu", default="H100", help="GPU profile (default: H100)")
    p_analyze.add_argument("--ctx", type=int, default=4096, help="Context length")
    p_analyze.add_argument("--batch", type=int, default=1, help="Batch size")
    p_analyze.add_argument("--wbits", type=int, default=16, help="Weight precision bits")
    p_analyze.add_argument("--kvbits", type=int, default=16, help="KV cache precision bits")
    p_analyze.add_argument("--json", type=str, help="Save results as JSON")

    # ── crossover ──
    p_cross = sub.add_parser("crossover", help="Plot W↔K crossover analysis")
    p_cross.add_argument("model", nargs="?", default="llama-3-70b")
    p_cross.add_argument("--config", type=str, help="Path to HuggingFace config.json")
    p_cross.add_argument("--gpu", default="H100")
    p_cross.add_argument("--wbits", type=int, default=16)
    p_cross.add_argument("--kvbits", type=int, default=16)
    p_cross.add_argument("--max-batch", type=int, default=256)
    p_cross.add_argument("--save", type=str, help="Save figure to path")

    # ── compare ──
    p_compare = sub.add_parser("compare", help="Compare multiple models")
    p_compare.add_argument("models", nargs="+", help="Model names to compare")
    p_compare.add_argument("--gpu", default="H100")
    p_compare.add_argument("--ctx", type=int, default=4096)
    p_compare.add_argument("--batch", type=int, default=1)
    p_compare.add_argument("--wbits", type=int, default=16)
    p_compare.add_argument("--kvbits", type=int, default=16)
    p_compare.add_argument("--save", type=str, help="Save comparison chart")

    # ── figures ──
    p_fig = sub.add_parser("figures", help="Generate all paper figures")
    p_fig.add_argument("--outdir", default="figures", help="Output directory")

    args = parser.parse_args()

    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "crossover":
        cmd_crossover(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "figures":
        cmd_figures(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
