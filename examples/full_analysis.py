#!/usr/bin/env python3
"""
Example: Full I/O bottleneck analysis for a research report.

This script demonstrates the complete workflow — analyzing a model,
finding its crossover point, and generating all key visualizations.
"""

from llm_io_survey import (
    profile_model,
    from_hf_config,
    get_gpu,
    plot_roofline,
    plot_crossover,
    plot_kv_scaling,
    plot_waterfall,
    plot_tpot_heatmap,
    plot_model_comparison,
    plot_hw_scaling,
)
from pathlib import Path

OUT = Path("example_output")
OUT.mkdir(exist_ok=True)


def main():
    # ── 1. Basic model analysis ──────────────────────────────────
    print("=" * 60)
    print("  LLM I/O Bottleneck Analysis Example")
    print("=" * 60)

    model = profile_model("llama-3-70b")
    gpu = get_gpu("H100")

    # Compare FP16 vs INT4 at different batch sizes
    for wbits in [16, 4]:
        for B in [1, 32, 128]:
            s = model.summary(gpu, seq_len=4096, batch_size=B,
                              weight_bits=wbits, kv_bits=16)
            print(f"  W{wbits:>2d} B={B:<4d} │ W={s['weight_io_per_step_gb']:>6.1f}GB "
                  f"K={s['kv_io_per_step_gb']:>6.1f}GB "
                  f"│ TPOT={s['tpot_ms']:>6.1f}ms │ {s['dominant_flow']}")

    # ── 2. Find the exact crossover for different configs ────────
    print("\n─── Crossover Analysis ────────────────────────────────")
    for ctx in [512, 2048, 4096, 16384, 65536]:
        b_cross = model.crossover_batch_size(ctx, weight_bits=16, kv_bits=16)
        b_max = model.max_batch_size(gpu, ctx, num_gpus=2)
        ctx_label = f"{ctx//1024}K" if ctx >= 1024 else str(ctx)
        print(f"  ctx={ctx_label:>5s} │ W↔K crossover at B={b_cross:>6.1f} "
              f"│ Max B (2×H100)={b_max:>4d}")

    # ── 3. How quantization helps: TPOT across settings ──────────
    print("\n─── Quantization Impact (B=1, ctx=4096) ───────────────")
    for wb in [16, 8, 4]:
        for kvb in [16, 8, 4]:
            tpot = model.tpot_ms(4096, 1, gpu, wb, kvb)
            dom = model.dominant_flow(4096, 1, wb, kvb)
            print(f"  W{wb:>2d}/KV{kvb:>2d} │ TPOT={tpot:>6.1f}ms │ {dom}")

    # ── 4. Generate all visualizations ───────────────────────────
    print(f"\n─── Generating Figures → {OUT}/ ────────────────────────")

    plot_roofline("H100", save_path=str(OUT / "roofline.png"))
    print("  ✓ roofline.png")

    plot_crossover("llama-3-70b", "H100", save_path=str(OUT / "crossover.png"))
    print("  ✓ crossover.png")

    plot_kv_scaling(save_path=str(OUT / "kv_scaling.png"))
    print("  ✓ kv_scaling.png")

    plot_waterfall("llama-3-70b", "H100", batch_size=32,
                   save_path=str(OUT / "waterfall.png"))
    print("  ✓ waterfall.png")

    plot_hw_scaling(save_path=str(OUT / "hw_scaling.png"))
    print("  ✓ hw_scaling.png")

    plot_tpot_heatmap("llama-3-70b", "H100",
                      save_path=str(OUT / "tpot_heatmap.png"))
    print("  ✓ tpot_heatmap.png")

    plot_model_comparison(save_path=str(OUT / "model_comparison.png"))
    print("  ✓ model_comparison.png")

    print(f"\n✓ Done! All output in {OUT}/\n")


if __name__ == "__main__":
    main()
