"""
llm-io-survey: I/O Bottleneck Analysis Toolkit for LLM Inference.

Companion code for "I/O for LLM Inference: A Survey of Storage and Memory Bottlenecks".

Quick start:
    from llm_io_survey import profile_model, plot_crossover
    p = profile_model("llama-3-70b")
    print(p.summary())
    plot_crossover("llama-3-70b")
"""

__version__ = "0.2.0"

from .model_profile import ModelProfile, from_hf_config, get_model, KNOWN_MODELS
from .hardware import GPUProfile, get_gpu, GPU_PROFILES
from .visualize import (
    plot_roofline,
    plot_crossover,
    plot_kv_scaling,
    plot_waterfall,
    plot_hw_scaling,
    plot_tpot_heatmap,
    plot_model_comparison,
)

# Convenience alias
profile_model = get_model
