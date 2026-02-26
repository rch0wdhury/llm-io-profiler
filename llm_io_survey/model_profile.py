"""
Model I/O profiling: parse model configs and compute per-token data movement.

Implements Equations 5-7 from the survey paper:
  W = L * (4*d^2 + alpha*d*d_ff) * b_w/8          (weight I/O, Eq. 5)
  K = L * 2 * B * s * n_kv * d_h * b_kv/8          (KV cache I/O, Eq. 6)
  A = L * O(B*n_h*N^2 + B*N*d_ff) * b_a/8          (activation I/O, Eq. 7)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .hardware import GPUProfile, get_gpu


# ── Model Configuration ──────────────────────────────────────────────

@dataclass
class ModelProfile:
    """Parsed model architecture with I/O-relevant parameters."""
    name: str
    num_layers: int               # L
    hidden_size: int              # d_model
    num_attention_heads: int      # n_h
    num_kv_heads: int             # n_kv (for GQA/MQA; equals n_h for MHA)
    intermediate_size: int        # d_ff
    vocab_size: int = 32000
    head_dim: Optional[int] = None  # d_h (inferred if not set)
    num_experts: int = 1          # total experts (1 = dense)
    num_experts_per_tok: int = 1  # active experts per token
    gated_ffn: bool = True        # SwiGLU-style gated FFN (alpha=3)
    total_params_b: Optional[float] = None  # override total params in billions
    # MLA-specific (DeepSeek-V2/V3 style)
    mla_latent_dim: Optional[int] = None  # d_c + d_R if MLA is used
    tie_word_embeddings: bool = False      # if True, input/output embeddings share weights

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 1

    @property
    def is_mla(self) -> bool:
        return self.mla_latent_dim is not None

    @property
    def attention_type(self) -> str:
        if self.is_mla:
            return "MLA"
        if self.num_kv_heads == 1:
            return "MQA"
        if self.num_kv_heads == self.num_attention_heads:
            return "MHA"
        return "GQA"

    @property
    def alpha(self) -> int:
        """FFN weight multiplier: 3 for gated (SwiGLU), 2 for standard."""
        return 3 if self.gated_ffn else 2

    @property
    def params_per_layer_attn(self) -> int:
        """Attention projection parameters per layer (Q, K, V, O)."""
        d = self.hidden_size
        qo_dim = self.num_attention_heads * self.head_dim  # may differ from d
        kv_dim = self.num_kv_heads * self.head_dim
        # Q: d → qo_dim, O: qo_dim → d, K: d → kv_dim, V: d → kv_dim
        return d * qo_dim + qo_dim * d + 2 * d * kv_dim

    @property
    def params_per_layer_ffn(self) -> int:
        """FFN parameters per layer (single expert)."""
        return self.alpha * self.hidden_size * self.intermediate_size

    @property
    def params_per_layer(self) -> int:
        """Total params per layer including all experts."""
        ffn = self.params_per_layer_ffn
        if self.is_moe:
            ffn = ffn * self.num_experts  # all expert weights exist
        return self.params_per_layer_attn + ffn

    @property
    def total_params(self) -> float:
        """Total parameter count (storage footprint)."""
        if self.total_params_b is not None:
            return self.total_params_b * 1e9
        # Approximate: layers + embeddings
        layer_params = self.num_layers * self.params_per_layer
        # With tied embeddings: single copy. Otherwise: input + output.
        if self.tie_word_embeddings:
            embed_params = self.vocab_size * self.hidden_size
        else:
            embed_params = 2 * self.vocab_size * self.hidden_size
        return layer_params + embed_params

    @property
    def active_params_per_token(self) -> float:
        """
        Parameters actually loaded per decode step (MoE-aware).
        For I/O: input embedding lookup is a single-row gather (negligible),
        but lm_head (output projection) is a full GEMV. So we count 1 copy
        of vocab*d regardless of tying.
        """
        attn = self.params_per_layer_attn
        ffn = self.params_per_layer_ffn * self.num_experts_per_tok
        layer_active = attn + ffn
        # Only lm_head is loaded as a full matrix during decode
        lm_head_params = self.vocab_size * self.hidden_size
        return self.num_layers * layer_active + lm_head_params

    # ── I/O Equations from the paper ─────────────────────────────

    def weight_io_bytes(self, weight_bits: int = 16) -> float:
        """
        Equation 5: W = L * (4*d^2 + alpha*d*d_ff) * b_w/8
        Returns total bytes loaded from HBM per forward pass.
        For MoE: only active expert weights are loaded per token.

        Note: This uses ``active_params_per_token`` which excludes the input
        embedding (a single-row gather, negligible I/O) but includes lm_head.
        The result is slightly smaller than ``weight_footprint_bytes`` — e.g.,
        ~139 GB vs ~141 GB for Llama-3 70B FP16.  Table 7 in the paper uses
        the full footprint as an approximation.
        """
        return self.active_params_per_token * (weight_bits / 8)

    def weight_footprint_bytes(self, weight_bits: int = 16) -> float:
        """Total weight storage (all params, including inactive MoE experts)."""
        return self.total_params * (weight_bits / 8)

    def kv_cache_per_token_per_layer_bytes(self, kv_bits: int = 16) -> float:
        """
        KV bytes stored per token per layer.
        MLA: uses mla_latent_dim instead of 2*n_kv*d_h.
        Standard: 2 * n_kv * d_h * (b_kv/8)
        """
        if self.is_mla:
            return self.mla_latent_dim * (kv_bits / 8)
        return 2 * self.num_kv_heads * self.head_dim * (kv_bits / 8)

    def kv_cache_bytes(self, seq_len: int, batch_size: int = 1,
                       kv_bits: int = 16) -> float:
        """
        Equation 6: K = L * 2 * B * s * n_kv * d_h * b_kv/8
        Total KV cache size for given batch and sequence length.
        """
        per_tok_per_layer = self.kv_cache_per_token_per_layer_bytes(kv_bits)
        return self.num_layers * seq_len * batch_size * per_tok_per_layer

    def kv_io_per_step_bytes(self, seq_len: int, batch_size: int = 1,
                             kv_bits: int = 16) -> float:
        """KV cache bytes read per decode step (= full KV cache size)."""
        return self.kv_cache_bytes(seq_len, batch_size, kv_bits)

    def activation_io_prefill_bytes(self, seq_len: int, batch_size: int = 1,
                                    act_bits: int = 16,
                                    flash_attention: bool = True) -> float:
        """
        Equation 7: A_prefill = L * O(B*n_h*N^2 + B*N*d_ff) * b_a/8
        With FlashAttention, the N^2 term is eliminated.
        """
        bytes_per_elem = act_bits / 8
        ffn_term = batch_size * seq_len * self.intermediate_size * bytes_per_elem
        if flash_attention:
            attn_term = 0  # kept in SRAM
        else:
            attn_term = (batch_size * self.num_attention_heads
                         * seq_len * seq_len * bytes_per_elem)
        return self.num_layers * (attn_term + ffn_term)

    def total_decode_io_bytes(self, seq_len: int, batch_size: int = 1,
                              weight_bits: int = 16, kv_bits: int = 16) -> float:
        """Total bytes moved per decode step: W + K."""
        w = self.weight_io_bytes(weight_bits)
        k = self.kv_io_per_step_bytes(seq_len, batch_size, kv_bits)
        return w + k

    # ── Derived metrics ──────────────────────────────────────────

    def arithmetic_intensity_decode(self, batch_size: int = 1,
                                    weight_bits: int = 16) -> float:
        """
        I_decode ≈ B (FLOP/byte) for weight-dominated regime.
        More precisely: 2*B*active_params / (active_params * b_w/8)
        """
        flops = 2 * batch_size * self.active_params_per_token
        data_bytes = self.weight_io_bytes(weight_bits)
        return flops / data_bytes if data_bytes > 0 else float("inf")

    def tpot_ms(self, seq_len: int, batch_size: int = 1,
                gpu: GPUProfile | str = "H100",
                weight_bits: int = 16, kv_bits: int = 16) -> float:
        """
        Equation 9: TPOT ≈ (W + K(s)) / β_eff
        Returns estimated time-per-output-token in milliseconds.
        """
        if isinstance(gpu, str):
            gpu = get_gpu(gpu)
        total_bytes = self.total_decode_io_bytes(seq_len, batch_size,
                                                  weight_bits, kv_bits)
        bw_bytes_per_sec = gpu.hbm_bandwidth_sustained_tbps * 1e12
        return (total_bytes / bw_bytes_per_sec) * 1000

    def crossover_batch_size(self, seq_len: int,
                             weight_bits: int = 16,
                             kv_bits: int = 16) -> float:
        """
        Batch size B where KV I/O equals weight I/O: B*K_seq ≈ W.
        Returns float (fractional batch size at crossover).
        """
        w = self.weight_io_bytes(weight_bits)
        k_per_seq = self.kv_cache_bytes(seq_len, batch_size=1, kv_bits=kv_bits)
        if k_per_seq == 0:
            return float("inf")
        return w / k_per_seq

    def max_batch_size(self, gpu: GPUProfile | str = "H100",
                       seq_len: int = 4096, num_gpus: int = 1,
                       weight_bits: int = 16, kv_bits: int = 16) -> int:
        """
        Maximum batch size before HBM is exhausted:
        B_max = (HBM_total - W_footprint) / K_per_seq
        """
        if isinstance(gpu, str):
            gpu = get_gpu(gpu)
        total_hbm = gpu.hbm_capacity_gb * 1e9 * num_gpus
        w_footprint = self.weight_footprint_bytes(weight_bits)
        k_per_seq = self.kv_cache_bytes(seq_len, batch_size=1, kv_bits=kv_bits)
        if k_per_seq == 0:
            return 9999
        available = total_hbm - w_footprint
        if available <= 0:
            return 0
        return int(available / k_per_seq)

    def dominant_flow(self, seq_len: int, batch_size: int = 1,
                      weight_bits: int = 16, kv_bits: int = 16) -> str:
        """Identify which I/O flow dominates at given operating point."""
        w = self.weight_io_bytes(weight_bits)
        k = self.kv_io_per_step_bytes(seq_len, batch_size, kv_bits)
        if w > k:
            return "W (weight-bound)"
        return "K (KV-cache-bound)"

    def summary(self, gpu: GPUProfile | str = "H100",
                seq_len: int = 4096, batch_size: int = 1,
                weight_bits: int = 16, kv_bits: int = 16) -> dict:
        """Generate a full I/O summary dictionary."""
        if isinstance(gpu, str):
            gpu = get_gpu(gpu)
        w = self.weight_io_bytes(weight_bits)
        k = self.kv_io_per_step_bytes(seq_len, batch_size, kv_bits)
        return {
            "model": self.name,
            "gpu": gpu.name,
            "attention_type": self.attention_type,
            "is_moe": self.is_moe,
            "total_params_B": self.total_params / 1e9,
            "active_params_B": self.active_params_per_token / 1e9,
            "weight_footprint_gb": self.weight_footprint_bytes(weight_bits) / 1e9,
            "weight_io_per_step_gb": w / 1e9,
            "kv_per_token_per_layer_bytes": self.kv_cache_per_token_per_layer_bytes(kv_bits),
            "kv_cache_total_gb": self.kv_cache_bytes(seq_len, batch_size, kv_bits) / 1e9,
            "kv_io_per_step_gb": k / 1e9,
            "total_decode_io_gb": (w + k) / 1e9,
            "dominant_flow": self.dominant_flow(seq_len, batch_size, weight_bits, kv_bits),
            "arithmetic_intensity": self.arithmetic_intensity_decode(batch_size, weight_bits),
            "ridge_point": gpu.ridge_point,
            "tpot_ms": self.tpot_ms(seq_len, batch_size, gpu, weight_bits, kv_bits),
            "crossover_batch_size": self.crossover_batch_size(seq_len, weight_bits, kv_bits),
            "max_batch_size": self.max_batch_size(gpu, seq_len, 1, weight_bits, kv_bits),
            "batch_size": batch_size,
            "seq_len": seq_len,
            "weight_bits": weight_bits,
            "kv_bits": kv_bits,
        }


# ── Parsing HuggingFace config.json ──────────────────────────────────

def from_hf_config(config_path: str | Path, name: str | None = None) -> ModelProfile:
    """
    Parse a HuggingFace model config.json into a ModelProfile.

    Supports: LlamaForCausalLM, MistralForCausalLM, MixtralForCausalLM,
    Qwen2ForCausalLM, Phi3ForCausalLM, GemmaForCausalLM, GPT2LMHeadModel,
    and generic transformer configs.
    """
    path = Path(config_path)
    with open(path) as f:
        cfg = json.load(f)

    model_type = cfg.get("model_type", "unknown")
    arch = cfg.get("architectures", ["unknown"])[0] if cfg.get("architectures") else "unknown"
    inferred_name = name or cfg.get("_name_or_path", model_type)

    # Common fields with sensible fallbacks
    L = cfg.get("num_hidden_layers", cfg.get("n_layer", 32))
    d = cfg.get("hidden_size", cfg.get("n_embd", 4096))
    n_h = cfg.get("num_attention_heads", cfg.get("n_head", 32))
    n_kv = cfg.get("num_key_value_heads", n_h)  # defaults to MHA
    d_ff = cfg.get("intermediate_size", cfg.get("n_inner", 4 * d))
    vocab = cfg.get("vocab_size", 32000)
    d_h = cfg.get("head_dim", None)

    # Detect gated FFN (SwiGLU)
    hidden_act = cfg.get("hidden_act", "silu")
    gated = hidden_act in ("silu", "swiglu") or "gate_proj" in str(cfg)

    # MoE detection
    num_experts = cfg.get("num_local_experts", cfg.get("num_experts", 1))
    experts_per_tok = cfg.get("num_experts_per_tok",
                               cfg.get("num_selected_experts", 1))

    # MLA detection (DeepSeek-V2/V3 style)
    mla_dim = None
    if "kv_lora_rank" in cfg:
        # DeepSeek-style MLA
        kv_lora = cfg.get("kv_lora_rank", 512)
        rope_head = cfg.get("qk_rope_head_dim", 64)
        mla_dim = kv_lora + rope_head

    # Tied embeddings detection
    tie_embeds = cfg.get("tie_word_embeddings", False)

    return ModelProfile(
        name=inferred_name,
        num_layers=L,
        hidden_size=d,
        num_attention_heads=n_h,
        num_kv_heads=n_kv,
        intermediate_size=d_ff,
        vocab_size=vocab,
        head_dim=d_h,
        num_experts=num_experts,
        num_experts_per_tok=experts_per_tok,
        gated_ffn=gated,
        mla_latent_dim=mla_dim,
        tie_word_embeddings=tie_embeds,
    )


# ── Pre-defined model profiles (from Table 2 of the paper) ──────────

KNOWN_MODELS: dict[str, ModelProfile] = {
    "llama-3.2-1b": ModelProfile("Llama-3.2 1B", 16, 2048, 32, 8, 8192, 128256, gated_ffn=True, tie_word_embeddings=True),
    "llama-3.2-3b": ModelProfile("Llama-3.2 3B", 28, 3072, 24, 8, 8192, 128256, gated_ffn=True, tie_word_embeddings=True),
    "llama-3-8b": ModelProfile("Llama-3 8B", 32, 4096, 32, 8, 14336, 128256, gated_ffn=True),
    "llama-3-70b": ModelProfile("Llama-3 70B", 80, 8192, 64, 8, 28672, 128256, gated_ffn=True),
    "llama-3-405b": ModelProfile("Llama-3 405B", 126, 16384, 128, 8, 53248, 128256, gated_ffn=True),
    "llama-2-7b": ModelProfile("Llama-2 7B", 32, 4096, 32, 32, 11008, 32000, gated_ffn=True),
    "llama-2-70b": ModelProfile("Llama-2 70B", 80, 8192, 64, 8, 28672, 32000, gated_ffn=True),
    "mistral-7b": ModelProfile("Mistral 7B", 32, 4096, 32, 8, 14336, 32768, gated_ffn=True),
    "phi-3-mini": ModelProfile("Phi-3 Mini 3.8B", 32, 3072, 32, 32, 8192, 32064, gated_ffn=True),
    "gemma-2-9b": ModelProfile("Gemma-2 9B", 42, 3584, 16, 8, 14336, 256000, head_dim=256, gated_ffn=True, tie_word_embeddings=True),
    "gemma-2-27b": ModelProfile("Gemma-2 27B", 46, 4608, 32, 16, 36864, 256000, head_dim=128, gated_ffn=True, tie_word_embeddings=True),
    "qwen-2.5-72b": ModelProfile("Qwen-2.5 72B", 80, 8192, 64, 8, 29568, 152064, gated_ffn=True),
    "mixtral-8x7b": ModelProfile(
        "Mixtral 8x7B", 32, 4096, 32, 8, 14336, 32000,
        gated_ffn=True, num_experts=8, num_experts_per_tok=2,
    ),
    "mixtral-8x22b": ModelProfile(
        "Mixtral 8x22B", 56, 6144, 48, 8, 16384, 32000,
        gated_ffn=True, num_experts=8, num_experts_per_tok=2,
    ),
    "deepseek-v3": ModelProfile(
        "DeepSeek-V3 671B", 61, 7168, 128, 128, 18432, 129280,
        gated_ffn=True, num_experts=257, num_experts_per_tok=9,
        mla_latent_dim=576,
        total_params_b=671,
    ),
    "gpt-3-175b": ModelProfile("GPT-3 175B", 96, 12288, 96, 96, 49152, 50257, gated_ffn=False),
}


def get_model(name: str) -> ModelProfile:
    """Look up a known model profile by name (case-insensitive, partial match)."""
    key = name.lower().replace(" ", "-")
    if key in KNOWN_MODELS:
        return KNOWN_MODELS[key]
    for k, v in KNOWN_MODELS.items():
        if key in k or key in v.name.lower().replace(" ", "-"):
            return v
    available = ", ".join(KNOWN_MODELS.keys())
    raise KeyError(f"Unknown model '{name}'. Available: {available}\n"
                   f"Or provide a HuggingFace config.json path.")
