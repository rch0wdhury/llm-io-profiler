"""
Hardware specifications for GPU memory hierarchy analysis.

Each GPU profile contains the bandwidth, capacity, and compute specs
needed for roofline modeling and I/O bottleneck analysis.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GPUProfile:
    """Hardware specification for a single GPU or accelerator.

    Note: The ``hbm_*`` fields are used generically for the device's primary
    memory tier, even on non-HBM hardware (e.g., LPDDR5X on Apple UMA,
    GDDR6X on RTX 4090).
    """
    name: str
    # HBM
    hbm_capacity_gb: float
    hbm_bandwidth_tbps: float  # TB/s peak
    # Compute
    fp16_tflops: float  # peak FP16/BF16 TFLOP/s
    int8_tops: Optional[float] = None  # peak INT8 TOP/s
    # On-chip
    sram_mb: float = 50.0
    sram_bandwidth_tbps: float = 12.0
    # Interconnect
    nvlink_bandwidth_gbps: float = 900.0  # GB/s bidirectional
    pcie_bandwidth_gbps: float = 64.0  # GB/s x16
    # Efficiency factor (sustained / peak)
    sustained_bw_ratio: float = 0.80

    @property
    def hbm_bandwidth_sustained_tbps(self) -> float:
        return self.hbm_bandwidth_tbps * self.sustained_bw_ratio

    @property
    def hbm_bandwidth_sustained_gbps(self) -> float:
        return self.hbm_bandwidth_sustained_tbps * 1000

    @property
    def ridge_point(self) -> float:
        """Ridge point I* = peak_compute / peak_bandwidth (FLOP/byte)."""
        compute_flops = self.fp16_tflops * 1e12
        bw_bytes = self.hbm_bandwidth_tbps * 1e12
        return compute_flops / bw_bytes

    @property
    def ridge_point_sustained(self) -> float:
        """Ridge point using sustained bandwidth."""
        compute_flops = self.fp16_tflops * 1e12
        bw_bytes = self.hbm_bandwidth_sustained_tbps * 1e12
        return compute_flops / bw_bytes


# ── Pre-defined GPU profiles ──────────────────────────────────────────

GPU_PROFILES: dict[str, GPUProfile] = {
    "V100": GPUProfile(
        name="NVIDIA V100 SXM2",
        hbm_capacity_gb=32,
        hbm_bandwidth_tbps=0.9,
        fp16_tflops=125,
        sram_mb=6.144,
        sram_bandwidth_tbps=8.0,
        nvlink_bandwidth_gbps=300,
        pcie_bandwidth_gbps=32,
    ),
    "A100": GPUProfile(
        name="NVIDIA A100 SXM",
        hbm_capacity_gb=80,
        hbm_bandwidth_tbps=2.0,
        fp16_tflops=312,
        int8_tops=624,
        sram_mb=40,
        sram_bandwidth_tbps=19.5,
        nvlink_bandwidth_gbps=600,
        pcie_bandwidth_gbps=64,
    ),
    "H100": GPUProfile(
        name="NVIDIA H100 SXM",
        hbm_capacity_gb=80,
        hbm_bandwidth_tbps=3.35,
        fp16_tflops=990,
        int8_tops=1980,
        sram_mb=50,
        sram_bandwidth_tbps=12.0,
        nvlink_bandwidth_gbps=900,
        pcie_bandwidth_gbps=64,
    ),
    "B200": GPUProfile(
        name="NVIDIA B200",
        hbm_capacity_gb=192,
        hbm_bandwidth_tbps=4.8,
        fp16_tflops=2250,
        int8_tops=4500,
        sram_mb=64,
        sram_bandwidth_tbps=15.0,
        nvlink_bandwidth_gbps=1800,
        pcie_bandwidth_gbps=64,
    ),
    "A770": GPUProfile(
        name="Intel Arc A770 (16GB)",
        hbm_capacity_gb=16,
        hbm_bandwidth_tbps=0.56,
        fp16_tflops=35,
        sram_mb=2,
        sram_bandwidth_tbps=2.0,
        nvlink_bandwidth_gbps=0,
        pcie_bandwidth_gbps=32,
    ),
    "RTX_4090": GPUProfile(
        name="NVIDIA RTX 4090",
        hbm_capacity_gb=24,
        hbm_bandwidth_tbps=1.008,
        fp16_tflops=330,
        sram_mb=72,
        sram_bandwidth_tbps=10.0,
        nvlink_bandwidth_gbps=0,
        pcie_bandwidth_gbps=32,
    ),
    "M2_Ultra": GPUProfile(
        name="Apple M2 Ultra (192GB)",
        hbm_capacity_gb=192,
        hbm_bandwidth_tbps=0.8,
        fp16_tflops=27,
        sram_mb=48,
        sram_bandwidth_tbps=4.0,
        nvlink_bandwidth_gbps=0,
        pcie_bandwidth_gbps=0,
    ),
    "M4_Max": GPUProfile(
        name="Apple M4 Max (128GB UMA)",
        hbm_capacity_gb=128,
        hbm_bandwidth_tbps=0.546,
        fp16_tflops=34,
        sram_mb=48,
        sram_bandwidth_tbps=4.0,
        nvlink_bandwidth_gbps=0,
        pcie_bandwidth_gbps=0,
    ),
    "M4_Ultra": GPUProfile(
        name="Apple M4 Ultra (192GB UMA)",
        hbm_capacity_gb=192,
        hbm_bandwidth_tbps=0.819,
        fp16_tflops=68,
        sram_mb=96,
        sram_bandwidth_tbps=8.0,
        nvlink_bandwidth_gbps=0,
        pcie_bandwidth_gbps=0,
    ),
}


def get_gpu(name: str) -> GPUProfile:
    """Look up a GPU profile by name (case-insensitive, partial match)."""
    key = name.upper().replace(" ", "_").replace("-", "_")
    if key in GPU_PROFILES:
        return GPU_PROFILES[key]
    # Fuzzy match
    for k, v in GPU_PROFILES.items():
        if key in k or key in v.name.upper().replace(" ", "_"):
            return v
    available = ", ".join(GPU_PROFILES.keys())
    raise KeyError(f"Unknown GPU '{name}'. Available: {available}")
