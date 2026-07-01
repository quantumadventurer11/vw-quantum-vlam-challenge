"""Metric computation for §5.5 benchmarks.

Shared utilities for Phase 3 evaluation. No heavy ML imports — numpy only.
All aggregation, derived-metric, and reporting helpers live here so the
notebook stays focused on orchestration.
"""

from __future__ import annotations

import numpy as np
from typing import Any

# US average grid carbon intensity (EPA 2023).
US_GRID_CO2E_KG_PER_KWH: float = 0.386


def aggregate(values: list[float]) -> dict[str, Any]:
    """Compute mean ± std for a list of per-run scalar values."""
    arr = np.asarray(values, dtype=float)
    return {
        "mean":   float(arr.mean()),
        "std":    float(arr.std()),
        "n_runs": int(len(arr)),
        "values": [float(v) for v in arr],
    }


def aggregate_runs(run_results: list[dict]) -> dict[str, dict]:
    """
    Given a list of per-seed result dicts (as returned by the evaluation
    loop), return a dict of {metric_key: aggregate(…)} across all seeds.
    """
    keys = [
        "l1_error_mean",
        "inference_time_ms_mean",
        "peak_mem_mib_mean",
        "avg_power_w",
        "total_kwh",
        "co2e_g",
    ]
    return {k: aggregate([r[k] for r in run_results]) for k in keys}


def efficiency_gain_pct(t_baseline_ms: float, t_compressed_ms: float) -> float:
    """
    Wall-clock efficiency gain:
        (t_baseline - t_compressed) / t_baseline × 100 %
    Positive = compressed model is faster.
    """
    return (t_baseline_ms - t_compressed_ms) / max(t_baseline_ms, 1e-9) * 100.0


def param_efficiency_gain_pct(n_params_baseline: int, n_params_core: int) -> float:
    """
    Parameter-count proxy for efficiency gain (FLOPs proxy):
        (n_base - n_core) / n_base × 100 %
    """
    return (n_params_baseline - n_params_core) / max(n_params_baseline, 1) * 100.0


def model_compression_ratio(
    n_total: int,
    n_nontarget: int,
    n_core: int,
) -> float:
    """
    Whole-model compression ratio when compressed layers are stored as MPS cores:
        n_total / (n_nontarget + n_core)
    """
    return n_total / max(n_nontarget + n_core, 1)


def co2e_grams(kwh: float) -> float:
    """Convert kWh energy consumption to grams of CO2-equivalent."""
    return kwh * US_GRID_CO2E_KG_PER_KWH * 1000.0


def delta_dict(cond_a: dict, cond_b: dict, key: str = "mean") -> dict:
    """
    Compute absolute and relative change B - A for a given aggregate metric.
    Useful for the ablation table (e.g. ΔL1 error from adding TN compression).
    """
    a_val = cond_a[key]
    b_val = cond_b[key]
    return {
        "absolute": b_val - a_val,
        "relative_pct": (b_val - a_val) / max(abs(a_val), 1e-12) * 100.0,
    }


def flag_benchmark(label: str, value: float, threshold: float, better: str = "lower") -> str:
    """
    Return a one-line status string flagging whether a benchmark target is met.
    *better* = 'lower' | 'higher'
    """
    if better == "lower":
        ok = value <= threshold
    else:
        ok = value >= threshold
    status = "[OK  ]" if ok else "[WARN]"
    return f"{status}  {label}: {value:.4g}  (target {'≤' if better=='lower' else '≥'} {threshold})"
