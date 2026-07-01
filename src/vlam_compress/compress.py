"""MPS/MPO tensor-network compression of transformer weight matrices.

Uses torch.linalg.svd (GPU-accelerated via cuSOLVER) for SVD decompositions
and quimb for TN structure bookkeeping and validation.
Follows the CompactifAI [arXiv:2401.14109] methodology adapted for OpenVLA-7B
(Vicuna-v1.5 / LLaMA-2 backbone).

Public API
----------
choose_reshape_dims(m, n, n_sites) -> tuple
mps_decompose(W, bond_dim, n_sites)  -> (cores, shape_info)
mps_reconstruct(cores, output_shape) -> Tensor
count_core_params(cores)             -> int
frobenius_error(W, W_hat)            -> float
find_compression_targets(model, ...)  -> dict[str, nn.Module]
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# Linear-layer suffixes targeted in the LLM backbone (Vicuna-v1.5 / LLaMA-2).
DEFAULT_TARGET_SUFFIXES: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)

# Skip layers below this parameter count (e.g. small bias-only or scalar layers).
MIN_LAYER_PARAMS: int = 1_000_000


# ── Reshape utilities ─────────────────────────────────────────────────────────

def _best_divisor_near(n: int, target: float) -> int:
    """Return the divisor of n whose distance to *target* is smallest."""
    best, best_dist = 1, abs(target - 1)
    for d in range(1, int(math.isqrt(n)) + 1):
        if n % d == 0:
            for candidate in (d, n // d):
                dist = abs(target - candidate)
                if dist < best_dist:
                    best, best_dist = candidate, dist
    return best


def _factorize_into(x: int, k: int) -> tuple[int, ...]:
    """
    Split integer x into k factors whose product equals x,
    choosing them as equal as possible (geometric-mean heuristic).
    """
    dims: list[int] = []
    rem = x
    for i in range(k, 1, -1):
        target = rem ** (1.0 / i)
        d = _best_divisor_near(rem, target)
        dims.append(d)
        rem //= d
    dims.append(rem)
    return tuple(dims)


def choose_reshape_dims(m: int, n: int, n_sites: int = 4) -> tuple[int, ...]:
    """
    Factorize a weight matrix shape (m, n) into *n_sites* approximately-equal
    dimensions for MPS decomposition.

    Row dimensions and column dimensions are factored separately, preserving
    the physical interpretation of the tensor network (row sites represent the
    output space, column sites the input space).

    Returns a tuple of length n_sites with product == m * n.

    Examples
    --------
    >>> choose_reshape_dims(4096, 4096, 4)
    (64, 64, 64, 64)          # 64^4 = 4096^2  ✓
    >>> choose_reshape_dims(11008, 4096, 4)
    (86, 128, 64, 64)         # 86*128=11008, 64*64=4096  ✓
    """
    half = n_sites // 2
    row_dims = _factorize_into(m, half)
    col_dims = _factorize_into(n, n_sites - half)
    return row_dims + col_dims


# ── Core MPS decomposition ────────────────────────────────────────────────────

def mps_decompose(
    W: torch.Tensor,
    bond_dim: int,
    n_sites: int = 4,
) -> tuple[list[torch.Tensor], dict]:
    """
    Decompose a 2-D weight matrix W ∈ R^(m × n) into an MPS (Matrix Product
    State) with at most *bond_dim* virtual indices per bond.

    Algorithm: standard left-to-right SVD sweep on the reshaped tensor, with
    singular-value truncation at each bond.  SVD is computed by
    ``torch.linalg.svd`` which dispatches to cuSOLVER on CUDA tensors.

    Parameters
    ----------
    W        : (m, n) tensor — the weight matrix to compress.
    bond_dim : maximum number of singular values kept per bond (χ).
    n_sites  : number of MPS sites (default 4).

    Returns
    -------
    cores      : list of k tensors with shapes
                 [1, d₀, r₀], [r₀, d₁, r₁], …, [r_{k-2}, d_{k-1}, 1]
    shape_info : dict with metadata needed for reconstruction.
    """
    m, n = W.shape
    orig_dtype = W.dtype
    reshape_dims = choose_reshape_dims(m, n, n_sites)

    # Work in float32 for numerical stability; results cast back at the end.
    T = W.detach().float().reshape(reshape_dims)

    cores: list[torch.Tensor] = []
    left_bond = 1
    # Fold left: shape is [left_bond * d_site, remaining]
    current = T.reshape(reshape_dims[0], -1)   # [d0, d1·d2·…·dk-1]

    for site in range(n_sites - 1):
        d_phys = reshape_dims[site]

        U, S, Vh = torch.linalg.svd(current, full_matrices=False)

        r = min(bond_dim, S.shape[0])
        U, S, Vh = U[:, :r], S[:r], Vh[:r, :]

        cores.append(U.reshape(left_bond, d_phys, r))

        # Absorb singular values into the right factor.
        remaining = S.unsqueeze(1) * Vh          # [r, remaining_elements]

        if site < n_sites - 2:
            d_next = reshape_dims[site + 1]
            current = remaining.reshape(r * d_next, -1)
        else:
            # Last core: [r, d_{k-1}, 1]
            cores.append(remaining.reshape(r, reshape_dims[-1], 1))

        left_bond = r

    actual_ranks = [int(c.shape[2]) for c in cores[:-1]]
    shape_info = {
        "original_shape": [m, n],
        "reshape_dims": list(reshape_dims),
        "bond_dim": bond_dim,
        "actual_ranks": actual_ranks,
    }

    # Cast cores back to the original dtype.
    cores = [c.to(orig_dtype) for c in cores]
    return cores, shape_info


def mps_reconstruct(
    cores: list[torch.Tensor],
    output_shape: tuple[int, int],
) -> torch.Tensor:
    """
    Contract MPS cores from left to right to reconstruct the full matrix.

    Parameters
    ----------
    cores        : list of k tensors (shapes as returned by ``mps_decompose``).
    output_shape : (m, n) target shape for the reconstructed matrix.

    Returns
    -------
    W_hat : (m, n) tensor — the reconstructed (approximate) weight matrix.
    """
    # cores[0]: [1, d0, r0]  →  squeeze left boundary  →  [d0, r0]
    result = cores[0].squeeze(0)

    # Middle cores: [r_{i-1}, d_i, r_i]
    for core in cores[1:-1]:
        result = torch.tensordot(result, core, dims=([-1], [0]))
        # After: result has shape [d0, …, d_i, r_i]

    # Last core: [r_{k-2}, d_{k-1}, 1]  →  squeeze right boundary  →  [r_{k-2}, d_{k-1}]
    result = torch.tensordot(result, cores[-1].squeeze(-1), dims=([-1], [0]))
    # result: [d0, d1, …, d_{k-1}]

    return result.reshape(output_shape)


# ── Metrics ───────────────────────────────────────────────────────────────────

def count_core_params(cores: list[torch.Tensor]) -> int:
    """Total number of parameters stored in the MPS cores."""
    return sum(int(c.numel()) for c in cores)


def frobenius_error(W_orig: torch.Tensor, W_hat: torch.Tensor) -> float:
    """
    Relative Frobenius-norm reconstruction error:
        ||W - W_hat||_F / ||W||_F
    Computed in float32 for precision.
    """
    diff = torch.linalg.norm((W_orig.float() - W_hat.float()))
    orig = torch.linalg.norm(W_orig.float())
    return float(diff / (orig + 1e-12))


def compression_ratio(n_orig: int, n_cores: int) -> float:
    """Ratio of original parameter count to compressed (core) parameter count."""
    return n_orig / max(n_cores, 1)


# ── Model-level utilities ─────────────────────────────────────────────────────

def find_compression_targets(
    model: nn.Module,
    target_suffixes: tuple[str, ...] = DEFAULT_TARGET_SUFFIXES,
    min_params: int = MIN_LAYER_PARAMS,
) -> dict[str, nn.Module]:
    """
    Walk the model and return a dict of {full_name: module} for every
    ``nn.Linear`` whose name ends with one of *target_suffixes* and whose
    weight has at least *min_params* parameters.
    """
    targets: dict[str, nn.Module] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(sfx) for sfx in target_suffixes):
            continue
        if module.weight.numel() < min_params:
            continue
        targets[name] = module
    return targets
