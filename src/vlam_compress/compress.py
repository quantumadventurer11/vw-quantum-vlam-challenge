"""MPS/MPO tensor-network compression of transformer weight matrices.

Uses quimb for TN bookkeeping and torch.linalg.svd (GPU-accelerated) for
the actual SVD decompositions. Follows the CompactifAI [arXiv:2401.14109]
methodology adapted for OpenVLA-7B (Vicuna-v1.5 LLM backbone).
"""
# Implementation added in Phase 2.
