# Quantum-Inspired Tensor-Network Compression of OpenVLA-7B

**VW Group / Quantum Insider — Global Quantum + AI Challenge 2026**  
Sub-track: Compression | Domain: Robotics

---

## License

**Code**: MIT — see [LICENSE](LICENSE).

> **Model weights**: The OpenVLA-7B weights used in this project are governed by
> the **LLaMA-2 Community License (non-commercial research use only)**. They are
> NOT covered by the MIT license above. You must accept the LLaMA-2 license on
> HuggingFace Hub before downloading: <https://huggingface.co/openvla/openvla-7b>

---

## What This Is

We apply quantum-inspired **Matrix Product State (MPS) / Matrix Product Operator
(MPO) tensor-network decomposition** (via [quimb](https://github.com/jcmgray/quimb))
to compress OpenVLA-7B weight matrices, then benchmark the compressed model against
the INT8 bitsandbytes baseline required by the challenge (§5.4 Robotics Compression).

Key results:
- **Compression vs. Accuracy**: Pareto curve across bond dimensions {16, 32, 64, 128}
- **Efficiency Gain**: Wall-clock inference time reduction vs. INT8 baseline (mean ± std, 3 runs)
- **MuJoCo visualization**: Humanoid figure driven by compressed model outputs
- **PennyLane appendix**: Hardware-feasibility analysis (qubit count, circuit depth)

---

## Requirements

- Python 3.10+
- CUDA-capable GPU with ≥ 24 GB VRAM (tested on single GPU)
- CUDA 11.8 or later

---

## Installation

```bash
git clone <this-repo>
cd vw-quantum-vlam-challenge
pip install -e .[dev]
```

For exact reproducibility (pinned versions from our test environment):

```bash
pip install -r requirements-pinned.txt   # generated after pip freeze — see Phase 7
```

---

## Quick Start

```bash
make baseline    # Phase 1: INT8 baseline metrics
make compress    # Phase 2: TN compression (all bond dims)
make eval        # Phase 3: evaluation + ablation
make visualize   # Phase 4: MuJoCo video
make report      # Phase 6: compile PDF report
make repro       # Full end-to-end reproducibility test
```

Individual scripts:

```bash
python scripts/run_baseline.py
python scripts/compress_model.py --chi 64
python scripts/run_eval.py --chi 64
python scripts/visualize.py --chi 64
```

---

## Hardware & Seeds

Hardware used: see `configs/seeds.yaml` (fill in GPU details before Phase 3 runs).

Random seeds for 3 independent runs: **42, 1337, 2024** (documented in `configs/seeds.yaml`).

---

## Repository Layout

```
src/vlam_compress/      # installable package
scripts/                # entry-point scripts (run_baseline, compress_model, run_eval, visualize)
configs/                # hyperparameters, seeds, hardware declaration
results/                # outputs (metrics JSON, plots, videos)
checkpoints/            # compressed model checkpoints (gitignored — too large)
docs/                   # PROJECT_PLAN.md, ATTRIBUTIONS.md, appendix_pennylane.md
tests/                  # unit tests
```

---

## Attribution

See [docs/ATTRIBUTIONS.md](docs/ATTRIBUTIONS.md) for a full log of all external
repositories, papers, and datasets used.

Key references:
- [CompactifAI](https://arxiv.org/abs/2401.14109) — MPS/MPO methodology [1]
- [OpenVLA](https://github.com/openvla/openvla) — reference model [MIT + LLaMA-2]
- [Open X-Embodiment](https://arxiv.org/abs/2310.08864) — reference dataset [5]
- [quimb](https://github.com/jcmgray/quimb) — tensor network library [MIT]
- [dm_control](https://github.com/google-deepmind/dm_control) — MuJoCo environments [Apache 2.0]
