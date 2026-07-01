# Project Plan: Quantum-Inspired Tensor-Network Compression of OpenVLA-7B

**Challenge**: Global Quantum + AI Challenge 2026 — VW Group Enterprise Track  
**Sub-track**: Compression | **Domain**: Robotics  
**Date written**: 2026-06-30  
**Author**: Benjamin Brumm

---

## 0. Configuration at a Glance

| Parameter | Value |
|---|---|
| Reference model | OpenVLA-7B (`openvla/openvla-7b`) |
| Reference dataset | Open X-Embodiment [5] (Apache 2.0) |
| Accepted baseline | INT8 quantization via bitsandbytes |
| Compression method | MPS/MPO tensor-network decomposition via **quimb** |
| Training framework | PyTorch + HuggingFace Transformers |
| Hardware | Google Colab (free tier or Colab Pro; T4/A100 GPU provisioned per session) |
| Memory strategy | 8-bit baseline load + gradient/activation checkpointing |
| Bonus appendix | PennyLane: hardware-feasibility mapping (qubit count, circuit depth) |

> **Execution environment**: The development machine has no NVIDIA GPU (Intel UHD
> integrated graphics only). All GPU-dependent work (Phases 1–5) runs on **Google
> Colab** (free tier or Colab Pro). Phases 1–5 are delivered as Jupyter notebooks in
> `notebooks/`; each is self-contained with `!pip install` cells at the top because
> Colab sessions start fresh. Notebooks install the repo package via
> `!pip install -e .` so all reusable logic in `src/vlam_compress/` remains the
> single source of truth. `dm_control` / MuJoCo dependency resolution is verified
> fresh inside Colab — no local workaround assumptions carry over.
> Phases 0, 6, and 7 do not require GPU and run locally via Makefile as before.

> **License note**: OpenVLA-7B *code* is MIT. The *model weights* inherit the
> **LLaMA-2 Community License (non-commercial research use only)**. Every
> deliverable that touches the weights — README, report, repo description —
> must state this clearly. We do NOT claim a fully open-source release;
> we claim open-source *code* with research-only weights.

---

## 1. Requirement Coverage Matrix

This table proves every required output (§5.2) and every benchmark (§5.5) is
covered by a specific phase. Nothing is silently dropped.

### §5.2 Required Outputs

| §5.2 Requirement | Covered In | Notes |
|---|---|---|
| Functional QI component integrated into VLAM compression | Phase 2 | MPS/MPO decomposition of weight matrices via quimb |
| Public repo + README for clean-environment replication (deps, seeds, hardware) | Phase 7 | README.md with pip-installable package, pinned deps, documented seeds |
| Quantitative results vs. INT8 baseline, mean ± std across at least 3 runs | Phase 3 | All metrics tabulated for 3 independent evaluation runs |
| 4-8 page technical report (method, theory, setup, results, ablation) | Phase 6 | LaTeX report; ablation isolates TN component vs. INT8-only |

### §5.5 Performance Benchmarks

| §5.5 Benchmark | Measured As | Guidance Threshold | Covered In |
|---|---|---|---|
| Efficiency Gain | Wall-clock inference time reduction (%) vs. INT8 baseline; also FLOPs proxy via parameter count | ≥10% improvement | Phase 3 |
| Compression vs. Accuracy | Parameter reduction ratio + action-prediction accuracy on held-out OXE split; Pareto curve across bond dimensions | ≤5% accuracy drop at ≥2x compression | Phase 3 |
| Latency on Reference Profile | End-to-end inference time (ms) on Colab GPU (T4/A100), clearly stated; compressed model must be ≤100 ms | ≤100 ms on stated hardware | Phase 3 |
| Reproducibility | End-to-end clean-env test by running Colab notebooks via "Run all"; seeds + hardware + hyperparams in repo | Full clean-env replication | Phase 7 |
| Quantum Justification | Ablation: (A) INT8-only baseline vs. (B) INT8 + TN compression; quantifies the delta attributable to TN step | Ablation mandatory | Phase 3 |

### §4.2 Secondary Objectives

| Secondary Objective | Status | Notes |
|---|---|---|
| Energy efficiency (kWh / CO2e) | **Included** (Phase 3) | Measured via `pynvml` during benchmark runs |
| Cross-track generalizability | **Out of scope** | Would require CARLA + LLaVA setup — too much scope creep |
| Open-source release (Apache 2.0 / MIT) | **Partial — see caveat** | Repository code is MIT. Model weights are **LLaMA-2 Community License (non-commercial research use only)** and cannot be redistributed under an open license. README, report, and repo description will all state this explicitly. We satisfy the "open-source code" objective but not "open-source weights." |
| Hardware pathway (qubit count, circuit depth) | **Included** (Phase 5) | PennyLane appendix |

### Items in the Challenge Statement NOT Covered — Flagged

| Item | Decision |
|---|---|
| §5.1 Quadratic Attention Complexity bottleneck | Out of scope for Compression sub-track; not selected |
| §5.1 RL Alignment Training Cost bottleneck | Out of scope; not selected |
| §5.1 Control Safety Guarantees bottleneck | Out of scope; not selected |
| §4.1 Lyapunov-based stability analysis | Not in scope; we target Compression sub-track only |
| nuScenes [6] dataset | Referenced in report for context; not used (Robotics domain uses OXE) |
| Real physical quantum hardware | Not required per §5.3; simulation fully accepted |

---

## 2. Phases

---

### Phase 0: Environment Setup & Repo Foundation

**Duration estimate**: 1-2 days | **GPU time**: ~0.5 GPU-hours

#### Deliverables
- `pyproject.toml` or `requirements.txt` with pinned dependencies:
  `torch`, `transformers`, `bitsandbytes`, `accelerate`, `quimb`, `tntorch`,
  `mujoco`, `dm_control`, `pennylane`, `pynvml`, `datasets`, `tensorflow-datasets`
- `src/` package structure (`vlam_compress/`)
- `notebooks/` directory scaffold (one `.ipynb` stub per phase 1–5)
- `Makefile` with targets: `install`, `test`, `repro` (GPU phases delegated to
  notebooks; `repro` documents the Colab workflow rather than running locally)
- `.gitignore`, `LICENSE` (MIT for code; model weight license caveat in README)
- `docs/ATTRIBUTIONS.md` initialised
- GitHub repo is public

#### Verification
- `pip install -e .[dev]` completes in a clean local venv with no errors (CPU only; no GPU required for scaffolding)
- `python -c "import vlam_compress; print('ok')"` succeeds locally
- Inside Colab: `!pip install -e . && python -c "import torch; print(torch.cuda.get_device_name(0))"` confirms GPU and package install
- Inside Colab: `!python -c "import quimb; import mujoco; print('ok')"` confirms all core GPU-phase deps

#### Challenge sections satisfied
- §5.2 (public repo prerequisite), §5.5 Reproducibility (foundation)

---

### Phase 1: Baseline Reproduction

**Duration estimate**: 2-3 days | **GPU time**: ~3-5 GPU-hours  
**Execution**: Google Colab — `notebooks/phase1_baseline.ipynb`

#### Task Description
Load OpenVLA-7B in INT8 via bitsandbytes (the accepted baseline per §5.4 Robotics
Compression), establish all measurements that will be compared against in Phase 3.
The notebook is self-contained: the first cell installs all dependencies via
`!pip install` and installs the repo package via `!pip install -e .` (after cloning
from GitHub or uploading via Drive).

#### Steps
1. Download `openvla/openvla-7b` from HuggingFace Hub using
   `AutoModelForVision2Seq.from_pretrained(..., load_in_8bit=True)`.
   Note: weights are subject to the LLaMA-2 Community License — download
   implies acceptance of that license.
2. Select a **held-out evaluation split** from Open X-Embodiment:
   - Use the `bridge_dataset` subset (tabletop manipulation; 7-DoF actions;
     well-structured language instructions). Small enough to evaluate locally,
     representative enough for the compression track.
   - Hold out 200 episodes (never used in any fine-tuning).
3. Define **task accuracy proxy**: mean L1 action-prediction error on held-out
   trajectories. This is the standard proxy used by the OpenVLA authors when no
   physical robot is available. Report success-rate proxies where feasible.
4. Run exactly **3 independent inference passes** (different episode subsets,
   different PyTorch random seeds documented in `configs/seeds.yaml`) and record:
   - Total parameter count (INT8 active parameters)
   - Peak GPU memory (MiB)
   - Per-sample inference time (ms), mean ± std
   - Action-prediction L1 error on held-out split, mean ± std
   - Wall-clock time per 100-sample batch
   - GPU power draw (W) via `pynvml`; compute total kWh
5. Save results to `results/baseline_metrics.json`.

#### Verification (inside Colab)
- Parameter count matches expected ~7.5 B for OpenVLA-7B architecture.
- INT8 peak memory is below the Colab GPU VRAM budget (T4: 16 GB; A100: 40 GB).
- Inference time per sample is in a reasonable range for a 7B model on the Colab GPU.
- `results/baseline_metrics.json` is written to Drive or downloaded; contains all five metric types.
- Notebook runs end-to-end via "Run all" with no manual intervention (seeds from `configs/seeds.yaml`).

#### Challenge sections satisfied
- §5.4 (accepted baseline established), §5.5 (Compression vs. Accuracy baseline,
  Latency baseline, Efficiency Gain baseline)

---

### Phase 2: Tensor-Network Compression Implementation

**Duration estimate**: 4-6 days | **GPU time**: ~6-16 GPU-hours  
**Execution**: Google Colab — `notebooks/phase2_compression.ipynb`

#### Task Description
Implement MPS (Matrix Product State) / MPO (Matrix Product Operator) decomposition
of OpenVLA-7B weight matrices using quimb, following the CompactifAI [1] methodology.
This is the **quantum-inspired component** required by §5.2.
All compression logic is implemented in `src/vlam_compress/compress.py`; the notebook
imports it and orchestrates the bond-dimension sweep and checkpoint saving.

#### Theoretical Motivation (for report)
MPS/MPO decomposition represents a weight matrix W in R^(m x n) as a product of
low-rank tensors with bond dimension X (chi). The number of parameters scales as
O(X^2 * d) where d is the physical dimension, vs. O(m * n) for the full matrix.
For transformer weight matrices (typically 4096x4096 or 4096x11008), this yields
tunable compression at the cost of bounded approximation error controlled by X.
This is the quantum-inspired analog of the MPO representation used in quantum
many-body physics.

#### Steps
1. **Identify compression targets**: linear layers in OpenVLA-7B's LLM backbone
   (Vicuna-v1.5 / LLaMA-2 style): `q_proj`, `k_proj`, `v_proj`, `o_proj`,
   `gate_proj`, `up_proj`, `down_proj`. Vision encoder (SigLIP) layers may be
   left uncompressed initially; the ablation (Phase 3) will test both.
2. **Implement `compress.py`**:
   - For each target weight matrix W:
     - Reshape into a higher-order tensor (e.g., for 4096x4096: reshape to
       [64, 64, 64, 64] or similar physically motivated shape)
     - Apply quimb `TensorNetwork` SVD-based MPS decomposition at bond dimension X
     - Store compressed representation as list of core tensors
     - Reconstruct approximate W_hat by contracting the TN
     - Replace original weight with W_hat (in-place substitution)
3. **Sweep bond dimensions**: X in {16, 32, 64, 128} to generate Pareto curve.
   - X=16: ~8-12x compression (aggressive)
   - X=32: ~4-6x compression
   - X=64: ~2-3x compression (primary target for ≤5% accuracy drop)
   - X=128: ~1.5-2x compression (high-fidelity reference point)
4. **Post-compression INT8 quantization**: apply bitsandbytes INT8 to reconstructed
   weights for maximum memory efficiency (TN compression + INT8 combined pipeline).
5. Save compressed model checkpoints to `checkpoints/compressed_chi{X}/`.

#### Verification (inside Colab)
- Compressed model loads without error and runs a forward pass on a sample input.
- Parameter counts match theoretical values for each X.
- For X=64, reconstruction error (Frobenius norm ratio ||W - W_hat|| / ||W||) < 5% per layer.
- All four bond-dimension checkpoints are saved to Google Drive or downloaded as archives.
- Notebook runs end-to-end via "Run all."

#### Challenge sections satisfied
- §5.2 (functional QI component), §4.1 (reduced parameter count), §5.1 (Model Footprint),
  §5.5 (Quantum Justification — the TN decomposition is the QI component)

**Compute note**: SVD-based MPS decomposition runs on CPU or GPU. For a single 7B
model with ~250 transformer linear layers, expect 30-120 min per bond dimension on
a Colab GPU (T4 or A100); GPU-accelerated SVD via `torch.linalg.svd` (cuSOLVER) is
preferred. Total: ~4-8 hours across all four X values. Colab Pro session limits apply;
plan to run one or two bond dimensions per session and save checkpoints to Drive between
sessions.

---

### Phase 3: Evaluation, Ablation & Pareto Analysis

**Duration estimate**: 3-4 days | **GPU time**: ~8-12 GPU-hours  
**Execution**: Google Colab — `notebooks/phase3_evaluation.ipynb`

#### Task Description
Produce all §5.5 benchmark numbers with mean ± std across at least 3 runs, and
the mandatory ablation isolating the TN component.
Evaluation logic lives in `src/vlam_compress/metrics.py`; the notebook loads
checkpoints from Drive, runs all benchmark conditions, and writes result JSONs.

#### Steps
1. **Inference benchmark** (3 independent runs per model variant).
   For each X in {16, 32, 64, 128} and for the INT8 baseline:
   - Load model, run on held-out 200-episode OXE split (seeds from `configs/seeds.yaml`)
   - Record: inference time (ms/sample), total parameter count, peak memory,
     action-prediction L1 error, GPU power draw (pynvml)
   - Each "run" uses a different random evaluation ordering and seed

2. **Efficiency Gain** (§5.5):
   - Wall-clock inference time: `(t_baseline - t_compressed) / t_baseline * 100%`
   - FLOPs proxy: `(params_baseline - params_compressed) / params_baseline * 100%`
   - Report as mean ± std across 3 runs; target ≥10%

3. **Compression vs. Accuracy** (§5.5):
   - Parameter reduction ratio: `params_baseline / params_compressed`
   - Task accuracy change: `delta_accuracy = accuracy_compressed - accuracy_baseline`
   - Generate Pareto curve: x-axis = compression ratio, y-axis = accuracy
   - Export as `results/pareto_curve.png` and `results/pareto_data.json`

4. **Latency on Reference Profile** (§5.5):
   - Measure end-to-end inference time (load sample -> tokenize -> forward pass ->
     decode action) for X=64 compressed model
   - Target ≤100 ms per sample on stated GPU
   - Clearly state hardware: GPU model, VRAM, driver version, CUDA version

5. **Ablation Study** (mandatory per §5.5 Quantum Justification).
   Three conditions, each run at least 3 times:
   - **(A) INT8-only baseline** (Phase 1 numbers): no TN, bitsandbytes only
   - **(B) INT8 + TN compression (LLM layers only)**: TN on attention + FFN, not vision
   - **(C) INT8 + TN compression (full model)**: TN on all linear layers
   This isolates the QI (TN) contribution from the INT8 baseline and shows whether
   applying TN to the vision encoder adds further value.

6. **Energy** (§4.2 secondary):
   - `pynvml` GPU power x runtime -> Joules -> kWh
   - CO2e estimate using US average grid intensity (0.386 kg CO2e/kWh)

7. Save all results to `results/eval_summary.json` and `results/ablation.json`.

#### Verification (inside Colab)
- `results/eval_summary.json` contains all five §5.5 benchmark fields for all model variants.
- Each metric has `mean`, `std`, and `n_runs >= 3` fields.
- Pareto curve covers at least 4 compression levels.
- Ablation table clearly shows delta L1 error attributable to TN step.
- If efficiency gain < 10% or accuracy drop > 5%, this is flagged and explained (not silently hidden).
- All result files are saved to Drive and committed to the repo as artifacts.

#### Challenge sections satisfied
- §5.2 (quantitative results, mean ± std, at least 3 runs), §5.5 (all five benchmarks),
  §5.3 (any degradation fully reported and discussed)

**Compute note**: Inference-only evaluation is fast. A 200-sample batch at ~300 ms/sample
= ~1 min per run. Budget ~8-12 GPU-hours total across all model variants and 3 runs each.
Latency results are stated for the Colab GPU type used (T4 or A100); the hardware profile
(GPU model, VRAM, CUDA version, driver) is logged by the notebook and included in the report.

---

### Phase 4: MuJoCo 3D Visualization

**Duration estimate**: 3-4 days | **GPU time**: ~1-2 GPU-hours  
**Execution**: Google Colab — `notebooks/phase4_mujoco_viz.ipynb`

#### Task Description
Drive a MuJoCo humanoid figure with action outputs from the compressed OpenVLA-7B
model (X=64 checkpoint) and render a 3D visualization of the executed task.
`dm_control` and MuJoCo installation are handled by the notebook's setup cell
(`!pip install dm-control mujoco`); dependency resolution is verified fresh inside
Colab without assumptions from any prior local environment.

#### Architecture

```
OXE sample (image + language instruction)
          |
  Compressed OpenVLA-7B (X=64)
          |
  7-DoF action vector [dx, dy, dz, droll, dpitch, dyaw, gripper]
          |
  Joint mapping layer  (action -> humanoid arm DoF subset)
          |
  MuJoCo dm_control humanoid (humanoid.xml, 21 DoF total)
          |
  3D rendered frames -> MP4 + GIF
```

**Mapping rationale**: OpenVLA produces 7-DoF end-effector delta actions for a
robot arm. The MuJoCo humanoid has 21 DoF. We map the 7 action dimensions to the
right arm (shoulder 3 DoF, elbow 1 DoF, wrist 2 DoF, hand aperture 1 DoF), leaving
the lower body stabilised by a PD controller. The humanoid's right arm traces the
compressed model's intended end-effector trajectory, giving a physically interpretable
3D demonstration.

#### Steps
1. Set up dm_control `humanoid` task environment.
2. Implement `src/vlam_compress/mujoco_bridge.py`:
   - `load_compressed_model(chi)` loads checkpoint
   - `sample_to_action(image, lang)` runs inference, returns 7-dim action vector
   - `action_to_joint_command(action)` maps to humanoid arm joints
   - `run_episode(n_steps=200)` rollout loop, returns frame array
3. Implement `scripts/visualize.py`:
   - Runs a tabletop pick-and-place scenario (image from OXE bridge dataset)
   - Records 200 physics steps at 25 Hz
   - Outputs `results/demo_compressed_chi64.mp4` and `.gif`
4. Annotate rendered video with overlays: action values, compression ratio, language instruction.

#### Verification (inside Colab)
- Notebook runs end-to-end via "Run all."
- Output video `demo_compressed_chi64.mp4` is at least 5 seconds; humanoid arm is visibly moving.
- The arm trajectory is non-trivial (not stationary, responds differently to different instructions).
- Side-by-side comparison: INT8 baseline arm trajectory vs. compressed model arm trajectory on same input.
- Video is downloaded from Colab and committed to `results/`.

#### Challenge sections satisfied
- User requirement: "3D visualization showing a humanoid figure executing a task, driven
  by the compressed model's outputs, using MuJoCo"

**Compute note**: MuJoCo physics and rendering run on CPU inside Colab. Inference for
each frame requires one GPU forward pass (~300 ms on T4). 200 steps ≈ 1 min wall-clock
per episode. Budget ~2 GPU-hours for development and rendering iterations.

---

### Phase 5: PennyLane Hardware-Feasibility Appendix (Bonus)

**Duration estimate**: 2-3 days | **GPU time**: 0 (CPU-only PennyLane simulation)  
**Execution**: Google Colab — `notebooks/phase5_pennylane.ipynb` (CPU runtime sufficient; no GPU needed)

#### Task Description
Map the achieved MPS compression (bond dimension X, tensor ranks) to estimates of
near-term quantum hardware requirements, satisfying §4.2 "Hardware pathway."

#### Steps
1. **MPS to quantum circuit mapping**:
   - An MPS with bond dimension X on a chain of n sites can be prepared by a
     quantum circuit with n qubits and O(n * X^2) two-qubit gates.
   - For each compressed layer, compute:
     - Required qubit count: q = log2(X) * n_cores_per_layer
     - Circuit depth estimate: d = O(n_layers * X^2 * log2(m))
2. **Noise sensitivity analysis**:
   - Use PennyLane to simulate a small toy MPS circuit (2-4 qubits) under
     depolarising noise at p in {0.001, 0.01, 0.1}.
   - Report fidelity degradation vs. noise level.
3. **Feasibility table**: for each X in {16, 32, 64, 128}:
   - Qubit count
   - Estimated circuit depth
   - Required T2 coherence time
   - Comment on current hardware capability (e.g., IBM Heron r2: 133 qubits)
4. Write as `docs/appendix_pennylane.md`; embed abbreviated version in report appendix.

#### Verification (inside Colab)
- PennyLane simulation runs for the toy circuit without error.
- Feasibility table is complete for all four X values.
- At least one X value is within reach of near-term hardware (qubit count ≤ 1000).
- Notebook runs end-to-end via "Run all"; output markdown is downloaded and committed to `docs/appendix_pennylane.md`.

#### Challenge sections satisfied
- §4.2 (Hardware pathway: qubit count, circuit depth, noise sensitivity),
  §5.3 (simulation environment and version stated explicitly)

---

### Phase 6: Technical Report

**Duration estimate**: 3-4 days | **GPU time**: 0

#### Task Description
Produce the 4-8 page technical report required by §5.2.

#### Structure

```
1. Introduction & Problem Motivation                  (~0.5 pages)
2. Quantum-Inspired Method: MPS/MPO                   (~1.5 pages)
   2.1 Theoretical Foundation
   2.2 Algorithm: TN Compression of Linear Layers
   2.3 Why This Provides QI Advantage (§5.5 Quantum Justification)
3. Experimental Setup                                 (~0.5 pages)
   3.1 Model: OpenVLA-7B [license caveat stated here]
   3.2 Dataset: Open X-Embodiment (bridge_dataset)
   3.3 Hardware & Resource Declaration
4. Results                                            (~2 pages)
   4.1 Efficiency Gain (Table: mean ± std, 3 runs)
   4.2 Compression vs. Accuracy (Pareto curve figure)
   4.3 Latency on Reference Profile
   4.4 Energy Efficiency (kWh, CO2e)
5. Ablation Study                                     (~1 page)
   5.1 INT8-only vs. INT8+TN (LLM only) vs. INT8+TN (Full)
   5.2 Sensitivity to Bond Dimension X
6. MuJoCo Demonstration                               (~0.5 pages)
7. Conclusion & Limitations [license caveat stated here] (~0.5 pages)
Appendix A: PennyLane Hardware Feasibility
Appendix B: Resource Declaration
Appendix C: License Statement
   - Code: MIT
   - Model weights: LLaMA-2 Community License (non-commercial research use only)
References: all seven from §7 of challenge statement + our repo
```

#### Resource Declaration (§6) — filled in during Phase 3
- GPU type and count
- Total GPU-hours consumed
- Simulation environment: quimb vX.Y, PennyLane vX.Y
- Estimated energy consumption (kWh)

#### Verification
- Report is 4-8 pages (not counting appendices).
- All §5.5 benchmarks present in tables with mean ± std.
- Ablation section isolates TN contribution.
- All seven challenge-statement references cited correctly.
- License caveat for model weights appears in Section 3.1 and Appendix C.
- Resource Declaration is complete.

#### Challenge sections satisfied
- §5.2 (4-8 page technical report), §5.5 (Quantum Justification, ablation),
  §6 (Resource Declaration)

---

### Phase 7: Repo Polish & Final Validation

**Duration estimate**: 1-2 days | **GPU time**: ~1 GPU-hour (clean-env replication test)

#### Task Description
Ensure the repository satisfies §5.5 Reproducibility: full clean-environment
replication with no manual intervention.

#### Steps
1. **README.md** (root) — two-path replication instructions:
   ```
   # Local (no GPU — scaffolding and report only)
   git clone <repo>
   cd vw-quantum-vlam-challenge
   pip install -e .[dev]
   make test        # unit tests for compress.py, metrics.py, mujoco_bridge.py

   # GPU phases (Phases 1–5) — Google Colab
   # 1. Open each notebook in notebooks/ via Google Colab.
   # 2. The first cell clones this repo and runs: !pip install -e .
   # 3. Run all cells. Seeds are fixed in configs/seeds.yaml [42, 1337, 2024].
   # 4. Download result files from Colab to results/ and commit.
   ```
   README must include prominently:
   - Code license: MIT
   - **Model weights license: LLaMA-2 Community License (non-commercial research use
     only). Users must accept the LLaMA-2 license before downloading weights via
     HuggingFace Hub.**
2. **`configs/`**: YAML files with all hyperparameters, random seeds (3 per run),
   and a `hardware_profile.yaml` that records the Colab GPU type, VRAM, CUDA version,
   and driver used during the submission runs.
3. **`configs/seeds.yaml`**: seeds [42, 1337, 2024] documented.
4. **`docs/ATTRIBUTIONS.md`**: final review — every repo, paper, and dataset listed.
5. **Reproducibility test**: open `notebooks/phase3_evaluation.ipynb` in a fresh
   Colab session, run all cells with the committed `configs/seeds.yaml`, and verify
   `results/eval_summary.json` matches committed baseline values within floating-point
   tolerance. This replaces the local `make repro` target as the canonical replication check.
6. **Health check before final push**:
   - All imports resolve locally (`pip install -e .[dev]`)
   - `make test` passes (unit tests for TN compression, action mapping, metric calculation)
   - README installation instructions succeed
   - No hardcoded absolute paths in any notebook or source file
7. **Final commit + tag**: `v1.0.0-submission`

#### Verification
- Notebooks run end-to-end via "Run all" in a fresh Colab session (reproducibility standard).
- `results/eval_summary.json` is reproducible within floating-point tolerance for documented seeds.
- ATTRIBUTIONS.md contains at least 8 entries (repos + papers + dataset).
- Report PDF compiles without errors.
- README's first prominent section after the title includes the model weight license caveat.
- `configs/hardware_profile.yaml` documents the exact Colab GPU environment used for submission results.

#### Challenge sections satisfied
- §5.2 (public repo + README), §5.5 Reproducibility (full clean-env replication via Colab notebooks),
  §5.3 (all resource assumptions explicitly stated)

---

## 3. Timeline Summary

| Phase | Description | Wall-Clock Days | GPU-Hours (estimate) |
|---|---|---|---|
| 0 | Environment Setup | 1-2 | 0.5 |
| 1 | Baseline Reproduction | 2-3 | 3-5 |
| 2 | TN Compression Implementation | 4-6 | 6-16 |
| 3 | Evaluation & Ablation | 3-4 | 8-12 |
| 4 | MuJoCo Visualization | 3-4 | 1-2 |
| 5 | PennyLane Appendix | 2-3 | 0 |
| 6 | Technical Report | 3-4 | 0 |
| 7 | Repo Polish & Validation | 1-2 | 1 |
| **Total** | | **~19-28 days** | **~20-36 GPU-hours** |

**Compute note**: Phase 2 dominates. All GPU-hours are consumed on Google Colab
(free tier or Colab Pro). SVD decomposition of 7B model weight matrices benefits
strongly from GPU-accelerated `torch.linalg.svd` (cuSOLVER). On a T4 (free tier),
each bond-dimension pass may take 1-2 hours; on an A100 (Colab Pro), 30-60 min.
Plan to checkpoint to Google Drive between Colab sessions to avoid losing progress.

---

## 4. External Repository Log (Initial Scouting)

Full entries in `docs/ATTRIBUTIONS.md`. License status summary:

| Repo | License | Role |
|---|---|---|
| `openvla/openvla` | MIT (code); LLaMA-2 Community (weights) | Reference model, loading/inference code |
| `jcmgray/quimb` | MIT | MPS/MPO tensor network decomposition |
| `rballester/tntorch` | MIT | Tensor-train decomposition utilities (comparison) |
| `google-deepmind/dm_control` | Apache 2.0 | MuJoCo humanoid environment |
| `google-deepmind/mujoco` | Apache 2.0 | Physics engine |
| `google-deepmind/open_x_embodiment` | Apache 2.0 | Reference dataset |
| `TimDettmers/bitsandbytes` | MIT | INT8 baseline quantization |
| CompactifAI [1] | arXiv only (no public code repo found) | Methodology reference; implemented from paper |

**License flag (repeated from Section 0)**: OpenVLA-7B model *weights* are subject to
the LLaMA-2 Community License — non-commercial research use only. This caveat must
appear in: (1) this plan, (2) README.md, (3) the technical report (Sections 3.1 and
Appendix C), (4) ATTRIBUTIONS.md. We do NOT claim a fully open-source weight release.

---

## 5. Key Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| TN compression does not achieve ≥10% efficiency gain at ≤5% accuracy drop | Medium | Sweep 4 bond dimensions; publish Pareto curve; §5.5 explicitly says submissions are not penalized for accuracy degradation provided the tradeoff is clearly characterized |
| Inference time exceeds 100 ms on GPU | Low | GPU inference for a compressed 7B model should be well under 100 ms; if not, use FP16 reconstruction of compressed weights |
| quimb SVD of 4096x4096 matrices is too slow | Medium | Use PyTorch `torch.linalg.svd` (GPU-accelerated) wrapped in the same interface; quimb used for TN bookkeeping only |
| MuJoCo humanoid action mapping produces implausible motion | Medium | Fall back to robot arm model (Franka Panda MJCF) if humanoid mapping is not convincing within time budget; will notify before doing so |
| Open X-Embodiment download is slow or requires GCS auth | Medium | Use `tensorflow_datasets` RLDS loader; cache locally; download only `bridge_dataset` subset (~50 GB, not the full 4 TB corpus) |
| LLaMA-2 base license blocks open-source weight release | Certain | Flag clearly everywhere (this plan, README, report). Code is MIT. Sufficient for competition; does not affect reproducibility since weights download from HuggingFace Hub under accepted license. |

---

## 6. Commit Discipline

- Commit after every meaningful unit: Phase 0 setup, baseline script, compression
  script at each X value, evaluation script, MuJoCo integration, report draft.
- Commit message format: `[phaseN] <verb> <what>`
  Example: `[phase1] add INT8 baseline inference benchmark`
- Health check before every push: imports resolve, `make test` passes, README
  installation instructions are current.
- No giant end-of-project commits.

---

## 7. References

[1] Tomut, A. et al. (2024). CompactifAI: Extreme Compression of Large Language Models using Quantum-Inspired Tensor Networks. arXiv:2401.14109.  
[2] Liu, J. et al. (2024). Towards Provably Efficient Quantum Algorithms for Large-Scale Machine-Learning Models. Nature Communications, 15(1), 434.  
[3] NVIDIA & Wang, Y. et al. (2025). Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail. arXiv:2511.00088.  
[4] NVIDIA Technical Blog (2025). Introducing NVFP4 for Efficient and Accurate Low-Precision Inference.  
[5] Open X-Embodiment Collaboration (2023). Open X-Embodiment: Robotic Learning Datasets and RT-X Models. arXiv:2310.08864.  
[6] Caesar, H. et al. (2020). nuScenes: A Multimodal Dataset for Autonomous Driving. CVPR 2020, pp. 11618-11628.  
[7] Black, K. et al. (2024). π₀: A Vision-Language-Action Flow Model for General Robot Control. arXiv:2410.24164.
