# TODO — VW Quantum+AI Challenge 2026 (Compression / Robotics)

Audit date: 2026-07-03. Derived from `docs/PROJECT_PLAN.md` deliverables vs. actual repo state.

## Status

| Phase | Status | Key commits | Blocker / Next |
|---|---|---|---|
| 0 — Scaffold | ~Complete (gaps) | `bde7b4c` | `tests/` empty; phase 4/5 notebook stubs missing; tf/tfds still in pyproject |
| 1 — INT8 baseline | Notebook stable; **no artifacts committed** | `4c2d6a8`→`1a8de10` | Full Colab run; commit `baseline_metrics.json` + `hardware_profile.yaml` |
| 2 — TN compression | Notebook drafted, **not stabilized** | `4c2d6a8` | Phase-1 fixes (pins, eager attn) never ported; checkpoint-transfer contradiction |
| 3 — Eval & ablation | Notebook drafted, **stale data loading** | `4c2d6a8` | Still on tfds/GCS (migration `1a8de10` only touched phase 1); efficiency-gain design flaw |
| 4 — MuJoCo viz | Not started | — | `mujoco_bridge.py` is a 6-line stub; no notebook |
| 5 — PennyLane | Not started | — | No notebook, no appendix |
| 6 — Report | Not started | — | `docs/report/` missing (Makefile `report` target dangles) |
| 7 — Polish | Not started | — | README points at stub scripts that `raise NotImplementedError` |

**Note on memory staleness**: memory marks Phase 1 "COMPLETE", but `results/` contains only `.gitkeep` and `configs/hardware_profile.yaml` does not exist — the notebook is stabilized but there is no committed evidence of a successful end-to-end run. Treat Phase 1 as "code done, run pending."

---

## Phase 0 — residual items

- [x] `pyproject.toml`, `src/vlam_compress/` package, `Makefile`, `.gitignore`, `LICENSE`, `docs/ATTRIBUTIONS.md`, `docs/PROJECT_PLAN.md`
- [x] Notebook scaffolds for phases 1–3
- [ ] Create notebook stubs `notebooks/phase4_mujoco_viz.ipynb` and `notebooks/phase5_pennylane.ipynb` (plan: one stub per phase 1–5)
- [ ] Remove `tensorflow` and `tensorflow-datasets` from `pyproject.toml` dependencies — tfds was dropped in `1a8de10` (protobuf gencode clash); they remain only because phase 3 is unmigrated
- [ ] Resolve repo visibility: plan/§5.2 require a **public** repo, but phase 1 notebook cell 3 clones a **private** repo via `GH_TOKEN` (`quantumadventurer11/vw-quantum-vlam-challenge`). Decide: keep private until submission, then flip public and strip the token path (tracked again in Phase 7)

## Phase 1 — INT8 baseline

- [x] Notebook implemented: INT8 load (`load_in_8bit=True`, `attn_implementation="eager"`), exact pins (`transformers==4.40.1`, `tokenizers==0.19.1`, `timm==0.9.10`, `sentencepiece==0.1.99`), subprocess-based pip installs
- [x] HF Hub data loading: `lerobot/bridge` streaming + PyAV frame decode + `meta/tasks.jsonl` task map
- [x] 3-seed eval loop, all 5 metric types, warm-up pass, verification checklist, download cell
- [x] `predict_action` / `norm_stats` / `UNNORM_KEY="bridge_orig"` runtime diagnostics (cell 11)
- [ ] **Execute full "Run all" in Colab** and confirm the verification checklist passes (param count ~7.5B, peak mem < VRAM, 5 metric types, n_runs=3)
- [ ] Commit `results/baseline_metrics.json`
- [ ] Commit `configs/hardware_profile.yaml` (written by cell 12)
- [ ] Fix duplicated cell-number comments (two cells labeled "Cell 3", two labeled "Cell 10") — cosmetic but confuses the "run cells in order" instruction
- [ ] Reconcile **episodes vs. frames**: plan says "200 held-out episodes"; the notebook evaluates 200 random *frames* from the `train` split (`seeds.yaml` `max_steps_per_episode: 50` is never used). Update plan/config/notebook wording to one consistent semantic
- [ ] Reword the "held-out" claim: BridgeData V2 is in OpenVLA's *training* mix (`bridge_orig` norm key exists precisely because of this), so these frames are not held out from model training. State honestly: "evaluation on training-distribution data, held out from any fine-tuning we perform" (we do none) — required by §5.3 honesty standard

## Phase 2 — TN compression

- [x] `src/vlam_compress/compress.py`: `choose_reshape_dims`, `mps_decompose` (SVD sweep w/ truncation), `mps_reconstruct`, `count_core_params`, `frobenius_error`, `compression_ratio`, `find_compression_targets`
- [x] Notebook: target discovery (~224 layers), quimb cross-validation cell, χ-sweep with per-layer stats, cores.pt checkpointing, forward-pass smoke test, INT8 re-quantization demo, merge-with-previous-session logic, verification checklist
- [x] **Port Phase-1 stabilization fixes**: exact `transformers==4.40.1` + tokenizers/timm/sentencepiece pins; subprocess `_pip` install pattern; version-verify cell; `attn_implementation="eager"` in both `from_pretrained` branches
- [x] Replace `REPO_URL` placeholder with GH_TOKEN Colab-secret clone (matches phase 1)
- [x] **Resolve checkpoint-transfer**: Google Drive mount cell added; `CHECKPOINTS_DIR` points to Drive; download cell updated (no more tar.gz — checkpoints persist on Drive for Phase 3)
- [x] Fix `find_compression_targets` for INT8 path: `_LINEAR_TYPES = (nn.Linear, Linear8bitLt)` in `compress.py` (conditional import; falls back gracefully if bitsandbytes absent)
- [ ] Verify `get_weight_fp16`'s `quant_state` branch: `dequantize_blockwise(w.data, w.quant_state)` is the 4-bit/blockwise scheme; `Linear8bitLt` stores row-wise `CB`/`SCB`. Test dequantization on a small model before trusting it, or route Int8Params to the CB/SCB path
- [ ] Sanity-check the cell-13 dummy forward pass against real `processor(...)` output (raw `torch.zeros(1,3,224,224)` pixel values may not match OpenVLA's expected input pipeline/resolution)
- [ ] **Run the sweep in Colab** for χ ∈ {16, 32, 64, 128}; confirm per-layer Frobenius error < 5% at χ=64 (plan verification gate)
- [ ] Persist checkpoints per the chosen transfer mechanism; commit `results/compression_sweep_stats.json`

## Phase 3 — evaluation, ablation, Pareto

- [x] `src/vlam_compress/metrics.py`: aggregate, efficiency gains, whole-model compression ratio, CO2e, delta_dict, flag_benchmark
- [x] Notebook: χ-sweep eval, 3-condition ablation (A/B/C) with on-the-fly vision compression for C, INT8-aware weight dequant helpers, Pareto data+plot, verification checklist, download cell
- [x] **Migrate data loading to HF Hub**: replaced tfds/GCS with `lerobot/bridge` streaming via `datasets`+PyAV; GCS auth cell replaced with no-op; `hf_item_to_sample` + task-map loading copied from Phase 1
- [x] Port Phase-1 pins + `attn_implementation="eager"` to install and model-load cells; version-verify cell added
- [x] Replace `REPO_URL` placeholder with GH_TOKEN Colab-secret clone
- [x] Wire checkpoints to Drive: `CHECKPOINTS_DIR = CHECKPOINTS_BASE`; Drive mount cell added (Phase 3 reads from same path Phase 2 wrote)
- [ ] **Resolve the efficiency-gain design flaw before burning GPU hours**: `apply_tn_patches` reconstructs full-shape FP16 `W_hat` and swaps it in — same matmul shape as baseline, so wall-clock latency will NOT improve (FP16 nn.Linear vs INT8 kernels may even be *faster*, but memory goes up and "compression" exists only in the stored cores). Options: (a) present param-count/FLOPs proxy as the primary efficiency metric with honest framing (§5.5 permits characterized tradeoffs), (b) implement a factorized `MPSLinear` module that contracts cores at inference (real FLOP reduction at small χ), (c) reconstruct + re-quantize to INT8 for memory parity. This choice affects Phase 2 checkpoint format — decide first
- [ ] Latency ≤100 ms target reality check: `predict_action` is autoregressive (7 action tokens); plan's own estimate is ~300 ms/sample on T4. Patched variants won't beat that. Decide mitigation (A100 runs, or report the miss with explanation as §5.5 allows) and document
- [ ] `results/*.png` is gitignored but the plan requires committing `pareto_curve.png` — use `git add -f` for final artifacts (already hinted in .gitignore comment) and note it in the download cell
- [ ] Run full evaluation in Colab (baseline reuse + 4 χ × 3 seeds + condition C × 3 seeds); commit `eval_summary.json`, `ablation.json`, `pareto_data.json`, `pareto_curve.png`

## Phase 4 — MuJoCo visualization

- [x] Create `notebooks/phase4_mujoco_viz.ipynb` — 21 cells; install dm-control/mujoco, EGL headless setup, Drive mount, full rollout + side-by-side comparison + arm trajectory plot + download
- [x] Implement `src/vlam_compress/mujoco_bridge.py`: `make_env`, `action_to_joint_command` (7→3 right arm via `_ACTION_MAP`), `pd_lower_body`, `run_episode`, `load_compressed_model`, `restore_patches`, `encode_video`, `make_side_by_side`, overlay helpers
- [x] PD controller for lower-body stabilization (`pd_lower_body` — proportional toward post-reset standing target)
- [x] Implement `scripts/visualize.py` as CLI wrapper around `mujoco_bridge` (argparse; supports `--chi`, `--n-steps`, `--seed`, `--no-baseline`)
- [ ] **Run in Colab** — requires chi=64 checkpoint from Phase 2; render ≥5 s video; commit `results/demo_compressed_chi64.mp4` + `.gif` (`git add -f`)
- [ ] Fallback documented in plan: Franka Panda MJCF if humanoid motion is implausible (notify before switching)

## Phase 5 — PennyLane appendix

- [ ] Create `notebooks/phase5_pennylane.ipynb` (CPU runtime)
- [ ] Qubit-count and circuit-depth estimates per χ ∈ {16, 32, 64, 128} from actual Phase-2 core ranks (`layer_stats.json`)
- [ ] Toy MPS-preparation circuit (2–4 qubits) under depolarizing noise p ∈ {0.001, 0.01, 0.1}; fidelity table
- [ ] Feasibility table incl. T2 estimate and comparison to current hardware (e.g., IBM Heron r2); verify ≥1 χ value ≤1000 qubits
- [ ] Write and commit `docs/appendix_pennylane.md`

## Phase 6 — technical report

- [ ] Create `docs/report/report.tex` with the plan's section skeleton (Makefile `report` target already expects this path)
- [ ] Sections 1–7 per plan; license caveat in §3.1 **and** Appendix C; all seven challenge-statement references
- [ ] Results tables (mean ± std, n=3) + Pareto figure from Phase-3 JSONs; ablation section isolating TN delta
- [ ] Appendix B Resource Declaration: GPU type(s), total GPU-hours, quimb/PennyLane versions, kWh
- [ ] **Start a GPU-hours ledger now** (e.g., `docs/gpu_hours_log.md`, one line per Colab session) — Appendix B needs the total and nothing currently tracks it
- [ ] Verify 4–8 pages excluding appendices; `make report` compiles cleanly

## Phase 7 — repo polish & validation

- [ ] Rewrite `README.md`: Quick Start currently points at `make baseline/compress/eval/visualize` → stub scripts that `raise NotImplementedError`; no mention of the `notebooks/` Colab workflow (the actual execution path); Repository Layout omits `notebooks/`. Implement the two-path structure from the plan
- [ ] Decide fate of `scripts/*.py` and the corresponding Makefile targets (`baseline`, `compress`, `eval`, `visualize`, `repro`): implement thin CLI wrappers around `vlam_compress` or delete and document Colab-only replication
- [ ] Generate `requirements-pinned.txt` from the Colab environment (README already references it)
- [ ] Update `configs/seeds.yaml` hardware block — placeholders still say "e.g. NVIDIA RTX 4090", 24 GB (pre-Colab-pivot relic); should reflect T4/A100 and defer to `hardware_profile.yaml`
- [ ] Write unit tests (`tests/` has only an empty `__init__.py`; `make test` collects nothing): `mps_decompose`→`mps_reconstruct` round-trip error bounds; `choose_reshape_dims` product invariant (incl. 4096×11008); `count_core_params` vs theoretical; `frobenius_error`; `metrics.aggregate`/`delta_dict`/`efficiency_gain_pct`; `find_compression_targets` on a toy model (incl. Linear8bitLt if feasible on CPU); action→joint mapping
- [ ] Make repo public; remove `GH_TOKEN` clone path from all notebooks; verify no hardcoded tokens/paths remain
- [ ] ATTRIBUTIONS.md final pass: add `lerobot/bridge` (HF mirror actually used — currently only google-deepmind/open_x_embodiment is listed), `huggingface/datasets`, PyAV; confirm ≥8 entries
- [ ] Fresh-Colab reproducibility run of `phase3_evaluation.ipynb`; verify `eval_summary.json` matches committed values within FP tolerance
- [ ] Final health check (imports, `make test`, README instructions) → tag `v1.0.0-submission`

---

## Critical path (ordered)

1. **Decide the efficiency-measurement architecture** (Phase 3 flaw above). It dictates Phase-2 checkpoint format and whether the ≥10% wall-clock target is claimable — everything downstream depends on it.
2. **Decide checkpoint transfer mechanism** (Drive / HF Hub / LFS) — blocks both Phase 2 output and Phase 3 input.
3. Stabilize phase 2 notebook (pins, eager attn, REPO_URL, INT8 target detection) — mechanical, ~1 session.
4. Colab session(s): run Phase 1 to completion → commit baseline artifacts; run Phase 2 sweep → persist 4 checkpoints + sweep stats.
5. Migrate phase 3 to HF Hub loading + stabilize → run full eval + ablation → commit result JSONs/PNG.
6. Phase 4 (needs χ=64 checkpoint) and Phase 5 (needs only layer_stats; can run any time after Phase 2) — parallelizable.
7. Phase 6 report (needs 3, 4, 5) → Phase 7 polish, repro test, tag. Budget ≥2 days for report + polish.

## Decisions

| # | Question | Decision |
|---|---|---|
| 1 | Efficiency metric | **Param-count/storage proxy (option a)**. cores.pt files are genuinely smaller (50–200 MB vs ~14 GB FP16); report storage compression ratio and parameter count reduction. Frame wall-clock latency honestly: "factorized inference is future work." §5.5 permits characterized tradeoffs. |
| 2 | Checkpoint transfer | **Google Drive mount**. `drive.mount('/content/drive')` in each notebook; write cores.pt to Drive from Phase 2, read from Phase 3/4. No LFS, no HF Hub write permissions, no broken `git add checkpoints/`. |
| 3 | Phase 3 data loading | **Migrate to HF Hub** (identical to Phase 1 fix). Copy `hf_item_to_sample` + task-map loading; drop tfds and GCS auth. |
| 6 | Repo visibility | **Keep private until submission**, then flip public and strip `GH_TOKEN` clone path (Phase 7 checklist). |

## Open questions

3. **"Held-out" framing**: bridge is in OpenVLA's training data — how do we frame the L1 proxy honestly without weakening the submission?
4. **200 episodes vs 200 frames**: which does the evaluation actually claim? Plan, config, and notebook currently disagree.
5. **lerobot/bridge action convention vs `bridge_orig` norm stats**: has anyone verified the HF mirror's `action` field matches the ordering/scaling OpenVLA's unnormalized outputs are compared against? A units mismatch would silently inflate L1. Worth a 5-minute sanity check on magnitudes during the Phase-1 run.
7. **scripts/ + Makefile relics**: implement or remove? README currently advertises them.
8. **Condition C (full-model TN)**: Phase 2 never compresses vision layers, so Phase 3 recomputes them on-the-fly every session (~non-reproducible across sessions unless seeded and slow). Should Phase 2 optionally emit a `cores_vision.pt` instead?
9. **GPU-hours tracking**: no mechanism exists; Appendix B requires a total. Ledger file or Colab-cell logging?
