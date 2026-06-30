# Attributions & External Sources

This file tracks every external repository, paper, and dataset used or substantially
referenced in this project. Updated continuously as work progresses — not reconstructed
at the end.

---

## Repositories

### Core Pipeline

| Repository | URL | License | Usage |
|---|---|---|---|
| openvla/openvla | https://github.com/openvla/openvla | MIT (code); **LLaMA-2 Community License (weights — non-commercial research use only)** | Reference model (OpenVLA-7B); loading/inference code patterns; `AutoModelForVision2Seq` setup |
| jcmgray/quimb | https://github.com/jcmgray/quimb | MIT | Primary library for MPS/MPO tensor network decomposition of weight matrices |
| rballester/tntorch | https://github.com/rballester/tntorch | MIT | Tensor-train decomposition utilities; used for cross-validation of compression results |
| TimDettmers/bitsandbytes | https://github.com/TimDettmers/bitsandbytes | MIT | INT8 quantization baseline (`load_in_8bit=True` via HuggingFace Transformers) |

### Evaluation & Environment

| Repository | URL | License | Usage |
|---|---|---|---|
| google-deepmind/mujoco | https://github.com/google-deepmind/mujoco | Apache 2.0 | Physics engine for 3D humanoid visualization |
| google-deepmind/dm_control | https://github.com/google-deepmind/dm_control | Apache 2.0 | MuJoCo humanoid environment (`humanoid.xml`); dm_control suite for task setup |
| google-deepmind/open_x_embodiment | https://github.com/google-deepmind/open_x_embodiment | Apache 2.0 | Reference dataset (Open X-Embodiment); RLDS data loading scripts |

### Supporting Tools

| Repository | URL | License | Usage |
|---|---|---|---|
| huggingface/transformers | https://github.com/huggingface/transformers | Apache 2.0 | Model loading, tokenization, `AutoModelForVision2Seq` |
| huggingface/accelerate | https://github.com/huggingface/accelerate | Apache 2.0 | INT8 quantization integration, device management |
| PennyLaneAI/pennylane | https://github.com/PennyLaneAI/pennylane | Apache 2.0 | Hardware-feasibility appendix: qubit count / circuit depth analysis |
| pytorch/pytorch | https://github.com/pytorch/pytorch | BSD-style | Training framework; `torch.linalg.svd` for GPU-accelerated SVD |

---

## Papers Cited

All papers are from the challenge statement §7 unless noted.

| Citation | Full Reference | arXiv / DOI | Role in Project |
|---|---|---|---|
| [1] CompactifAI | Tomut, A. et al. (2024). CompactifAI: Extreme Compression of Large Language Models using Quantum-Inspired Tensor Networks. | arXiv:2401.14109 | Primary methodological reference for MPS/MPO weight compression. No public code repo found — implementing from paper. |
| [2] Nature Comms QML | Liu, J. et al. (2024). Towards Provably Efficient Quantum Algorithms for Large-Scale Machine-Learning Models. Nature Communications, 15(1), 434. | doi:10.1038/s41467-023-43957-x | Theoretical motivation for quantum-inspired advantage in ML; cited in QI justification section |
| [3] Alpamayo-R1 | NVIDIA & Wang, Y. et al. (2025). Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail. | arXiv:2511.00088 | Context for VLAM landscape; AD track reference model (not our track) |
| [4] NVFP4 | NVIDIA Technical Blog (2025). Introducing NVFP4 for Efficient and Accurate Low-Precision Inference. | NVIDIA Developer Blog | Classical quantization context; counterpoint to TN compression |
| [5] Open X-Embodiment | Open X-Embodiment Collaboration (2023). Open X-Embodiment: Robotic Learning Datasets and RT-X Models. | arXiv:2310.08864 | Reference dataset for Robotics Compression sub-track |
| [6] nuScenes | Caesar, H. et al. (2020). nuScenes: A Multimodal Dataset for Autonomous Driving. CVPR 2020, pp. 11618-11628. | — | AD track dataset (not our track); cited for completeness per challenge statement |
| [7] π₀ | Black, K. et al. (2024). π₀: A Vision-Language-Action Flow Model for General Robot Control. | arXiv:2410.24164 | Alternative reference model for Robotics track; VLA landscape context |

---

## Datasets

| Dataset | Source | License | Usage |
|---|---|---|---|
| Open X-Embodiment (`bridge_dataset` subset) | https://robotics-transformer-x.github.io | Apache 2.0 | Held-out evaluation split (200 episodes); action-prediction accuracy metric |
| OpenVLA-7B model weights | https://huggingface.co/openvla/openvla-7b | **LLaMA-2 Community License — non-commercial research use only** | Reference model for compression; users must accept LLaMA-2 license on HuggingFace Hub before downloading |

---

## License Compatibility Summary

| Dependency | License | Permissive for research? |
|---|---|---|
| quimb | MIT | Yes |
| tntorch | MIT | Yes |
| bitsandbytes | MIT | Yes |
| HuggingFace Transformers | Apache 2.0 | Yes |
| HuggingFace Accelerate | Apache 2.0 | Yes |
| PennyLane | Apache 2.0 | Yes |
| MuJoCo | Apache 2.0 | Yes |
| dm_control | Apache 2.0 | Yes |
| Open X-Embodiment dataset | Apache 2.0 | Yes |
| PyTorch | BSD-style | Yes |
| OpenVLA-7B code | MIT | Yes |
| **OpenVLA-7B weights** | **LLaMA-2 Community License** | **Research only — non-commercial** |

All *code* dependencies are permissive. Model weights are research-use-only.
This is acceptable for competition submission but must be disclosed in README and report.

---

*Last updated: 2026-06-30 (Phase 0 — initial scouting)*
