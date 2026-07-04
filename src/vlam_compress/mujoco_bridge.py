"""MuJoCo bridge: maps compressed OpenVLA-7B action outputs to humanoid joints.

Architecture
------------
OXE sample (image + language)
    → Compressed OpenVLA-7B (χ=64)
    → 7-DoF delta action [dx, dy, dz, droll, dpitch, dyaw, gripper]
    → action_to_joint_command()
    → 21-DoF humanoid ctrl (right arm driven; lower body PD-stabilised)
    → dm_control humanoid physics
    → rendered frames → MP4 + GIF

Humanoid actuator layout (dm_control humanoid.xml, 21 total)
-------------------------------------------------------------
 0  abdomen_y       9  left_hip_x     15  right_shoulder1
 1  abdomen_z      10  left_hip_z     16  right_shoulder2
 2  abdomen_x      11  left_hip_y     17  right_elbow
 3  right_hip_x    12  left_knee      18  left_shoulder1
 4  right_hip_z    13  left_ankle_x   19  left_shoulder2
 5  right_hip_y    14  left_ankle_y   20  left_elbow
 6  right_knee
 7  right_ankle_x
 8  right_ankle_y
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:
    import dm_control.suite as suite
    HAS_DM_CONTROL = True
except ImportError:
    HAS_DM_CONTROL = False

# ── Humanoid joint constants ──────────────────────────────────────────────────

ACTUATOR_COUNT  = 21
RIGHT_SHOULDER1 = 15
RIGHT_SHOULDER2 = 16
RIGHT_ELBOW     = 17
RIGHT_ARM_IDXS  = (RIGHT_SHOULDER1, RIGHT_SHOULDER2, RIGHT_ELBOW)

# ── 7-DoF → 3 right-arm-joint linear mapping ─────────────────────────────────
# Rows = OpenVLA dims [dx, dy, dz, droll, dpitch, dyaw, gripper]
# Cols = arm joints   [shoulder1, shoulder2, elbow]
_ACTION_MAP = np.array([
    [1.0, 0.0, 0.0],   # dx      → shoulder1 (coronal rotation)
    [0.0, 1.0, 0.0],   # dy      → shoulder2 (sagittal rotation)
    [0.0, 0.0, 1.0],   # dz      → elbow flex/extend
    [0.4, 0.0, 0.0],   # droll   → shoulder1 (secondary)
    [0.0, 0.4, 0.0],   # dpitch  → shoulder2 (secondary)
    [0.2, 0.0, 0.3],   # dyaw    → shoulder1 + elbow (secondary)
    [0.0, 0.0, 0.0],   # gripper → no hand DoF in standard humanoid
], dtype=np.float32)   # (7, 3)

ACTION_SCALE = 0.15    # scale applied to delta before integration
ARM_CLIP     = 1.0     # actuator control range [-1, 1]
PD_KP        = 0.8     # lower-body PD proportional gain

# ── Render constants ──────────────────────────────────────────────────────────

RENDER_WIDTH    = 480
RENDER_HEIGHT   = 480
SIM_FREQ        = 500   # dm_control humanoid default physics frequency (Hz)
RENDER_FPS      = 25
STEPS_PER_FRAME = SIM_FREQ // RENDER_FPS   # 20 physics steps per rendered frame


# ── Environment ───────────────────────────────────────────────────────────────

def make_env(random_state: int = 0):
    """Create a dm_control humanoid stand environment."""
    if not HAS_DM_CONTROL:
        raise ImportError(
            "dm_control not installed.\n"
            "Run: pip install dm-control mujoco\n"
            "Set MUJOCO_GL=egl (Colab GPU) or MUJOCO_GL=osmesa (headless CPU)."
        )
    return suite.load("humanoid", "stand", task_kwargs={"random": random_state})


# ── Action mapping ────────────────────────────────────────────────────────────

def action_to_joint_command(
    action: np.ndarray,
    arm_state: np.ndarray,
) -> np.ndarray:
    """
    Map a 7-DoF OpenVLA delta action to a 21-DoF humanoid control signal.

    Integrates deltas into ``arm_state`` (modified in place) so the arm
    accumulates position over the episode. Lower-body actuators are zero
    here; call ``pd_lower_body`` separately and fill ctrl[:15].

    Parameters
    ----------
    action    : (7,) array  — [dx, dy, dz, droll, dpitch, dyaw, gripper]
    arm_state : (3,) array  — mutable integrated state for [shoulder1, shoulder2, elbow]

    Returns
    -------
    ctrl : (21,) float64 array with arm joints filled; lower body zeros
    """
    action = np.asarray(action, dtype=np.float32).ravel()[:7]
    delta  = (action @ _ACTION_MAP) * ACTION_SCALE       # (3,)
    arm_state[:] = np.clip(arm_state + delta, -ARM_CLIP, ARM_CLIP)

    ctrl = np.zeros(ACTUATOR_COUNT, dtype=np.float64)
    for idx, val in zip(RIGHT_ARM_IDXS, arm_state):
        ctrl[idx] = float(val)
    return ctrl


def pd_lower_body(physics, standing_target: np.ndarray) -> np.ndarray:
    """
    Proportional controller for the 15 lower-body actuators (indices 0-14).
    Drives toward ``standing_target`` (captured once after env.reset()).
    Returns a (15,) clipped control array.
    """
    current = np.array(physics.data.actuator_length[:15], dtype=np.float64)
    return np.clip(PD_KP * (standing_target - current), -1.0, 1.0)


# ── Episode rollout ───────────────────────────────────────────────────────────

def run_episode(
    predict_fn: Callable,
    get_sample_fn: Callable,
    n_steps: int = 200,
    random_state: int = 0,
    render_width: int = RENDER_WIDTH,
    render_height: int = RENDER_HEIGHT,
    camera_id: int = 1,
) -> dict:
    """
    Run a humanoid episode driven by ``predict_fn`` and return rendered frames.

    Parameters
    ----------
    predict_fn    : callable(image, language) → np.ndarray[7]
                    Wraps model.predict_action — runs on GPU.
    get_sample_fn : callable() → {"image": PIL.Image, "language": str}
                    Returns one evaluation frame from the dataset.
    n_steps       : number of physics+render cycles (200 steps ≈ 8 s at 25 fps).
    random_state  : seed for dm_control env initialisation.
    render_width  : rendered frame width in pixels.
    render_height : rendered frame height in pixels.
    camera_id     : dm_control camera index (1 = side view for humanoid).

    Returns
    -------
    dict with keys:
        frames         — list of H×W×3 uint8 arrays (one per rendered frame)
        actions        — list of 7-element lists
        arm_trajectory — list of 3-element lists (shoulder1, shoulder2, elbow)
        language       — instruction string used for this episode
        n_frames       — len(frames)
    """
    env = make_env(random_state)
    env.reset()
    physics = env.physics

    standing_target = np.array(physics.data.actuator_length[:15], dtype=np.float64)

    arm_state  = np.zeros(3, dtype=np.float32)
    frames:    list[np.ndarray] = []
    actions:   list[list]       = []
    arm_traj:  list[list]       = []

    sample   = get_sample_fn()
    image    = sample["image"]
    language = sample["language"]

    for step in range(n_steps):
        action = np.asarray(predict_fn(image, language), dtype=np.float32)

        ctrl       = action_to_joint_command(action, arm_state)
        ctrl[:15]  = pd_lower_body(physics, standing_target)
        env.step(ctrl)

        if step % STEPS_PER_FRAME == 0:
            frame = physics.render(
                height=render_height, width=render_width, camera_id=camera_id,
            )
            frames.append(frame)

        actions.append(action.tolist())
        arm_traj.append(arm_state.copy().tolist())

    return {
        "frames":         frames,
        "actions":        actions,
        "arm_trajectory": arm_traj,
        "language":       language,
        "n_frames":       len(frames),
    }


# ── Overlay rendering ─────────────────────────────────────────────────────────

def add_overlay(
    frame: np.ndarray,
    lines: list[str],
    origin: tuple[int, int] = (8, 8),
    line_height: int = 17,
    color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Draw text lines onto a rendered frame using PIL. Returns a new uint8 array."""
    from PIL import Image, ImageDraw
    img  = Image.fromarray(frame.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    x, y = origin
    for line in lines:
        draw.text((x, y), line, fill=color)
        y += line_height
    return np.array(img)


def make_overlay_fn(
    label: str,
    language: str,
    arm_traj: list[list],
    compression_ratio: Optional[float] = None,
) -> Callable:
    """Return a per-frame overlay callable suitable for ``encode_video``."""
    cr_str  = f"CR={compression_ratio:.1f}x" if compression_ratio else ""
    lang_str = f'"{language[:38]}"' if len(language) > 38 else f'"{language}"'

    def _overlay(frame_idx: int, frame: np.ndarray) -> np.ndarray:
        traj_idx = min(frame_idx * STEPS_PER_FRAME, len(arm_traj) - 1)
        arm = arm_traj[traj_idx]
        lines = [label, lang_str, f"arm [{arm[0]:+.2f} {arm[1]:+.2f} {arm[2]:+.2f}]"]
        if cr_str:
            lines.append(cr_str)
        return add_overlay(frame, lines)

    return _overlay


# ── Video encoding ────────────────────────────────────────────────────────────

def encode_video(
    frames: list[np.ndarray],
    output_path: str | Path,
    fps: int = RENDER_FPS,
    overlay_fn: Optional[Callable] = None,
) -> Path:
    """
    Encode frames to MP4 (H.264) and GIF using imageio.

    Requires: ``pip install imageio[ffmpeg]``

    Parameters
    ----------
    frames      : list of H×W×3 uint8 arrays
    output_path : destination .mp4 path; GIF written to same stem + .gif
    fps         : frames per second for MP4
    overlay_fn  : optional callable(frame_idx, frame) → annotated frame

    Returns
    -------
    Path to the written MP4 file
    """
    import imageio.v3 as iio

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    annotated = (
        [overlay_fn(i, f) for i, f in enumerate(frames)]
        if overlay_fn is not None
        else list(frames)
    )

    iio.imwrite(str(output_path), annotated, fps=fps, codec="libx264")

    gif_path = output_path.with_suffix(".gif")
    iio.imwrite(str(gif_path), annotated[::2], duration=1000 // (fps // 2))

    return output_path


def make_side_by_side(
    frames_a: list[np.ndarray],
    frames_b: list[np.ndarray],
    label_a: str = "INT8 baseline",
    label_b: str = "TN chi=64",
    divider_px: int = 4,
) -> list[np.ndarray]:
    """
    Horizontally concatenate two frame sequences with a thin divider.
    Truncates to the shorter sequence length.
    """
    n   = min(len(frames_a), len(frames_b))
    h   = frames_a[0].shape[0]
    div = np.full((h, divider_px, 3), 200, dtype=np.uint8)

    out = []
    for i in range(n):
        fa = add_overlay(frames_a[i], [label_a], color=(220, 220, 80))
        fb = add_overlay(frames_b[i], [label_b], color=(80, 220, 220))
        out.append(np.concatenate([fa, div, fb], axis=1))
    return out


# ── Checkpoint loading ────────────────────────────────────────────────────────

def load_compressed_model(
    chi: int,
    checkpoints_base: str | Path,
    model_id: str = "openvla/openvla-7b",
):
    """
    Load OpenVLA-7B in INT8 and patch target layers with χ-compressed weights.

    Parameters
    ----------
    chi              : bond dimension of the checkpoint to load
    checkpoints_base : directory containing compressed_chi{chi}/cores.pt
                       (Google Drive path set in Phase 2/3 notebooks)
    model_id         : HuggingFace model ID

    Returns
    -------
    (model, processor, saved_modules)
    Pass ``saved_modules`` to ``restore_patches()`` to undo compression.
    """
    import torch
    import torch.nn as nn
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
    from vlam_compress.compress import mps_reconstruct

    cores_path = Path(checkpoints_base) / f"compressed_chi{chi}" / "cores.pt"
    if not cores_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {cores_path}\n"
            "Run Phase 2 in Colab first to generate cores."
        )

    print(f"Loading processor ...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    print("Loading model in INT8 ...")
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        attn_implementation="eager",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print(f"Loading chi={chi} cores from {cores_path} ...")
    cores_dict  = torch.load(cores_path, map_location="cpu")
    all_modules = dict(model.named_modules())
    saved: dict = {}

    with torch.no_grad():
        for layer_name, layer_cores in cores_dict.items():
            if "." not in layer_name:
                continue
            parent_name, child_name = layer_name.rsplit(".", 1)
            parent   = all_modules.get(parent_name)
            orig_mod = getattr(parent, child_name, None) if parent else None
            if orig_mod is None or not hasattr(orig_mod, "weight"):
                continue

            device = orig_mod.weight.device
            W_hat  = mps_reconstruct(
                [c.to(device, dtype=torch.float32) for c in layer_cores],
                tuple(orig_mod.weight.shape),
            ).to(torch.float16)

            has_bias = getattr(orig_mod, "bias", None) is not None
            new_mod  = nn.Linear(
                W_hat.shape[1], W_hat.shape[0],
                bias=has_bias, device=device, dtype=torch.float16,
            )
            new_mod.weight = nn.Parameter(W_hat)
            if has_bias:
                new_mod.bias = nn.Parameter(orig_mod.bias.data.to(torch.float16))

            saved[layer_name] = (parent, child_name, orig_mod)
            setattr(parent, child_name, new_mod)

    print(f"Patched {len(saved)} layers with chi={chi} weights.")
    return model, processor, saved


def restore_patches(saved: dict) -> None:
    """Restore original INT8 modules replaced by ``load_compressed_model``."""
    for _, (parent, child_name, orig_mod) in saved.items():
        setattr(parent, child_name, orig_mod)
