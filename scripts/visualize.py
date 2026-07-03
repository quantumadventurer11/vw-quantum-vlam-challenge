"""Phase 4: MuJoCo 3D visualization.

Drives a dm_control humanoid with compressed OpenVLA-7B outputs and renders
a side-by-side video (INT8 baseline vs. TN-compressed) to results/.

Usage
-----
    python scripts/visualize.py --chi 64 --n-steps 200 --seed 42

Requires a Phase 2 checkpoint on Google Drive (or local checkpoints/).
Set MUJOCO_GL=egl on headless GPU machines (Colab uses this automatically).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def _build_predict_fn(model, processor, unnorm_key: str = "bridge_orig"):
    """Return a predict_fn(image, language) → np.ndarray[7] closure."""
    import torch

    def predict_fn(image, language) -> np.ndarray:
        inputs = processor(language, image).to("cuda:0")
        with torch.no_grad():
            action = model.predict_action(
                **inputs, unnorm_key=unnorm_key, do_sample=False
            )
        return np.array(action, dtype=np.float32)

    return predict_fn


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 MuJoCo visualization")
    parser.add_argument("--chi",           type=int,  default=64)
    parser.add_argument("--n-steps",       type=int,  default=200)
    parser.add_argument("--seed",          type=int,  default=42)
    parser.add_argument("--checkpoints",   type=str,  default="checkpoints")
    parser.add_argument("--results",       type=str,  default="results")
    parser.add_argument("--hf-dataset",   type=str,  default="lerobot/bridge")
    parser.add_argument("--unnorm-key",   type=str,  default="bridge_orig")
    parser.add_argument("--model-id",     type=str,  default="openvla/openvla-7b")
    parser.add_argument("--no-baseline",  action="store_true",
                        help="Skip INT8 baseline episode (faster)")
    args = parser.parse_args()

    from vlam_compress.mujoco_bridge import (
        run_episode, encode_video, make_side_by_side,
        make_overlay_fn, restore_patches, load_compressed_model,
        RENDER_FPS,
    )

    results_dir = Path(args.results)
    results_dir.mkdir(exist_ok=True)

    # ── Load dataset sample ───────────────────────────────────────────────────
    print("Loading dataset sample ...")
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    import json as _json

    hf_ds    = load_dataset(args.hf_dataset, split="train", streaming=True)
    task_map: dict = {}
    try:
        tf = hf_hub_download(args.hf_dataset, filename="meta/tasks.jsonl")
        with open(tf) as fh:
            for line in fh:
                d = _json.loads(line.strip())
                task_map[d["task_index"]] = d["task"]
    except Exception:
        pass

    from vlam_compress.mujoco_bridge import _ACTION_MAP  # noqa – just to confirm import
    from datasets import load_dataset  # already imported above

    np.random.seed(args.seed)
    sample_iter = iter(hf_ds.shuffle(seed=args.seed, buffer_size=1000).take(1))
    _raw        = next(sample_iter)

    # Import hf_item_to_sample — defined in phase notebooks; replicate minimally here
    def _to_sample(item):
        from PIL import Image as _PIL
        import io as _io
        img = None
        for k in ("observation.image", "observation.images.image_0", "image"):
            if k in item:
                img = item[k]; break
        if isinstance(img, dict):
            _b = img.get("bytes")
            if _b:
                img = _PIL.open(_io.BytesIO(_b))
        action = np.array(item["action"], dtype=np.float32).ravel()[:7]
        lang   = (item.get("language_instruction") or item.get("task") or "pick up the object")
        if isinstance(lang, (bytes, bytearray)):
            lang = lang.decode()
        return {"image": img, "language": str(lang), "action_gt": action}

    fixed_sample = _to_sample(_raw)
    get_sample_fn = lambda: fixed_sample

    # ── Load compressed model ─────────────────────────────────────────────────
    print(f"Loading chi={args.chi} compressed model ...")
    model, processor, saved = load_compressed_model(
        args.chi, args.checkpoints, model_id=args.model_id
    )
    predict_compressed = _build_predict_fn(model, processor, args.unnorm_key)

    # ── Run compressed episode ────────────────────────────────────────────────
    print(f"Running compressed episode ({args.n_steps} steps) ...")
    result_c = run_episode(
        predict_fn=predict_compressed,
        get_sample_fn=get_sample_fn,
        n_steps=args.n_steps,
        random_state=args.seed,
    )
    print(f"  Rendered {result_c['n_frames']} frames.")

    # ── Run INT8 baseline episode ─────────────────────────────────────────────
    result_b = None
    if not args.no_baseline:
        print("Restoring INT8 weights for baseline episode ...")
        restore_patches(saved)
        predict_baseline = _build_predict_fn(model, processor, args.unnorm_key)

        print(f"Running INT8 baseline episode ({args.n_steps} steps) ...")
        result_b = run_episode(
            predict_fn=predict_baseline,
            get_sample_fn=get_sample_fn,
            n_steps=args.n_steps,
            random_state=args.seed,
        )
        print(f"  Rendered {result_b['n_frames']} frames.")

    # ── Load compression ratio from sweep stats ───────────────────────────────
    cr = None
    stats_path = results_dir / "compression_sweep_stats.json"
    if stats_path.exists():
        with open(stats_path) as fh:
            stats = json.load(fh)
        cr = stats.get("sweep_stats", {}).get(str(args.chi), {}).get(
            "layer_compression_ratio_mean"
        )

    # ── Encode videos ─────────────────────────────────────────────────────────
    overlay_c = make_overlay_fn(
        label=f"TN chi={args.chi}",
        language=result_c["language"],
        arm_traj=result_c["arm_trajectory"],
        compression_ratio=cr,
    )

    out_c = results_dir / f"demo_compressed_chi{args.chi}.mp4"
    encode_video(result_c["frames"], out_c, fps=RENDER_FPS, overlay_fn=overlay_c)
    print(f"Written: {out_c}  (+ .gif)")

    if result_b is not None:
        overlay_b = make_overlay_fn(
            label="INT8 baseline",
            language=result_b["language"],
            arm_traj=result_b["arm_trajectory"],
        )
        sbs_frames = make_side_by_side(
            result_b["frames"], result_c["frames"],
            label_a="INT8 baseline",
            label_b=f"TN chi={args.chi}",
        )
        out_sbs = results_dir / f"demo_side_by_side_chi{args.chi}.mp4"
        encode_video(sbs_frames, out_sbs, fps=RENDER_FPS)
        print(f"Written: {out_sbs}  (+ .gif)")

    print("Done.")


if __name__ == "__main__":
    main()
