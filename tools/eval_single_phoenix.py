#!/usr/bin/env python3
"""Single-file Phoenix pipeline.

Given one text prompt file and one model checkpoint:
1) run sample.generate (1 sample),
2) load prediction,
3) convert rot6d -> axis-angle when needed,
4) convert angles -> xyz (SMPL-X),
5) compute Phoenix DTW metrics,
6) render MP4 with metrics overlay (top-right).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eval.eval_phoenix import compute_part_dtw_metrics, smpl_x
from eval.phoenix_qualitative_eval import (
    _unwrap_npy_object,
    angles_from_results_file,
    normalize_angles_dim,
    save_comparison_mp4,
    to_smplx_joints,
)
from utils.rotation_conversions import matrix_to_axis_angle, rotation_6d_to_matrix


def _infer_results_dir(output_dir: Path, input_text: Path) -> Path:
    stem = input_text.name.replace('.txt', '').replace(' ', '_').replace('.', '')
    return Path(f"{str(output_dir)}_{stem}")


def _rot6d_motion_to_angles(results_dict: dict[str, Any], sample_idx: int) -> np.ndarray:
    if "motion" not in results_dict:
        raise KeyError("results.npy does not contain 'motion' and no 'motion_emb' was found.")
    motion = np.asarray(results_dict["motion"])
    if motion.ndim != 4:
        raise ValueError(f"Expected results['motion'] shape [B,J,6,T], got {motion.shape}")

    if sample_idx >= motion.shape[0]:
        raise IndexError(f"sample_idx={sample_idx} out of range for B={motion.shape[0]}")

    sample = motion[sample_idx]  # [J,6,T]
    if sample.shape[1] != 6:
        raise ValueError(f"Expected rot6d features on axis=1, got shape {sample.shape}")

    sample_tj6 = np.transpose(sample, (2, 0, 1)).astype(np.float32)  # [T,J,6]
    d6 = torch.from_numpy(sample_tj6)
    rot_m = rotation_6d_to_matrix(d6)          # [T,J,3,3]
    aa = matrix_to_axis_angle(rot_m).numpy()   # [T,J,3]
    flat = aa.reshape(aa.shape[0], -1)

    # Match Phoenix accepted dims exactly.
    if flat.shape[1] == 172:
        return flat
    if flat.shape[1] == 169:
        return flat
    if flat.shape[1] == 133:
        return flat

    raise ValueError(
        f"Converted rot6d->axis-angle has unsupported dim {flat.shape[1]} (expected 133/169/172)."
    )


def _load_pred_angles(results_file: Path, sample_idx: int) -> np.ndarray:
    raw = np.load(results_file, allow_pickle=True)
    data = _unwrap_npy_object(raw)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in {results_file}, got {type(data)}")

    if "motion_emb" in data:
        return angles_from_results_file(str(results_file), sample_idx=sample_idx)

    return _rot6d_motion_to_angles(data, sample_idx)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-file Phoenix eval with MP4 metrics overlay.")
    p.add_argument("--model-path", type=Path, required=True, help="Checkpoint model path.")
    p.add_argument("--input-text", type=Path, default=Path("annotations/training/sample_single.txt"),
                   help="Single-text file to translate.")
    p.add_argument("--gt-file", type=Path, required=True, help="Ground-truth angles .npy for metric computation.")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory root.")
    p.add_argument("--output-mp4", type=Path, default=None, help="Optional explicit mp4 output path.")
    p.add_argument("--sample-idx", type=int, default=0)
    p.add_argument("--guidance-param", type=float, default=2.5)
    p.add_argument("--motion-length", type=float, default=6.0)
    p.add_argument("--device", choices=["cpu", "cuda"], default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--fps", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "sample.generate",
        "--model_path", str(args.model_path),
        "--output_dir", str(args.output_dir),
        "--input_text", str(args.input_text),
        "--num_repetitions", "1",
        "--guidance_param", str(args.guidance_param),
        "--motion_length", str(args.motion_length),
        "--device", args.device,
    ]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    results_dir = _infer_results_dir(args.output_dir, args.input_text)
    results_file = results_dir / "results.npy"
    if not results_file.exists():
        raise FileNotFoundError(f"Could not find generated results file: {results_file}")

    device = torch.device(args.device)
    gt_angles = np.asarray(np.load(args.gt_file), dtype=np.float32)
    pred_angles = _load_pred_angles(results_file, sample_idx=args.sample_idx)

    gt_xyz = to_smplx_joints(normalize_angles_dim(gt_angles), device=device)
    pred_xyz = to_smplx_joints(normalize_angles_dim(pred_angles), device=device)

    t = min(len(gt_xyz), len(pred_xyz))
    if t == 0:
        raise ValueError("Empty sequence after conversion")
    gt_eval, pred_eval = gt_xyz[:t], pred_xyz[:t]

    body_idx = list(smpl_x.joint_part2idx["upper_body"])
    lhand_idx = list(smpl_x.joint_part2idx["lhand"])
    rhand_idx = list(smpl_x.joint_part2idx["rhand"])

    m_body = compute_part_dtw_metrics(pred_eval, gt_eval, body_idx)
    m_lhand = compute_part_dtw_metrics(pred_eval, gt_eval, lhand_idx)
    m_rhand = compute_part_dtw_metrics(pred_eval, gt_eval, rhand_idx)

    metrics = {
        "frames_used": int(t),
        "body": m_body,
        "lhand": m_lhand,
        "rhand": m_rhand,
    }

    metrics_text = (
        f"body JPE {m_body['DTW_JPE']:.3f} | PA {m_body['DTW_PA_JPE']:.3f}\n"
        f"lhand JPE {m_lhand['DTW_JPE']:.3f} | PA {m_lhand['DTW_PA_JPE']:.3f}\n"
        f"rhand JPE {m_rhand['DTW_JPE']:.3f} | PA {m_rhand['DTW_PA_JPE']:.3f}"
    )

    output_mp4 = args.output_mp4 or (args.output_dir / "single_eval.mp4")
    save_comparison_mp4(
        gt_joints=gt_eval,
        pred_joints=pred_eval,
        output_mp4=output_mp4,
        fps=args.fps,
        title=args.input_text.stem,
        hide_lower_body=True,
        metrics_text=metrics_text,
    )

    metrics_json = args.output_dir / "single_eval_metrics.json"
    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved video: {output_mp4}")
    print(f"Saved metrics: {metrics_json}")


if __name__ == "__main__":
    main()
