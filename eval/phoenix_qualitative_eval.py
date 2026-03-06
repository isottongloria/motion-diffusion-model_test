#!/usr/bin/env python3
"""Qualitative Phoenix evaluation: angles -> SMPL-X joints -> 3-panel MP4.

Pipeline (kept intentionally rigid to match training/evaluation alignment):
1) Load GT angles and predicted angles (from results.npy['motion_emb']).
2) Normalize feature dim to 169 (133 -> prepend 36 zeros, 172 -> crop to 169).
3) Slice fixed SMPL-X blocks and run neutral SMPL-X forward.
4) Render GT / prediction / overlap as a short video.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.animation import FFMpegWriter, FuncAnimation

from mGPT.utils.human_models import smpl_x

FIXED_SHAPE = np.array(
    [-0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172,
     0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842],
    dtype=np.float32,
)

LOWER_BODY_INDICES = [1, 2, 3, 4, 5, 6, 14, 15, 16, 17, 18, 19]
VISIBLE_NO_LOWER_BODY_INDICES = [idx for idx in range(137) if idx not in set(LOWER_BODY_INDICES)]
LHAND_INDICES = list(range(25, 45))
RHAND_INDICES = list(range(45, 65))


def _unwrap_npy_object(obj: Any) -> Any:
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        if obj.shape == ():
            return obj.item()
        if obj.size == 1:
            return obj.reshape(-1)[0]
    return obj


def load_gloss_timeline(csv_path: Path, nframes: int) -> list[str]:
    df = pd.read_csv(csv_path)
    if "text" not in df.columns or "length" not in df.columns:
        raise ValueError(f"{csv_path} must contain columns: text,length")

    labels = [""] * nframes
    cur = 0
    for _, row in df.iterrows():
        g = str(row["text"]).strip()
        ln = int(row["length"])
        if ln <= 0:
            continue
        end = min(cur + ln, nframes)
        for t in range(cur, end):
            labels[t] = g
        cur = end
        if cur >= nframes:
            break
    return labels


def angles_from_results_file(results_path: str, sample_idx: int = 0) -> np.ndarray:
    """Load results.npy dict and return one sample in [T, D]."""
    raw = np.load(results_path, allow_pickle=True)
    data = _unwrap_npy_object(raw)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in .npy, found: {type(data)}")

    if "motion_emb" not in data:
        raise KeyError(f"Missing 'motion_emb' in results file. Keys: {list(data.keys())}")

    angles = np.asarray(data["motion_emb"])  # required key per qualitative-eval spec
    angles = np.squeeze(angles)

    if angles.ndim == 4:
        angles = np.squeeze(angles[sample_idx])
    elif angles.ndim == 3:
        angles = np.squeeze(angles[sample_idx])
    elif angles.ndim != 2:
        raise ValueError(f"Expected 2/3/4 dims, got shape {angles.shape}")

    if angles.shape[1] in (133, 169, 172):
        td = angles
    elif angles.shape[0] in (133, 169, 172):
        td = angles.T
    else:
        raise ValueError(
            f"Unsupported 2D angle shape {angles.shape}: cannot locate D in {{133,169,172}}"
        )
    return np.asarray(td, dtype=np.float32)


def normalize_angles_dim(angles: np.ndarray) -> np.ndarray:
    if angles.ndim != 2:
        raise ValueError(f"Expected [T,D], got {angles.shape}")

    t, d = angles.shape
    if d == 133:
        return np.concatenate([np.zeros((t, 36), dtype=np.float32), angles], axis=1)
    if d == 169:
        return angles
    if d == 172:
        return angles[:, :169]
    raise ValueError(f"Unsupported feature dim {d}. Expected one of 133, 169, 172")


def to_smplx_joints(features: np.ndarray, device: torch.device) -> np.ndarray:
    """Convert [T,133|169|172] to [T,137,3] with fixed slicing and neutral SMPL-X."""
    full_169 = normalize_angles_dim(features).astype(np.float32)
    t = full_169.shape[0]

    root_pose = torch.from_numpy(full_169[:, 0:3]).to(device)
    body_pose = torch.from_numpy(full_169[:, 3:66]).to(device)
    lhand_pose = torch.from_numpy(full_169[:, 66:111]).to(device)
    rhand_pose = torch.from_numpy(full_169[:, 111:156]).to(device)
    jaw_pose = torch.from_numpy(full_169[:, 156:159]).to(device)
    expr = torch.from_numpy(full_169[:, 159:169]).to(device)

    betas = torch.from_numpy(FIXED_SHAPE).to(device).unsqueeze(0).repeat(t, 1)
    eye_pose = torch.zeros((t, 3), dtype=torch.float32, device=device)

    smplx_layer = copy.deepcopy(smpl_x.layer["neutral"]).to(device)
    smplx_layer.eval()

    with torch.no_grad():
        out = smplx_layer(
            betas=betas,
            body_pose=body_pose,
            global_orient=root_pose,
            right_hand_pose=rhand_pose,
            left_hand_pose=lhand_pose,
            jaw_pose=jaw_pose,
            leye_pose=eye_pose,
            reye_pose=eye_pose,
            expression=expr,
        )
        joints = out.joints[:, smpl_x.joint_idx, :]

    return joints.detach().cpu().numpy()


def build_joint_colors(njoints: int = 137) -> np.ndarray:
    colors = np.array([[0.25, 0.25, 0.25]] * njoints)
    colors[LHAND_INDICES] = np.array([0.10, 0.35, 0.95])
    colors[RHAND_INDICES] = np.array([0.90, 0.10, 0.10])
    return colors


def prepare_data(joints: np.ndarray, hide_lower_body: bool) -> tuple[np.ndarray, np.ndarray]:
    data = joints.copy()
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    visible_joint_mask = np.ones(data.shape[1], dtype=bool)
    if hide_lower_body:
        visible_joint_mask[:] = False
        visible_joint_mask[np.array(VISIBLE_NO_LOWER_BODY_INDICES, dtype=int)] = True

    return data, visible_joint_mask


def save_comparison_mp4(
    gt_joints: np.ndarray,
    pred_joints: np.ndarray,
    output_mp4: Path,
    fps: int = 20,
    title: str | None = None,
    hide_lower_body: bool = True,
    gloss_labels: list[str] | None = None,
    metrics_text: str | None = None,
):
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    gt_data, vis_mask = prepare_data(gt_joints, hide_lower_body=hide_lower_body)
    pred_data, _ = prepare_data(pred_joints, hide_lower_body=hide_lower_body)

    nframes = min(gt_data.shape[0], pred_data.shape[0])
    gt_visible = gt_data[:nframes, vis_mask, :]
    pred_visible = pred_data[:nframes, vis_mask, :]

    both = gt_visible
    mins = both.reshape(-1, 3).min(axis=0)
    maxs = both.reshape(-1, 3).max(axis=0)
    center = (mins + maxs) / 2.0
    radius = np.max(maxs - mins) * 0.5 + 1e-6

    all_colors = build_joint_colors(gt_data.shape[1])
    vis_colors = all_colors[vis_mask]

    fig = plt.figure(figsize=(18, 6), dpi=100)
    gloss_text = fig.text(
        0.5, 0.03, "",
        ha="center", va="center",
        fontsize=14, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="black", alpha=0.65, edgecolor="none"),
    )
    metric_overlay = fig.text(
        0.98, 0.96, metrics_text or "",
        ha="right", va="top",
        fontsize=10, color="white",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="black", alpha=0.65, edgecolor="none"),
    )
    axs = [
        fig.add_subplot(1, 3, 1, projection="3d"),
        fig.add_subplot(1, 3, 2, projection="3d"),
        fig.add_subplot(1, 3, 3, projection="3d"),
    ]

    def style_ax(ax, subtitle):
        ax.view_init(elev=110, azim=-90)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.grid(True, alpha=0.25)
        ax.set_title(subtitle)

    def update(frame_idx: int):
        for ax in axs:
            ax.clear()

        style_ax(axs[0], "Ground Truth")
        axs[0].scatter(
            gt_visible[frame_idx, :, 0],
            gt_visible[frame_idx, :, 1],
            gt_visible[frame_idx, :, 2],
            s=7,
            c=vis_colors,
            alpha=0.95,
        )

        style_ax(axs[1], "Model Output")
        axs[1].scatter(
            pred_visible[frame_idx, :, 0],
            pred_visible[frame_idx, :, 1],
            pred_visible[frame_idx, :, 2],
            s=7,
            c=vis_colors,
            alpha=0.95,
        )

        style_ax(axs[2], "Overlap")
        axs[2].scatter(
            gt_visible[frame_idx, :, 0],
            gt_visible[frame_idx, :, 1],
            gt_visible[frame_idx, :, 2],
            s=7,
            c="black",
            alpha=0.70,
            label="GT",
        )
        axs[2].scatter(
            pred_visible[frame_idx, :, 0],
            pred_visible[frame_idx, :, 1],
            pred_visible[frame_idx, :, 2],
            s=7,
            c="limegreen",
            alpha=0.55,
            label="Pred",
        )
        axs[2].legend(loc="upper right")

        if title:
            fig.suptitle(f"{title} | frame {frame_idx + 1}/{nframes}")
        if gloss_labels is not None and frame_idx < len(gloss_labels):
            g = gloss_labels[frame_idx] if gloss_labels[frame_idx] else "<no gloss>"
            gloss_text.set_text(f"Gloss: {g}")
        else:
            gloss_text.set_text("")
        metric_overlay.set_text(metrics_text or "")

    anim = FuncAnimation(fig, update, frames=nframes, interval=1000 / max(fps, 1), repeat=False)
    writer = FFMpegWriter(fps=fps, bitrate=3500)
    anim.save(str(output_mp4), writer=writer)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qualitative Phoenix eval: GT vs prediction MP4.")
    parser.add_argument("--gt-file", type=Path, required=True, help="Ground-truth .npy angles.")
    parser.add_argument("--results-file", type=Path, required=True, help="results.npy containing dict['motion_emb'].")
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index for batched result arrays.")
    parser.add_argument("--output-mp4", type=Path, required=True, help="Output MP4 path.")
    parser.add_argument("--fps", type=int, default=20, help="Output FPS.")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"), choices=["cpu", "cuda"])
    parser.add_argument("--ffmpeg-path", type=str, default="", help="Optional explicit ffmpeg path.")
    parser.add_argument("--show-full-body", action="store_true", help="Show lower body (default hidden).")
    parser.add_argument("--input-gt-local-txt", type=Path, default=None, help="Optional CSV with text,length columns.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.ffmpeg_path:
        plt.rcParams["animation.ffmpeg_path"] = args.ffmpeg_path

    device = torch.device(args.device)

    gt_angles = np.asarray(np.load(args.gt_file), dtype=np.float32)
    pred_angles = angles_from_results_file(str(args.results_file), sample_idx=args.sample_idx)

    gt_joints = to_smplx_joints(gt_angles, device=device)
    pred_joints = to_smplx_joints(pred_angles, device=device)

    nframes_vis = min(gt_joints.shape[0], pred_joints.shape[0])
    gloss_labels = None
    if args.input_gt_local_txt is not None:
        gloss_labels = load_gloss_timeline(args.input_gt_local_txt, nframes=nframes_vis)

    save_comparison_mp4(
        gt_joints=gt_joints,
        pred_joints=pred_joints,
        output_mp4=args.output_mp4,
        fps=args.fps,
        title=args.gt_file.stem,
        hide_lower_body=not args.show_full_body,
        gloss_labels=gloss_labels,
    )

    print(f"Saved MP4: {args.output_mp4}")


if __name__ == "__main__":
    main()
