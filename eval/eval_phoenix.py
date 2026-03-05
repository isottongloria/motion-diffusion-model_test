#!/usr/bin/env python3
"""Evaluate Phoenix-style DTW-JPE metrics from xyz joint coordinates."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


def _ensure_mgpt_parent_on_path() -> Path:
    if importlib.util.find_spec("mGPT") is not None:
        spec = importlib.util.find_spec("mGPT")
        if spec is not None and spec.submodule_search_locations:
            return Path(next(iter(spec.submodule_search_locations))).parent

    env_parent_raw = os.environ.get("MGPT_PARENT_PATH")
    env_parent = Path(env_parent_raw) if env_parent_raw else None
    candidates = []
    if env_parent is not None:
        candidates.append(env_parent)

    script_parent = Path(__file__).resolve().parent
    candidates.extend([
        script_parent / "visualize",
        script_parent.parent / "visualize",
        Path.cwd() / "visualize",
        script_parent,
    ])

    for parent in candidates:
        if (parent / "mGPT").is_dir():
            sys.path.insert(0, str(parent))
            return parent

    raise ModuleNotFoundError(
        "Could not locate 'mGPT'. Set MGPT_PARENT_PATH to the directory containing mGPT/ "
        "or run with PYTHONPATH pointing to that directory (e.g., PYTHONPATH=./visualize)."
    )


def _import_soke_metric_functions():
    _ensure_mgpt_parent_on_path()

    from mGPT.utils.human_models import smpl_x

    dtw_path = Path(__file__).resolve().parent
    local_candidate = dtw_path / "mGPT" / "metrics" / "dtw.py"
    if local_candidate.is_file():
        dtw_file = local_candidate
    else:
        m = importlib.util.find_spec("mGPT")
        assert m is not None and m.submodule_search_locations is not None
        mgpt_dir = Path(next(iter(m.submodule_search_locations)))
        dtw_file = mgpt_dir / "metrics" / "dtw.py"

    spec = importlib.util.spec_from_file_location("soke_dtw_module", dtw_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load SOKE dtw module from: {dtw_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.dtw, module.l2_dist_align, smpl_x


dtw, l2_dist_align, smpl_x = _import_soke_metric_functions()


def _unwrap_npy_object(obj: Any) -> Any:
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        if obj.shape == ():
            return obj.item()
        if obj.size == 1:
            return obj.reshape(-1)[0]
    return obj


def _parse_indices(text: str) -> list[int]:
    text = text.strip()
    if text.startswith("["):
        values = json.loads(text)
        return [int(v) for v in values]
    if not text:
        return []
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def load_xyz(path: Path, key: str | None = None, sample_idx: int = 0) -> np.ndarray:
    raw = np.load(path, allow_pickle=True)
    data = _unwrap_npy_object(raw)

    if isinstance(data, dict):
        if key is None:
            if "xyz" in data:
                arr = np.asarray(data["xyz"])
            else:
                raise KeyError(
                    f"{path} contains a dict. Pass --key. Available keys: {list(data.keys())}"
                )
        else:
            if key not in data:
                raise KeyError(f"Key '{key}' not found in {path}. Keys: {list(data.keys())}")
            arr = np.asarray(data[key])
    else:
        arr = np.asarray(data)

    arr = np.squeeze(arr)

    if arr.ndim == 4:
        arr = arr[sample_idx]
    elif arr.ndim != 3:
        raise ValueError(f"Expected 3D/4D input, got shape {arr.shape}")

    if arr.shape[-1] != 3:
        raise ValueError(f"Expected last dim=3 for xyz, got shape {arr.shape}")

    return arr.astype(np.float32)


def _validate_indices(indices: list[int], njoints: int, part_name: str) -> None:
    if not indices:
        raise ValueError(f"{part_name} indices are empty")
    if min(indices) < 0 or max(indices) >= njoints:
        raise ValueError(
            f"{part_name} indices out of range for J={njoints}: min={min(indices)}, max={max(indices)}"
        )


def compute_part_dtw_metrics(pred_xyz: np.ndarray, gt_xyz: np.ndarray, wanted_indices: list[int]) -> dict[str, float]:
    dist_jpe = lambda x, y: l2_dist_align(x, y, wanted=wanted_indices, align_idx=0)
    dtw_jpe = float(dtw(pred_xyz, gt_xyz, dist_jpe)[0])

    dist_pa = lambda x, y: l2_dist_align(x, y, wanted=wanted_indices, align_idx=None)
    dtw_pa_jpe = float(dtw(pred_xyz, gt_xyz, dist_pa)[0])

    return {"DTW_JPE": dtw_jpe, "DTW_PA_JPE": dtw_pa_jpe}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DTW-JPE / DTW-PA-JPE from xyz joints.")
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--gt-key", type=str, default=None)
    parser.add_argument("--pred-key", type=str, default=None)
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--body-indices", type=str, default=None)
    parser.add_argument("--lhand-indices", type=str, default=None)
    parser.add_argument("--rhand-indices", type=str, default=None)

    args = parser.parse_args()

    gt_xyz = load_xyz(args.gt, key=args.gt_key, sample_idx=args.sample_idx)
    pred_xyz = load_xyz(args.pred, key=args.pred_key, sample_idx=args.sample_idx)

    t = min(len(gt_xyz), len(pred_xyz))
    if t == 0:
        raise ValueError("Empty sequence after loading")
    if len(gt_xyz) != len(pred_xyz):
        print(f"[WARN] Length mismatch GT={len(gt_xyz)} PRED={len(pred_xyz)}. Using first {t} frames.")

    gt_xyz = gt_xyz[:t]
    pred_xyz = pred_xyz[:t]

    if gt_xyz.shape[1] != pred_xyz.shape[1]:
        raise ValueError(f"Joint count mismatch: GT J={gt_xyz.shape[1]} vs PRED J={pred_xyz.shape[1]}")

    body_indices = list(smpl_x.joint_part2idx["upper_body"]) if args.body_indices is None else _parse_indices(args.body_indices)
    lhand_indices = list(smpl_x.joint_part2idx["lhand"]) if args.lhand_indices is None else _parse_indices(args.lhand_indices)
    rhand_indices = list(smpl_x.joint_part2idx["rhand"]) if args.rhand_indices is None else _parse_indices(args.rhand_indices)

    njoints = gt_xyz.shape[1]
    _validate_indices(body_indices, njoints, "body")
    _validate_indices(lhand_indices, njoints, "lhand")
    _validate_indices(rhand_indices, njoints, "rhand")

    report = {
        "frames_used": t,
        "joint_count": njoints,
        "body": compute_part_dtw_metrics(pred_xyz, gt_xyz, body_indices),
        "lhand": compute_part_dtw_metrics(pred_xyz, gt_xyz, lhand_indices),
        "rhand": compute_part_dtw_metrics(pred_xyz, gt_xyz, rhand_indices),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
