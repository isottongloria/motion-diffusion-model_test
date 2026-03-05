#!/usr/bin/env python3
"""Phoenix metrics runner from `motion_emb`.

Modes:
1) Manifest mode: evaluate existing (gt_file, results_file) pairs.
2) Config mode: run sample.generate per item, then xyz conversion + DTW metrics.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable
import numpy as np
import torch


def _import_eval_phoenix():
    try:
        from eval.eval_phoenix import compute_part_dtw_metrics, smpl_x  # type: ignore
        return compute_part_dtw_metrics, smpl_x
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from eval.eval_phoenix import compute_part_dtw_metrics, smpl_x  # type: ignore
        return compute_part_dtw_metrics, smpl_x


compute_part_dtw_metrics, smpl_x = _import_eval_phoenix()

FIXED_SHAPE = np.array(
    [-0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172,
     0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842],
    dtype=np.float32,
)


def _unwrap_npy_object(obj: Any) -> Any:
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        if obj.shape == ():
            return obj.item()
        if obj.size == 1:
            return obj.reshape(-1)[0]
    return obj


def _angles_from_results_file(results_path: Path, sample_idx: int = 0) -> np.ndarray:
    raw = np.load(results_path, allow_pickle=True)
    data = _unwrap_npy_object(raw)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in {results_path}, found {type(data)}")
    if "motion_emb" not in data:
        raise KeyError(f"Missing 'motion_emb' in {results_path}. Keys: {list(data.keys())}")

    angles = np.squeeze(np.asarray(data["motion_emb"]))
    if angles.ndim in (3, 4):
        angles = np.squeeze(angles[sample_idx])
    elif angles.ndim != 2:
        raise ValueError(f"Expected 2/3/4 dims, got {angles.shape} in {results_path}")

    if angles.shape[1] in (133, 169, 172):
        td = angles
    elif angles.shape[0] in (133, 169, 172):
        td = angles.T
    else:
        raise ValueError(f"Unsupported angle shape {angles.shape} in {results_path}")
    return np.asarray(td, dtype=np.float32)


def _normalize_angles_dim(angles: np.ndarray) -> np.ndarray:
    t, d = angles.shape
    if d == 133:
        return np.concatenate([np.zeros((t, 36), dtype=np.float32), angles], axis=1)
    if d == 169:
        return angles
    if d == 172:
        return angles[:, :169]
    raise ValueError(f"Unsupported feature dim {d}; expected 133/169/172")


def _to_smplx_joints(features: np.ndarray, device: torch.device) -> np.ndarray:
    full_169 = _normalize_angles_dim(features).astype(np.float32)
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


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std())}


def _write_metrics_json(out_path: Path, *, num_samples: int, device: str,
                        summary: dict[str, dict[str, float]], per_sample: list[dict[str, Any]],
                        config: Path | None = None, manifest: Path | None = None) -> None:
    payload: dict[str, Any] = {
        "num_samples": num_samples,
        "device": device,
        "summary": summary,
        "per_sample": per_sample,
    }
    if config is not None:
        payload["config"] = str(config)
    if manifest is not None:
        payload["manifest"] = str(manifest)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_arg(cmd: list[str], key: str, value: Any) -> None:
    flag = f"--{key}"
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
    elif value is None:
        return
    elif isinstance(value, list):
        for item in value:
            cmd.extend([flag, str(item)])
    else:
        cmd.extend([flag, str(value)])


def _csv_stem(path_value: str | None) -> str:
    return Path(path_value).stem if path_value else ""


def _infer_results_file(generate_args: dict[str, Any]) -> Path:
    out_dir = generate_args.get("output_dir")
    in_csv = generate_args.get("input_gt_local_txt")
    if not out_dir or not in_csv:
        raise ValueError("Cannot infer results path without output_dir and input_gt_local_txt")
    stem = _csv_stem(str(in_csv))
    candidate_concat = Path(f"{str(out_dir)}_{stem}") / "results.npy"
    candidate_join = Path(str(out_dir)) / f"_{stem}" / "results.npy"
    if candidate_concat.exists():
        return candidate_concat
    if candidate_join.exists():
        return candidate_join
    return candidate_concat


def _run_generate(generate_args: dict[str, Any], dry_run: bool, stream_logs: bool) -> Path:
    cmd = [sys.executable, "-m", "sample.generate"]
    for k, v in generate_args.items():
        _append_arg(cmd, k, v)
    print("[RUN]", " ".join(cmd))
    if not dry_run:
        if stream_logs:
            subprocess.run(cmd, check=True)
        else:
            proc = subprocess.run(cmd, text=True, capture_output=True)
            if proc.returncode != 0:
                print(proc.stdout)
                print(proc.stderr)
                raise subprocess.CalledProcessError(proc.returncode, cmd)
    return _infer_results_file(generate_args)


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Manifest empty: {path}")
    for req in ("gt_file", "results_file"):
        if req not in rows[0]:
            raise ValueError(f"Manifest missing column: {req}")
    return rows


def _compute_row_metrics(row: dict[str, Any], device: torch.device) -> dict[str, Any]:
    body_idx = list(smpl_x.joint_part2idx["upper_body"])
    lhand_idx = list(smpl_x.joint_part2idx["lhand"])
    rhand_idx = list(smpl_x.joint_part2idx["rhand"])

    sid = str(row.get("id", "sample"))
    gt_file = Path(str(row["gt_file"]))
    results_file = Path(str(row["results_file"]))
    sidx = int(row.get("sample_idx", 0) or 0)

    gt_angles = np.asarray(np.load(gt_file), dtype=np.float32)
    pred_angles = _angles_from_results_file(results_file, sample_idx=sidx)
    gt_xyz = _to_smplx_joints(gt_angles, device)
    pred_xyz = _to_smplx_joints(pred_angles, device)

    t = min(len(gt_xyz), len(pred_xyz))
    if t == 0:
        raise ValueError(f"Empty sequence for {sid}")
    gt_eval, pred_eval = gt_xyz[:t], pred_xyz[:t]

    return {
        "id": sid,
        "gt_file": str(gt_file),
        "results_file": str(results_file),
        "frames_used": int(t),
        "body": compute_part_dtw_metrics(pred_eval, gt_eval, body_idx),
        "lhand": compute_part_dtw_metrics(pred_eval, gt_eval, lhand_idx),
        "rhand": compute_part_dtw_metrics(pred_eval, gt_eval, rhand_idx),
    }


def _summary_from_per_sample(per_sample: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    acc = {k: [] for k in [
        "body_DTW_JPE", "body_DTW_PA_JPE",
        "lhand_DTW_JPE", "lhand_DTW_PA_JPE",
        "rhand_DTW_JPE", "rhand_DTW_PA_JPE",
    ]}
    for sample in per_sample:
        acc["body_DTW_JPE"].append(sample["body"]["DTW_JPE"])
        acc["body_DTW_PA_JPE"].append(sample["body"]["DTW_PA_JPE"])
        acc["lhand_DTW_JPE"].append(sample["lhand"]["DTW_JPE"])
        acc["lhand_DTW_PA_JPE"].append(sample["lhand"]["DTW_PA_JPE"])
        acc["rhand_DTW_JPE"].append(sample["rhand"]["DTW_JPE"])
        acc["rhand_DTW_PA_JPE"].append(sample["rhand"]["DTW_PA_JPE"])
    return {k: _mean_std(v) for k, v in acc.items()}


def _compute_dataset_metrics(rows: list[dict[str, Any]], device: torch.device, verbose: bool,
                             on_update: Callable[[int, dict[str, dict[str, float]], list[dict[str, Any]]], None] | None = None) -> dict[str, Any]:
    per_sample: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        sample_metrics = _compute_row_metrics(row, device)
        per_sample.append(sample_metrics)
        if verbose:
            print(
                f"[{i+1}/{len(rows)}] {sample_metrics['id']}: "
                f"bodyPA={sample_metrics['body']['DTW_PA_JPE']:.3f} "
                f"lPA={sample_metrics['lhand']['DTW_PA_JPE']:.3f} "
                f"rPA={sample_metrics['rhand']['DTW_PA_JPE']:.3f}"
            )
        summary = _summary_from_per_sample(per_sample)
        if on_update is not None:
            on_update(i + 1, summary, per_sample)
    return {"num_samples": len(per_sample), "summary": _summary_from_per_sample(per_sample), "per_sample": per_sample}


def _print_summary_table(summary: dict[str, dict[str, float]]) -> None:
    keys = ["body_DTW_JPE", "body_DTW_PA_JPE", "lhand_DTW_JPE", "lhand_DTW_PA_JPE", "rhand_DTW_JPE", "rhand_DTW_PA_JPE"]
    print("\nQuantitative metrics (average on full set)")
    print("| Metric | Mean | Std |")
    print("|---|---:|---:|")
    for k in keys:
        print(f"| {k} | {summary[k]['mean']:.4f} | {summary[k]['std']:.4f} |")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config must be a JSON object")
    return data


def _load_id_list(path: Path) -> list[str]:
    ids = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            ids.append(s)
    if not ids:
        raise ValueError(f"No valid ids found in list: {path}")
    return ids


def _build_samples_from_id_list(config: dict[str, Any]) -> list[dict[str, Any]]:
    spec = config.get("dataset_list")
    if not isinstance(spec, dict):
        return []

    ids_txt = Path(str(spec["ids_txt"]))
    csv_dir = Path(str(spec["input_gt_local_txt_dir"]))
    text_dir = Path(str(spec["input_text_dir"]))
    gt_dir = Path(str(spec["gt_file_dir"]))

    csv_suffix = str(spec.get("csv_suffix", ".csv"))
    text_suffix = str(spec.get("text_suffix", "_simple.txt"))
    gt_suffix = str(spec.get("gt_suffix", ".npy"))

    samples = []
    for sid in _load_id_list(ids_txt):
        samples.append({
            "id": sid,
            "input_gt_local_txt": str(csv_dir / f"{sid}{csv_suffix}"),
            "input_text": str(text_dir / f"{sid}{text_suffix}"),
            "gt_file": str(gt_dir / f"{sid}{gt_suffix}"),
        })
    return samples


def _rows_from_config(cfg: dict[str, Any], dry_run: bool, keep_generated: bool, stream_generate_logs: bool,
                      on_row_ready: Callable[[dict[str, Any]], None] | None = None, limit: int | None = None) -> tuple[list[dict[str, Any]], tempfile.TemporaryDirectory[str] | None]:
    gen = cfg.get("generate", {})
    if not isinstance(gen, dict):
        raise ValueError("config.generate must be an object")
    base_args = gen.get("args", gen)
    if not isinstance(base_args, dict):
        raise ValueError("config.generate.args must be an object")

    samples = cfg.get("samples")
    if samples is None:
        samples = []
    if not isinstance(samples, list):
        raise ValueError("config.samples must be a list")

    samples = [*samples, *_build_samples_from_id_list(cfg)]
    if not samples:
        raise ValueError("No samples provided. Use config.samples and/or config.dataset_list")

    tmp_ctx = None
    if not keep_generated:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="phoenix_test_tmp_")
        base_args = dict(base_args)
        base_args["output_dir"] = tmp_ctx.name

    rows: list[dict[str, Any]] = []
    for i, sample in enumerate(samples):
        if limit is not None and len(rows) >= limit:
            break
        if not isinstance(sample, dict) or "gt_file" not in sample:
            raise ValueError(f"Invalid sample at index {i}; require object with gt_file")
        merged = dict(base_args)
        for key in ["input_gt_local_txt", "input_text", "text_prompt", "sample_split", "output_dir", "num_samples", "num_repetitions", "guidance_param"]:
            if key in sample:
                merged[key] = sample[key]

        results_file = Path(str(sample.get("results_file"))) if sample.get("results_file") else _run_generate(merged, dry_run=dry_run, stream_logs=stream_generate_logs)
        row = {
            "id": sample.get("id", f"sample_{i}"),
            "gt_file": sample["gt_file"],
            "results_file": str(results_file),
            "sample_idx": sample.get("sample_idx", 0),
        }
        rows.append(row)
        if on_row_ready is not None:
            on_row_ready(row)
    return rows, tmp_ctx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phoenix test-set metrics from motion_emb")
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--device", choices=["cpu", "cuda"], default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--keep-generated", action="store_true")
    args = p.parse_args()
    if (args.manifest is None) == (args.config is None):
        raise ValueError("Use exactly one of --manifest or --config")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    cfg_out: Path | None = None
    out_path: Path | None = None
    tmp_ctx: tempfile.TemporaryDirectory[str] | None = None
    try:
        per_sample_live: list[dict[str, Any]] = []

        if args.config:
            cfg = _load_json(args.config)
            out = cfg.get("output_json")
            if isinstance(out, str) and out:
                cfg_out = Path(out)

            if args.dry_run:
                _rows_from_config(cfg, dry_run=True, keep_generated=args.keep_generated, stream_generate_logs=args.verbose, limit=args.limit)
                print("[DRY RUN] Commands printed; metrics not computed.")
                return

            out_path = args.output_json or cfg_out or Path("phoenix_metrics.json")
            target_count = args.limit

            def _on_row_ready(row: dict[str, Any]) -> None:
                if target_count is not None and len(per_sample_live) >= target_count:
                    return
                sample_metrics = _compute_row_metrics(row, device)
                per_sample_live.append(sample_metrics)
                _write_metrics_json(
                    out_path,
                    num_samples=len(per_sample_live),
                    device=args.device,
                    summary=_summary_from_per_sample(per_sample_live),
                    per_sample=per_sample_live,
                    config=args.config,
                    manifest=args.manifest,
                )

            _rows_from_config(
                cfg,
                dry_run=False,
                keep_generated=args.keep_generated,
                stream_generate_logs=args.verbose,
                on_row_ready=_on_row_ready,
                limit=args.limit,
            )
            report = {"num_samples": len(per_sample_live), "summary": _summary_from_per_sample(per_sample_live), "per_sample": per_sample_live}
        else:
            rows = _load_manifest(args.manifest)
            if args.limit is not None:
                rows = rows[:args.limit]
            if args.dry_run:
                print("[DRY RUN] Commands printed; metrics not computed.")
                return
            out_path = args.output_json or cfg_out or Path("phoenix_metrics.json")

            def _on_metrics_update(num_samples: int, summary: dict[str, dict[str, float]], per_sample: list[dict[str, Any]]) -> None:
                _write_metrics_json(
                    out_path,
                    num_samples=num_samples,
                    device=args.device,
                    summary=summary,
                    per_sample=per_sample,
                    config=args.config,
                    manifest=args.manifest,
                )

            report = _compute_dataset_metrics(rows, device=device, verbose=args.verbose, on_update=_on_metrics_update)

        _print_summary_table(report["summary"])
        _write_metrics_json(
            out_path,
            num_samples=report["num_samples"],
            device=args.device,
            summary=report["summary"],
            per_sample=report["per_sample"],
            config=args.config,
            manifest=args.manifest,
        )
        print(f"\nSaved metrics JSON: {out_path}")
    finally:
        if tmp_ctx is not None:
            tmp_root = Path(tmp_ctx.name)
            tmp_ctx.cleanup()
            if tmp_root.exists():
                shutil.rmtree(tmp_root, ignore_errors=True)
            print(f"[CLEANUP] Temporary generation directory removed: {tmp_root}")


if __name__ == "__main__":
    main()
