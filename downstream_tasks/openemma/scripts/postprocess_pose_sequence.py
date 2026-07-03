#!/usr/bin/env python3
import argparse
import json
import pickle
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def natural_sort_key(path):
    text = Path(path).name
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def path_length(points):
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def rotation_2d(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def sim2_umeyama(source, target):
    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    src_centered = source - src_mean
    tgt_centered = target - tgt_mean
    covariance = (tgt_centered.T @ src_centered) / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.eye(2)
    if np.linalg.det(u @ vt) < 0:
        sign[-1, -1] = -1.0
    rotation = u @ sign @ vt
    src_var = float(np.mean(np.sum(src_centered**2, axis=1)))
    scale = float(np.trace(np.diag(singular_values) @ sign) / src_var) if src_var > 1e-12 else 1.0
    translation = tgt_mean - scale * (rotation @ src_mean)
    aligned = (scale * (rotation @ source.T)).T + translation
    return aligned, scale, rotation, translation


def load_pose_json(path):
    pose_path = Path(path)
    payload = json.loads(pose_path.read_text(encoding="utf-8"))
    records = payload.get("poses", payload)
    if not isinstance(records, list) or not records:
        raise ValueError(f"No poses found in {pose_path}")
    matrices = np.array([record["matrix"] if isinstance(record, dict) else record for record in records], dtype=float)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
        raise ValueError(f"Expected 4x4 pose matrices, got {matrices.shape}")
    if not np.isfinite(matrices).all():
        raise ValueError(f"Pose file contains non-finite values: {pose_path}")
    return payload, records, matrices


def load_deepaccident_gt(calib_dir):
    paths = sorted(Path(calib_dir).glob("*.pkl"), key=natural_sort_key)
    if not paths:
        raise FileNotFoundError(f"No calibration pkl files found in {calib_dir}")
    positions = []
    for path in paths:
        with path.open("rb") as handle:
            calibration = pickle.load(handle)
        positions.append(np.array(calibration["ego_to_world"], dtype=float)[:3, 3])
    return np.array(positions, dtype=float)


def gt_for_records(records, gt_xyz, count):
    source_indices = []
    for record in records[:count]:
        if isinstance(record, dict) and "source_index" in record:
            source_indices.append(int(record["source_index"]))
    if len(source_indices) == count:
        if max(source_indices) >= len(gt_xyz):
            raise ValueError(f"source_index {max(source_indices)} exceeds GT length {len(gt_xyz)}")
        return gt_xyz[source_indices], source_indices
    return gt_xyz[:count], None


def moving_average(points, window):
    if window <= 1:
        return points.copy()
    if window % 2 == 0:
        raise ValueError("--smooth-window must be odd")
    pad = window // 2
    padded = np.pad(points, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    smoothed = np.empty_like(points)
    for dim in range(points.shape[1]):
        smoothed[:, dim] = np.convolve(padded[:, dim], kernel, mode="valid")
    smoothed[0] = points[0]
    smoothed[-1] = points[-1]
    return smoothed


def clip_steps(points, max_step):
    if max_step is None or max_step <= 0 or len(points) < 2:
        return points.copy(), 0
    clipped = [points[0].copy()]
    clipped_count = 0
    for step in np.diff(points, axis=0):
        step_norm = float(np.linalg.norm(step[:2]))
        if step_norm > max_step:
            step = step * (max_step / step_norm)
            clipped_count += 1
        clipped.append(clipped[-1] + step)
    return np.array(clipped, dtype=float), clipped_count


def robust_step_scale(source_xy, target_xy):
    src_step = np.linalg.norm(np.diff(source_xy, axis=0), axis=1)
    tgt_step = np.linalg.norm(np.diff(target_xy, axis=0), axis=1)
    valid = (src_step > 1e-9) & (tgt_step > 1e-9)
    if not np.any(valid):
        return 1.0
    ratios = tgt_step[valid] / src_step[valid]
    return float(np.median(ratios))


def infer_frame_interval(records, fallback):
    timestamps = []
    for record in records:
        if isinstance(record, dict) and "timestamp" in record:
            timestamps.append(float(record["timestamp"]))
    if len(timestamps) >= 2:
        deltas = np.diff(timestamps)
        deltas = deltas[deltas > 1e-9]
        if len(deltas):
            return float(np.median(deltas))
    return fallback


def yaw_matrix(yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return matrix


def build_matrices(positions):
    matrices = []
    last_yaw = 0.0
    for idx, position in enumerate(positions):
        if idx + 1 < len(positions):
            delta = positions[idx + 1, :2] - positions[idx, :2]
        elif idx > 0:
            delta = positions[idx, :2] - positions[idx - 1, :2]
        else:
            delta = np.array([1.0, 0.0], dtype=float)
        if np.linalg.norm(delta) > 1e-9:
            last_yaw = float(np.arctan2(delta[1], delta[0]))
        matrix = yaw_matrix(last_yaw)
        matrix[:3, 3] = position
        matrices.append(matrix)
    return np.array(matrices, dtype=float)


def estimate_scale_and_align(args, positions, gt_xyz, records, count):
    gt_used = None
    source_indices = None
    xy = positions[:, :2]
    z = positions[:, 2:3]
    scale = args.scale
    rotation = np.eye(2, dtype=float)
    translation = np.zeros(2, dtype=float)
    scale_note = args.scale_mode

    if args.scale_mode.startswith("gt_"):
        if gt_xyz is None:
            raise ValueError(f"--gt-calib-dir is required for --scale-mode {args.scale_mode}")
        gt_used, source_indices = gt_for_records(records, gt_xyz, count)
        gt_xy = gt_used[:, :2]
        target_path_length = path_length(gt_xy)
        if args.scale_mode == "gt_path":
            src_len = path_length(xy)
            scale = target_path_length / src_len if src_len > 1e-9 else 1.0
            aligned_xy = xy * scale
            aligned_z = z * scale
        elif args.scale_mode == "gt_median_step":
            scale = robust_step_scale(xy, gt_xy)
            aligned_xy = xy * scale
            aligned_z = z * scale
        elif args.scale_mode == "gt_sim2":
            aligned_xy, scale, rotation, translation = sim2_umeyama(xy, gt_xy)
            aligned_z = z * scale
        else:
            raise ValueError(f"Unsupported scale mode: {args.scale_mode}")
    elif args.scale_mode == "target_path":
        src_len = path_length(xy)
        if args.target_path_length is None:
            raise ValueError("--target-path-length is required for --scale-mode target_path")
        target_path_length = float(args.target_path_length)
        scale = target_path_length / src_len if src_len > 1e-9 else 1.0
        aligned_xy = xy * scale
        aligned_z = z * scale
        scale_note = f"target_path:{target_path_length}"
    elif args.scale_mode == "target_speed":
        if args.target_speed_mps is None:
            raise ValueError("--target-speed-mps is required for --scale-mode target_speed")
        frame_interval = infer_frame_interval(records[:count], args.frame_interval_s)
        target_path_length = args.target_speed_mps * frame_interval * max(0, count - 1)
        src_len = path_length(xy)
        scale = target_path_length / src_len if src_len > 1e-9 else 1.0
        aligned_xy = xy * scale
        aligned_z = z * scale
        scale_note = f"target_speed:{args.target_speed_mps}"
    elif args.scale_mode == "factor":
        target_path_length = None
        aligned_xy = xy * scale
        aligned_z = z * scale
    elif args.scale_mode == "none":
        scale = 1.0
        target_path_length = None
        aligned_xy = xy.copy()
        aligned_z = z.copy()
    else:
        raise ValueError(f"Unsupported scale mode: {args.scale_mode}")

    aligned = np.concatenate([aligned_xy, aligned_z], axis=1)
    return aligned, {
        "scale_mode": scale_note,
        "scale": float(scale),
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "used_source_indices": source_indices is not None,
        "gt_path_length": path_length(gt_used[:, :2]) if gt_used is not None else None,
        "target_path_length": target_path_length,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pose", required=True)
    parser.add_argument("--output-pose", required=True)
    parser.add_argument("--output-plot", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--gt-calib-dir", default=None)
    parser.add_argument(
        "--scale-mode",
        choices=["none", "factor", "target_path", "target_speed", "gt_path", "gt_median_step", "gt_sim2"],
        default="none",
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--target-path-length", type=float, default=None)
    parser.add_argument("--target-speed-mps", type=float, default=None)
    parser.add_argument("--frame-interval-s", type=float, default=0.5)
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--smooth-passes", type=int, default=1)
    parser.add_argument("--max-step-median-factor", type=float, default=None)
    parser.add_argument("--max-step-mps", type=float, default=None)
    parser.add_argument("--rescale-after-smoothing", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    payload, records, matrices = load_pose_json(args.input_pose)
    count = len(records)
    if args.max_frames is not None:
        count = min(count, args.max_frames)
    records = records[:count]
    positions = matrices[:count, :3, 3].copy()
    raw_positions = positions.copy()

    gt_xyz = load_deepaccident_gt(args.gt_calib_dir) if args.gt_calib_dir else None
    raw_len = path_length(raw_positions[:, :2])

    if args.max_step_median_factor is not None:
        raw_steps = np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)
        valid_steps = raw_steps[raw_steps > 1e-9]
        if len(valid_steps):
            max_step = float(np.median(valid_steps) * args.max_step_median_factor)
            positions, pre_clipped = clip_steps(positions, max_step)
        else:
            pre_clipped = 0
    else:
        pre_clipped = 0

    for _ in range(max(0, args.smooth_passes)):
        positions = moving_average(positions, args.smooth_window)

    aligned, scale_info = estimate_scale_and_align(args, positions, gt_xyz, records, count)
    before_final_len = path_length(aligned[:, :2])

    if args.max_step_mps is not None:
        frame_interval = infer_frame_interval(records, args.frame_interval_s)
        aligned, post_clipped = clip_steps(aligned, args.max_step_mps * frame_interval)
    else:
        post_clipped = 0

    for _ in range(max(0, args.smooth_passes)):
        aligned = moving_average(aligned, args.smooth_window)

    if args.rescale_after_smoothing and scale_info.get("target_path_length"):
        smoothed_len = path_length(aligned[:, :2])
        if smoothed_len > 1e-9:
            correction = scale_info["target_path_length"] / smoothed_len
            origin = aligned[0].copy()
            aligned = origin + (aligned - origin) * correction
            scale_info["post_smoothing_rescale"] = float(correction)

    matrices_out = build_matrices(aligned)
    processed_records = []
    for idx, record in enumerate(records):
        processed = dict(record) if isinstance(record, dict) else {}
        processed["matrix"] = matrices_out[idx].tolist()
        processed_records.append(processed)

    output_payload = dict(payload) if isinstance(payload, dict) else {}
    output_payload["name"] = args.name or output_payload.get("name", Path(args.input_pose).stem) + "_postprocessed"
    output_payload["source_pose_file"] = str(Path(args.input_pose).resolve())
    output_payload["source"] = f"{output_payload.get('source', 'pose sequence')} + scale/smoothing postprocess"
    output_payload["poses"] = processed_records
    output_payload["postprocess"] = {
        **scale_info,
        "smooth_window": args.smooth_window,
        "smooth_passes": args.smooth_passes,
        "pre_clip_step_count": int(pre_clipped),
        "post_clip_step_count": int(post_clipped),
        "raw_path_length_xy": float(raw_len),
        "smoothed_scaled_path_length_xy_before_final_clip": float(before_final_len),
        "final_path_length_xy": path_length(aligned[:, :2]),
    }

    output_pose = Path(args.output_pose)
    output_pose.parent.mkdir(parents=True, exist_ok=True)
    output_pose.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    plot_path = Path(args.output_plot) if args.output_plot else output_pose.with_suffix(".jpg")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 7))
    plt.plot(raw_positions[:, 0], raw_positions[:, 1], "k:", label="raw")
    plt.plot(aligned[:, 0], aligned[:, 1], "r-", label="postprocessed")
    if gt_xyz is not None:
        gt_used, _ = gt_for_records(records, gt_xyz, count)
        plt.plot(gt_used[:, 0], gt_used[:, 1], "b--", label="GT")
    plt.axis("equal")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.title(output_payload["name"])
    plt.savefig(plot_path, dpi=160)
    plt.close()

    summary = {
        "output_pose": str(output_pose),
        "plot": str(plot_path),
        "pose_count": int(count),
        **output_payload["postprocess"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
