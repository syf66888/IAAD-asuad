#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def load_pose_matrices(pose_file):
    pose_path = Path(pose_file)
    payload = json.loads(pose_path.read_text(encoding="utf-8"))
    poses = payload.get("poses", payload)
    if not isinstance(poses, list) or not poses:
        raise ValueError(f"No poses found in {pose_path}")
    matrices = np.array([record["matrix"] if isinstance(record, dict) else record for record in poses], dtype=float)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
        raise ValueError(f"Expected 4x4 pose matrices in {pose_path}, got {matrices.shape}")
    if not np.isfinite(matrices).all():
        raise ValueError(f"Pose file contains non-finite values: {pose_path}")
    return payload, poses, matrices


def select_axes(xyz, axes):
    if len(axes) != 2 or any(axis not in AXIS_INDEX for axis in axes):
        raise ValueError(f"Axes must be two letters from xyz, got {axes}")
    return xyz[:, [AXIS_INDEX[axes[0]], AXIS_INDEX[axes[1]]]]


def path_length(points):
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def rotation_2d(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def heading_aligned_transform(source, target, allow_scale):
    src_delta = source[1] - source[0]
    tgt_delta = target[1] - target[0]
    src_heading = np.arctan2(src_delta[1], src_delta[0])
    tgt_heading = np.arctan2(tgt_delta[1], tgt_delta[0])
    rotation = rotation_2d(tgt_heading - src_heading)
    scale = 1.0
    if allow_scale:
        src_len = path_length(source)
        tgt_len = path_length(target)
        scale = tgt_len / src_len if src_len > 1e-9 else 1.0
    aligned = (scale * (rotation @ (source - source[0]).T)).T + target[0]
    return aligned, scale, rotation


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


def error_summary(errors):
    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "p95": float(np.percentile(errors, 95)),
        "max": float(np.max(errors)),
        "final": float(errors[-1]),
    }


def wrapped_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def segment_headings(points):
    deltas = np.diff(points, axis=0)
    return np.arctan2(deltas[:, 1], deltas[:, 0])


def align_records(estimated_records, gt_records, estimated_xyz, gt_xyz):
    has_source_indices = all(isinstance(record, dict) and "source_index" in record for record in estimated_records)
    if has_source_indices:
        source_indices = [int(record["source_index"]) for record in estimated_records]
        if max(source_indices) < len(gt_xyz):
            return estimated_xyz, gt_xyz[source_indices], source_indices
    count = min(len(estimated_xyz), len(gt_xyz))
    return estimated_xyz[:count], gt_xyz[:count], None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimated-pose", required=True)
    parser.add_argument("--gt-pose", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default="pose_json_accuracy")
    parser.add_argument("--estimated-axes", default="xy")
    parser.add_argument("--gt-axes", default="xy")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, estimated_records, estimated_matrices = load_pose_matrices(args.estimated_pose)
    _, gt_records, gt_matrices = load_pose_matrices(args.gt_pose)
    estimated_xyz = estimated_matrices[:, :3, 3]
    gt_xyz = gt_matrices[:, :3, 3]
    estimated_xyz, gt_xyz, source_indices = align_records(estimated_records, gt_records, estimated_xyz, gt_xyz)
    count = min(len(estimated_xyz), len(gt_xyz))
    if args.max_frames is not None:
        count = min(count, args.max_frames)
    estimated_xy = select_axes(estimated_xyz[:count], args.estimated_axes)
    gt_xy = select_axes(gt_xyz[:count], args.gt_axes)

    if count < 3:
        raise ValueError("Need at least three poses to evaluate trajectory accuracy")

    origin_heading_aligned, heading_scale, _ = heading_aligned_transform(estimated_xy, gt_xy, allow_scale=False)
    path_scale_aligned, path_scale, _ = heading_aligned_transform(estimated_xy, gt_xy, allow_scale=True)
    sim2_aligned, sim2_scale, sim2_rotation, sim2_translation = sim2_umeyama(estimated_xy, gt_xy)

    origin_heading_errors = np.linalg.norm(origin_heading_aligned - gt_xy, axis=1)
    path_scale_errors = np.linalg.norm(path_scale_aligned - gt_xy, axis=1)
    sim2_errors = np.linalg.norm(sim2_aligned - gt_xy, axis=1)

    gt_step = np.linalg.norm(np.diff(gt_xy, axis=0), axis=1)
    est_step = np.linalg.norm(np.diff(estimated_xy, axis=0), axis=1)
    scaled_est_step = sim2_scale * est_step
    valid_step = gt_step > 1e-6
    step_abs_errors = np.abs(scaled_est_step - gt_step)
    step_rel_errors = step_abs_errors[valid_step] / gt_step[valid_step]

    gt_turn = wrapped_angle(np.diff(segment_headings(gt_xy)))
    est_turn = wrapped_angle(np.diff(segment_headings(estimated_xy)))
    turn_errors_deg = np.abs(np.rad2deg(wrapped_angle(est_turn - gt_turn)))

    metrics = {
        "pose_count": int(count),
        "estimated_axes": args.estimated_axes,
        "gt_axes": args.gt_axes,
        "used_source_indices": source_indices is not None,
        "gt_path_length": path_length(gt_xy),
        "estimated_path_length_raw": path_length(estimated_xy),
        "path_length_ratio_est_over_gt": path_length(estimated_xy) / path_length(gt_xy),
        "heading_alignment_no_scale_ate": error_summary(origin_heading_errors),
        "heading_alignment_path_scale": float(path_scale),
        "heading_alignment_with_path_scale_ate": error_summary(path_scale_errors),
        "sim2_scale": float(sim2_scale),
        "sim2_rotation_deg": float(np.rad2deg(np.arctan2(sim2_rotation[1, 0], sim2_rotation[0, 0]))),
        "sim2_translation_xy": sim2_translation.tolist(),
        "sim2_ate": error_summary(sim2_errors),
        "sim2_step_abs_error": error_summary(step_abs_errors),
        "sim2_step_rel_error": {
            "mean": float(np.mean(step_rel_errors)),
            "median": float(np.median(step_rel_errors)),
            "p95": float(np.percentile(step_rel_errors, 95)),
            "max": float(np.max(step_rel_errors)),
        },
        "turn_increment_error_deg": {
            "mean": float(np.mean(turn_errors_deg)),
            "median": float(np.median(turn_errors_deg)),
            "p95": float(np.percentile(turn_errors_deg, 95)),
            "max": float(np.max(turn_errors_deg)),
        },
    }

    metrics_path = output_dir / f"{args.name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    csv_path = output_dir / f"{args.name}_per_frame.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "gt_x",
                "gt_y",
                "est_raw_x",
                "est_raw_y",
                "est_sim2_x",
                "est_sim2_y",
                "sim2_position_error",
            ]
        )
        for idx in range(count):
            writer.writerow(
                [
                    idx,
                    gt_xy[idx, 0],
                    gt_xy[idx, 1],
                    estimated_xy[idx, 0],
                    estimated_xy[idx, 1],
                    sim2_aligned[idx, 0],
                    sim2_aligned[idx, 1],
                    sim2_errors[idx],
                ]
            )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(gt_xy[:, 0], gt_xy[:, 1], "k-", label="GT")
    axes[0].plot(sim2_aligned[:, 0], sim2_aligned[:, 1], "r--", label="Estimated after Sim(2)")
    axes[0].scatter(gt_xy[0, 0], gt_xy[0, 1], c="g", label="start")
    axes[0].scatter(gt_xy[-1, 0], gt_xy[-1, 1], c="b", label="end")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[0].set_title("Trajectory")
    axes[1].plot(sim2_errors, "r-", label="Sim(2) position error")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("2D error")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    axes[1].set_title("Per-frame Error")
    fig.tight_layout()
    plot_path = output_dir / f"{args.name}_plot.jpg"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    print(json.dumps({"metrics": str(metrics_path), "plot": str(plot_path), "csv": str(csv_path), **metrics}, indent=2))


if __name__ == "__main__":
    main()
