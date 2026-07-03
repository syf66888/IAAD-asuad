#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def natural_sort_key(path):
    text = Path(path).name
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def collect_images(image_dir):
    paths = [path for path in Path(image_dir).iterdir() if path.suffix.lower() in IMAGE_SUFFIXES]
    paths = sorted(paths, key=natural_sort_key)
    if len(paths) < 2:
        raise FileNotFoundError(f"Need at least two images in {image_dir}")
    return paths


def load_intrinsics(path):
    values = [float(item) for item in Path(path).read_text(encoding="utf-8").split()]
    if len(values) < 4:
        raise ValueError(f"Expected fx fy cx cy in {path}")
    fx, fy, cx, cy = values[:4]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)
    return K


def load_pose_positions(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    poses = payload.get("poses", payload)
    matrices = np.array([record["matrix"] if isinstance(record, dict) else record for record in poses], dtype=float)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
        raise ValueError(f"Expected 4x4 matrices in {path}")
    return matrices[:, :3, 3]


def pair_ground_translation(prev_gray, curr_gray, K, camera_height_m, args, orb, matcher):
    height, width = prev_gray.shape[:2]
    mask_prev = np.zeros_like(prev_gray)
    y0 = int(height * args.road_roi_ymin)
    y1 = int(height * args.road_roi_ymax)
    x0 = int(width * args.road_roi_xmin)
    x1 = int(width * args.road_roi_xmax)
    mask_prev[y0:y1, x0:x1] = 255

    kp_prev, des_prev = orb.detectAndCompute(prev_gray, mask_prev)
    kp_curr, des_curr = orb.detectAndCompute(curr_gray, None)
    stats = {
        "keypoints_prev": len(kp_prev) if kp_prev is not None else 0,
        "keypoints_curr": len(kp_curr) if kp_curr is not None else 0,
        "matches": 0,
        "homography_inliers": 0,
        "status": "failed",
    }
    if des_prev is None or des_curr is None or len(kp_prev) < args.min_matches or len(kp_curr) < args.min_matches:
        stats["status"] = "not_enough_keypoints"
        return None, stats

    matches = matcher.knnMatch(des_prev, des_curr, k=2)
    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < args.ratio * second.distance:
            good.append(first)
    stats["matches"] = len(good)
    if len(good) < args.min_matches:
        stats["status"] = "not_enough_matches"
        return None, stats

    pts_prev = np.float32([kp_prev[match.queryIdx].pt for match in good])
    pts_curr = np.float32([kp_curr[match.trainIdx].pt for match in good])
    H, inlier_mask = cv2.findHomography(pts_prev, pts_curr, cv2.RANSAC, args.ransac_thresh)
    if H is None or inlier_mask is None:
        stats["status"] = "homography_failed"
        return None, stats
    inliers = int(inlier_mask.sum())
    stats["homography_inliers"] = inliers
    if inliers < args.min_inliers:
        stats["status"] = "not_enough_inliers"
        return None, stats

    try:
        retval, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)
    except cv2.error:
        stats["status"] = "decompose_failed"
        return None, stats
    if retval <= 0:
        stats["status"] = "decompose_empty"
        return None, stats

    candidates = []
    for translation in translations:
        t = np.asarray(translation, dtype=float).reshape(3) * camera_height_m
        # OpenCV camera axes: x right, y down, z forward. Use ground-plane motion magnitude.
        ground_step = float(np.linalg.norm([t[2], t[0]]))
        vertical_ratio = abs(float(t[1])) / max(ground_step, 1e-9)
        if np.isfinite(ground_step) and args.min_ground_step <= ground_step <= args.max_ground_step:
            candidates.append((vertical_ratio, ground_step, t.tolist()))
    if not candidates:
        stats["status"] = "no_valid_solution"
        return None, stats
    candidates.sort(key=lambda item: item[0])
    stats["status"] = "ok"
    stats["vertical_ratio"] = candidates[0][0]
    stats["translation"] = candidates[0][2]
    return candidates[0][1], stats


def path_length(points):
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--pose-file", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--camera-height-m", type=float, default=1.5)
    parser.add_argument("--frame-interval-s", type=float, default=1 / 30)
    parser.add_argument("--road-roi-ymin", type=float, default=0.50)
    parser.add_argument("--road-roi-ymax", type=float, default=0.98)
    parser.add_argument("--road-roi-xmin", type=float, default=0.10)
    parser.add_argument("--road-roi-xmax", type=float, default=0.90)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--min-matches", type=int, default=30)
    parser.add_argument("--min-inliers", type=int, default=20)
    parser.add_argument("--ransac-thresh", type=float, default=3.0)
    parser.add_argument("--min-ground-step", type=float, default=0.005)
    parser.add_argument("--max-ground-step", type=float, default=4.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = collect_images(args.image_dir)
    K = load_intrinsics(args.calib)
    positions = load_pose_positions(args.pose_file)
    usable = min(len(image_paths), len(positions))
    image_paths = image_paths[:usable]
    positions = positions[:usable]

    orb = cv2.ORB_create(nfeatures=args.max_features, fastThreshold=7)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    ground_steps = []
    droid_steps = []
    pair_stats = []
    prev = cv2.imread(str(image_paths[0]), cv2.IMREAD_GRAYSCALE)
    for idx in range(1, usable):
        curr = cv2.imread(str(image_paths[idx]), cv2.IMREAD_GRAYSCALE)
        if curr is None or prev is None:
            raise FileNotFoundError(image_paths[idx])
        ground_step, stats = pair_ground_translation(prev, curr, K, args.camera_height_m, args, orb, matcher)
        droid_step = float(np.linalg.norm(positions[idx, :2] - positions[idx - 1, :2]))
        stats["pair_index"] = idx - 1
        stats["droid_step_raw"] = droid_step
        if ground_step is not None and droid_step > 1e-9:
            stats["ground_step_m"] = float(ground_step)
            stats["scale"] = float(ground_step / droid_step)
            ground_steps.append(float(ground_step))
            droid_steps.append(droid_step)
        pair_stats.append(stats)
        prev = curr

    valid_scales = [stat["scale"] for stat in pair_stats if stat.get("status") == "ok" and "scale" in stat]
    if not valid_scales:
        scale = 1.0
    else:
        scale = float(np.median(valid_scales))
    raw_path = path_length(positions)
    scaled_path = raw_path * scale
    valid_ground_path = float(np.sum(ground_steps)) if ground_steps else 0.0

    summary = {
        "camera_height_m": args.camera_height_m,
        "pose_count": int(usable),
        "pair_count": int(max(0, usable - 1)),
        "valid_pair_count": int(len(valid_scales)),
        "valid_pair_ratio": float(len(valid_scales) / max(1, usable - 1)),
        "scale_median": scale,
        "scale_mean": float(np.mean(valid_scales)) if valid_scales else None,
        "scale_p10": float(np.percentile(valid_scales, 10)) if valid_scales else None,
        "scale_p90": float(np.percentile(valid_scales, 90)) if valid_scales else None,
        "droid_raw_path_length": raw_path,
        "height_scaled_path_length": scaled_path,
        "valid_ground_step_sum_m": valid_ground_path,
        "median_ground_step_m": float(np.median(ground_steps)) if ground_steps else None,
        "median_droid_step_raw": float(np.median(droid_steps)) if droid_steps else None,
        "frame_interval_s": args.frame_interval_s,
        "mean_speed_mps_height_scaled": float(scaled_path / ((usable - 1) * args.frame_interval_s)) if usable > 1 else 0.0,
    }
    (output_dir / "ground_height_scale_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "ground_height_scale_pair_stats.json").write_text(json.dumps(pair_stats, indent=2), encoding="utf-8")

    if valid_scales:
        plt.figure(figsize=(10, 4))
        plt.plot(valid_scales, ".", markersize=3)
        plt.axhline(scale, color="r", linestyle="--", label=f"median scale={scale:.3f}")
        plt.xlabel("valid pair index")
        plt.ylabel("height-derived scale")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "scale_distribution.jpg", dpi=160)
        plt.close()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
