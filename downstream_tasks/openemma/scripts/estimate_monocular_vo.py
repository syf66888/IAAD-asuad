#!/usr/bin/env python3
import argparse
import csv
import json
import re
from math import atan2, cos, sin, sqrt
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def natural_sort_key(path):
    text = Path(path).name
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def collect_images(image_dir):
    image_dir = Path(image_dir)
    image_paths = []
    for suffix in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_paths.extend(image_dir.glob(suffix))
        image_paths.extend(image_dir.glob(suffix.upper()))
    image_paths = sorted(set(image_paths), key=natural_sort_key)
    if len(image_paths) < 2:
        raise FileNotFoundError(f"Need at least two images in {image_dir}")
    return image_paths


def autogen_intrinsics(first_image, focal_scale):
    image = cv2.imread(str(first_image))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {first_image}")
    height, width = image.shape[:2]
    focal = focal_scale * width
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def normalize(vector, fallback):
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9 or not np.isfinite(norm):
        return fallback.copy(), False
    return vector / norm, True


def yaw_matrix(yaw):
    c = cos(yaw)
    s = sin(yaw)
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return matrix


def yaw_to_quaternion_xyzw(yaw):
    return [0.0, 0.0, sin(yaw / 2.0), cos(yaw / 2.0)]


def estimate_pair_motion(prev_gray, curr_gray, orb, matcher, K, args):
    kp_prev, des_prev = orb.detectAndCompute(prev_gray, None)
    kp_curr, des_curr = orb.detectAndCompute(curr_gray, None)
    stats = {
        "keypoints_prev": len(kp_prev) if kp_prev is not None else 0,
        "keypoints_curr": len(kp_curr) if kp_curr is not None else 0,
        "matches": 0,
        "essential_inliers": 0,
        "pose_inliers": 0,
        "status": "failed",
    }

    if des_prev is None or des_curr is None or len(kp_prev) < args.min_matches or len(kp_curr) < args.min_matches:
        stats["status"] = "not_enough_keypoints"
        return None, None, stats, None

    knn = matcher.knnMatch(des_prev, des_curr, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < args.ratio * second.distance:
            good.append(first)

    stats["matches"] = len(good)
    if len(good) < args.min_matches:
        stats["status"] = "not_enough_matches"
        return None, None, stats, None

    pts_prev = np.float32([kp_prev[match.queryIdx].pt for match in good])
    pts_curr = np.float32([kp_curr[match.trainIdx].pt for match in good])

    essential, essential_mask = cv2.findEssentialMat(
        pts_prev,
        pts_curr,
        K,
        method=cv2.RANSAC,
        prob=args.ransac_prob,
        threshold=args.ransac_thresh,
    )
    if essential is None:
        stats["status"] = "essential_failed"
        return None, None, stats, None

    essential = essential[:3, :3]
    stats["essential_inliers"] = int(essential_mask.sum()) if essential_mask is not None else 0
    if stats["essential_inliers"] < args.min_inliers:
        stats["status"] = "not_enough_essential_inliers"
        return None, None, stats, None

    pose_inliers, rotation, translation, pose_mask = cv2.recoverPose(
        essential,
        pts_prev,
        pts_curr,
        K,
        mask=essential_mask,
    )
    stats["pose_inliers"] = int(pose_inliers)
    if pose_inliers < args.min_inliers:
        stats["status"] = "not_enough_pose_inliers"
        return None, None, stats, None

    t = translation.reshape(3)
    # Map OpenCV camera translation direction to the ego plane:
    # camera z -> ego forward x, camera x -> ego right, so ego left y is -camera x.
    motion_ego = np.array([t[2], -t[0]], dtype=float)
    if motion_ego[0] < 0:
        motion_ego *= -1.0

    motion_ego, ok = normalize(motion_ego, np.array([1.0, 0.0], dtype=float))
    if not ok:
        stats["status"] = "degenerate_translation"
        return None, rotation, stats, None

    stats["status"] = "ok"
    debug = (kp_prev, kp_curr, good, pose_mask)
    return motion_ego, rotation, stats, debug


def write_debug_matches(prev_image, curr_image, debug, output_path):
    if debug is None:
        return
    kp_prev, kp_curr, matches, pose_mask = debug
    if pose_mask is not None:
        mask = pose_mask.reshape(-1).astype(bool)
        matches = [match for match, keep in zip(matches, mask) if keep]
    preview = cv2.drawMatches(
        prev_image,
        kp_prev,
        curr_image,
        kp_curr,
        matches[:80],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(output_path), preview)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--frame-interval-s", type=float, default=0.5)
    parser.add_argument("--focal-scale", type=float, default=0.8)
    parser.add_argument("--step-scale", type=float, default=1.0)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--min-matches", type=int, default=25)
    parser.add_argument("--min-inliers", type=int, default=20)
    parser.add_argument("--ransac-prob", type=float, default=0.999)
    parser.add_argument("--ransac-thresh", type=float, default=1.0)
    parser.add_argument("--smooth-alpha", type=float, default=0.35)
    parser.add_argument("--debug-matches", type=int, default=3)
    args = parser.parse_args()

    image_paths = collect_images(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or Path(args.image_dir).name

    K = autogen_intrinsics(image_paths[0], args.focal_scale)
    (output_dir / "calib_autogen.txt").write_text(
        f"{K[0, 0]:.9f} {K[1, 1]:.9f} {K[0, 2]:.9f} {K[1, 2]:.9f}\n",
        encoding="utf-8",
    )
    (output_dir / "calib_autogen.json").write_text(
        json.dumps({"camera_intrinsic": K.tolist(), "source": "autogenerated_from_image_size"}, indent=2),
        encoding="utf-8",
    )

    orb = cv2.ORB_create(nfeatures=args.max_features, fastThreshold=7)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    first_image = cv2.imread(str(image_paths[0]))
    prev_gray = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
    position = np.zeros(3, dtype=float)
    yaw = 0.0
    prev_motion_ego = np.array([1.0, 0.0], dtype=float)

    pose_records = []
    pair_stats = []
    trajectory = []

    initial_pose = yaw_matrix(yaw)
    initial_pose[:3, 3] = position
    pose_records.append(
        {
            "timestamp": 0.0,
            "image": image_paths[0].name,
            "matrix": initial_pose.tolist(),
        }
    )
    trajectory.append(position.copy())

    for index, image_path in enumerate(image_paths[1:], start=1):
        curr_image = cv2.imread(str(image_path))
        if curr_image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        curr_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)

        motion_ego, _, stats, debug = estimate_pair_motion(prev_gray, curr_gray, orb, matcher, K, args)
        if motion_ego is None:
            motion_ego = prev_motion_ego.copy()
        else:
            blended = args.smooth_alpha * motion_ego + (1.0 - args.smooth_alpha) * prev_motion_ego
            motion_ego, _ = normalize(blended, prev_motion_ego)
            prev_motion_ego = motion_ego.copy()

        world_delta = np.array(
            [
                cos(yaw) * motion_ego[0] - sin(yaw) * motion_ego[1],
                sin(yaw) * motion_ego[0] + cos(yaw) * motion_ego[1],
                0.0,
            ],
            dtype=float,
        )
        position = position + args.step_scale * world_delta
        if np.linalg.norm(world_delta[:2]) > 1e-9:
            yaw = atan2(world_delta[1], world_delta[0])

        pose = yaw_matrix(yaw)
        pose[:3, 3] = position
        timestamp = index * args.frame_interval_s
        pose_records.append({"timestamp": timestamp, "image": image_path.name, "matrix": pose.tolist()})
        trajectory.append(position.copy())

        stats["frame_index"] = index
        stats["image"] = image_path.name
        stats["used_fallback_motion"] = debug is None or stats["status"] != "ok"
        pair_stats.append(stats)

        if index <= args.debug_matches:
            write_debug_matches(
                first_image if index == 1 else prev_image,
                curr_image,
                debug,
                output_dir / f"{name}_matches_{index - 1:03d}_{index:03d}.jpg",
            )

        prev_image = curr_image
        prev_gray = curr_gray

    trajectory = np.array(trajectory, dtype=float)
    np.save(output_dir / f"{name}_trajectory_xyz.npy", trajectory)

    pose_payload = {
        "name": name,
        "source_image_dir": str(Path(args.image_dir).resolve()),
        "coordinate_system": "ego_world_x_forward_y_left_z_up",
        "monocular_scale_note": "Scale is arbitrary because only monocular images were used.",
        "camera_intrinsic": K.tolist(),
        "poses": pose_records,
        "pair_stats": pair_stats,
    }
    pose_json = output_dir / f"{name}_poses.json"
    pose_json.write_text(json.dumps(pose_payload, indent=2), encoding="utf-8")

    tum_path = output_dir / f"{name}_poses_tum.txt"
    with open(tum_path, "w", encoding="utf-8") as handle:
        for record in pose_records:
            matrix = np.array(record["matrix"], dtype=float)
            tx, ty, tz = matrix[:3, 3]
            yaw_value = atan2(matrix[1, 0], matrix[0, 0])
            qx, qy, qz, qw = yaw_to_quaternion_xyzw(yaw_value)
            handle.write(
                f"{record['timestamp']:.9f} {tx:.9f} {ty:.9f} {tz:.9f} "
                f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"
            )

    with open(output_dir / f"{name}_pair_stats.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_index",
                "image",
                "status",
                "used_fallback_motion",
                "keypoints_prev",
                "keypoints_curr",
                "matches",
                "essential_inliers",
                "pose_inliers",
            ],
        )
        writer.writeheader()
        writer.writerows(pair_stats)

    plt.figure(figsize=(7, 7))
    plt.plot(trajectory[:, 0], trajectory[:, 1], "b-", label="Estimated monocular VO")
    plt.scatter(trajectory[0, 0], trajectory[0, 1], c="g", label="start")
    plt.scatter(trajectory[-1, 0], trajectory[-1, 1], c="r", label="end")
    plt.axis("equal")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.title(f"{name} pseudo trajectory")
    plt.savefig(output_dir / f"{name}_trajectory.jpg", dpi=160)
    plt.close()

    ok_pairs = sum(1 for item in pair_stats if item["status"] == "ok")
    summary = {
        "image_count": len(image_paths),
        "pose_count": len(pose_records),
        "ok_pair_count": ok_pairs,
        "fallback_pair_count": len(pair_stats) - ok_pairs,
        "pose_json": str(pose_json),
        "pose_tum": str(tum_path),
    }
    (output_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
