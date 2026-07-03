#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def natural_sort_key(path):
    text = Path(path).name
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def yaw_matrix(yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return matrix


def yaw_to_quaternion_xyzw(yaw):
    return [0.0, 0.0, float(np.sin(yaw / 2.0)), float(np.cos(yaw / 2.0))]


def build_pose_matrices_from_positions(positions):
    matrices = []
    for idx, position in enumerate(positions):
        if len(positions) == 1:
            delta = np.array([1.0, 0.0], dtype=float)
        elif idx == 0:
            delta = positions[1, :2] - positions[0, :2]
        else:
            delta = positions[idx, :2] - positions[idx - 1, :2]
            if np.linalg.norm(delta) < 1e-9 and idx + 1 < len(positions):
                delta = positions[idx + 1, :2] - positions[idx, :2]
        yaw = float(np.arctan2(delta[1], delta[0])) if np.linalg.norm(delta) > 1e-9 else 0.0
        matrix = yaw_matrix(yaw)
        matrix[:3, 3] = position
        matrices.append(matrix)
    return matrices


def select_trajectory_plane(raw_positions, plane, scale):
    axis_lookup = {"x": 0, "y": 1, "z": 2}
    if len(plane) != 2 or any(axis not in axis_lookup for axis in plane):
        raise ValueError(f"Invalid trajectory plane {plane}; expected two letters from xyz")
    selected = raw_positions[:, [axis_lookup[plane[0]], axis_lookup[plane[1]]]].copy()
    selected *= float(scale)
    positions = np.zeros((len(raw_positions), 3), dtype=float)
    positions[:, :2] = selected
    return positions


def image_stream(image_dir, calib, stride):
    fx, fy, cx, cy = calib[:4]
    K = np.eye(3, dtype=float)
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy

    image_paths = sorted(Path(image_dir).glob("*"), key=natural_sort_key)
    image_paths = [path for path in image_paths if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    image_paths = image_paths[::stride]

    for t, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
        image = cv2.resize(image, (w1, h1))
        image = image[: h1 - h1 % 8, : w1 - w1 % 8]
        tensor = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy], dtype=torch.float32)
        intrinsics[0::2] *= image.shape[1] / w0
        intrinsics[1::2] *= image.shape[0] / h0
        yield t, tensor[None], intrinsics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--droid-root", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default="droid_sequence")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--frame-interval-s", type=float, default=0.5)
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--filter-thresh", type=float, default=2.4)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--keyframe-thresh", type=float, default=4.0)
    parser.add_argument("--frontend-thresh", type=float, default=16.0)
    parser.add_argument("--backend-thresh", type=float, default=22.0)
    parser.add_argument("--trajectory-plane", default="xy")
    parser.add_argument("--trajectory-scale", type=float, default=1.0)
    parser.add_argument("--asynchronous", action="store_true")
    args = parser.parse_args()

    droid_root = Path(args.droid_root).resolve()
    sys.path.insert(0, str(droid_root / "droid_slam"))
    sys.path.insert(0, str(droid_root))
    from droid import Droid
    from droid_async import DroidAsync

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calib = np.loadtxt(args.calib, delimiter=" ")
    stream_for_shape = list(image_stream(args.image_dir, calib, args.stride))
    if not stream_for_shape:
        raise FileNotFoundError(f"No images found in {args.image_dir}")

    droid_args = SimpleNamespace(
        weights=args.weights,
        buffer=args.buffer,
        image_size=[stream_for_shape[0][1].shape[2], stream_for_shape[0][1].shape[3]],
        disable_vis=True,
        beta=0.3,
        filter_thresh=args.filter_thresh,
        warmup=args.warmup,
        keyframe_thresh=args.keyframe_thresh,
        frontend_thresh=args.frontend_thresh,
        frontend_window=25,
        frontend_radius=2,
        frontend_nms=1,
        backend_thresh=args.backend_thresh,
        backend_radius=2,
        backend_nms=3,
        upsample=False,
        asynchronous=args.asynchronous,
        frontend_device="cuda",
        backend_device="cuda",
        stereo=False,
    )

    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    droid = DroidAsync(droid_args) if args.asynchronous else Droid(droid_args)
    with torch.no_grad():
        for t, image, intrinsics in stream_for_shape:
            droid.track(t, image, intrinsics=intrinsics)
        traj = droid.terminate(stream_for_shape)

    traj = np.asarray(traj, dtype=float)
    np.save(output_dir / f"{args.name}_droid_traj_raw.npy", traj)
    finite = np.isfinite(traj).all(axis=1)
    if not finite.all():
        print(f"Warning: DROID produced {finite.sum()} finite rows out of {len(finite)}")

    raw_positions = traj[:, :3]
    positions = select_trajectory_plane(raw_positions, args.trajectory_plane, args.trajectory_scale)
    matrices = build_pose_matrices_from_positions(positions)
    pose_records = []
    for idx, matrix in enumerate(matrices):
        pose_records.append(
            {
                "timestamp": idx * args.frame_interval_s * args.stride,
                "image_index": idx,
                "source_index": idx * args.stride,
                "matrix": matrix.tolist(),
            }
        )

    pose_json = output_dir / f"{args.name}_poses.json"
    pose_payload = {
        "name": args.name,
        "source_image_dir": str(Path(args.image_dir).resolve()),
        "source": "DROID-SLAM monocular",
        "monocular_scale_note": "Scale is arbitrary unless externally aligned.",
        "trajectory_plane": args.trajectory_plane,
        "trajectory_scale": args.trajectory_scale,
        "poses": pose_records,
        "raw_traj_format": "tx ty tz q...",
        "finite_pose_rows": int(finite.sum()),
        "pose_count": int(len(traj)),
    }
    pose_json.write_text(json.dumps(pose_payload, indent=2), encoding="utf-8")

    tum_path = output_dir / f"{args.name}_poses_tum.txt"
    with tum_path.open("w", encoding="utf-8") as handle:
        for record in pose_records:
            matrix = np.array(record["matrix"], dtype=float)
            yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
            qx, qy, qz, qw = yaw_to_quaternion_xyzw(yaw)
            tx, ty, tz = matrix[:3, 3]
            handle.write(
                f"{record['timestamp']:.9f} {tx:.9f} {ty:.9f} {tz:.9f} "
                f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"
            )

    plt.figure(figsize=(7, 7))
    plt.plot(positions[:, 0], positions[:, 1], "b-", label="DROID-SLAM")
    plt.scatter(positions[0, 0], positions[0, 1], c="g", label="start")
    plt.scatter(positions[-1, 0], positions[-1, 1], c="r", label="end")
    plt.axis("equal")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.title(args.name)
    plot_path = output_dir / f"{args.name}_trajectory.jpg"
    plt.savefig(plot_path, dpi=160)
    plt.close()

    summary = {
        "pose_json": str(pose_json),
        "pose_tum": str(tum_path),
        "plot": str(plot_path),
        "pose_count": int(len(traj)),
        "finite_pose_rows": int(finite.sum()),
        "trajectory_plane": args.trajectory_plane,
        "trajectory_scale": args.trajectory_scale,
        "path_length_selected_xy": float(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1).sum())
        if len(positions) > 1
        else 0.0,
        "path_length_raw_xyz": float(np.linalg.norm(np.diff(raw_positions, axis=0), axis=1).sum())
        if len(raw_positions) > 1
        else 0.0,
    }
    (output_dir / f"{args.name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
