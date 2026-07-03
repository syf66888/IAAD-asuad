#!/usr/bin/env python3
import argparse
import json
import pickle
import re
from pathlib import Path

import cv2
import numpy as np


DYNAMIC_CLASSES = {
    "car",
    "truck",
    "bus",
    "van",
    "motorcycle",
    "bicycle",
    "bike",
    "pedestrian",
    "person",
    "walker",
}


def natural_sort_key(path):
    text = Path(path).name
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def parse_label_file(path):
    objects = []
    with open(path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    for line in lines[1:]:
        fields = line.split()
        if len(fields) < 13:
            continue
        class_name = fields[0].lower()
        if class_name not in DYNAMIC_CLASSES:
            continue
        try:
            values = [float(item) for item in fields[1:10]]
            track_id = int(float(fields[10]))
        except ValueError:
            continue
        if track_id == -100:
            continue
        visible = fields[-1].lower() == "true"
        if not visible:
            continue
        x, y, z, length, width, height, yaw, _, _ = values
        objects.append(
            {
                "class_name": class_name,
                "center": np.array([x, y, z], dtype=float),
                "size": np.array([length, width, height], dtype=float),
                "yaw": yaw,
                "track_id": track_id,
            }
        )
    return objects


def load_camera_projection(calib_path, camera):
    with open(calib_path, "rb") as handle:
        calibration = pickle.load(handle)

    intrinsic = np.array(calibration[f"intrinsic_{camera}"], dtype=float)
    lidar_to_ego = np.array(calibration["lidar_to_ego"], dtype=float)
    lidar_to_camera = np.array(calibration[f"lidar_to_{camera}"], dtype=float)
    camera_to_ego = lidar_to_ego @ np.linalg.inv(lidar_to_camera)
    ego_to_camera = np.linalg.inv(camera_to_ego)
    return intrinsic, ego_to_camera


def box_corners_ego(obj):
    length, width, height = obj["size"]
    x_offsets = np.array([-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5]) * length
    y_offsets = np.array([-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5]) * width
    z_offsets = np.array([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5]) * height
    local = np.stack([x_offsets, y_offsets, z_offsets], axis=1)

    cos_yaw = np.cos(obj["yaw"])
    sin_yaw = np.sin(obj["yaw"])
    rotation = np.array(
        [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    return (rotation @ local.T).T + obj["center"]


def project_carla_camera(points_ego, intrinsic, ego_to_camera):
    points_h = np.concatenate([points_ego, np.ones((len(points_ego), 1), dtype=float)], axis=1)
    points_camera = (ego_to_camera @ points_h.T).T[:, :3]
    in_front = points_camera[:, 0] > 1e-3
    if not np.any(in_front):
        return np.zeros((0, 2), dtype=float)

    points_camera = points_camera[in_front]
    projected = (intrinsic @ points_camera.T).T
    valid = np.abs(projected[:, 2]) > 1e-6
    if not np.any(valid):
        return np.zeros((0, 2), dtype=float)
    projected = projected[valid]
    return projected[:, :2] / projected[:, 2:3]


def mask_image(image, label_path, calib_path, camera, dilation, fill_mode):
    objects = parse_label_file(label_path)
    intrinsic, ego_to_camera = load_camera_projection(calib_path, camera)
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    for obj in objects:
        corners = box_corners_ego(obj)
        points = project_carla_camera(corners, intrinsic, ego_to_camera)
        if len(points) < 2:
            continue
        x0, y0 = np.floor(points.min(axis=0)).astype(int)
        x1, y1 = np.ceil(points.max(axis=0)).astype(int)
        if x1 < 0 or y1 < 0 or x0 >= width or y0 >= height:
            continue
        x0 = max(0, x0 - dilation)
        y0 = max(0, y0 - dilation)
        x1 = min(width - 1, x1 + dilation)
        y1 = min(height - 1, y1 + dilation)
        if x1 <= x0 or y1 <= y0:
            continue
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)

    output = image.copy()
    if fill_mode == "black":
        output[mask > 0] = 0
    elif fill_mode == "blur":
        blurred = cv2.GaussianBlur(output, (35, 35), 0)
        output[mask > 0] = blurred[mask > 0]
    else:
        raise ValueError(f"Unsupported fill mode: {fill_mode}")
    return output, mask, len(objects)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--calib-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--camera", default="Camera_Front")
    parser.add_argument("--dilation", type=int, default=8)
    parser.add_argument("--fill-mode", choices=["black", "blur"], default="blur")
    parser.add_argument("--preview-count", type=int, default=5)
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    calib_dir = Path(args.calib_dir)
    output_dir = Path(args.output_dir)
    masked_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    previews_dir = output_dir / "previews"
    masked_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(image_dir.glob("*.jpg"), key=natural_sort_key)
    stats = []
    for idx, image_path in enumerate(image_paths):
        stem = image_path.stem
        label_path = label_dir / f"{stem}.txt"
        calib_path = calib_dir / f"{stem}.pkl"
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        if not label_path.exists() or not calib_path.exists():
            raise FileNotFoundError(f"Missing label/calib for {image_path.name}")

        masked, mask, object_count = mask_image(
            image,
            label_path,
            calib_path,
            args.camera,
            args.dilation,
            args.fill_mode,
        )
        cv2.imwrite(str(masked_dir / image_path.name), masked)
        cv2.imwrite(str(masks_dir / image_path.name), mask)
        coverage = float(np.count_nonzero(mask)) / float(mask.size)
        stats.append({"image": image_path.name, "object_count": object_count, "mask_coverage": coverage})

        if idx < args.preview_count:
            overlay = image.copy()
            red = np.zeros_like(overlay)
            red[:, :, 2] = 255
            overlay[mask > 0] = cv2.addWeighted(overlay, 0.45, red, 0.55, 0)[mask > 0]
            cv2.imwrite(str(previews_dir / image_path.name), overlay)

    summary = {
        "image_count": len(image_paths),
        "mean_mask_coverage": float(np.mean([item["mask_coverage"] for item in stats])) if stats else 0.0,
        "fill_mode": args.fill_mode,
        "masked_dir": str(masked_dir),
        "masks_dir": str(masks_dir),
    }
    (output_dir / "mask_stats.json").write_text(
        json.dumps({"summary": summary, "frames": stats}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
