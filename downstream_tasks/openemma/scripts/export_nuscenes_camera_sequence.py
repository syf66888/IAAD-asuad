#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from nuscenes import NuScenes
from pyquaternion import Quaternion


def transform_matrix(translation, rotation):
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = Quaternion(rotation).rotation_matrix
    matrix[:3, 3] = np.array(translation, dtype=float)
    return matrix


def link_or_copy(src, dst, copy_files):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_files:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def find_scene(nusc, scene_name):
    for scene in nusc.scene:
        if scene["name"] == scene_name:
            return scene
    raise ValueError(f"Scene not found: {scene_name}")


def collect_camera_chain(nusc, scene, camera, keyframes_only):
    first_sample = nusc.get("sample", scene["first_sample_token"])
    sd_token = first_sample["data"][camera]
    records = []
    while sd_token:
        sample_data = nusc.get("sample_data", sd_token)
        if not keyframes_only or sample_data["is_key_frame"]:
            records.append(sample_data)
        sd_token = sample_data["next"]
    if not records:
        raise ValueError(f"No camera records found for {scene['name']} {camera}")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--camera", default="CAM_FRONT")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--keyframes-only", action="store_true")
    parser.add_argument("--copy-images", action="store_true")
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    scene = find_scene(nusc, args.scene_name)
    camera_records = collect_camera_chain(nusc, scene, args.camera, args.keyframes_only)

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    first_timestamp = camera_records[0]["timestamp"]
    pose_records = []
    intrinsics = None
    for idx, sample_data in enumerate(camera_records):
        source_path = Path(args.dataroot) / sample_data["filename"]
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        dst_name = f"{args.scene_name}_{args.camera}_{idx:06d}{source_path.suffix.lower()}"
        dst_path = image_dir / dst_name
        link_or_copy(source_path, dst_path, args.copy_images)

        ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
        calibrated_sensor = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
        if intrinsics is None:
            intrinsics = np.array(calibrated_sensor["camera_intrinsic"], dtype=float)
        matrix = transform_matrix(ego_pose["translation"], ego_pose["rotation"])
        pose_records.append(
            {
                "timestamp": float((sample_data["timestamp"] - first_timestamp) * 1e-6),
                "image_index": idx,
                "source_index": idx,
                "image": dst_name,
                "source_filename": sample_data["filename"],
                "sample_data_token": sample_data["token"],
                "sample_token": sample_data["sample_token"],
                "is_key_frame": bool(sample_data["is_key_frame"]),
                "matrix": matrix.tolist(),
            }
        )

    calib_path = output_dir / "calib.txt"
    calib_path.write_text(
        f"{intrinsics[0, 0]:.9f} {intrinsics[1, 1]:.9f} {intrinsics[0, 2]:.9f} {intrinsics[1, 2]:.9f}\n",
        encoding="utf-8",
    )
    (output_dir / "calib.json").write_text(
        json.dumps({"camera_intrinsic": intrinsics.tolist(), "camera": args.camera}, indent=2),
        encoding="utf-8",
    )

    pose_path = output_dir / "gt_ego_poses.json"
    pose_payload = {
        "name": f"{args.scene_name}_{args.camera}_{'keyframes' if args.keyframes_only else 'all'}",
        "source": "NuScenes ego_pose",
        "dataroot": str(Path(args.dataroot).resolve()),
        "scene_name": args.scene_name,
        "camera": args.camera,
        "keyframes_only": bool(args.keyframes_only),
        "poses": pose_records,
    }
    pose_path.write_text(json.dumps(pose_payload, indent=2), encoding="utf-8")

    metadata = {
        "image_dir": str(image_dir.resolve()),
        "gt_pose": str(pose_path.resolve()),
        "calib": str(calib_path.resolve()),
        "scene_name": args.scene_name,
        "camera": args.camera,
        "frame_count": len(pose_records),
        "keyframe_count": sum(record["is_key_frame"] for record in pose_records),
        "duration_s": pose_records[-1]["timestamp"] - pose_records[0]["timestamp"],
        "mean_fps": (len(pose_records) - 1) / (pose_records[-1]["timestamp"] - pose_records[0]["timestamp"])
        if len(pose_records) > 1 and pose_records[-1]["timestamp"] > pose_records[0]["timestamp"]
        else None,
        "image_size_note": "original NuScenes camera image resolution",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
