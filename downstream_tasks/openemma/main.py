import argparse
import base64
import importlib.util
import json
import os
import pickle
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from math import atan2
from pathlib import Path, PurePosixPath

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from nuscenes import NuScenes
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, MllamaForConditionalGeneration, Qwen2VLForConditionalGeneration

from utils import (
    EstimateCurvatureFromTrajectory,
    IntegrateCurvatureForPoints,
    OverlayTrajectory,
    WriteImageSequenceToVideo,
)

OBS_LEN = 10
FUT_LEN = 10
TTL_LEN = OBS_LEN + FUT_LEN
FRAME_INTERVAL_S = 0.5
METRIC_HORIZON_STEPS = (
    ("ade_1s", int(round(1.0 / FRAME_INTERVAL_S))),
    ("ade_3s", int(round(3.0 / FRAME_INTERVAL_S))),
    ("ade_5s", int(round(5.0 / FRAME_INTERVAL_S))),
)
DEFAULT_LOCAL_QWEN_PATH = Path("/mnt/ubm_code_nas/gac_liulian/gac_syf/experiments/qwen2-vl")
DEFAULT_SCENES = ["scene-0103", "scene-1077"]


@dataclass
class DeepAccidentFrame:
    image_ref: str
    calib_ref: str
    label_ref: str
    frame_index: int


@dataclass
class DeepAccidentScene:
    name: str
    token: str
    category: str
    scenario: str
    frames: list[DeepAccidentFrame]


@dataclass
class ImageSequenceScene:
    name: str
    token: str
    category: str
    image_paths: list[str]
    ego_poses: list[dict]
    camera_params: list[dict]


class DeepAccidentSource:
    def __init__(self, dataroot):
        self.source_path = self._resolve_source_path(Path(dataroot))
        self.is_zip = self.source_path.is_file() and self.source_path.suffix.lower() == ".zip"
        self._archive = zipfile.ZipFile(self.source_path) if self.is_zip else None

    @staticmethod
    def _looks_like_extracted_root(root_path):
        if not root_path.is_dir():
            return False
        try:
            return any(child.is_dir() and child.name.startswith("type") for child in root_path.iterdir())
        except FileNotFoundError:
            return False

    @classmethod
    def _resolve_source_path(cls, dataroot_path):
        if dataroot_path.is_file() and dataroot_path.suffix.lower() == ".zip":
            return dataroot_path

        if dataroot_path.is_dir():
            if cls._looks_like_extracted_root(dataroot_path):
                return dataroot_path

            extracted_root = dataroot_path / "DeepAccident_data"
            if cls._looks_like_extracted_root(extracted_root):
                return extracted_root

            canonical_archive = dataroot_path / "DeepAccident.zip"
            if canonical_archive.is_file():
                return canonical_archive

            zip_candidates = sorted(path for path in dataroot_path.glob("*.zip") if "deepaccident" in path.name.lower())
            if len(zip_candidates) == 1:
                return zip_candidates[0]

        raise FileNotFoundError(
            "Could not resolve a DeepAccident source. Pass either the extracted dataset root, "
            "a directory containing `DeepAccident.zip`, or the zip file itself."
        )

    def close(self):
        if self._archive is not None:
            self._archive.close()
            self._archive = None

    def iter_files(self):
        if self.is_zip:
            for rel_path in self._archive.namelist():
                if not rel_path.endswith("/"):
                    yield rel_path
            return

        for path in self.source_path.rglob("*"):
            if path.is_file():
                yield path.relative_to(self.source_path).as_posix()

    @staticmethod
    def _swap_sensor_path(rel_path, current_sensor, target_sensor, new_suffix):
        parts = list(PurePosixPath(rel_path).parts)
        sensor_index = parts.index(current_sensor)
        parts[sensor_index] = target_sensor
        parts[-1] = f"{Path(parts[-1]).stem}{new_suffix}"
        return PurePosixPath(*parts).as_posix()

    @staticmethod
    def _parse_image_path(rel_path, view, camera):
        parts = PurePosixPath(rel_path).parts
        if not parts or not rel_path.lower().endswith(".jpg"):
            return None

        try:
            view_index = parts.index(view)
        except ValueError:
            return None

        if view_index < 1 or view_index + 3 >= len(parts):
            return None
        if parts[view_index + 1] != camera:
            return None

        category = parts[view_index - 1]
        scenario = parts[view_index + 2]

        try:
            frame_index = int(Path(parts[-1]).stem.rsplit("_", 1)[-1])
        except ValueError:
            return None

        return category, scenario, frame_index

    def build_scenes(self, view, camera):
        scenes_by_key = {}

        for rel_path in self.iter_files():
            parsed = self._parse_image_path(rel_path, view, camera)
            if parsed is None:
                continue

            category, scenario, frame_index = parsed
            scene_key = (category, scenario)
            frame = DeepAccidentFrame(
                image_ref=rel_path,
                calib_ref=self._swap_sensor_path(rel_path, camera, "calib", ".pkl"),
                label_ref=self._swap_sensor_path(rel_path, camera, "label", ".txt"),
                frame_index=frame_index,
            )
            scenes_by_key.setdefault(scene_key, []).append(frame)

        scenes = []
        for (category, scenario), frames in sorted(scenes_by_key.items()):
            frames.sort(key=lambda item: item.frame_index)
            scene_name = f"{category}__{scenario}"
            scenes.append(
                DeepAccidentScene(
                    name=scene_name,
                    token=f"{category}/{scenario}",
                    category=category,
                    scenario=scenario,
                    frames=frames,
                )
            )
        return scenes

    def read_bytes(self, rel_path):
        if self.is_zip:
            return self._archive.read(rel_path)
        return (self.source_path / rel_path).read_bytes()

    def load_calibration(self, calib_ref):
        payload = self.read_bytes(calib_ref)
        if self.is_zip:
            return pickle.loads(payload)
        with open(self.source_path / calib_ref, "rb") as handle:
            return pickle.load(handle)

    def load_image_pil(self, image_ref):
        image = Image.open(BytesIO(self.read_bytes(image_ref)))
        return image.convert("RGB")

    def load_image_base64(self, image_ref):
        return base64.b64encode(self.read_bytes(image_ref)).decode("utf-8")

    def load_image_bgr(self, image_ref):
        if self.is_zip:
            return cv2.imdecode(np.frombuffer(self.read_bytes(image_ref), dtype=np.uint8), cv2.IMREAD_COLOR)
        return cv2.imread(str(self.source_path / image_ref))


def natural_sort_key(path):
    text = Path(path).name
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def collect_image_sequence(image_dir):
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image sequence directory not found: {image_dir}")

    image_paths = []
    for suffix in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_paths.extend(str(path) for path in image_dir.glob(suffix))
        image_paths.extend(str(path) for path in image_dir.glob(suffix.upper()))

    image_paths = sorted(set(image_paths), key=natural_sort_key)
    if not image_paths:
        raise FileNotFoundError(f"No image files found in {image_dir}")
    return image_paths


def quaternion_xyzw_to_matrix(quaternion):
    qx, qy, qz, qw = [float(value) for value in quaternion]
    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-12:
        return np.eye(3)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def load_pose_sequence(pose_file):
    pose_path = Path(pose_file)
    if not pose_path.exists():
        raise FileNotFoundError(f"Pose file not found: {pose_path}")

    poses = []
    if pose_path.suffix.lower() == ".json":
        payload = json.loads(pose_path.read_text(encoding="utf-8"))
        records = payload.get("poses", payload) if isinstance(payload, dict) else payload
        for record in records:
            if isinstance(record, dict) and "matrix" in record:
                matrix = np.array(record["matrix"], dtype=float)
                pose_record = dict(record)
            elif isinstance(record, (list, tuple)):
                matrix = np.array(record, dtype=float)
                pose_record = {}
            else:
                raise ValueError(f"Unsupported pose record in {pose_path}: {record}")
            if matrix.shape != (4, 4):
                raise ValueError(f"Pose matrix must be 4x4 in {pose_path}, got {matrix.shape}")
            pose_record["matrix"] = matrix
            poses.append(pose_record)
    else:
        with open(pose_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) != 8:
                    raise ValueError(f"TUM pose rows must have 8 fields, got: {line}")
                _, tx, ty, tz, qx, qy, qz, qw = fields
                matrix = np.eye(4, dtype=float)
                matrix[:3, :3] = quaternion_xyzw_to_matrix([qx, qy, qz, qw])
                matrix[:3, 3] = [float(tx), float(ty), float(tz)]
                poses.append({"matrix": matrix})

    if not poses:
        raise ValueError(f"No poses loaded from {pose_path}")

    pose_array = np.array([pose["matrix"] for pose in poses], dtype=float)
    if not np.isfinite(pose_array).all():
        raise ValueError(f"Pose file contains non-finite values: {pose_path}")
    return poses


def load_image_sequence_intrinsics(intrinsics_file, first_image_path):
    if intrinsics_file:
        path = Path(intrinsics_file)
        if not path.exists():
            raise FileNotFoundError(f"Intrinsics file not found: {path}")
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if "camera_intrinsic" in payload:
                return np.array(payload["camera_intrinsic"], dtype=float)
            if "K" in payload:
                return np.array(payload["K"], dtype=float)
            fx, fy, cx, cy = (payload[key] for key in ("fx", "fy", "cx", "cy"))
            return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)

        values = [float(item) for item in path.read_text(encoding="utf-8").split()]
        if len(values) == 4:
            fx, fy, cx, cy = values
            return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)
        if len(values) == 9:
            return np.array(values, dtype=float).reshape(3, 3)
        raise ValueError(f"Unsupported intrinsics format in {path}")

    image = cv2.imread(str(first_image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read first image for auto intrinsics: {first_image_path}")
    height, width = image.shape[:2]
    focal = 0.8 * width
    return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=float)


def build_image_sequence_camera_param(camera_intrinsic):
    camera_to_ego = np.eye(4, dtype=float)
    # OpenCV camera axes: x right, y down, z forward. Ego axes: x forward, y left, z up.
    camera_to_ego[:3, :3] = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=float,
    )
    return {
        "matrix": camera_to_ego,
        "camera_intrinsic": camera_intrinsic,
        "camera_coordinate": "opencv",
    }


def build_image_sequence_scenes(args):
    image_paths = collect_image_sequence(args.dataroot)
    if not args.pose_file:
        raise ValueError("--pose-file is required when --dataset-type image_sequence")

    ego_poses = load_pose_sequence(args.pose_file)
    usable_length = min(len(image_paths), len(ego_poses))
    if usable_length < len(image_paths) or usable_length < len(ego_poses):
        print(
            "Image/pose length mismatch for image_sequence: "
            f"{len(image_paths)} images, {len(ego_poses)} poses. Using first {usable_length} pairs."
        )
        image_paths = image_paths[:usable_length]
        ego_poses = ego_poses[:usable_length]

    camera_intrinsic = load_image_sequence_intrinsics(args.image_sequence_intrinsics, image_paths[0])
    camera_param = build_image_sequence_camera_param(camera_intrinsic)
    scene_name = args.image_sequence_name or Path(args.dataroot).name
    scene = ImageSequenceScene(
        name=scene_name,
        token=str(Path(args.dataroot).resolve()),
        category=args.image_sequence_category,
        image_paths=image_paths,
        ego_poses=ego_poses,
        camera_params=[camera_param for _ in image_paths],
    )
    return [scene]


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def is_existing_path(path_str):
    return isinstance(path_str, str) and Path(path_str).exists()


def format_seconds(seconds):
    return f"{float(seconds):g}"


def build_metric_horizon_steps(frame_interval_s):
    return (
        ("ade_1s", max(1, int(round(1.0 / frame_interval_s)))),
        ("ade_3s", max(1, int(round(3.0 / frame_interval_s)))),
        ("ade_5s", max(1, int(round(5.0 / frame_interval_s)))),
    )


def resolve_dataset_type(args):
    if args.dataset_type != "auto":
        return args.dataset_type

    dataroot_path = Path(args.dataroot)
    if getattr(args, "pose_file", None):
        return "image_sequence"

    if dataroot_path.is_file() and dataroot_path.suffix.lower() == ".zip":
        return "deepaccident"

    if (dataroot_path / args.version / "sample.json").exists():
        return "nuscenes"

    if DeepAccidentSource._looks_like_extracted_root(dataroot_path):
        return "deepaccident"

    if DeepAccidentSource._looks_like_extracted_root(dataroot_path / "DeepAccident_data"):
        return "deepaccident"

    if dataroot_path.is_dir():
        if (dataroot_path / "DeepAccident.zip").is_file():
            return "deepaccident"
        if any("deepaccident" in path.name.lower() for path in dataroot_path.glob("*.zip")):
            return "deepaccident"

    return "nuscenes"


def extract_translation(pose_record):
    if isinstance(pose_record, dict):
        if "translation" in pose_record:
            return np.array(pose_record["translation"][:3], dtype=float)
        if "matrix" in pose_record:
            return np.array(pose_record["matrix"], dtype=float)[:3, 3]

    pose_matrix = np.array(pose_record, dtype=float)
    if pose_matrix.shape == (4, 4):
        return pose_matrix[:3, 3]
    raise ValueError(f"Unsupported pose record format: {type(pose_record)}")


def infer_headings_from_velocities(velocities):
    headings = np.zeros(len(velocities), dtype=float)
    fallback = 0.0
    for idx, velocity in enumerate(velocities):
        speed = np.linalg.norm(velocity[:2])
        if speed > 1e-9:
            fallback = atan2(velocity[1], velocity[0])
        headings[idx] = fallback
    return np.unwrap(headings)


def build_motion_arrays(ego_poses, ego_poses_world, frame_interval_s):
    ego_velocities = np.zeros_like(ego_poses_world)
    ego_velocities[1:] = ego_poses_world[1:] - ego_poses_world[:-1]
    if len(ego_velocities) > 1:
        ego_velocities[0] = ego_velocities[1]

    ego_curvatures = EstimateCurvatureFromTrajectory(ego_poses_world)
    ego_headings = infer_headings_from_velocities(ego_velocities)

    if not all(isinstance(pose, dict) for pose in ego_poses):
        return ego_velocities, ego_curvatures, ego_headings

    has_velocity = all("velocity" in pose for pose in ego_poses)
    has_heading = all("heading_rad" in pose for pose in ego_poses)
    has_speed = all("speed_per_step" in pose or "speed_mps" in pose for pose in ego_poses)
    has_curvature = all("curvature" in pose for pose in ego_poses)

    if has_heading:
        ego_headings = np.unwrap(np.array([float(pose["heading_rad"]) for pose in ego_poses], dtype=float))

    if has_velocity:
        ego_velocities = np.array([pose["velocity"][:3] for pose in ego_poses], dtype=float)
    elif has_heading and has_speed:
        speeds = []
        for pose in ego_poses:
            if "speed_per_step" in pose:
                speeds.append(float(pose["speed_per_step"]))
            else:
                speeds.append(float(pose["speed_mps"]) * frame_interval_s)
        speeds = np.array(speeds, dtype=float)
        ego_velocities = np.zeros_like(ego_poses_world)
        ego_velocities[:, 0] = speeds * np.cos(ego_headings)
        ego_velocities[:, 1] = speeds * np.sin(ego_headings)

    if has_curvature:
        ego_curvatures = np.array([float(pose["curvature"]) for pose in ego_poses], dtype=float)

    return ego_velocities, ego_curvatures, ego_headings


def build_deepaccident_ego_pose(calibration):
    return {"matrix": np.array(calibration["ego_to_world"], dtype=float)}


def build_deepaccident_camera_params(calibration, camera_name):
    intrinsic_key = f"intrinsic_{camera_name}"
    lidar_to_camera_key = f"lidar_to_{camera_name}"

    if intrinsic_key not in calibration or lidar_to_camera_key not in calibration:
        raise KeyError(
            f"DeepAccident calibration is missing {intrinsic_key} or {lidar_to_camera_key}."
        )

    lidar_to_ego = np.array(calibration["lidar_to_ego"], dtype=float)
    lidar_to_camera = np.array(calibration[lidar_to_camera_key], dtype=float)
    camera_to_ego = lidar_to_ego @ np.linalg.inv(lidar_to_camera)

    return {
        "matrix": camera_to_ego,
        "camera_intrinsic": np.array(calibration[intrinsic_key], dtype=float),
        "camera_coordinate": "carla",
    }


def load_image_rgb(image_source):
    if isinstance(image_source, Image.Image):
        return image_source.convert("RGB")
    if isinstance(image_source, (bytes, bytearray)):
        return Image.open(BytesIO(image_source)).convert("RGB")
    return Image.open(image_source).convert("RGB")


def materialize_image_inputs(image_refs, dataset_type, dataset_source, args):
    refs = normalize_images(image_refs)
    single_input = not isinstance(image_refs, (list, tuple))
    prepared = []

    for image_ref in refs:
        if dataset_type == "deepaccident":
            if is_gpt_model(args.model_path):
                prepared.append(dataset_source.load_image_base64(image_ref))
            else:
                prepared.append(dataset_source.load_image_pil(image_ref))
            continue

        if is_gpt_model(args.model_path):
            with open(image_ref, "rb") as image_file:
                prepared.append(base64.b64encode(image_file.read()).decode("utf-8"))
        else:
            prepared.append(image_ref)

    if single_input:
        return prepared[0]
    return prepared


def load_visualization_image(image_ref, dataset_type, dataset_source):
    if dataset_type == "deepaccident":
        return dataset_source.load_image_bgr(image_ref)
    return cv2.imread(str(image_ref))


def is_qwen_model(model_path, local_model_path=None):
    lowered = model_path.lower()
    return "qwen" in lowered or bool(local_model_path) or is_existing_path(model_path)


def is_llava_model(model_path):
    return "llava" in model_path.lower()


def is_llama_model(model_path):
    lowered = model_path.lower()
    return "llama" in lowered and "llava" not in lowered


def is_gpt_model(model_path):
    return "gpt" in model_path.lower()


def normalize_images(images):
    if images is None:
        return []
    if isinstance(images, (list, tuple)):
        return list(images)
    return [images]


def safe_model_tag(model_path, local_model_path=None):
    reference = local_model_path or model_path
    name = Path(reference).name or reference
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def has_flash_attn():
    return importlib.util.find_spec("flash_attn") is not None


def resolve_torch_dtype(dtype_name):
    if dtype_name == "auto":
        return None
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def get_model_device(model):
    device = getattr(model, "device", None)
    if device is not None and str(device) != "meta":
        return device
    return next(model.parameters()).device


def load_extra_scene_text(args):
    parts = []
    if args.extra_scene_text:
        parts.append(args.extra_scene_text.strip())
    if args.extra_scene_text_file:
        extra_path = Path(args.extra_scene_text_file)
        if not extra_path.exists():
            raise FileNotFoundError(f"Extra scene text file not found: {extra_path}")
        parts.append(extra_path.read_text(encoding="utf-8").strip())
    return "\n".join(part for part in parts if part)


def build_external_guidance(extra_scene_text, stage):
    if not extra_scene_text:
        return ""
    if stage == "intent":
        return (
            "External safety note from a human annotator about unusual or hazardous events: "
            f"{extra_scene_text}\n"
            "When describing the ego intent, absorb this note into the high-level maneuver choice when it "
            "does not conflict with the image evidence, lane markings, or traffic rules.\n"
        )
    return (
        "External safety note from a human annotator: "
        f"{extra_scene_text}\n"
        "Treat this note as high-priority planning guidance when it does not conflict with "
        "the image evidence, lane markings, or traffic rules.\n"
    )


def stage_uses_extra_scene_text(args, stage):
    placement = getattr(args, "extra_scene_text_placement", "motion")
    return placement in {stage, "both"}


def resolve_qwen_source(args):
    candidates = []
    if args.local_model_path:
        candidates.append(Path(args.local_model_path))
    if is_existing_path(args.model_path):
        candidates.append(Path(args.model_path))

    env_path = os.environ.get("OPENEMMA_QWEN_PATH")
    if env_path:
        candidates.append(Path(env_path))

    if DEFAULT_LOCAL_QWEN_PATH.exists():
        candidates.append(DEFAULT_LOCAL_QWEN_PATH)

    seen = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return str(resolved)

    if args.allow_hf_fallback:
        return "Qwen/Qwen2-VL-7B-Instruct"

    raise FileNotFoundError(
        "Could not find a local Qwen2-VL model directory. "
        "Set --local-model-path or OPENEMMA_QWEN_PATH."
    )


def resolve_qwen_attn(args):
    if args.attn_implementation != "auto":
        return args.attn_implementation
    if args.prefer_flash_attn and has_flash_attn():
        return "flash_attention_2"
    return None


def load_qwen_model(args):
    model_source = resolve_qwen_source(args)
    processor_kwargs = {}
    if args.min_pixels is not None:
        processor_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels is not None:
        processor_kwargs["max_pixels"] = args.max_pixels

    dtype_candidates = [resolve_torch_dtype(args.qwen_dtype)]
    if dtype_candidates[0] is torch.bfloat16:
        dtype_candidates.append(torch.float16)
    if None not in dtype_candidates:
        dtype_candidates.append(None)

    attn_candidates = [resolve_qwen_attn(args)]
    if attn_candidates[0] == "flash_attention_2":
        attn_candidates.append(None)
    elif attn_candidates[0] is None:
        attn_candidates = [None]

    last_error = None
    for dtype in dtype_candidates:
        for attn_impl in attn_candidates:
            load_kwargs = {"device_map": "auto"}
            if dtype is not None:
                load_kwargs["torch_dtype"] = dtype
            if attn_impl is not None:
                load_kwargs["attn_implementation"] = attn_impl

            try:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model_source, **load_kwargs)
                processor = AutoProcessor.from_pretrained(model_source, **processor_kwargs)
                print(
                    f"Loaded Qwen2-VL from {model_source} "
                    f"(dtype={dtype or 'auto'}, attn={attn_impl or 'default'})."
                )
                return model, processor, None, model_source
            except Exception as exc:
                last_error = exc
                print(
                    "Qwen load attempt failed with "
                    f"dtype={dtype or 'auto'}, attn={attn_impl or 'default'}: {exc}"
                )

    raise RuntimeError(f"Failed to load Qwen2-VL from {model_source}: {last_error}")


def load_llama_model(args):
    model = MllamaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    print(f"Loaded Llama vision model from {args.model_path}.")
    return model, processor, None, args.model_path


def load_llava_model(args):
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    disable_torch_init()
    model_ref = "liuhaotian/llava-v1.6-mistral-7b" if args.model_path == "llava" else args.model_path
    tokenizer, model, processor, _ = load_pretrained_model(model_ref, None, "llava-v1.6-mistral-7b")
    print(f"Loaded LLaVA model from {model_ref}.")
    return model, processor, tokenizer, model_ref


def load_model_bundle(args):
    if is_qwen_model(args.model_path, args.local_model_path):
        return load_qwen_model(args)
    if is_llava_model(args.model_path):
        return load_llava_model(args)
    if is_llama_model(args.model_path):
        return load_llama_model(args)
    if is_gpt_model(args.model_path):
        print("Using GPT API mode, no local model weights will be loaded.")
        return None, None, None, args.model_path
    raise ValueError(f"Unsupported model path: {args.model_path}")


def build_qwen_message(prompt, images):
    content = []
    for image in normalize_images(images):
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def get_openai_client():
    from openai import OpenAI

    return OpenAI()


def get_yolo3d_inference():
    from openemma.YOLO3D.inference import yolo3d_nuScenes

    return yolo3d_nuScenes


def vlm_inference(text=None, images=None, sys_message=None, processor=None, model=None, tokenizer=None, args=None):
    if is_llama_model(args.model_path):
        image = load_image_rgb(normalize_images(images)[-1])
        message = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        input_text = processor.apply_chat_template(message, add_generation_prompt=True)
        inputs = processor(image, input_text, add_special_tokens=False, return_tensors="pt")
        inputs = inputs.to(get_model_device(model))
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        output_text = processor.decode(output[0])
        match = re.findall(
            r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>",
            output_text,
            re.DOTALL,
        )
        return match[0].strip() if match else output_text

    if is_qwen_model(args.model_path, args.local_model_path):
        message = build_qwen_message(text, images)
        text_prompt = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(message)
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(get_model_device(model))
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        trimmed_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]

    if is_llava_model(args.model_path):
        from llava.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_PLACEHOLDER,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token

        prompt_text = text
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in prompt_text:
            if model.config.mm_use_im_start_end:
                prompt_text = re.sub(IMAGE_PLACEHOLDER, image_token_se, prompt_text)
            else:
                prompt_text = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, prompt_text)
        elif model.config.mm_use_im_start_end:
            prompt_text = image_token_se + "\n" + prompt_text
        else:
            prompt_text = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text

        conv = conv_templates["mistral_instruct"].copy()
        conv.append_message(conv.roles[0], prompt_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        device = get_model_device(model)
        input_ids = tokenizer_image_token(
            prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(device)
        image = load_image_rgb(normalize_images(images)[-1])
        image_tensor = process_images([image], processor, model.config)[0]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).to(device=device, dtype=torch.float16),
                image_sizes=[image.size],
                do_sample=True,
                temperature=0.2,
                top_p=None,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    if is_gpt_model(args.model_path):
        client = get_openai_client()
        prompt_messages = [
            {
                "role": "user",
                "content": [*map(lambda item: {"image": item, "resize": 768}, normalize_images(images)), text],
            }
        ]
        if sys_message is not None:
            prompt_messages.append({"role": "system", "content": sys_message})
        result = client.chat.completions.create(
            model=args.gpt_model_name,
            messages=prompt_messages,
            max_tokens=400,
        )
        return result.choices[0].message.content

    raise ValueError(f"Unsupported model path for inference: {args.model_path}")


def SceneDescription(obs_images, processor=None, model=None, tokenizer=None, args=None):
    prompt = (
        "You are an autonomous driving labeller. You have access to front-view camera "
        f"images of a car taken at a {format_seconds(args.frame_interval_s)} second interval "
        f"over the past {format_seconds(OBS_LEN * args.frame_interval_s)} seconds. "
        "Imagine you are driving the car. Describe the driving scene according to "
        "traffic lights, movements of other cars or pedestrians, unusual hazards, and lane markings."
    )
    return vlm_inference(
        text=prompt,
        images=obs_images,
        processor=processor,
        model=model,
        tokenizer=tokenizer,
        args=args,
    )


def DescribeObjects(obs_images, processor=None, model=None, tokenizer=None, args=None):
    prompt = (
        "You are an autonomous driving labeller. You have access to front-view camera "
        f"images of a vehicle taken at a {format_seconds(args.frame_interval_s)} second interval "
        f"over the past {format_seconds(OBS_LEN * args.frame_interval_s)} seconds. "
        "Imagine you are driving the car. List two or three road users or obstacles "
        "that deserve attention, specify their image locations, summarize what they "
        "are doing, and explain why they matter."
    )
    return vlm_inference(
        text=prompt,
        images=obs_images,
        processor=processor,
        model=model,
        tokenizer=tokenizer,
        args=args,
    )


def DescribeOrUpdateIntent(
    obs_images,
    prev_intent=None,
    extra_scene_text=None,
    processor=None,
    model=None,
    tokenizer=None,
    args=None,
):
    external_guidance = ""
    if stage_uses_extra_scene_text(args, "intent"):
        external_guidance = build_external_guidance(extra_scene_text, "intent")

    if prev_intent is None:
        prompt = (
            "You are an autonomous driving labeller. You have access to front-view camera "
            f"images of a vehicle taken at a {format_seconds(args.frame_interval_s)} second interval "
            f"over the past {format_seconds(OBS_LEN * args.frame_interval_s)} seconds. "
            f"{external_guidance}"
            "Imagine you are driving the car. Based on lane markings and the movement of "
            "other cars and pedestrians, describe the desired intent of the ego car. "
            "Should it go straight, turn left, turn right, maintain speed, slow down, or speed up?"
        )
    else:
        prompt = (
            "You are an autonomous driving labeller. You have access to front-view camera "
            f"images of a vehicle taken at a {format_seconds(args.frame_interval_s)} second interval "
            f"over the past {format_seconds(OBS_LEN * args.frame_interval_s)} seconds. "
            f"{external_guidance}"
            f"{format_seconds(args.frame_interval_s)} seconds ago your intent was: {prev_intent}. "
            "Based on the updated scene, "
            "do you keep or change that intent? Explain the current intent."
        )
    return vlm_inference(
        text=prompt,
        images=obs_images,
        processor=processor,
        model=model,
        tokenizer=tokenizer,
        args=args,
    )


def GenerateMotion(
    obs_images,
    obs_waypoints,
    obs_velocities,
    obs_curvatures,
    given_intent,
    extra_scene_text=None,
    processor=None,
    model=None,
    tokenizer=None,
    args=None,
):
    scene_description = None
    object_description = None
    intent_description = None

    if args.method == "openemma":
        scene_description = SceneDescription(
            obs_images,
            processor=processor,
            model=model,
            tokenizer=tokenizer,
            args=args,
        )
        object_description = DescribeObjects(
            obs_images,
            processor=processor,
            model=model,
            tokenizer=tokenizer,
            args=args,
        )
        intent_description = DescribeOrUpdateIntent(
            obs_images,
            prev_intent=given_intent,
            extra_scene_text=extra_scene_text,
            processor=processor,
            model=model,
            tokenizer=tokenizer,
            args=args,
        )
        print(f"Scene Description: {scene_description}")
        print(f"Object Description: {object_description}")
        print(f"Intent Description: {intent_description}")

    obs_waypoints_str = ", ".join(f"[{x[0]:.2f},{x[1]:.2f}]" for x in obs_waypoints)
    obs_velocities_norm = np.linalg.norm(obs_velocities, axis=1)
    scaled_curvatures = obs_curvatures * 100
    obs_speed_curvature_str = ", ".join(
        f"[{speed:.1f},{curvature:.1f}]"
        for speed, curvature in zip(obs_velocities_norm, scaled_curvatures)
    )
    print(f"Observed Waypoints: {obs_waypoints_str}")
    print(f"Observed Speed and Curvature: {obs_speed_curvature_str}")

    sys_message = (
        "You are an autonomous driving labeller. You have access to front-view camera "
        "images of a vehicle, a sequence of past speeds, a sequence of past curvatures, "
        "and optional driving rationales. Each speed-curvature pair is represented as "
        "[v, k], where v is speed and k is curvature. Positive curvature means turning left, "
        "negative curvature means turning right, and curvature close to zero means driving straight. "
        "Follow common-sense traffic rules, stay near the center of the lane, maintain safe distance, "
        "and obey lane markings. Predict future speeds and curvatures for the next 10 timesteps. "
        "Return exactly 10 bracketed numeric pairs and nothing else, for example: "
        "[1.0, 0.0], [1.0, 0.0], [0.8, -0.2], [0.8, -0.2], [0.7, -0.1], "
        "[0.7, 0.0], [0.8, 0.0], [0.8, 0.1], [0.9, 0.1], [1.0, 0.0]."
    )

    external_guidance = ""
    if stage_uses_extra_scene_text(args, "motion"):
        external_guidance = build_external_guidance(extra_scene_text, "motion")

    if args.method == "openemma":
        prompt = (
            "These are frames from a video taken by a front-view camera mounted on a car. "
            f"The images are sampled every {format_seconds(args.frame_interval_s)} seconds.\n"
            f"The scene is described as follows: {scene_description}\n"
            f"The identified critical objects are: {object_description}\n"
            f"The car's intent is: {intent_description}\n"
            f"{external_guidance}"
            f"The {format_seconds(OBS_LEN * args.frame_interval_s)}-second historical ego speeds "
            f"and curvatures are: {obs_speed_curvature_str}\n"
            "Infer the association between these numbers and the image sequence. Generate the predicted "
            "future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2], "
            "..., [speed_10, curvature_10]. Do not include explanations, numbering, markdown, or units. "
            "Future speeds and curvatures:"
        )
    else:
        prompt = (
            "These are frames from a video taken by a front-view camera mounted on a car. "
            f"The images are sampled every {format_seconds(args.frame_interval_s)} seconds.\n"
            f"{external_guidance}"
            f"The {format_seconds(OBS_LEN * args.frame_interval_s)}-second historical ego speeds "
            f"and curvatures are: {obs_speed_curvature_str}\n"
            "Infer the association between these numbers and the image sequence. Generate the predicted "
            "future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2], "
            "..., [speed_10, curvature_10]. Do not include explanations, numbering, markdown, or units. "
            "Future speeds and curvatures:"
        )

    result = ""
    for _ in range(3):
        result = vlm_inference(
            text=prompt,
            images=obs_images,
            sys_message=sys_message,
            processor=processor,
            model=model,
            tokenizer=tokenizer,
            args=args,
        )
        lowered = result.lower()
        if "unable" not in lowered and "sorry" not in lowered and "[" in result:
            break

    return result, scene_description, object_description, intent_description


def parse_speed_curvature_pairs(prediction_text):
    coordinates = re.findall(r"\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]", prediction_text)
    return [[float(speed), float(curvature)] for speed, curvature in coordinates[:FUT_LEN]]


def select_observation_images(obs_images, args):
    if is_gpt_model(args.model_path):
        return obs_images
    if is_qwen_model(args.model_path, args.local_model_path) and args.image_input_mode == "sequence":
        return obs_images
    return obs_images[-1]


def rollout_future_trajectory(curvatures, speeds, initial_position, initial_heading):
    if len(curvatures) == 0:
        return np.zeros((0, 3))

    # Duplicate the first action so the integrated samples line up with t=0.5s, 1.0s, ..., 5.0s.
    aligned_curvatures = np.concatenate(([curvatures[0]], curvatures))
    aligned_speeds = np.concatenate(([speeds[0]], speeds))
    aligned_xy = IntegrateCurvatureForPoints(
        aligned_curvatures,
        aligned_speeds,
        initial_position,
        initial_heading,
        len(aligned_speeds),
    )[1:]

    pred_traj = np.zeros((len(curvatures), 3))
    pred_traj[:, :2] = aligned_xy
    return pred_traj


def compute_horizon_ade(gt_future, pred_traj, horizon_steps):
    if len(gt_future) < horizon_steps or len(pred_traj) < horizon_steps:
        return None
    return float(np.mean(np.linalg.norm(gt_future[:horizon_steps] - pred_traj[:horizon_steps], axis=1)))


def compute_scene_metrics(values):
    valid_values = [value for value in values if value is not None]
    if not valid_values:
        return None
    return float(np.mean(valid_values))


def resolve_output_dir(args):
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if args.save_folder:
            output_root = Path(args.save_folder)
        else:
            output_root = Path(f"{safe_model_tag(args.model_path, args.local_model_path)}_results") / args.method
        output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_completed_scene_names(output_dir):
    results_path = output_dir / "ade_results.jsonl"
    if not results_path.exists():
        return set()

    completed = set()
    with open(results_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            scene_name = record.get("name")
            if scene_name:
                completed.add(scene_name)
    return completed


def scene_is_selected(scene_name, selected_scenes):
    if not selected_scenes or selected_scenes == ["all"]:
        return True
    selected_set = set(selected_scenes)
    if scene_name in selected_set:
        return True
    return any(selected in scene_name for selected in selected_set)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="qwen")
    parser.add_argument("--local-model-path", type=str, default=str(DEFAULT_LOCAL_QWEN_PATH))
    parser.add_argument("--plot", type=str2bool, default=True)
    parser.add_argument("--dataroot", type=str, default="datasets/NuScenes")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--dataset-type", choices=["auto", "nuscenes", "deepaccident", "image_sequence"], default="auto")
    parser.add_argument("--method", type=str, default="openemma")
    parser.add_argument("--save-folder", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--scene-names", nargs="*", default=None)
    parser.add_argument("--max-windows-per-scene", type=int, default=None)
    parser.add_argument("--image-input-mode", choices=["last_frame", "sequence"], default="last_frame")
    parser.add_argument("--extra-scene-text", type=str, default=None)
    parser.add_argument("--extra-scene-text-file", type=str, default=None)
    parser.add_argument("--extra-scene-text-placement", choices=["motion", "intent", "both"], default="motion")
    parser.add_argument("--frame-interval-s", type=float, default=FRAME_INTERVAL_S)
    parser.add_argument(
        "--min-required-future-steps",
        type=int,
        default=FUT_LEN,
        help=(
            "Minimum future pose steps required to create a planning window. "
            "The model still predicts 10 future actions; shorter scenes can be "
            "used for ADE@1s/ADE@3s when this is set below 10."
        ),
    )
    parser.add_argument("--deepaccident-view", type=str, default="ego_vehicle")
    parser.add_argument("--deepaccident-camera", type=str, default="Camera_Front")
    parser.add_argument("--pose-file", type=str, default=None)
    parser.add_argument("--image-sequence-name", type=str, default=None)
    parser.add_argument("--image-sequence-category", type=str, default="monocular_pose_estimate")
    parser.add_argument("--image-sequence-intrinsics", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--gpt-model-name", type=str, default="gpt-4o-2024-11-20")
    parser.add_argument("--qwen-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", choices=["auto", "flash_attention_2", "sdpa", "eager"], default="auto")
    parser.add_argument("--prefer-flash-attn", type=str2bool, default=False)
    parser.add_argument("--allow-hf-fallback", type=str2bool, default=False)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    args = parser.parse_args()

    output_dir = resolve_output_dir(args)
    extra_scene_text = load_extra_scene_text(args)
    dataset_type = resolve_dataset_type(args)
    metric_horizon_steps = build_metric_horizon_steps(args.frame_interval_s)
    args.min_required_future_steps = max(1, min(FUT_LEN, int(args.min_required_future_steps)))
    required_scene_length = OBS_LEN + args.min_required_future_steps

    print(f"Model path argument: {args.model_path}")
    print(f"Resolved dataset type: {dataset_type}")
    print(f"Results will be written to: {output_dir}")

    model, processor, tokenizer, model_source = load_model_bundle(args)
    print(f"Resolved model source: {model_source}")

    dataset_source = None
    if dataset_type == "nuscenes":
        dataset_source = NuScenes(version=args.version, dataroot=args.dataroot)
        scenes = dataset_source.scene
    elif dataset_type == "deepaccident":
        dataset_source = DeepAccidentSource(args.dataroot)
        scenes = dataset_source.build_scenes(args.deepaccident_view, args.deepaccident_camera)
    elif dataset_type == "image_sequence":
        scenes = build_image_sequence_scenes(args)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    print(f"Number of scenes in dataset: {len(scenes)}")

    if args.scene_names is None:
        selected_scenes = DEFAULT_SCENES if dataset_type == "nuscenes" else ["all"]
    else:
        selected_scenes = args.scene_names

    if selected_scenes:
        print(f"Selected scenes: {selected_scenes}")

    completed_scene_names = load_completed_scene_names(output_dir)
    if completed_scene_names:
        print(f"Resuming run with {len(completed_scene_names)} completed scenes already recorded.")

    try:
        for scene in scenes:
            if dataset_type == "nuscenes":
                token = scene["token"]
                first_sample_token = scene["first_sample_token"]
                last_sample_token = scene["last_sample_token"]
                name = scene["name"]
                scene_category = None
            elif dataset_type == "deepaccident":
                token = scene.token
                name = scene.name
                scene_category = scene.category
            else:
                token = scene.token
                name = scene.name
                scene_category = scene.category

            if not scene_is_selected(name, selected_scenes):
                continue
            if name in completed_scene_names:
                print(f"Skipping completed scene {name}")
                continue

            front_camera_images = []
            ego_poses = []
            camera_params = []

            if dataset_type == "nuscenes":
                curr_sample_token = first_sample_token

                while True:
                    sample = dataset_source.get("sample", curr_sample_token)
                    cam_front_data = dataset_source.get("sample_data", sample["data"]["CAM_FRONT"])
                    front_camera_images.append(os.path.join(dataset_source.dataroot, cam_front_data["filename"]))
                    ego_poses.append(dataset_source.get("ego_pose", cam_front_data["ego_pose_token"]))
                    camera_params.append(dataset_source.get("calibrated_sensor", cam_front_data["calibrated_sensor_token"]))

                    if curr_sample_token == last_sample_token:
                        break
                    curr_sample_token = sample["next"]
            elif dataset_type == "deepaccident":
                for frame in scene.frames:
                    front_camera_images.append(frame.image_ref)
                    calibration = dataset_source.load_calibration(frame.calib_ref)
                    ego_poses.append(build_deepaccident_ego_pose(calibration))
                    camera_params.append(build_deepaccident_camera_params(calibration, args.deepaccident_camera))
            else:
                front_camera_images = list(scene.image_paths)
                ego_poses = list(scene.ego_poses)
                camera_params = list(scene.camera_params)

            scene_length = len(front_camera_images)
            print(f"Scene {name} has {scene_length} frames")

            if scene_length <= required_scene_length:
                print(
                    f"Scene {name} has {scene_length} frames; needs more than "
                    f"{required_scene_length} frames for {args.min_required_future_steps} "
                    "required future steps, skipping."
                )
                continue

            ego_poses_world = np.array([extract_translation(ego_poses[idx]) for idx in range(scene_length)])
            ego_velocities, ego_curvatures, ego_headings = build_motion_arrays(
                ego_poses,
                ego_poses_world,
                args.frame_interval_s,
            )
            ego_velocities_norm = np.linalg.norm(ego_velocities, axis=1)
            estimated_points = IntegrateCurvatureForPoints(
                ego_curvatures,
                ego_velocities_norm,
                ego_poses_world[0],
                ego_headings[0],
                scene_length,
            )

            if args.plot:
                plt.plot(ego_poses_world[:, 0], ego_poses_world[:, 1], "r-", label="GT")
                plt.quiver(
                    ego_poses_world[:, 0],
                    ego_poses_world[:, 1],
                    ego_velocities[:, 0],
                    ego_velocities[:, 1],
                    color="b",
                )
                plt.plot(estimated_points[:, 0], estimated_points[:, 1], "g-", label="Reconstruction")
                plt.legend()
                plt.savefig(output_dir / f"{name}_interpolation.jpg")
                plt.close()

            ego_traj_world = [extract_translation(ego_poses[idx]) for idx in range(scene_length)]

            prev_intent = None
            cam_images_sequence = []
            scene_metric_lists = {metric_name: [] for metric_name, _ in metric_horizon_steps}
            num_windows = scene_length - required_scene_length

            if args.max_windows_per_scene is not None:
                num_windows = min(num_windows, args.max_windows_per_scene)

            for i in range(num_windows):
                obs_images = front_camera_images[i:i + OBS_LEN]
                obs_ego_poses = ego_poses[i:i + OBS_LEN]
                obs_camera_params = camera_params[i:i + OBS_LEN]
                obs_ego_traj_world = ego_traj_world[i:i + OBS_LEN]
                fut_ego_traj_world = ego_traj_world[i + OBS_LEN:min(i + TTL_LEN, scene_length)]
                obs_ego_velocities = ego_velocities[i:i + OBS_LEN]
                obs_ego_curvatures = ego_curvatures[i:i + OBS_LEN]
                obs_ego_headings = ego_headings[i:i + OBS_LEN]
                fut_start_world = obs_ego_traj_world[-1]
                curr_image = obs_images[-1]

                img = load_visualization_image(curr_image, dataset_type, dataset_source)
                if is_gpt_model(args.model_path) and dataset_type == "nuscenes":
                    yolo3d_nuScenes = get_yolo3d_inference()
                    img = yolo3d_nuScenes(img, calib=obs_camera_params[-1])[0]

                prompt_images = materialize_image_inputs(
                    select_observation_images(obs_images, args),
                    dataset_type,
                    dataset_source,
                    args,
                )

                prediction = ""
                scene_description = None
                object_description = None
                updated_intent = None
                speed_curvature_pred = []
                for _ in range(3):
                    prediction, scene_description, object_description, updated_intent = GenerateMotion(
                        prompt_images,
                        obs_ego_traj_world,
                        obs_ego_velocities,
                        obs_ego_curvatures,
                        prev_intent,
                        extra_scene_text=extra_scene_text,
                        processor=processor,
                        model=model,
                        tokenizer=tokenizer,
                        args=args,
                    )
                    prev_intent = updated_intent
                    speed_curvature_pred = parse_speed_curvature_pairs(
                        prediction.replace("Future speeds and curvatures:", "").strip()
                    )
                    if speed_curvature_pred:
                        if len(speed_curvature_pred) < FUT_LEN:
                            parsed_len = len(speed_curvature_pred)
                            last_action = speed_curvature_pred[-1]
                            speed_curvature_pred.extend([last_action] * (FUT_LEN - len(speed_curvature_pred)))
                            print(
                                f"Only parsed {parsed_len} future actions; "
                                f"repeated last action {last_action} to reach {FUT_LEN} steps."
                            )
                        break

                if not speed_curvature_pred:
                    print(f"Skipping frame window {i} in {name}: could not parse prediction.")
                    continue

                print(f"Got {len(speed_curvature_pred)} future actions: {speed_curvature_pred}")

                fut_ego_traj_world = np.array(fut_ego_traj_world)
                pred_len = min(FUT_LEN, len(speed_curvature_pred), len(fut_ego_traj_world))
                if pred_len <= 0:
                    print(f"Skipping frame window {i} in {name}: no future GT points.")
                    continue
                pred_curvatures = np.array(speed_curvature_pred)[:, 1] / 100
                pred_speeds = np.array(speed_curvature_pred)[:, 0]
                pred_traj = rollout_future_trajectory(
                    pred_curvatures,
                    pred_speeds,
                    fut_start_world,
                    obs_ego_headings[-1],
                )

                if pred_len >= 2:
                    OverlayTrajectory(
                        img,
                        pred_traj.tolist(),
                        obs_camera_params[-1],
                        obs_ego_poses[-1],
                        color=(255, 0, 0),
                        args=args,
                    )

                ade = np.mean(np.linalg.norm(fut_ego_traj_world[:pred_len] - pred_traj[:pred_len], axis=1))

                window_horizon_metrics = {}
                for metric_name, horizon_steps in metric_horizon_steps:
                    metric_value = compute_horizon_ade(fut_ego_traj_world, pred_traj, horizon_steps)
                    scene_metric_lists[metric_name].append(metric_value)
                    window_horizon_metrics[metric_name] = metric_value

                if args.plot:
                    cam_images_sequence.append(img.copy())
                    cv2.imwrite(str(output_dir / f"{name}_{i}_front_cam.jpg"), img)

                    plt.plot(fut_ego_traj_world[:, 0], fut_ego_traj_world[:, 1], "r-", label="GT")
                    plt.plot(pred_traj[:, 0], pred_traj[:, 1], "b-", label="Pred")
                    plt.legend()
                    plt.title(f"Scene: {name}, Frame: {i}, ADE: {ade}")
                    plt.savefig(output_dir / f"{name}_{i}_traj.jpg")
                    plt.close()

                    np.save(output_dir / f"{name}_{i}_pred_traj.npy", pred_traj)
                    np.save(output_dir / f"{name}_{i}_pred_curvatures.npy", pred_curvatures)
                    np.save(output_dir / f"{name}_{i}_pred_speeds.npy", pred_speeds)

                    with open(output_dir / f"{name}_{i}_logs.txt", "w", encoding="utf-8") as handle:
                        handle.write(f"Dataset Type: {dataset_type}\n")
                        if scene_category is not None:
                            handle.write(f"Scene Category: {scene_category}\n")
                        handle.write(f"Scene Description: {scene_description}\n")
                        handle.write(f"Object Description: {object_description}\n")
                        handle.write(f"Intent Description: {updated_intent}\n")
                        handle.write(f"Extra Scene Text: {extra_scene_text}\n")
                        handle.write(f"Extra Scene Text Placement: {args.extra_scene_text_placement}\n")
                        handle.write(f"Average Displacement Error: {ade}\n")
                        for metric_name, metric_value in window_horizon_metrics.items():
                            label = metric_name.replace("_", "@").upper()
                            metric_text = "N/A" if metric_value is None else f"{metric_value}"
                            handle.write(f"{label}: {metric_text}\n")

            mean_horizon_metrics = {
                metric_name: compute_scene_metrics(scene_metric_lists[metric_name])
                for metric_name, _ in metric_horizon_steps
            }

            valid_horizon_metrics = [value for value in mean_horizon_metrics.values() if value is not None]
            if not valid_horizon_metrics:
                print(f"No valid predictions were produced for scene {name}, skipping metrics output.")
                continue

            avg_ade = float(np.mean(valid_horizon_metrics))
            result = {
                "name": name,
                "token": token,
                "dataset_type": dataset_type,
                "metric_sampling_interval_s": args.frame_interval_s,
                "ade_1s": mean_horizon_metrics["ade_1s"],
                "ade_3s": mean_horizon_metrics["ade_3s"],
                "ade_5s": mean_horizon_metrics["ade_5s"],
                "ade1s": mean_horizon_metrics["ade_1s"],
                "ade3s": mean_horizon_metrics["ade_3s"],
                "ade5s": mean_horizon_metrics["ade_5s"],
                "ade_1s_num_windows": sum(value is not None for value in scene_metric_lists["ade_1s"]),
                "ade_3s_num_windows": sum(value is not None for value in scene_metric_lists["ade_3s"]),
                "ade_5s_num_windows": sum(value is not None for value in scene_metric_lists["ade_5s"]),
                "avgade": avg_ade,
                "method": args.method,
                "image_input_mode": args.image_input_mode,
                "extra_scene_text": bool(extra_scene_text),
                "extra_scene_text_placement": args.extra_scene_text_placement if extra_scene_text else "none",
                "min_required_future_steps": args.min_required_future_steps,
            }
            if scene_category is not None:
                result["scene_category"] = scene_category
            if dataset_type == "image_sequence":
                result["pose_file"] = str(Path(args.pose_file).resolve())
                result["pose_note"] = "monocular pseudo-pose; metric scale is arbitrary unless externally aligned"

            with open(output_dir / "ade_results.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(result))
                handle.write("\n")

            if args.plot and cam_images_sequence:
                WriteImageSequenceToVideo(cam_images_sequence, str(output_dir / name))
    finally:
        if isinstance(dataset_source, DeepAccidentSource):
            dataset_source.close()


if __name__ == "__main__":
    main()
