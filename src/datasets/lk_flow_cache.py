"""Offline sparse LK tracks with the historical four motion statistics.

Tracks are computed on correctly decoded RGB at the same pre-crop resize as
the video augmentation. Training selects starting points inside its RGB crop.
The output is [mean dx, mean dy, mean magnitude, cos(circular mean angle)],
in final-crop pixels. Frame zero has no incoming transition and is all zero.

Offline full-frame corner detection and LK boundary context are not identical
to re-detecting and tracking on every random crop. This intentionally restores
the old estimator/statistic units, not its normalized-RGB-to-uint8 bug or its
incorrect temporal placement in the visual tokens.
"""
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .flow_cache import _atomic_json, _digest, source_fingerprint, uniform_indices, SAMPLING_VERSION

SCHEMA_VERSION = 1
MAX_CORNERS = 200
LK_PARAMETERS = {"winSize": (21, 21), "maxLevel": 3, "criteria": (3, 30, 0.01),
                 "flags": 0, "minEigThreshold": 1e-4}


def resized_shape(source_size, resize_short_side=224):
    """Match video_functional.get_resize_sizes and its integer rounding."""
    height, width = [int(value) for value in source_size]
    size = int(resize_short_side)
    if min(height, width, size) < 1:
        raise ValueError("Source dimensions and resize_short_side must be positive")
    if width < height:
        return int(size * height / width), size
    return size, int(size * width / height)


def _validate_tracks(record, num_frames=None):
    tracks, valid = np.asarray(record["tracks"]), np.asarray(record["valid"])
    expected = (num_frames if num_frames is not None else len(tracks), MAX_CORNERS, 4)
    if (tracks.shape != expected or tracks.dtype != np.float32
            or valid.shape != expected[:2] or valid.dtype != np.bool_
            or not np.isfinite(tracks).all()):
        raise ValueError("LK tracks must be finite float32 [S,200,4] with bool [S,200] validity")
    if len(tracks) < 1 or valid[0].any() or np.any(tracks[0] != 0):
        raise ValueError("The first LK frame must have no incoming flow")
    if np.any(tracks[~valid] != 0):
        raise ValueError("Invalid LK track padding must be zero")
    shape = tuple(record["resized_size"])
    if len(shape) != 2 or any(int(v) != v or v < 1 for v in shape):
        raise ValueError("LK resized_size must be positive integer height,width")
    if valid.any():
        starts = tracks[valid, :2]
        if (starts < 0).any() or (starts[:, 0] >= shape[1]).any() or (starts[:, 1] >= shape[0]).any():
            raise ValueError("LK starting points must lie inside the resized image")
    return tracks, valid


def compute_sparse_lk(rgb_frames, resize_short_side=224):
    """Correct uint8 RGB [S,H,W,3] -> sparse forward tracks, without torch.

    The old augmentation calls Resize(..., interpolation='nearest'), whose
    PIL branch actually selects BILINEAR. Reproduce that existing behavior.
    LK options below are the explicit equivalents of the old OpenCV defaults.
    """
    import cv2
    from PIL import Image

    frames = np.asarray(rgb_frames)
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8 or frames.shape[0] < 1:
        raise ValueError("LK cache input must be uint8 RGB [S,H,W,3] with at least one frame")
    height, width = resized_shape(frames.shape[1:3], resize_short_side)
    bilinear = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    gray = []
    for frame in frames:
        if frame.shape[:2] == (height, width):
            resized = frame
        else:
            resized = np.asarray(Image.fromarray(frame).resize((width, height), bilinear))
        gray.append(cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY))
    tracks = np.zeros((len(frames), MAX_CORNERS, 4), dtype=np.float32)
    valid = np.zeros((len(frames), MAX_CORNERS), dtype=np.bool_)
    for t in range(1, len(frames)):
        points = cv2.goodFeaturesToTrack(gray[t - 1], maxCorners=MAX_CORNERS,
                                        qualityLevel=0.01, minDistance=30)
        if points is None:
            continue
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(gray[t - 1], gray[t], points, None,
                                                       **LK_PARAMETERS)
        if next_points is None or status is None:
            continue
        starts = points.reshape(-1, 2)
        ends = next_points.reshape(-1, 2)
        good = (status.reshape(-1) == 1) & np.isfinite(starts).all(1) & np.isfinite(ends).all(1)
        count = int(good.sum())
        tracks[t, :count] = np.concatenate((starts[good], ends[good]), axis=1)
        valid[t, :count] = True
    result = {"tracks": tracks, "valid": valid, "resized_size": (height, width)}
    _validate_tracks(result)
    return result


def crop_lk_statistics_numpy(record, crop_box=(0.0, 0.0, 1.0, 1.0),
                             horizontal_flip=False, output_size=(224, 224)):
    """Crop sparse starting points; reduce displacement in final image pixels.

    crop_box=(top,left,height,width) is normalized to the full resized image.
    Endpoints may leave the crop, just as a tracked point can leave an image;
    selection is by the point's previous-frame location. No corner detection
    or optical-flow estimation occurs in this function.
    """
    tracks, valid = _validate_tracks(record)
    top, left, crop_h, crop_w = [float(v) for v in crop_box]
    if (not np.isfinite([top, left, crop_h, crop_w]).all() or min(top, left) < 0
            or min(crop_h, crop_w) <= 0 or top + crop_h > 1.000001 or left + crop_w > 1.000001):
        raise ValueError("crop_box must fit inside normalized full-frame bounds")
    out_h, out_w = [int(v) for v in output_size]
    if min(out_h, out_w) < 1:
        raise ValueError("output_size must contain positive height,width")
    height, width = record["resized_size"]
    x0, y0 = left * width, top * height
    x1, y1 = (left + crop_w) * width, (top + crop_h) * height
    selected = (valid & (tracks[:, :, 0] >= x0) & (tracks[:, :, 0] < x1)
                & (tracks[:, :, 1] >= y0) & (tracks[:, :, 1] < y1))
    scale = np.array([out_w / (crop_w * width), out_h / (crop_h * height)], dtype=np.float32)
    features = np.zeros((len(tracks), 4), dtype=np.float32)
    for t in range(1, len(tracks)):
        pair = tracks[t, selected[t]]
        if not len(pair):
            continue
        displacement = (pair[:, 2:] - pair[:, :2]) * scale
        if horizontal_flip:
            displacement[:, 0] *= -1
        dx, dy = displacement[:, 0], displacement[:, 1]
        magnitude = np.sqrt(dx ** 2 + dy ** 2)
        angles = np.arctan2(dy, dx)
        angle_mean = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
        # This is cos(mean circular angle), NOT mean(cos(angle)). A valid
        # stationary track has atan2(0,0)=0 and cosine 1, as in the old code.
        features[t] = (dx.mean(), dy.mean(), magnitude.mean(), np.cos(angle_mean))
    return features


def crop_lk_statistics(record, crop_box=(0.0, 0.0, 1.0, 1.0),
                       horizontal_flip=False, output_size=(224, 224)):
    """Torch wrapper used by dataset workers; preprocessing stays torch-free."""
    import torch
    return torch.from_numpy(crop_lk_statistics_numpy(record, crop_box, horizontal_flip, output_size))


class SparseLKFlowCache:
    """Separate source/config namespace; fail on missing or mismatched caches."""

    def __init__(self, cache_dir, visual_tsv_path, num_frames=32, resize_short_side=224, create=False):
        import cv2
        import PIL
        self.source = source_fingerprint(visual_tsv_path)
        self.num_frames = int(num_frames)
        self.resize_short_side = int(resize_short_side)
        if min(self.num_frames, self.resize_short_side) < 1:
            raise ValueError("num_frames and resize_short_side must be positive")
        self.config = {"schema_version": SCHEMA_VERSION, "num_frames": self.num_frames,
                       "algorithm": "SPARSE_PYRAMIDAL_LK", "sampling": SAMPLING_VERSION,
                       "resize_short_side": self.resize_short_side, "resize_interpolation": "PIL_BILINEAR",
                       "max_corners": MAX_CORNERS, "quality_level": 0.01, "min_distance": 30,
                       "lk_parameters": json.loads(json.dumps(LK_PARAMETERS)),
                       "opencv_version": cv2.__version__, "pillow_version": PIL.__version__,
                       "track_units": "resized_full_frame_pixels", "dtype": "float32",
                       "statistics": "mean_dx,mean_dy,mean_magnitude,cos_circular_mean_angle",
                       "output_units": "final_crop_pixels", "input": "decoded_RGB_uint8"}
        self.identity = _digest({"source": self.source, "config": self.config})
        self.directory = Path(cache_dir) / ("lk_" + _digest(self.source)[:16]) / _digest(self.config)[:16]
        self.manifest_path = self.directory / "manifest.json"
        if create and not self.manifest_path.exists():
            with Path(self.source["path"]).with_suffix(".lineidx").open() as handle:
                rows = sum(1 for line in handle if line.strip())
            _atomic_json(self.manifest_path, {"identity": self.identity, "source": self.source,
                         "config": self.config, "source_rows": rows, "status": "incomplete", "completed_rows": 0})
        if not self.manifest_path.is_file():
            raise FileNotFoundError("Run scripts/precompute_lk_flow.py for this LK source/config: " + str(self.manifest_path))
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if (self.manifest.get("identity") != self.identity or self.manifest.get("source") != self.source
                or self.manifest.get("config") != self.config):
            raise ValueError("LK cache manifest does not match source/sampling/config: " + str(self.manifest_path))

    def path(self, row_index):
        row_index = int(row_index)
        if row_index < 0 or row_index >= int(self.manifest["source_rows"]):
            raise IndexError("LK cache source row out of range: {}".format(row_index))
        return self.directory / "{:08d}.npz".format(row_index)

    def load(self, row_index, source_size=None, expected_key=None, row_sha256=None):
        path = self.path(row_index)
        try:
            with np.load(str(path), allow_pickle=False) as record:
                metadata = json.loads(str(record["metadata"].item()))
                result = {"tracks": record["tracks"], "valid": record["valid"],
                          "resized_size": tuple(metadata["resized_size"])}
        except Exception as exc:
            raise RuntimeError("Missing/corrupt LK cache; preprocessing must finish before training: " + str(path)) from exc
        if (metadata.get("identity") != self.identity or metadata.get("row_index") != int(row_index)
                or metadata.get("schema_version") != SCHEMA_VERSION):
            raise ValueError("Stale/mismatched LK record: " + str(path))
        _validate_tracks(result, self.num_frames)
        if metadata.get("sample_indices") != uniform_indices(metadata["source_frame_count"], self.num_frames):
            raise ValueError("LK temporal sampling mismatch: " + str(path))
        if result["resized_size"] != resized_shape(metadata["source_size"], self.resize_short_side):
            raise ValueError("LK resize geometry mismatch: " + str(path))
        if source_size is not None and list(source_size) != metadata.get("source_size"):
            raise ValueError("Cached and actual RGB dimensions differ: " + str(path))
        if expected_key is not None and str(expected_key) != metadata.get("clip_key"):
            raise ValueError("Cached and actual clip keys differ: " + str(path))
        digest = metadata.get("row_sha256", "")
        if not metadata.get("clip_key") or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Missing clip/source-row fingerprint: " + str(path))
        if row_sha256 is not None and row_sha256 != digest:
            raise ValueError("Source TSV row changed since LK preprocessing: " + str(path))
        return result

    def write(self, row_index, record, clip_key, source_size, source_frame_count, sample_indices, row_sha256):
        tracks, valid = _validate_tracks(record, self.num_frames)
        if tuple(record["resized_size"]) != resized_shape(source_size, self.resize_short_side):
            raise ValueError("Cannot cache tracks with a different RGB resize")
        if sample_indices != uniform_indices(source_frame_count, self.num_frames):
            raise ValueError("Cannot cache tracks from a different temporal sampling")
        if not str(clip_key) or len(row_sha256) != 64 or any(c not in "0123456789abcdef" for c in row_sha256):
            raise ValueError("A clip key and full source-row SHA256 are required")
        metadata = {"identity": self.identity, "schema_version": SCHEMA_VERSION, "row_index": int(row_index),
                    "clip_key": str(clip_key), "source_size": list(source_size),
                    "resized_size": list(record["resized_size"]), "source_frame_count": int(source_frame_count),
                    "sample_indices": sample_indices, "row_sha256": row_sha256}
        path = self.path(row_index)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=str(self.directory), prefix=path.stem + ".", suffix=".npz",
                                             delete=False) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(handle, tracks=tracks, valid=valid,
                                    metadata=np.array(json.dumps(metadata, sort_keys=True)))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temporary), str(path))
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def update_manifest(self, completed_rows, status, **extra):
        self.manifest.update(completed_rows=int(completed_rows), status=status, **extra)
        _atomic_json(self.manifest_path, self.manifest)
