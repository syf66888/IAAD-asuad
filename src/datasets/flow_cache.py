"""Validated, atomic dense DIS caches and crop-consistent motion statistics.

Cache arrays are [S,2,Hc,Wc], float16, forward displacement from frame t-1
to t expressed in fractions of the *full source* width/height. Frame 0 is
zero. The manifest binds each cache to source TSV contents/sampling/config.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

SCHEMA_VERSION = 1
SAMPLING_VERSION = "uniform-python-round-repeat-short-v2"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_fingerprint(tsv_path):
    """Fingerprint immutable TSV identity plus sampled bytes and full row index.

    Reading 3 small windows avoids hashing hundreds of GB at each worker start.
    Each completed clip also stores its full original TSV-row SHA256.
    """
    source = Path(tsv_path).resolve(strict=True)
    stat = source.stat()
    index = source.with_suffix(".lineidx")
    if not index.is_file():
        raise FileNotFoundError("Flow preprocessing needs the existing TSV .lineidx: " + str(index))
    sample = hashlib.sha256()
    block = 65536
    with source.open("rb") as handle:
        for offset in sorted(set([0, max(0, stat.st_size // 2 - block // 2), max(0, stat.st_size - block)])):
            handle.seek(offset)
            sample.update(str(offset).encode())
            sample.update(handle.read(block))
    return {"path": str(source), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sample_sha256": sample.hexdigest(), "lineidx_sha256": hashlib.sha256(index.read_bytes()).hexdigest()}


def uniform_indices(total_frames, num_frames):
    if num_frames < 1 or total_frames < 1:
        raise ValueError("Source and requested frame counts must be positive")
    if num_frames == 1:
        return [int(round((total_frames - 1) / 2.0))]
    step = (total_frames - 1) / float(num_frames - 1)
    return [int(round(i * step)) for i in range(num_frames)]


def _atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=str(path.parent),
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, sort_keys=True, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class DenseFlowCache:
    """One source/config namespace. Missing or stale records raise errors."""

    def __init__(self, cache_dir, visual_tsv_path, num_frames=32, flow_size=(64, 112), create=False):
        import cv2
        self.source = source_fingerprint(visual_tsv_path)
        self.num_frames = int(num_frames)
        self.flow_size = tuple(int(v) for v in flow_size)
        if self.num_frames < 1 or len(self.flow_size) != 2 or min(self.flow_size) < 32:
            raise ValueError("num_frames must be positive and flow_size=(height,width) must be >=32")
        self.config = {"num_frames": self.num_frames, "flow_size": list(self.flow_size),
                       "algorithm": "DIS_FAST", "opencv_version": cv2.__version__,
                       "sampling": SAMPLING_VERSION, "units": "dx/source_width,dy/source_height",
                       "dtype": "float16", "schema_version": SCHEMA_VERSION}
        self.identity = _digest({"source": self.source, "config": self.config})
        self.directory = Path(cache_dir) / _digest(self.source)[:16] / _digest(self.config)[:16]
        self.manifest_path = self.directory / "manifest.json"
        if create and not self.manifest_path.exists():
            self.directory.mkdir(parents=True, exist_ok=True)
            with Path(self.source["path"]).with_suffix(".lineidx").open() as handle:
                rows = sum(1 for line in handle if line.strip())
            _atomic_json(self.manifest_path, {"identity": self.identity, "source": self.source,
                         "config": self.config, "source_rows": rows, "status": "incomplete",
                         "completed_rows": 0})
        if not self.manifest_path.is_file():
            raise FileNotFoundError("No matching flow-cache manifest for this source/config. Run scripts/precompute_flow.py: "
                                    + str(self.manifest_path))
        self.manifest = json.loads(self.manifest_path.read_text())
        if (self.manifest.get("identity") != self.identity or self.manifest.get("source") != self.source
                or self.manifest.get("config") != self.config):
            raise ValueError("Flow-cache manifest does not match source/sampling/config: " + str(self.manifest_path))

    def path(self, row_index):
        row_index = int(row_index)
        if row_index < 0 or row_index >= int(self.manifest["source_rows"]):
            raise IndexError("Flow cache source row out of range: {}".format(row_index))
        return self.directory / "{:08d}.npz".format(row_index)

    def load(self, row_index, source_size=None, expected_key=None, row_sha256=None):
        path = self.path(row_index)
        try:
            with np.load(str(path), allow_pickle=False) as record:
                dense = record["flow"]
                metadata = json.loads(str(record["metadata"].item()))
        except Exception as exc:
            raise RuntimeError("Missing/corrupt dense flow; preprocessing must finish before training: " + str(path)) from exc
        expected_shape = (self.num_frames, 2) + self.flow_size
        if (metadata.get("identity") != self.identity or metadata.get("row_index") != int(row_index)
                or metadata.get("schema_version") != SCHEMA_VERSION):
            raise ValueError("Stale/mismatched flow record: " + str(path))
        if dense.shape != expected_shape or dense.dtype != np.float16 or not np.isfinite(dense).all():
            raise ValueError("Invalid dense flow shape/dtype/values: " + str(path))
        expected_indices = uniform_indices(metadata["source_frame_count"], self.num_frames)
        if metadata.get("sample_indices") != expected_indices or not np.all(dense[0] == 0):
            raise ValueError("Flow sampling/first-frame mismatch: " + str(path))
        if source_size is not None and list(source_size) != metadata.get("source_size"):
            raise ValueError("Cached and actual RGB dimensions differ: " + str(path))
        if expected_key is not None and str(expected_key) != metadata.get("clip_key"):
            raise ValueError("Cached and actual clip keys differ: " + str(path))
        if not metadata.get("clip_key") or len(metadata.get("row_sha256", "")) != 64:
            raise ValueError("Missing clip/source-row fingerprint: " + str(path))
        if row_sha256 is not None and row_sha256 != metadata["row_sha256"]:
            raise ValueError("Source TSV row changed since flow preprocessing: " + str(path))
        return dense.astype(np.float32)

    def write(self, row_index, dense, clip_key, source_size, source_frame_count, sample_indices, row_sha256):
        if dense.shape != (self.num_frames, 2) + self.flow_size or not np.isfinite(dense).all():
            raise ValueError("Cannot cache invalid dense flow")
        if sample_indices != uniform_indices(source_frame_count, self.num_frames):
            raise ValueError("Cannot cache features from a different temporal sampling")
        quantized = np.asarray(dense, dtype=np.float16)
        if not np.isfinite(quantized).all() or not np.all(quantized[0] == 0):
            raise ValueError("Flow cannot be represented as finite float16 with zero first frame")
        metadata = {"identity": self.identity, "schema_version": SCHEMA_VERSION, "row_index": int(row_index),
                    "clip_key": str(clip_key), "source_size": list(source_size), "source_frame_count": source_frame_count,
                    "sample_indices": sample_indices, "row_sha256": row_sha256}
        path = self.path(row_index)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=str(self.directory), prefix=path.stem + ".", suffix=".npz",
                                             delete=False) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(handle, flow=quantized, metadata=np.array(json.dumps(metadata, sort_keys=True)))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temporary), str(path))
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def update_manifest(self, completed_rows, status, **extra):
        self.manifest.update(completed_rows=int(completed_rows), status=status, **extra)
        _atomic_json(self.manifest_path, self.manifest)


def compute_dense_dis(rgb_frames, flow_size=(64, 112)):
    """RGB uint8 [S,H,W,3] -> normalized dense forward flow [S,2,Hc,Wc]."""
    import cv2
    if rgb_frames.ndim != 4 or rgb_frames.shape[-1] != 3 or rgb_frames.dtype != np.uint8:
        raise ValueError("DIS cache input must be uint8 RGB [S,H,W,3]")
    if rgb_frames.shape[0] < 1 or min(flow_size) < 32:
        raise ValueError("At least one RGB frame and flow dimensions >=32 are required")
    flow_h, flow_w = flow_size
    algorithm = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    gray = [cv2.cvtColor(cv2.resize(frame, (flow_w, flow_h), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_RGB2GRAY) for frame in rgb_frames]
    dense = np.zeros((len(gray), 2, flow_h, flow_w), dtype=np.float32)
    for t in range(1, len(gray)):
        if np.array_equal(gray[t - 1], gray[t]):
            continue
        flow = algorithm.calc(gray[t - 1], gray[t], None)
        if flow is None or not np.isfinite(flow).all():
            raise RuntimeError("DIS failed on frame pair {}->{}".format(t - 1, t))
        flow = flow / np.array([flow_w, flow_h], dtype=np.float32)
        dense[t] = flow.transpose(2, 0, 1)
    return dense


def crop_flow_statistics(dense, crop_box=(0.0, 0.0, 1.0, 1.0), horizontal_flip=False):
    """Sample the RGB crop and convert full-frame units to cropped-frame units.

    crop_box is (top,left,height,width), fractions of the resized full RGB
    frame. A following image resize does not change these normalized units.
    Motion magnitudes are computed AFTER x/y scale correction. No pixels from
    outside the RGB crop contribute except normal bilinear boundary sampling.
    """
    import cv2
    import torch
    dense = np.asarray(dense, dtype=np.float32)
    if dense.ndim != 4 or dense.shape[1] != 2 or not np.isfinite(dense).all():
        raise ValueError("Dense flow must be finite [S,2,H,W]")
    top, left, crop_h, crop_w = [float(v) for v in crop_box]
    if (not all(np.isfinite([top, left, crop_h, crop_w])) or min(top, left) < 0
            or min(crop_h, crop_w) <= 0 or top + crop_h > 1.000001 or left + crop_w > 1.000001):
        raise ValueError("crop_box must fit inside normalized full-frame bounds")
    height, width = dense.shape[-2:]
    out_h, out_w = max(1, round(height * crop_h)), max(1, round(width * crop_w))
    xs = (left + (np.arange(out_w, dtype=np.float32) + 0.5) * crop_w / out_w) * width - 0.5
    ys = (top + (np.arange(out_h, dtype=np.float32) + 0.5) * crop_h / out_h) * height - 0.5
    map_x, map_y = np.meshgrid(xs, ys)
    features = np.zeros((len(dense), 4), dtype=np.float32)
    for t, flow in enumerate(dense):
        dx = cv2.remap(flow[0], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) / crop_w
        dy = cv2.remap(flow[1], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) / crop_h
        if horizontal_flip:
            dx = -dx
        magnitude = np.sqrt(dx * dx + dy * dy)
        features[t] = (dx.mean(), dy.mean(), magnitude.mean(), magnitude.std())
    return torch.from_numpy(features)
