"""Aligned object and motion references for ImageNet-normalized video clips.

No clip-id cache is used: a video can be sampled/cropped differently every epoch.
YOLO weights must already exist locally. Detection is frozen, while ROI pooling
keeps gradients to the Swin feature grid. DIS runs on one CPU copy per batch.
"""

from pathlib import Path
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def denormalize_rgb(images):
    """Convert [B,S,3,H,W] ImageNet input to float RGB [0,1]."""
    if images.ndim != 5 or images.shape[2] != 3:
        raise ValueError("Expected normalized RGB images [B,S,3,H,W]")
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 1, 3, 1, 1)
    rgb = images.detach().float() * std + mean
    if not torch.isfinite(rgb).all():
        raise ValueError("Non-finite image values cannot be passed to vision extractors")
    return rgb.clamp(0.0, 1.0)


def temporal_sample_indices(source_length, sample_count, device=None):
    """Uniformly sample source timestamps, including the endpoints."""
    if source_length < 1 or sample_count < 1:
        raise ValueError("Temporal lengths must be positive")
    count = min(int(sample_count), int(source_length))
    return torch.linspace(0, source_length - 1, count, device=device).round().long()


def nearest_temporal_samples(source_length, target_length, selected_indices):
    """Map Swin temporal-bin centers to positions within selected RGB frames."""
    if target_length < 1 or source_length < 1:
        raise ValueError("Temporal lengths must be positive")
    centers = ((torch.arange(target_length, device=selected_indices.device).float() + 0.5)
               * source_length / target_length - 0.5).clamp(0, source_length - 1)
    return (centers[:, None] - selected_indices.float()[None, :]).abs().argmin(1)


def align_temporal_features(features, target_length):
    """Align [B,S,D] to Swin temporal bins; never pad temporal rows as pixels."""
    if features.ndim != 3 or target_length < 1 or features.shape[1] < 1:
        raise ValueError("Expected nonempty features [B,S,D] and target_length > 0")
    return F.interpolate(features.transpose(1, 2), size=target_length,
                         mode="linear", align_corners=False).transpose(1, 2)


def pool_object_references(feature_grid, detections, selected_indices, source_length):
    """Pool confidence-weighted ROIs with correct batch/time/channel alignment.

    detections has B*K entries of [N,5] (xyxy in [0,1], confidence), where
    K=len(selected_indices). Missing detections fall back to spatial mean.
    """
    from torchvision.ops import roi_align

    if feature_grid.ndim != 5:
        raise ValueError("Expected Swin feature grid [B,C,T,H,W]")
    batch, channels, time, height, width = feature_grid.shape
    sampled = len(selected_indices)
    if len(detections) != batch * sampled:
        raise ValueError("Detection count does not match batch and sampled frames")
    flat = feature_grid.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width)
    fallback = flat.mean(dim=(-2, -1))
    mapping = nearest_temporal_samples(source_length, time, selected_indices).tolist()
    rois, weights, owners = [], [], []
    for b in range(batch):
        for t, selected in enumerate(mapping):
            det = detections[b * sampled + selected]
            if det is None or det.numel() == 0:
                continue
            det = det.detach().to(device=flat.device, dtype=torch.float32)
            if det.ndim != 2 or det.shape[1] != 5 or not torch.isfinite(det).all():
                raise ValueError("Detections must be finite [N,5] xyxy+confidence tensors")
            boxes = det[:, :4].clamp(0, 1)
            valid = ((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                     & (det[:, 4] > 0))
            boxes, confidence = boxes[valid], det[valid, 4]
            if not len(boxes):
                continue
            boxes = boxes * boxes.new_tensor([width, height, width, height])
            owner = b * time + t
            rois.append(torch.cat([boxes.new_full((len(boxes), 1), owner), boxes], 1))
            weights.append(confidence)
            owners.append(torch.full((len(boxes),), owner, device=flat.device, dtype=torch.long))
    if not rois:
        return fallback.reshape(batch, time, channels)
    # float32 ROIAlign is stable under both fp16 and bf16 training, and the cast
    # preserves gradients to the original feature grid.
    pooled = roi_align(flat.float(), torch.cat(rois), output_size=(2, 2),
                       spatial_scale=1.0, sampling_ratio=2, aligned=True).mean((-2, -1))
    confidence, owner = torch.cat(weights), torch.cat(owners)
    sums = pooled.new_zeros((batch * time, channels)).index_add(
        0, owner, pooled * confidence[:, None])
    counts = pooled.new_zeros((batch * time,)).index_add(0, owner, confidence)
    result = torch.where(counts[:, None] > 0,
                         sums / counts.clamp_min(1e-6)[:, None], fallback.float())
    return result.to(feature_grid.dtype).reshape(batch, time, channels)


class VisualReferenceExtractor(nn.Module):
    """Frozen sampled YOLOv5u + efficient DIS references, aligned to Swin time."""

    def __init__(self, args):
        super().__init__()
        self.yolo_frames = int(getattr(args, "vision_yolo_frames", 8))
        self.yolo_batch_size = int(getattr(args, "vision_yolo_batch_size", 16))
        self.yolo_conf = float(getattr(args, "vision_yolo_conf", 0.25))
        self.yolo_imgsz = int(getattr(args, "vision_yolo_imgsz", 320))
        self.flow_backend = getattr(args, "vision_flow_backend", "dis")
        self.flow_size = int(getattr(args, "vision_flow_size", 112))
        self.max_detections = int(getattr(args, "vision_yolo_max_det", 32))
        if min(self.yolo_frames, self.yolo_batch_size, self.yolo_imgsz, self.max_detections) < 1:
            raise ValueError("YOLO sampling/batch/image-size/max-det settings must be positive")
        if not 0 <= self.yolo_conf <= 1:
            raise ValueError("vision_yolo_conf must be in [0,1]")
        if self.flow_backend not in ("dis", "lk", "none") or self.flow_size < 32:
            raise ValueError("Use vision_flow_backend=dis|lk|none and vision_flow_size >= 32")
        if self.flow_backend == "lk" and not getattr(args, "flow_cache_dir", ""):
            raise ValueError("LK experiment requires completed offline sparse tracks.")
        self.detector = None
        weights = getattr(args, "vision_yolo_weights", "yolov5su.pt")
        if weights:
            weights = Path(weights).expanduser()
            if weights.suffix.lower() != ".pt" or not weights.is_file():
                raise FileNotFoundError("A local pretrained Ultralytics .pt checkpoint is required: " + str(weights))
            try:
                from ultralytics import YOLO
                from ultralytics.cfg import DEFAULT_CFG_DICT
            except ImportError as exc:
                raise ImportError("Install ultralytics explicitly before enabling YOLO references") from exc
            yolo = YOLO(str(weights), task="detect")
            # Fuse before optimizer/checkpoint setup, in CPU float32. Otherwise
            # AutoBackend's first prediction replaces Conv/BN parameters and a
            # later state_dict cannot be restored into an unfused detector.
            yolo.model.float().eval().fuse(verbose=False)
            # Only the underlying nn.Module is registered. YOLO.train() launches
            # a training job, so the high-level wrapper must not be a child here.
            self.detector = yolo.model
            object.__setattr__(self, "_yolo", yolo)
            self._quantize_supported = "quantize" in DEFAULT_CFG_DICT
            self.detector.eval()
            self.detector.requires_grad_(False)
        self._flow = None

    def train(self, mode=True):
        super().train(mode)
        if self.detector is not None:
            self.detector.eval()
        return self

    @torch.no_grad()
    def _detect(self, rgb, selected):
        batch, _, _, height, width = rgb.shape
        if self.detector is None:
            return [rgb.new_empty((0, 5)) for _ in range(batch * len(selected))]
        # Torch 1.13 CUDA nearest upsampling (used by YOLO's neck) cannot
        # consume bf16. Caption autocast must never enter this frozen branch.
        # FP32 also avoids depending on global/native/deepspeed precision.
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            self.detector.float()
            return self._detect_fp32(rgb.float(), selected)

    def _detect_fp32(self, rgb, selected):
        batch, _, _, height, width = rgb.shape
        self.detector.eval()
        selected_rgb = rgb[:, selected].reshape(-1, 3, height, width)
        # Ultralytics tensor inputs skip letterboxing; resize explicitly to
        # stride multiples, and normalize boxes in that same coordinate space.
        scale = self.yolo_imgsz / max(height, width)
        det_h = max(32, int(math.ceil(height * scale / 32)) * 32)
        det_w = max(32, int(math.ceil(width * scale / 32)) * 32)
        selected_rgb = F.interpolate(selected_rgb, size=(det_h, det_w),
                                     mode="bilinear", align_corners=False)
        detections = []
        precision = {"quantize": None} if self._quantize_supported else {"half": False}
        for chunk in selected_rgb.split(self.yolo_batch_size):
            results = self._yolo.predict(
                source=chunk, device=str(chunk.device), imgsz=(det_h, det_w),
                conf=self.yolo_conf, iou=0.5, max_det=self.max_detections,
                verbose=False, save=False, **precision,
                classes=[0, 1, 2, 3, 5, 7, 9, 11])
            if len(results) != len(chunk):
                raise RuntimeError("YOLO result count differs from input frame count")
            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    detections.append(rgb.new_empty((0, 5)))
                else:
                    boxes = result.boxes.xyxy.detach().float()
                    boxes = boxes / boxes.new_tensor([det_w, det_h, det_w, det_h])
                    detections.append(torch.cat([boxes, result.boxes.conf.float()[:, None]], 1))
        return detections

    @torch.no_grad()
    def _motion(self, rgb):
        if self.flow_backend == "lk":
            raise RuntimeError("Offline LK features are mandatory; refusing an online DIS fallback.")
        batch, frames, _, height, width = rgb.shape
        features = np.zeros((batch, frames, 4), dtype=np.float32)
        if self.flow_backend == "none" or frames == 1:
            return torch.from_numpy(features).to(rgb.device)
        import cv2
        if self._flow is None:
            if not hasattr(cv2, "DISOpticalFlow_create"):
                raise RuntimeError("OpenCV with DISOpticalFlow_create is required for DIS references")
            self._flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        scale = self.flow_size / max(height, width)
        flow_h, flow_w = max(32, round(height * scale)), max(32, round(width * scale))
        reduced = F.interpolate(rgb.reshape(-1, 3, height, width), size=(flow_h, flow_w),
                                mode="bilinear", align_corners=False)
        # Clamp BEFORE casting. One transfer, instead of two GPU synchronizations
        # for every pair of frames.
        pixels = reduced.mul(255).round().clamp(0, 255).to(torch.uint8)
        pixels = pixels.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in pixels]
        for b in range(batch):
            for t in range(1, frames):
                previous, current = gray[b * frames + t - 1], gray[b * frames + t]
                if np.array_equal(previous, current):
                    continue
                flow = self._flow.calc(previous, current, None)
                if flow is None or flow.shape != (flow_h, flow_w, 2) or not np.isfinite(flow).all():
                    raise RuntimeError("DIS produced invalid flow at batch={}, frame={}".format(b, t))
                # Fraction-of-frame displacement: invariant to reduced flow
                # resolution. Signed x/y preserve horizontal/vertical direction.
                flow = flow / np.array([flow_w, flow_h], dtype=np.float32)
                magnitude = np.linalg.norm(flow, axis=-1)
                features[b, t] = (flow[..., 0].mean(), flow[..., 1].mean(),
                                  magnitude.mean(), magnitude.std())
        return torch.from_numpy(features).to(rgb.device)

    def forward(self, normalized_images, feature_grid, motion_features=None):
        if feature_grid.ndim != 5 or normalized_images.shape[0] != feature_grid.shape[0]:
            raise ValueError("Images and Swin grid must have matching batch dimensions")
        rgb = denormalize_rgb(normalized_images)
        selected = temporal_sample_indices(rgb.shape[1], self.yolo_frames, rgb.device)
        detections = self._detect(rgb, selected)
        objects = pool_object_references(feature_grid, detections, selected, rgb.shape[1])
        if motion_features is None:
            motion = self._motion(rgb)
        else:
            expected = (rgb.shape[0], rgb.shape[1], 4)
            if not isinstance(motion_features, torch.Tensor) or tuple(motion_features.shape) != expected:
                raise ValueError("Cached motion must be a tensor of shape {}".format(expected))
            if not torch.isfinite(motion_features).all():
                raise ValueError("Cached motion contains non-finite values")
            motion = motion_features.detach().to(device=feature_grid.device, dtype=torch.float32)
        motion = align_temporal_features(motion, feature_grid.shape[2])
        return objects, motion.to(device=feature_grid.device, dtype=feature_grid.dtype)
