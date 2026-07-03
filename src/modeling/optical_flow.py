import json
import os
import sys
from argparse import Namespace

import cv2
import numpy as np
import torch


FLOW_FEATURE_DIM = 4


def normalize_optical_flow_source(source):
    key = str(source or "none").strip().lower().replace("_", "-")
    if key in ("none", "off", "false", "0", "no"):
        return "none"
    if key in ("lk", "my", "sparse-lk", "lucas-kanade", "lucas_kanade"):
        return "lk"
    if key in ("sea-raft", "searaft"):
        return "sea_raft"
    raise ValueError(
        "Unsupported optical flow source '{}'. Use one of: none, lk, my, sea-raft.".format(source)
    )


def build_optical_flow_extractor(args, device=None):
    source = normalize_optical_flow_source(getattr(args, "optical_flow_source", "lk"))
    if source == "none":
        return None
    if source == "lk":
        return SparseLKFlowExtractor(args, device=device)
    if source == "sea_raft":
        return SeaRaftFlowExtractor(args, device=device)
    raise ValueError("Unsupported optical flow source '{}'.".format(source))


def _imagenet_unnormalize(images):
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device, dtype=images.dtype)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device, dtype=images.dtype)
    return images * std.view(1, 3, 1, 1, 1) + mean.view(1, 3, 1, 1, 1)


def _to_uint8_video(images, input_normalized=True):
    images = images.detach().float()
    if input_normalized:
        images = _imagenet_unnormalize(images)
    return images.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)


def _dense_flow_to_stats(flow):
    if isinstance(flow, np.ndarray):
        flow = torch.from_numpy(flow)
    flow = flow.detach().float()
    if flow.dim() == 4:
        flow = flow[0]
    if flow.dim() == 3 and flow.shape[0] == 2:
        u, v = flow[0], flow[1]
    elif flow.dim() == 3 and flow.shape[-1] == 2:
        u, v = flow[..., 0], flow[..., 1]
    else:
        raise ValueError("Expected flow shape [2,H,W] or [H,W,2], got {}".format(tuple(flow.shape)))

    mag = torch.sqrt(u * u + v * v)
    angles = torch.atan2(v, u)
    mean_angle = torch.atan2(torch.sin(angles).mean(), torch.cos(angles).mean())
    return torch.stack([u.mean(), v.mean(), mag.mean(), torch.cos(mean_angle)])


class BaseOpticalFlowExtractor(object):
    def __init__(self, args, device=None):
        self.device = device
        self.input_normalized = bool(getattr(args, "optical_flow_input_normalized", True))
        self.cache_enabled = bool(getattr(args, "optical_flow_cache", True))
        self.cache = {}

    def __call__(self, images, cache_keys=None):
        # images: [B, C, T, H, W]
        batch_size, _, num_frames, _, _ = images.shape
        num_pairs = max(num_frames - 1, 0)
        features = torch.zeros(
            batch_size, num_pairs, FLOW_FEATURE_DIM, device=images.device, dtype=torch.float32
        )
        for batch_idx in range(batch_size):
            key = self._make_cache_key(cache_keys, batch_idx)
            if self.cache_enabled and key is not None and key in self.cache:
                features[batch_idx] = self.cache[key].to(images.device)
                continue
            sample_feats = self._compute_video(images[batch_idx].unsqueeze(0))
            sample_feats = sample_feats.to(images.device, dtype=torch.float32)
            if self.cache_enabled and key is not None:
                self.cache[key] = sample_feats.detach().cpu()
            features[batch_idx] = sample_feats
        return features

    def _make_cache_key(self, cache_keys, batch_idx):
        if cache_keys is None:
            return None
        try:
            key = cache_keys[batch_idx]
        except Exception:
            return None
        if torch.is_tensor(key):
            key = key.detach().cpu().reshape(-1)
            if key.numel() == 1:
                return ("tensor", float(key.item()))
            return ("tensor", tuple(float(v) for v in key.tolist()))
        if isinstance(key, (int, float, str)):
            return ("value", key)
        return None

    def _compute_video(self, images):
        raise NotImplementedError


class SparseLKFlowExtractor(BaseOpticalFlowExtractor):
    """Current sparse Lucas-Kanade flow used by the repository."""

    def _compute_pair(self, prev_rgb, next_rgb):
        try:
            prev_gray = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
            next_gray = cv2.cvtColor(next_rgb, cv2.COLOR_RGB2GRAY)
            prev_pts = cv2.goodFeaturesToTrack(
                prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30
            )
            if prev_pts is None:
                return torch.zeros(FLOW_FEATURE_DIM)
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_pts, None)
            if next_pts is None or status is None:
                return torch.zeros(FLOW_FEATURE_DIM)
            good_prev = prev_pts[status == 1]
            good_next = next_pts[status == 1]
            if len(good_prev) == 0:
                return torch.zeros(FLOW_FEATURE_DIM)
            flow_vectors = (good_next - good_prev).reshape(-1, 2)
            return _dense_flow_to_stats(flow_vectors.reshape(1, -1, 2))
        except Exception as exc:
            print("Optical flow error: {}, using fallback values".format(exc))
            return torch.zeros(FLOW_FEATURE_DIM)

    def _compute_video(self, images):
        frames = _to_uint8_video(images, input_normalized=self.input_normalized)[0]
        frames = frames.permute(1, 2, 3, 0).cpu().numpy()
        features = [self._compute_pair(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]
        if not features:
            return torch.zeros((0, FLOW_FEATURE_DIM))
        return torch.stack(features)


class SeaRaftFlowExtractor(BaseOpticalFlowExtractor):
    """SEA-RAFT wrapper that converts dense flow to the same 4-D stats as LK flow."""

    def __init__(self, args, device=None):
        super(SeaRaftFlowExtractor, self).__init__(args, device=device)
        self.repo = getattr(args, "sea_raft_repo", "")
        self.cfg = getattr(args, "sea_raft_cfg", "")
        self.checkpoint = getattr(args, "sea_raft_checkpoint", "")
        self.url = getattr(args, "sea_raft_url", "")
        self.iters = int(getattr(args, "sea_raft_iters", 4))
        self.model = None
        self.InputPadder = None
        self._load_model()

    def _load_model(self):
        if not self.repo:
            raise ValueError("--sea_raft_repo is required when --optical_flow_source sea-raft")
        repo = os.path.abspath(self.repo)
        core = os.path.join(repo, "core")
        for path in (core, repo):
            if path and path not in sys.path:
                sys.path.insert(0, path)

        try:
            from raft import RAFT
            from utils.utils import InputPadder, load_ckpt
        except Exception as exc:
            raise ImportError(
                "Could not import SEA-RAFT from '{}'. Set --sea_raft_repo to the SEA-RAFT repo root.".format(repo)
            ) from exc

        self.InputPadder = InputPadder
        model_device = self.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.url:
            args = Namespace(device=str(model_device), url=self.url)
            self.model = RAFT.from_pretrained(self.url, args=args)
        else:
            if not self.cfg:
                raise ValueError("--sea_raft_cfg is required unless --sea_raft_url is provided")
            if not self.checkpoint:
                raise ValueError("--sea_raft_checkpoint is required unless --sea_raft_url is provided")
            with open(self.cfg, "r") as f:
                cfg_dict = json.load(f)
            raft_args = Namespace(**cfg_dict)
            raft_args.path = self.checkpoint
            raft_args.url = None
            raft_args.device = str(model_device)
            self.model = RAFT(raft_args)
            load_ckpt(self.model, self.checkpoint)

        self.model.to(model_device)
        self.model.eval()
        self.device = model_device
        for param in self.model.parameters():
            param.requires_grad = False

    def _compute_video(self, images):
        frames = _to_uint8_video(images, input_normalized=self.input_normalized)[0].float()
        features = []
        with torch.no_grad():
            for idx in range(frames.shape[1] - 1):
                image1 = frames[:, idx].unsqueeze(0).to(self.device)
                image2 = frames[:, idx + 1].unsqueeze(0).to(self.device)
                padder = self.InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                output = self.model(image1, image2, iters=self.iters, test_mode=True)
                flow = output["flow"][-1] if isinstance(output, dict) else output
                flow = padder.unpad(flow)[0]
                features.append(_dense_flow_to_stats(flow).cpu())
        if not features:
            return torch.zeros((0, FLOW_FEATURE_DIM))
        return torch.stack(features)
