"""Modular lightweight RGB feature frontends.

The dependency-free encoder is a fixed ablation. The primary dataset frontend
uses torchvision ResNet-18 only when an explicitly versioned local weight file
and torch installation are supplied; weights are never downloaded implicitly.
The transparent v1 SDF loss remains silhouette based and records these features
as a modular extension point rather than pretending an untrained decoder uses
them.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time

import numpy as np


@dataclass(frozen=True)
class FeatureEncoding:
    name: str
    feature_map: np.ndarray
    summary: tuple[float, ...]
    parameter_count: int
    estimated_flops: int
    runtime_ms: float
    weight_provenance: str


class AnalyticColorEdgeEncoder:
    """Frozen RGB/luminance/Sobel baseline; intentionally has no learned weights."""

    name = "analytic-rgb-luminance-sobel-v1"

    def encode(self, rgb: np.ndarray) -> FeatureEncoding:
        started = time.perf_counter()
        image = np.asarray(rgb, dtype=np.float32) / 255.0
        luminance = image @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        padded = np.pad(luminance, 1, mode="edge")
        gx = (
            -padded[:-2, :-2] + padded[:-2, 2:]
            -2*padded[1:-1, :-2] + 2*padded[1:-1, 2:]
            -padded[2:, :-2] + padded[2:, 2:]
        ) / 8.0
        gy = (
            -padded[:-2, :-2] - 2*padded[:-2, 1:-1] - padded[:-2, 2:]
            +padded[2:, :-2] + 2*padded[2:, 1:-1] + padded[2:, 2:]
        ) / 8.0
        feature_map = np.dstack((image, luminance, gx, gy, np.hypot(gx, gy))).astype(np.float32)
        summary = tuple(float(v) for v in np.concatenate((
            feature_map.mean(axis=(0, 1)), feature_map.std(axis=(0, 1)))))
        return FeatureEncoding(
            self.name, feature_map, summary, 0,
            int(rgb.shape[0]*rgb.shape[1]*45),
            (time.perf_counter()-started)*1000.0,
            "fixed analytic filters; not pretrained and not presented as a learned baseline",
        )


class LocalResNet18Encoder:
    """Optional frozen ImageNet ResNet-18 feature adapter with explicit weights."""

    name = "torchvision-resnet18-imagenet1k-v1-local-weights-only"

    def __init__(self, weights_path: str | Path, *, expected_sha256: str):
        path = Path(weights_path)
        if not path.is_file():
            raise FileNotFoundError("A local, versioned ResNet-18 weight file is required")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ValueError("Refusing unverified pretrained encoder weights")
        try:
            import torch
            from torchvision import models
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("torch and torchvision are required for this optional encoder") from exc
        self._torch = torch
        model = models.resnet18(weights=None)
        # This official torchvision checkpoint uses the legacy tar serializer,
        # which cannot be read with weights_only=True. The exact published file
        # is hash-verified above before invoking the legacy loader.
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
        self._body = torch.nn.Sequential(*list(model.children())[:-2]).eval()
        self._parameters = sum(p.numel() for p in self._body.parameters())

    def encode(self, rgb: np.ndarray) -> FeatureEncoding:  # pragma: no cover - optional dependency
        started = time.perf_counter()
        torch = self._torch
        image = torch.from_numpy(np.asarray(rgb)).permute(2, 0, 1).float()[None] / 255.0
        image = torch.nn.functional.interpolate(image, (224, 224), mode="bilinear", align_corners=False)
        mean = torch.tensor([.485, .456, .406])[None, :, None, None]
        std = torch.tensor([.229, .224, .225])[None, :, None, None]
        with torch.no_grad():
            value = self._body((image-mean)/std)[0].permute(1, 2, 0).numpy()
        summary = tuple(float(v) for v in value.mean(axis=(0, 1)))
        return FeatureEncoding(
            self.name, value, summary, self._parameters, 1_814_000_000,
            (time.perf_counter()-started)*1000.0,
            "explicit local torchvision ResNet-18 ImageNet state_dict; no implicit download",
        )
