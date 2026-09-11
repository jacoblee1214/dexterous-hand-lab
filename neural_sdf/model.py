"""Compact learned neural SDF with frozen RGB and local tactile features."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


MODES = (
    "learned_rgb_only",
    "learned_rgb_tactile_global",
    "learned_rgb_tactile_representative",
    "learned_rgb_tactile_finite_patch",
    "learned_rgb_tactile_reliability",
)


class FrozenResNet18GridEncoder(nn.Module):
    """ImageNet ResNet-18 through layer2; local file and hash are mandatory."""

    name = "torchvision-resnet18-layer2-imagenet1k-v1-frozen"
    output_channels = 16
    estimated_flops_per_frame = 555_000_000

    def __init__(self, weights_path: str | Path, expected_sha256: str):
        super().__init__()
        path = Path(weights_path)
        if not path.is_file():
            raise FileNotFoundError("Explicit local ResNet-18 weights are required")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ValueError("Refusing RGB weights with a non-frozen SHA-256")
        backbone = models.resnet18(weights=None)
        # The exact official legacy checkpoint is hash-verified before unpickling.
        backbone.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
        self.body = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2,
        ).eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None])

    @property
    def parameter_count(self):
        return sum(value.numel() for value in self.body.parameters())

    @torch.no_grad()
    def forward(self, rgb_uint8_or_float: torch.Tensor) -> torch.Tensor:
        value = rgb_uint8_or_float.float()
        if value.max() > 1.5:
            value = value/255.0
        value = F.interpolate(value, (224, 224), mode="bilinear", align_corners=False)
        feature = self.body((value-self.mean)/self.std)
        # A parameter-free grouped mean keeps local pretrained features while
        # making dense query evaluation practical on CPU.
        return feature.reshape(feature.shape[0], 16, 8, *feature.shape[-2:]).mean(dim=2)


class NeuralSDFModel(nn.Module):
    """Object-centric query decoder; no class, CAD, radius, or GT input."""

    def __init__(self, mode: str, *, rgb_channels=16, tactile_hidden=32,
                 decoder_hidden=96, fourier_frequencies=3):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown neural SDF mode: {mode}")
        self.mode = mode
        self.fourier_frequencies = int(fourier_frequencies)
        self.sensor_embedding = nn.Embedding(8, 4)
        self.finger_embedding = nn.Embedding(4, 4)
        self.tactile_element = nn.Sequential(
            nn.Linear(17, tactile_hidden), nn.SiLU(),
            nn.Linear(tactile_hidden, tactile_hidden), nn.SiLU(),
        )
        self.local_element = nn.Sequential(
            nn.Linear(11, tactile_hidden), nn.SiLU(),
            nn.Linear(tactile_hidden, tactile_hidden), nn.SiLU(),
        )
        query_dim = 3+6*self.fourier_frequencies
        decoder_input = query_dim+rgb_channels+5+4+tactile_hidden*2+1
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input, decoder_hidden), nn.SiLU(),
            nn.Linear(decoder_hidden, decoder_hidden), nn.SiLU(),
            nn.Linear(decoder_hidden, decoder_hidden//2), nn.SiLU(),
            nn.Linear(decoder_hidden//2, 1),
        )

    @property
    def trainable_parameter_count(self):
        return sum(value.numel() for value in self.parameters() if value.requires_grad)

    def _encode_query(self, query):
        normalized = query/.065
        parts = [normalized]
        for index in range(self.fourier_frequencies):
            frequency = math.pi*(2**index)
            parts.extend((torch.sin(normalized*frequency), torch.cos(normalized*frequency)))
        return torch.cat(parts, dim=-1)

    def _sample_rgb(self, query, feature_map, visibility_masks, intrinsic,
                    camera_from_object):
        batch, count, _ = query.shape
        ones = torch.ones((batch, count, 1), device=query.device, dtype=query.dtype)
        homogeneous = torch.cat((query, ones), dim=-1)
        camera = torch.bmm(homogeneous, camera_from_object.transpose(1, 2))[..., :3]
        z = camera[..., 2].clamp_min(1e-6)
        u = intrinsic[:, None, 0, 0]*camera[..., 0]/z+intrinsic[:, None, 0, 2]
        v = intrinsic[:, None, 1, 1]*camera[..., 1]/z+intrinsic[:, None, 1, 2]
        height, width = visibility_masks.shape[-2:]
        grid = torch.stack((2*u/max(1, width-1)-1, 2*v/max(1, height-1)-1), dim=-1)
        sample_grid = grid[:, :, None, :]
        rgb_feature = F.grid_sample(
            feature_map, sample_grid, mode="bilinear", padding_mode="zeros",
            align_corners=True).squeeze(-1).transpose(1, 2)
        visibility = F.grid_sample(
            visibility_masks, sample_grid, mode="nearest", padding_mode="zeros",
            align_corners=True).squeeze(-1).transpose(1, 2)
        in_view = ((grid[..., 0].abs() <= 1) & (grid[..., 1].abs() <= 1)
                   & (camera[..., 2] > 0)).to(query.dtype)
        out_of_view = (1-in_view)[..., None]
        visibility = torch.cat((visibility, out_of_view), dim=-1)
        ray = camera/(torch.linalg.norm(camera, dim=-1, keepdim=True)+1e-8)
        ray_geometry = torch.cat((ray, (camera[..., 2]/.30)[..., None]), dim=-1)
        return rgb_feature, visibility, ray_geometry

    def _tactile_features(self, query, tactile, coverage):
        position, normal = tactile["position"], tactile["normal"]
        valid = tactile["valid"].float()
        sensor = self.sensor_embedding(tactile["sensor_index"])
        finger = self.finger_embedding(tactile["finger_index"])
        element_input = torch.cat((
            position/.065, normal, tactile["scalar_n"]/5.0,
            tactile["indentation_m"]/.005, tactile["sigma_m"]/.01,
            sensor, finger,
        ), dim=-1)
        element = self.tactile_element(element_input)*valid[..., None]
        denominator = valid.sum(dim=1, keepdim=True).clamp_min(1)[..., None]
        global_mean = element.sum(dim=1, keepdim=True)/denominator
        masked = element.masked_fill(~tactile["valid"][..., None], -1e6)
        global_max = masked.max(dim=1, keepdim=True).values
        global_max = torch.where(valid.sum(dim=1, keepdim=True)[..., None] > 0,
                                 global_max, torch.zeros_like(global_max))
        global_feature = .5*(global_mean+global_max)
        global_feature = global_feature.expand(-1, query.shape[1], -1)

        relative = query[:, :, None, :]-position[:, None, :, :]
        rep_distance = torch.linalg.norm(relative, dim=-1)
        if self.mode in {"learned_rgb_tactile_finite_patch",
                         "learned_rgb_tactile_reliability"}:
            patch_relative = query[:, :, None, None, :]-tactile["patch"][:, None, :, :, :]
            patch_distance = torch.linalg.norm(patch_relative, dim=-1).min(dim=-1).values
        else:
            patch_distance = rep_distance
        direction = relative/(rep_distance[..., None]+1e-8)
        alignment = (direction*normal[:, None, :, :]).sum(dim=-1)
        sigma = tactile["sigma_m"][:, None, :, 0]
        near = torch.exp(-.5*(patch_distance/(.012+sigma))**2)
        local_input = torch.cat((
            relative/.065, (rep_distance/.065)[..., None],
            (patch_distance/.065)[..., None], alignment[..., None],
            tactile["scalar_n"][:, None, :, :].expand(-1, query.shape[1], -1, -1)/5.0,
            tactile["indentation_m"][:, None, :, :].expand(-1, query.shape[1], -1, -1)/.005,
            tactile["sigma_m"][:, None, :, :].expand(-1, query.shape[1], -1, -1)/.01,
            near[..., None], valid[:, None, :, None].expand(-1, query.shape[1], -1, -1),
        ), dim=-1)
        local = self.local_element(local_input)
        weights = near*valid[:, None, :]
        if self.mode == "learned_rgb_tactile_reliability":
            uncertainty = 1/(1+(sigma/.014)**2)
            weights = weights*uncertainty*coverage[:, None, :]
        weights = weights/(weights.sum(dim=-1, keepdim=True)+1e-8)
        local_feature = (local*weights[..., None]).sum(dim=2)
        return global_feature, local_feature

    def forward(self, query, model_input):
        rgb, visibility, ray = self._sample_rgb(
            query, model_input["rgb_feature_map"], model_input["visibility_masks"],
            model_input["intrinsic"], model_input["camera_from_object"])
        zero = torch.zeros((query.shape[0], query.shape[1], 32),
                           device=query.device, dtype=query.dtype)
        if self.mode == "learned_rgb_only":
            global_touch, local_touch = zero, zero
            # A true RGB-only ablation must not receive tactile availability or
            # coverage through a side channel.
            coverage_feature = model_input["coverage"].new_zeros(
                model_input["coverage"].shape)
        else:
            global_touch, local_touch = self._tactile_features(
                query, model_input["tactile"], model_input["coverage"])
            coverage_feature = model_input["coverage"]
            if self.mode == "learned_rgb_tactile_global":
                local_touch = zero
        value = torch.cat((self._encode_query(query), rgb, visibility, ray,
                           global_touch, local_touch,
                           coverage_feature[:, None, :].expand(-1, query.shape[1], -1)),
                          dim=-1)
        return self.decoder(value).squeeze(-1)
