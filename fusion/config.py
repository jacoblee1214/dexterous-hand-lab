"""Versioned configuration for geometric visuo-tactile sphere fusion."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import yaml


DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")
V2_CONFIG = Path(__file__).with_name("config_v2.yaml")


def load_fusion_config(path: str | Path = DEFAULT_CONFIG) -> Mapping:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("Fusion config must use schema_version 1")
    sphere = raw["sphere_experiment"]
    radius, diameter = float(sphere["radius_m"]), float(sphere["diameter_m"])
    if radius <= 0 or abs(diameter - 2.0 * radius) > 1e-12:
        raise ValueError("Fusion sphere diameter must equal exactly twice its radius")
    if sphere.get("radius_role") != "declared_known_radius_prior":
        raise ValueError("Fusion radius must be labeled as a declared prior")
    if min(float(raw["weights"][name]) for name in ("vision", "tactile")) <= 0:
        raise ValueError("Fusion weights must be positive")
    return MappingProxyType(raw)
