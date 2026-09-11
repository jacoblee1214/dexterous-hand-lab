"""Explicit compute-device selection for learned SDF training/evaluation."""

from __future__ import annotations

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    value = str(requested).strip().lower()
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value == "cpu":
        return torch.device("cpu")
    if value.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but PyTorch cannot access an NVIDIA GPU; "
                "refusing a silent CPU fallback")
        device = torch.device(value)
        torch.empty(1, device=device)
        return device
    raise ValueError("device must be auto, cpu, cuda, or cuda:N")


def device_metadata(device: torch.device) -> dict:
    result = {
        "requested_device_resolved_to": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if device.type == "cuda":
        index = device.index or 0
        properties = torch.cuda.get_device_properties(index)
        result.update({
            "cuda_device_index": index,
            "cuda_device_name": properties.name,
            "cuda_total_memory_bytes": int(properties.total_memory),
        })
    return result
