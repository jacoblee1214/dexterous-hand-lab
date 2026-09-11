"""One-epoch representative memory profile; not a checkpoint-selection run."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import resource
import time

import torch
import yaml

from .data import materialize_with_rgb_features, split_records
from .model import FrozenResNet18GridEncoder
from .train import DEFAULT_CONFIG, DEFAULT_OUTPUT, train_mode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-weights", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(DEFAULT_CONFIG).read_text())
    encoder = FrozenResNet18GridEncoder(
        args.feature_weights, config["rgb_encoder"]["weight_sha256"])
    records = materialize_with_rgb_features(encoder)
    profile_config = copy.deepcopy(config)
    profile_config["training"]["epochs"] = 1
    scratch = DEFAULT_OUTPUT.parent/"generated_cache/training_memory_profile_scratch"
    started = time.perf_counter()
    train_mode(
        "learned_rgb_tactile_reliability", split_records(records, "train"),
        split_records(records, "validation"), profile_config, scratch)
    payload = {
        "profile_kind": "representative_one_epoch_L4_training_process",
        "not_used_for_checkpoint_selection": True,
        "device": "cpu", "cuda_available": torch.cuda.is_available(),
        "elapsed_seconds": time.perf_counter()-started,
        "process_peak_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
        "batch_size": profile_config["training"]["batch_size"],
        "queries_per_sample": profile_config["training"]["queries_per_sample"],
    }
    (DEFAULT_OUTPUT/"training_memory_profile.json").write_text(
        json.dumps(payload, indent=2)+"\n")
    print(payload)


if __name__ == "__main__":
    main()
