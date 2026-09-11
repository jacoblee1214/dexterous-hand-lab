"""Deterministic CPU training for five lightweight neural-SDF ablations."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .data import collate_model_inputs, materialize_with_rgb_features, split_records
from .model import FrozenResNet18GridEncoder, MODES, NeuralSDFModel


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "neural_sdf/config_v1.yaml"
DEFAULT_OUTPUT = ROOT / "experiments/neural_sdf/learned_v1_20260911"


def _box_sdf(point, half, center=None):
    half = point.new_tensor(half)
    center = point.new_zeros(3) if center is None else point.new_tensor(center)
    q = (point-center).abs()-half
    return torch.linalg.norm(q.clamp_min(0), dim=-1)+torch.minimum(q.max(dim=-1).values,
                                                                  q.new_zeros(()))


def supervision_sdf(point, family, parameters):
    """Training/evaluation-only analytic supervision; never called by model."""
    if family == "sphere":
        return torch.linalg.norm(point, dim=-1)-float(parameters["radius_m"])
    if family == "cylinder":
        d = torch.stack((torch.linalg.norm(point[..., :2], dim=-1)-float(parameters["radius_m"]),
                         point[..., 2].abs()-float(parameters["half_height_m"])), dim=-1)
        return torch.linalg.norm(d.clamp_min(0), dim=-1)+torch.minimum(
            d.max(dim=-1).values, d.new_zeros(()))
    if family == "cuboid":
        return _box_sdf(point, parameters["half_extents_m"])
    if family == "rounded_box":
        radius = float(parameters["round_m"])
        half = np.asarray(parameters["half_extents_m"])-radius
        return _box_sdf(point, half.tolist())-radius
    if family == "asymmetric_l":
        a = _box_sdf(point, parameters["box_a_half_m"], parameters["box_a_center_m"])
        b = _box_sdf(point, parameters["box_b_half_m"], parameters["box_b_center_m"])
        return torch.minimum(a, b)
    raise ValueError(family)


def _choose_surface(record, label, count, generator):
    surface = record["supervision"]["surface_points"]
    labels = record["supervision"]["surface_labels"]
    candidates = torch.nonzero(labels == label, as_tuple=False).flatten()
    if not len(candidates):
        candidates = torch.arange(len(surface))
    indexes = candidates[torch.randint(len(candidates), (count,), generator=generator)]
    return surface[indexes]


def sample_queries(records, count, generator, device):
    queries, targets = [], []
    visible_count, occluded_count = count//4, count//4
    uniform_count = count-visible_count-occluded_count
    for record in records:
        visible = _choose_surface(record, 0, visible_count, generator)
        occluded = _choose_surface(record, 1, occluded_count, generator)
        near = torch.cat((visible, occluded))+torch.randn(
            visible_count+occluded_count, 3, generator=generator)*.0025
        uniform = torch.rand(uniform_count, 3, generator=generator)*.13-.065
        query = torch.cat((near, uniform)).to(device)
        target = supervision_sdf(
            query, record["supervision"]["family"],
            record["supervision"]["parameters"]).clamp(-.03, .03)
        queries.append(query); targets.append(target)
    return torch.stack(queries), torch.stack(targets)


def sample_surface_points(records, count, generator, device):
    points = []
    for record in records:
        visible = _choose_surface(record, 0, count//2, generator)
        occluded = _choose_surface(record, 1, count-count//2, generator)
        points.append(torch.cat((visible, occluded)))
    return torch.stack(points).to(device)


def _drop_tactile(model_input, full_probability, contact_probability, generator):
    result = {key: value for key, value in model_input.items() if key != "tactile"}
    result["tactile"] = {key: value.clone() for key, value in model_input["tactile"].items()}
    valid = result["tactile"]["valid"]
    original = valid.sum(dim=1)
    for batch_index in range(len(valid)):
        if torch.rand((), generator=generator) < full_probability:
            valid[batch_index] = False
        else:
            drop = torch.rand(valid.shape[1], generator=generator) < contact_probability
            valid[batch_index] &= ~drop.to(valid.device)
    remaining = valid.sum(dim=1)
    fraction = remaining.float()/original.clamp_min(1).float()
    result["coverage"] = result["coverage"]*fraction[:, None]
    return result


def tactile_contact_loss(model, model_input, temperature):
    if model.mode == "learned_rgb_only":
        return next(model.parameters()).new_zeros(())
    tactile, valid = model_input["tactile"], model_input["tactile"]["valid"]
    if not valid.any():
        return next(model.parameters()).new_zeros(())
    if model.mode in {"learned_rgb_tactile_finite_patch", "learned_rgb_tactile_reliability"}:
        batch, contacts, patch_count, _ = tactile["patch"].shape
        query = tactile["patch"].reshape(batch, contacts*patch_count, 3)
        values = model(query, model_input).abs().reshape(batch, contacts, patch_count)
        smooth_min = -temperature*torch.log(torch.exp(-values/temperature).mean(dim=-1)+1e-12)
        weights = valid.float()
        if model.mode == "learned_rgb_tactile_reliability":
            weights = weights/(1+(tactile["sigma_m"][..., 0]/.014)**2)
            weights = weights*model_input["coverage"]
        return (smooth_min*weights).sum()/weights.sum().clamp_min(1e-8)
    values = model(tactile["position"], model_input).abs()
    return (values*valid).sum()/valid.sum().clamp_min(1)


@torch.no_grad()
def validation_occluded_surface_error(model, records, batch_size, device):
    model.eval(); values = []
    for start in range(0, len(records), batch_size):
        subset = records[start:start+batch_size]
        model_input = collate_model_inputs(subset, device)
        point_sets = []
        for record in subset:
            labels = record["supervision"]["surface_labels"]
            points = record["supervision"]["surface_points"][labels == 1]
            indexes = torch.linspace(0, max(0, len(points)-1), 192).long()
            point_sets.append(points[indexes])
        query = torch.stack(point_sets).to(device)
        values.extend(model(query, model_input).abs().mean(dim=1).cpu().tolist())
    return float(np.mean(values))


def train_mode(mode, train_records, validation_records, config, output, *, device="cpu"):
    seed = int(config["seed"]); torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model_config, train_config = config["model"], config["training"]
    model = NeuralSDFModel(
        mode, rgb_channels=int(config["rgb_encoder"]["output_channels"]),
        tactile_hidden=int(model_config["tactile_hidden"]),
        decoder_hidden=int(model_config["decoder_hidden"]),
        fourier_frequencies=int(model_config["query_fourier_frequencies"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]))
    generator = torch.Generator().manual_seed(seed+sum(map(ord, mode)))
    best = math.inf; best_epoch = 0; logs = []
    started = time.perf_counter()
    for epoch in range(1, int(train_config["epochs"])+1):
        model.train(); permutation = torch.randperm(len(train_records), generator=generator)
        losses = []
        for start in range(0, len(train_records), int(train_config["batch_size"])):
            subset = [train_records[int(index)] for index in
                      permutation[start:start+int(train_config["batch_size"])] ]
            model_input = collate_model_inputs(subset, device)
            model_input = _drop_tactile(
                model_input, float(train_config["modality_dropout_probability"]),
                float(train_config["contact_dropout_probability"]), generator)
            query, target = sample_queries(
                subset, int(train_config["queries_per_sample"]), generator, device)
            prediction = model(query, model_input)
            sdf_loss = F.smooth_l1_loss(prediction, target, beta=.003)
            surface_query = sample_surface_points(subset, 64, generator, device)
            surface_loss = model(surface_query, model_input).abs().mean()
            batch_index = start//int(train_config["batch_size"])
            if batch_index % int(train_config["tactile_contact_loss_every_batches"]) == 0:
                contact_loss = tactile_contact_loss(
                    model, model_input,
                    float(train_config["loss"]["finite_patch_softmin_temperature_m"]))
            else:
                contact_loss = prediction.new_zeros(())
            weights = train_config["loss"]
            loss = (float(weights["sdf_l1"])*sdf_loss+
                    float(weights["surface"])*surface_loss+
                    float(weights["tactile_contact"])*contact_loss)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_config["gradient_clip_norm"]))
            optimizer.step(); losses.append(float(loss.detach()))
        validation = validation_occluded_surface_error(
            model, validation_records, int(train_config["batch_size"]), device)
        row = {"epoch": epoch, "training_loss": float(np.mean(losses)),
               "validation_occluded_surface_abs_sdf_m": validation}
        logs.append(row); print(mode, row, flush=True)
        payload = {
            "mode": mode, "epoch": epoch, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "validation_metric_m": validation,
            "source_parent_commit": config["parent_commit"],
            "configuration_version": config["configuration_version"],
        }
        mode_dir = output/"checkpoints"/mode; mode_dir.mkdir(parents=True, exist_ok=True)
        torch.save(payload, mode_dir/"last.pt")
        if validation < best:
            best, best_epoch = validation, epoch
            torch.save(payload, mode_dir/"best.pt")
    (output/"training_logs").mkdir(parents=True, exist_ok=True)
    (output/"training_logs"/f"{mode}.json").write_text(
        json.dumps({"mode": mode, "best_epoch": best_epoch,
                    "best_validation_metric_m": best,
                    "elapsed_seconds": time.perf_counter()-started,
                    "epochs": logs}, indent=2)+"\n")
    return {"mode": mode, "best_epoch": best_epoch, "best_validation_metric_m": best,
            "trainable_parameters": model.trainable_parameter_count}


def run(config_path, output, weights_path, *, device="cpu", force_data=False, modes=MODES):
    config = yaml.safe_load(Path(config_path).read_text())
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    encoder = FrozenResNet18GridEncoder(
        weights_path, config["rgb_encoder"]["weight_sha256"])
    records = materialize_with_rgb_features(
        encoder, device=device, force_data=force_data)
    train_records = split_records(records, "train")
    validation_records = split_records(records, "validation")
    summaries = [train_mode(mode, train_records, validation_records, config, output, device=device)
                 for mode in modes]
    summary_path = output/"training_summary.json"
    # Rebuild from per-mode logs/checkpoints instead of merging a stale summary.
    # This also makes separate resumable CLI invocations race-safe.
    merged = {}
    for mode in MODES:
        log_path = output/"training_logs"/f"{mode}.json"
        checkpoint_path = output/"checkpoints"/mode/"best.pt"
        if not (log_path.is_file() and checkpoint_path.is_file()):
            continue
        log = json.loads(log_path.read_text())
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["epoch"] != log["best_epoch"]:
            raise ValueError(f"Checkpoint/log mismatch for {mode}")
        merged[mode] = {
            "mode": mode, "best_epoch": int(log["best_epoch"]),
            "best_validation_metric_m": float(log["best_validation_metric_m"]),
            "trainable_parameters": NeuralSDFModel(
                mode, rgb_channels=int(config["rgb_encoder"]["output_channels"]),
                tactile_hidden=int(config["model"]["tactile_hidden"]),
                decoder_hidden=int(config["model"]["decoder_hidden"]),
                fourier_frequencies=int(config["model"]["query_fourier_frequencies"]),
            ).trainable_parameter_count,
        }
    summary_path.write_text(json.dumps({
        "configuration": config, "models": [merged[key] for key in sorted(merged)],
        "frozen_rgb_parameter_count": encoder.parameter_count,
        "frozen_rgb_estimated_flops_per_frame": encoder.estimated_flops_per_frame,
        "test_split_used_for_checkpoint_selection": False,
    }, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--feature-weights", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--mode", action="append", choices=MODES,
                        help="Train one or more ablations; default trains all")
    args = parser.parse_args()
    run(args.config, args.output, args.feature_weights, device=args.device,
        force_data=args.force_data, modes=tuple(args.mode) if args.mode else MODES)


if __name__ == "__main__":
    main()
