"""Training and evaluation loops."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from .data import DataBundle
from .losses import reconstruction_loss
from .metrics import clustering_scores
from .model import FusionBackbone
from .utils import plot_history, write_json

LOGGER = logging.getLogger("scmaf")


def _loader(
    bundle: DataBundle,
    config: Any,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    batch_size = min(int(config.batch_size), len(bundle.dataset))
    if batch_size < 2:
        raise ValueError("At least two samples are required because the model uses batch normalization")
    drop_last = shuffle and len(bundle.dataset) % batch_size == 1
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        bundle.dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=int(getattr(config, "num_workers", 0)),
        pin_memory=bool(getattr(config, "pin_memory", False)),
        generator=generator,
    )


def _move_features(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return batch["x1"].to(device, non_blocking=True), batch["x2"].to(device, non_blocking=True)


def _training_loss(
    outputs: dict[str, Any],
    batch: dict[str, Any],
    bundle: DataBundle,
    device: torch.device,
    dataset_type: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    if dataset_type == "multi_omics":
        types = bundle.modality_types
        if types is None:
            raise ValueError("Multi-omics modality types are missing")
        raw1 = batch["raw1"].to(device, non_blocking=True)
        raw2 = batch["raw2"].to(device, non_blocking=True)
        sf1 = batch["sf1"].to(device, non_blocking=True)
        sf2 = batch["sf2"].to(device, non_blocking=True)
        losses = {
            "direct1": reconstruction_loss(*outputs["direct1"], sf1, raw1, types[0]),
            "direct2": reconstruction_loss(*outputs["direct2"], sf2, raw2, types[1]),
            "fused1": reconstruction_loss(*outputs["fused1"], sf1, raw1, types[0]),
            "fused2": reconstruction_loss(*outputs["fused2"], sf2, raw2, types[1]),
        }
    else:
        x1, x2 = _move_features(batch, device)
        losses = {
            "direct1": F.mse_loss(outputs["direct1"], x1),
            "direct2": F.mse_loss(outputs["direct2"], x2),
            "fused1": F.mse_loss(outputs["fused1"], x1),
            "fused2": F.mse_loss(outputs["fused2"], x2),
        }
    total = sum(losses.values())
    return total, {name: float(value.detach()) for name, value in losses.items()}


@torch.no_grad()
def collect_embeddings(
    model: FusionBackbone,
    bundle: DataBundle,
    config: Any,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in _loader(bundle, config, shuffle=False, seed=seed):
        x1, x2 = _move_features(batch, device)
        embeddings.append(model.embed(x1, x2).cpu().numpy())
        labels.append(batch["label"].numpy())
    return np.concatenate(embeddings), np.concatenate(labels)


def evaluate(
    model: FusionBackbone,
    bundle: DataBundle,
    config: Any,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    embeddings, labels = collect_embeddings(model, bundle, config, device, seed)
    predicted = KMeans(
        n_clusters=bundle.n_classes,
        n_init=int(getattr(config, "kmeans_n_init", 20)),
        random_state=seed,
    ).fit_predict(embeddings)
    return clustering_scores(labels, predicted), embeddings, labels, predicted


def _checkpoint(
    model: FusionBackbone,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    config: Any,
    run_seed: int,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": vars(config).copy(),
        "run_seed": run_seed,
    }


def fit(
    model: FusionBackbone,
    bundle: DataBundle,
    config: Any,
    device: torch.device,
    output_dir: Path,
    seed: int,
    resume_path: Path | None = None,
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config.lr),
        amsgrad=True,
    )
    dataset_type = str(config.dataset_type).lower().replace("-", "_")
    history: list[dict[str, float]] = []
    best_loss = np.inf
    final_artifacts: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    started = time.perf_counter()
    start_epoch = 1

    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        seed = int(checkpoint.get("run_seed", seed))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint.get("metrics", {}).get("loss", np.inf))
        if start_epoch > int(config.epochs):
            raise ValueError(
                f"Checkpoint epoch {start_epoch - 1} is not earlier than target epoch {config.epochs}"
            )
        LOGGER.info("Resumed checkpoint %s at epoch %d", resume_path, start_epoch - 1)

    train_loader = _loader(bundle, config, shuffle=True, seed=seed)

    LOGGER.info(
        "Training %s on %s (%d samples, %d classes, %d parameters)",
        model.__class__.__name__,
        device,
        len(bundle.dataset),
        bundle.n_classes,
        sum(parameter.numel() for parameter in model.parameters()),
    )
    for epoch in range(start_epoch, int(config.epochs) + 1):
        model.train()
        total_loss = 0.0
        component_totals = {name: 0.0 for name in ("direct1", "direct2", "fused1", "fused2")}
        for batch in train_loader:
            x1, x2 = _move_features(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(x1, x2)
            loss, components = _training_loss(outputs, batch, bundle, device, dataset_type)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss encountered at epoch {epoch}: {loss.item()}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float(config.grad_clip_norm), error_if_nonfinite=False
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"Non-finite gradient encountered at epoch {epoch}: {gradient_norm.item()}"
                )
            optimizer.step()
            total_loss += float(loss.detach())
            for name, value in components.items():
                component_totals[name] += value

        should_evaluate = epoch == 1 or epoch == int(config.epochs) or epoch % int(config.print_every) == 0
        if not should_evaluate:
            continue
        average_loss = total_loss / len(train_loader)
        scores, embeddings, labels, predicted = evaluate(model, bundle, config, device, seed)
        final_artifacts = embeddings, labels, predicted
        record = {"epoch": epoch, "loss": average_loss, **scores}
        history.append(record)
        write_json(output_dir / "history.json", history)
        component_text = ", ".join(
            f"{name}={value / len(train_loader):.4f}" for name, value in component_totals.items()
        )
        LOGGER.info(
            "Epoch %d/%d | loss=%.4f (%s) | NMI=%.4f ARI=%.4f ACC=%.4f",
            epoch,
            config.epochs,
            average_loss,
            component_text,
            scores["nmi"],
            scores["ari"],
            scores["acc"],
        )
        if bool(config.save_model) and average_loss < best_loss:
            best_loss = average_loss
            torch.save(
                _checkpoint(model, optimizer, epoch, record, config, seed), output_dir / "best.pt"
            )
        if bool(config.save_model):
            torch.save(
                _checkpoint(model, optimizer, epoch, record, config, seed), output_dir / "last.pt"
            )

    elapsed = time.perf_counter() - started
    if not history or final_artifacts is None:
        raise RuntimeError("Training finished without an evaluation")
    final_metrics = {**history[-1], "training_seconds": elapsed, "seed": seed}
    embeddings, labels, predicted = final_artifacts
    np.savez_compressed(
        output_dir / "representations.npz",
        embedding=embeddings,
        label=labels,
        prediction=predicted,
    )
    write_json(output_dir / "history.json", history)
    write_json(output_dir / "metrics.json", final_metrics)
    plot_history(history, output_dir / "training_curves.pdf")
    if bool(config.save_model):
        torch.save(
            _checkpoint(model, optimizer, int(config.epochs), final_metrics, config, seed),
            output_dir / "last.pt",
        )
    LOGGER.info("Finished in %.2f seconds; results saved to %s", elapsed, output_dir)
    return final_metrics
