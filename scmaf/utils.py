"""Runtime, logging, plotting, and serialization helpers."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def configure_logging(
    output_dir: Path, verbose: bool = False, append: bool = False
) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("scmaf")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(
        output_dir / "training.log", mode="a" if append else "w", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(specification: str) -> torch.device:
    value = specification.strip().lower()
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value.isdigit():
        value = f"cuda:{value}"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device} was requested, but CUDA is not available")
    return device


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def plot_history(history: list[dict[str, float]], output_path: Path) -> None:
    if not history:
        return
    epochs = [item["epoch"] for item in history]
    figure, performance_axis = plt.subplots(figsize=(7, 5))
    loss_axis = performance_axis.twinx()
    colors = {"nmi": "#00B050", "ari": "#FFB400", "acc": "#00AEEF", "loss": "#FF3333"}
    lines = []
    for name in ("nmi", "ari", "acc"):
        line = performance_axis.plot(
            epochs, [item[name] for item in history], label=name.upper(), color=colors[name]
        )[0]
        lines.append(line)
    loss_line = loss_axis.plot(
        epochs, [item["loss"] for item in history], label="Loss", color=colors["loss"]
    )[0]
    lines.append(loss_line)
    performance_axis.set(xlabel="Epoch", ylabel="Clustering performance", ylim=(0, 1))
    loss_axis.set_ylabel("Loss")
    performance_axis.legend(lines, [line.get_label() for line in lines], loc="best")
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)

