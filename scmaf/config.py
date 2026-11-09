"""Command-line and YAML configuration handling."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULTS = {
    "dataset_type": "multi_omics",
    "epochs": 400,
    "batch_size": 256,
    "print_every": 50,
    "device": "auto",
    "lr": 1e-3,
    "seed": 0,
    "runs": 1,
    "output_dir": "outputs",
    "num_chunks": 2,
    "sigma1": 3.5,
    "sigma2": 2.0,
    "activation": "ReLU",
    "batch_norm": True,
    "dropout": 0.0,
    "attention_dropout": 0.1,
    "transformer_layers": 1,
    "transformer_ff_dim": 2048,
    "n_heads": 1,
    "filter1": True,
    "filter2": False,
    "f1": 2000,
    "f2": 2000,
    "data_norm": "standard",
    "view_keys": ["X1", "X2"],
    "label_key": "Y",
    "kmeans_n_init": 20,
    "num_workers": 0,
    "pin_memory": False,
    "save_model": True,
    "deterministic": True,
    "max_samples": None,
    "experiment_name": None,
    "verbose": False,
    "dry_run": False,
    "resume": None,
    "grad_clip_norm": 10.0,
}


def resolve_config_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.extend(
            [
                PROJECT_ROOT / path,
                PROJECT_ROOT / "config" / path,
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Configuration file not found: {value}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train scMAF on paired multi-omics or generic multi-view data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", "--config", default="config/multiomics/10xmalt.yaml")
    parser.add_argument(
        "--dataset-type", "--dataset_type", dest="dataset_type", choices=("multi_omics", "multi_view")
    )
    parser.add_argument("--dataset")
    parser.add_argument("--data-path", "--data_path", dest="data_path")
    parser.add_argument("--view-keys", "--view_keys", dest="view_keys", nargs=2)
    parser.add_argument("--label-key", "--label_key", dest="label_key")
    parser.add_argument("--data-type", "--data_type", dest="data_type", nargs=2, metavar=("VIEW1", "VIEW2"))
    parser.add_argument(
        "--data-norm",
        "--data_norm",
        dest="data_norm",
        choices=("standard", "min-max", "l2-norm", "none"),
    )
    parser.add_argument("-e", "--epochs", type=int)
    parser.add_argument("-b", "--batch-size", "--batch_size", dest="batch_size", type=int)
    parser.add_argument(
        "-p", "--print-every", "--print-num", "--print_num", dest="print_every", type=int
    )
    parser.add_argument("-d", "--device", help="auto, cpu, cuda, cuda:N, or legacy GPU index N")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--runs", "--train-time", "--train_time", dest="runs", type=int)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir")
    parser.add_argument("--experiment-name")
    parser.add_argument("--num-chunks", "--num_chunks", "-nc", dest="num_chunks", type=int)
    parser.add_argument("--sigma1", type=float)
    parser.add_argument("--sigma2", type=float)
    parser.add_argument("--filter1", action=argparse.BooleanOptionalAction)
    parser.add_argument("--filter2", action=argparse.BooleanOptionalAction)
    parser.add_argument("--f1", type=int)
    parser.add_argument("--f2", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--grad-clip-norm", type=float)
    parser.add_argument("--resume", help="Resume the first run from a .pt checkpoint")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction)
    parser.add_argument("--save-model", action=argparse.BooleanOptionalAction)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction)
    parser.add_argument("--dry-run", action="store_true", help="Run one epoch without checkpoints")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def parse_config(argv: Sequence[str] | None = None) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("-c", "--config", default="config/multiomics/10xmalt.yaml")
    preliminary, _ = pre_parser.parse_known_args(argv)
    config_path = resolve_config_path(preliminary.config)
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"The YAML root must be a mapping: {config_path}")

    parser = _parser()
    defaults = {**DEFAULTS, **loaded, "config": str(config_path)}
    parser.set_defaults(**defaults)
    config = parser.parse_args(argv)
    config.config = str(config_path)

    required = ["dataset", "data_path", "encoder_dim", "decoder_dim"]
    if config.dataset_type == "multi_omics":
        required.append("data_type")
    missing = [name for name in required if getattr(config, name, None) is None]
    if missing:
        parser.error(f"Missing required configuration value(s): {', '.join(missing)}")
    for name in ("epochs", "batch_size", "print_every", "runs", "num_chunks"):
        if int(getattr(config, name)) <= 0:
            parser.error(f"{name} must be positive")

    data_path = Path(config.data_path).expanduser()
    if not data_path.is_absolute():
        working_candidate = Path.cwd() / data_path
        project_candidate = PROJECT_ROOT / data_path
        data_path = working_candidate if working_candidate.exists() else project_candidate
    config.data_path = str(data_path.resolve())
    output_dir = Path(config.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    config.output_dir = str(output_dir.resolve())
    if config.resume is not None:
        resume_path = Path(config.resume).expanduser()
        if not resume_path.is_absolute():
            resume_path = Path.cwd() / resume_path
        if not resume_path.is_file():
            parser.error(f"Resume checkpoint not found: {resume_path}")
        config.resume = str(resume_path.resolve())
    if config.dry_run:
        config.epochs = 1
        config.runs = 1
        config.save_model = False
        config.print_every = 1
    return config

