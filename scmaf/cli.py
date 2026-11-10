"""scMAF command-line entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .config import parse_config
from .data import load_data
from .model import build_model
from .trainer import evaluate, fit
from .utils import configure_logging, resolve_device, seed_everything, write_json


def _aggregate(results: list[dict[str, float]]) -> dict[str, object]:
    metric_names = ("nmi", "ari", "acc")
    aggregate: dict[str, object] = {"runs": results}
    for name in metric_names:
        values = np.asarray([result[name] for result in results], dtype=np.float64)
        aggregate[name] = {"mean": float(values.mean()), "std": float(values.std())}
    return aggregate


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_config(argv)
    device = resolve_device(config.device)

    if config.evaluate_checkpoint is not None:
        checkpoint_path = Path(config.evaluate_checkpoint)
        logger = configure_logging(checkpoint_path.parent, verbose=config.verbose, append=True)
        logger.info("Evaluating checkpoint: %s", checkpoint_path)
        logger.info("Configuration: %s", config.config)
        seed_everything(int(config.seed), bool(config.deterministic))
        bundle = load_data(config)
        model = build_model(config, bundle.input_dims, bundle.modality_types)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        evaluation_seed = int(checkpoint.get("run_seed", config.seed))
        scores, _, _, _ = evaluate(model, bundle, config, device, evaluation_seed)
        result = {
            "checkpoint": str(checkpoint_path),
            "epoch": int(checkpoint.get("epoch", -1)),
            "seed": evaluation_seed,
            **scores,
        }
        result_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_evaluation.json")
        write_json(result_path, result)
        logger.info(
            "Checkpoint epoch %d | NMI=%.4f ARI=%.4f ACC=%.4f",
            result["epoch"],
            scores["nmi"],
            scores["ari"],
            scores["acc"],
        )
        logger.info("Evaluation saved to %s", result_path)
        return 0

    experiment_name = config.experiment_name or (
        f"{config.dataset}_epochs{config.epochs}_bs{config.batch_size}_lr{config.lr}"
    )
    experiment_dir = Path(config.output_dir) / experiment_name
    logger = configure_logging(
        experiment_dir, verbose=config.verbose, append=config.resume is not None
    )
    logger.info("Configuration: %s", config.config)
    logger.info("Dataset type: %s; data: %s", config.dataset_type, config.data_path)
    logger.info("Output directory: %s", experiment_dir)
    write_json(experiment_dir / "config.json", vars(config))

    seed_everything(int(config.seed), bool(config.deterministic))
    bundle = load_data(config)
    results: list[dict[str, float]] = []
    for run_index in range(int(config.runs)):
        run_seed = int(config.seed) + run_index
        seed_everything(run_seed, bool(config.deterministic))
        model = build_model(config, bundle.input_dims, bundle.modality_types)
        run_dir = experiment_dir if int(config.runs) == 1 else experiment_dir / f"run_{run_index + 1:02d}"
        logger.info("Starting run %d/%d with seed %d", run_index + 1, config.runs, run_seed)
        resume_path = Path(config.resume) if config.resume is not None and run_index == 0 else None
        results.append(
            fit(model, bundle, config, device, run_dir, run_seed, resume_path=resume_path)
        )

    summary = _aggregate(results)
    write_json(experiment_dir / "summary.json", summary)
    logger.info(
        "Summary | NMI %.4f±%.4f | ARI %.4f±%.4f | ACC %.4f±%.4f",
        summary["nmi"]["mean"],
        summary["nmi"]["std"],
        summary["ari"]["mean"],
        summary["ari"]["std"],
        summary["acc"]["mean"],
        summary["acc"]["std"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
