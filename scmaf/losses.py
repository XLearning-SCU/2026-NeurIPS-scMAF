"""Count-distribution reconstruction losses used by scMAF."""

from __future__ import annotations

import torch


def poisson_loss(mean: torch.Tensor, scale_factor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Element-wise Poisson negative log likelihood."""
    eps = torch.finfo(mean.dtype).eps
    scaled_mean = (mean * scale_factor).clamp(min=eps, max=1e8)
    return scaled_mean - target * torch.log(scaled_mean) + torch.lgamma(target + 1.0)


def zinb_loss(
    target: torch.Tensor,
    mean: torch.Tensor,
    dispersion: torch.Tensor,
    dropout_probability: torch.Tensor,
    scale_factor: torch.Tensor,
) -> torch.Tensor:
    """Element-wise zero-inflated negative-binomial negative log likelihood."""
    # eps=1e-10 rounds away next to 1.0 in float32. That made sigmoid outputs
    # equal to one and log(1-pi) become -inf after long training. Work in log
    # space and use a representable dtype-specific margin instead.
    eps = max(float(torch.finfo(mean.dtype).eps), 1e-7)
    scaled_mean = (mean * scale_factor).clamp(min=eps, max=1e8)
    dispersion = dispersion.clamp(min=eps, max=1e6)
    dropout_probability = dropout_probability.clamp(min=eps, max=1.0 - eps)

    log_mean = torch.log(scaled_mean)
    log_dispersion = torch.log(dispersion)
    log_sum = torch.log(scaled_mean + dispersion)
    negative_binomial = (
        torch.lgamma(dispersion)
        + torch.lgamma(target + 1.0)
        - torch.lgamma(target + dispersion)
        + dispersion * (log_sum - log_dispersion)
        + target * (log_sum - log_mean)
    )

    log_one_minus_pi = torch.log1p(-dropout_probability)
    nonzero_case = negative_binomial - log_one_minus_pi
    log_zero_nb = dispersion * (log_dispersion - log_sum)
    zero_case = -torch.logaddexp(torch.log(dropout_probability), log_one_minus_pi + log_zero_nb)
    return torch.where(target <= 1e-8, zero_case, nonzero_case)


def reconstruction_loss(
    mean: torch.Tensor,
    dispersion: torch.Tensor | None,
    dropout_probability: torch.Tensor | None,
    scale_factor: torch.Tensor,
    target: torch.Tensor,
    data_type: str,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute the modality-specific reconstruction loss."""
    modality = data_type.upper()
    if modality in {"RNA", "ADT"}:
        if dispersion is None or dropout_probability is None:
            raise ValueError(f"{modality} reconstruction requires dispersion and dropout probability")
        loss = zinb_loss(target, mean, dispersion, dropout_probability, scale_factor)
    elif modality == "ATAC":
        loss = poisson_loss(mean, scale_factor, target)
    else:
        raise ValueError(f"Unsupported modality type: {data_type!r}")

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"Unsupported reduction: {reduction!r}")

