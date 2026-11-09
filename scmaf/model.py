"""scMAF model definitions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    activations: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "prelu": nn.PReLU,
        "gelu": nn.GELU,
        "leakyrelu": nn.LeakyReLU,
        "elu": nn.ELU,
    }
    key = name.lower().replace("_", "")
    if key not in activations:
        raise ValueError(f"Unsupported activation: {name!r}")
    return activations[key]()


class MLPEncoder(nn.Module):
    def __init__(
        self,
        dimensions: Sequence[int],
        activation: str = "ReLU",
        batch_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(dimensions) < 2:
            raise ValueError("An encoder needs at least an input and output dimension")
        layers: list[nn.Module] = []
        for index, (input_dim, output_dim) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(nn.Linear(input_dim, output_dim))
            if index < len(dimensions) - 2:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(output_dim))
                layers.append(_activation(activation))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class DenseDecoder(nn.Module):
    """Decoder used for generic multi-view features."""

    def __init__(
        self,
        dimensions: Sequence[int],
        activation: str = "ReLU",
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if len(dimensions) < 2:
            raise ValueError("A decoder needs at least an input and output dimension")
        layers: list[nn.Module] = []
        for index, (input_dim, output_dim) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(nn.Linear(input_dim, output_dim))
            if index < len(dimensions) - 2:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(output_dim))
                layers.append(_activation(activation))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class CountDecoder(nn.Module):
    """ZINB decoder for RNA/ADT or Poisson decoder for ATAC."""

    def __init__(
        self,
        dimensions: Sequence[int],
        output_dim: int,
        data_type: str,
        activation: str = "ReLU",
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.data_type = data_type.upper()
        if self.data_type == "ATAC":
            self.atac_mean = nn.Linear(dimensions[0], output_dim)
            self.backbone = None
            return
        if self.data_type not in {"RNA", "ADT"}:
            raise ValueError(f"Unsupported modality type: {data_type!r}")
        if len(dimensions) < 2:
            raise ValueError("RNA/ADT decoders require at least one hidden layer")

        layers: list[nn.Module] = []
        for index, (input_dim, hidden_dim) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(nn.Linear(input_dim, hidden_dim))
            if index < len(dimensions) - 2:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(_activation(activation))
        self.backbone = nn.Sequential(*layers)
        final_dim = dimensions[-1]
        self.mean = nn.Linear(final_dim, output_dim)
        self.dispersion = nn.Sequential(nn.Linear(final_dim, output_dim), nn.Softplus())
        self.dropout_probability = nn.Sequential(nn.Linear(final_dim, output_dim), nn.Sigmoid())

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if self.data_type == "ATAC":
            logits = self.atac_mean(inputs).clamp(min=-11.5, max=13.8)
            mean = torch.exp(logits)
            return mean, None, None
        hidden = self.backbone(inputs)
        mean_logits = self.mean(hidden).clamp(min=-11.5, max=13.8)
        mean = torch.exp(mean_logits)
        dispersion = torch.clamp(self.dispersion(hidden), 1e-4, 1e4)
        dropout_probability = self.dropout_probability(hidden)
        return mean, dispersion, dropout_probability


class FusionBackbone(nn.Module):
    """Two encoders followed by token-level Transformer fusion."""

    def __init__(self, config: Any, input_dims: tuple[int, int]) -> None:
        super().__init__()
        encoder_dims = deepcopy(config.encoder_dim)
        if len(encoder_dims) != 2:
            raise ValueError("encoder_dim must define exactly two encoders")
        encoder1_dims = [input_dims[0], *encoder_dims[0]]
        encoder2_dims = [input_dims[1], *encoder_dims[1]]
        if encoder1_dims[-1] != encoder2_dims[-1]:
            raise ValueError("The two encoders must have the same latent dimension")

        self.latent_dim = int(encoder1_dims[-1])
        self.num_chunks = int(config.num_chunks)
        if self.num_chunks <= 0 or self.latent_dim % self.num_chunks != 0:
            raise ValueError("The latent dimension must be divisible by num_chunks")
        self.chunk_dim = self.latent_dim // self.num_chunks
        activation = getattr(config, "activation", "ReLU")
        batch_norm = bool(getattr(config, "batch_norm", True))
        dropout = float(getattr(config, "dropout", 0.0))
        self.noise_std = (float(config.sigma1), float(config.sigma2))

        self.encoder1 = MLPEncoder(encoder1_dims, activation, batch_norm, dropout)
        self.encoder2 = MLPEncoder(encoder2_dims, activation, batch_norm, dropout)
        self.cls_projector = nn.Sequential(
            nn.Linear(sum(input_dims), 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.chunk_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.chunk_dim,
            nhead=int(getattr(config, "n_heads", 1)),
            dim_feedforward=int(getattr(config, "transformer_ff_dim", 2048)),
            dropout=float(getattr(config, "attention_dropout", 0.1)),
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=int(getattr(config, "transformer_layers", 1)),
            enable_nested_tensor=False,
        )

    def _encode(
        self, x1: torch.Tensor, x2: torch.Tensor, add_noise: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if add_noise:
            x1 = x1 + torch.randn_like(x1) * self.noise_std[0]
            x2 = x2 + torch.randn_like(x2) * self.noise_std[1]
        batch_size = x1.shape[0]
        z1 = self.encoder1(x1)
        z2 = self.encoder2(x2)
        tokens1 = z1.reshape(batch_size, self.num_chunks, self.chunk_dim)
        tokens2 = z2.reshape(batch_size, self.num_chunks, self.chunk_dim)
        cls_token = self.cls_projector(torch.cat((x1, x2), dim=1)).unsqueeze(1)
        sequence = torch.cat((cls_token, tokens1, tokens2), dim=1)
        fused = self.transformer(sequence)[:, 0, :]
        return fused, z1, z2

    def embed(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        fused, _, _ = self._encode(x1, x2, add_noise=False)
        return fused


class MultiOmicsSCMAF(FusionBackbone):
    def __init__(
        self,
        config: Any,
        input_dims: tuple[int, int],
        modality_types: tuple[str, str],
    ) -> None:
        super().__init__(config, input_dims)
        decoder_dims = deepcopy(config.decoder_dim)
        if len(decoder_dims) != 2:
            raise ValueError("decoder_dim must define exactly two decoders")
        activation = getattr(config, "activation", "ReLU")
        batch_norm = bool(getattr(config, "batch_norm", True))
        direct1 = list(decoder_dims[0])
        direct2 = list(decoder_dims[1])
        if direct1[0] != self.latent_dim or direct2[0] != self.latent_dim:
            raise ValueError("Each decoder must start with the encoder latent dimension")
        fused1, fused2 = list(direct1), list(direct2)
        fused1[0] = self.chunk_dim
        fused2[0] = self.chunk_dim
        self.modality_types = tuple(value.upper() for value in modality_types)
        self.direct_decoder1 = CountDecoder(
            direct1, input_dims[0], self.modality_types[0], activation, batch_norm
        )
        self.direct_decoder2 = CountDecoder(
            direct2, input_dims[1], self.modality_types[1], activation, batch_norm
        )
        self.fused_decoder1 = CountDecoder(
            fused1, input_dims[0], self.modality_types[0], activation, batch_norm
        )
        self.fused_decoder2 = CountDecoder(
            fused2, input_dims[1], self.modality_types[1], activation, batch_norm
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> dict[str, object]:
        fused, z1, z2 = self._encode(x1, x2, add_noise=True)
        return {
            "embedding": fused,
            "direct1": self.direct_decoder1(z1),
            "direct2": self.direct_decoder2(z2),
            "fused1": self.fused_decoder1(fused),
            "fused2": self.fused_decoder2(fused),
        }


class MultiViewSCMAF(FusionBackbone):
    def __init__(self, config: Any, input_dims: tuple[int, int]) -> None:
        super().__init__(config, input_dims)
        decoder_dims = deepcopy(config.decoder_dim)
        if len(decoder_dims) != 2:
            raise ValueError("decoder_dim must define exactly two decoders")
        direct1, direct2 = list(decoder_dims[0]), list(decoder_dims[1])
        if direct1[0] != self.latent_dim or direct2[0] != self.latent_dim:
            raise ValueError("Each decoder must start with the encoder latent dimension")
        fused1, fused2 = list(direct1), list(direct2)
        fused1[0] = self.chunk_dim
        fused2[0] = self.chunk_dim
        activation = getattr(config, "activation", "ReLU")
        batch_norm = bool(getattr(config, "batch_norm", True))
        self.direct_decoder1 = DenseDecoder([*direct1, input_dims[0]], activation, batch_norm)
        self.direct_decoder2 = DenseDecoder([*direct2, input_dims[1]], activation, batch_norm)
        self.fused_decoder1 = DenseDecoder([*fused1, input_dims[0]], activation, batch_norm)
        self.fused_decoder2 = DenseDecoder([*fused2, input_dims[1]], activation, batch_norm)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> dict[str, torch.Tensor]:
        fused, z1, z2 = self._encode(x1, x2, add_noise=True)
        return {
            "embedding": fused,
            "direct1": self.direct_decoder1(z1),
            "direct2": self.direct_decoder2(z2),
            "fused1": self.fused_decoder1(fused),
            "fused2": self.fused_decoder2(fused),
        }


def build_model(config: Any, input_dims: tuple[int, int], modality_types=None) -> FusionBackbone:
    dataset_type = str(config.dataset_type).lower().replace("-", "_")
    if dataset_type == "multi_omics":
        if modality_types is None:
            raise ValueError("modality_types are required for a multi-omics model")
        return MultiOmicsSCMAF(config, input_dims, modality_types)
    if dataset_type == "multi_view":
        return MultiViewSCMAF(config, input_dims)
    raise ValueError(f"Unsupported dataset_type: {config.dataset_type!r}")
