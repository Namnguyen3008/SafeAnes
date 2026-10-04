"""UC04 HypoFormer artifact loader and tensor inference contract.

Inputs must already be aligned and preprocessed as the UC04 notebook expects.
This module returns raw logits; it does not claim calibrated probabilities.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


NUM_KEYS = ("MAP", "SBP", "DBP", "HR", "SPO2", "RR", "ETCO2")
STATIC_KEYS = ("age", "bmi", "asa", "emop", "sex_male")
NUM_CENTER = (75, 120, 65, 75, 98, 14, 35)
NUM_SCALE = (20, 30, 20, 20, 3, 6, 10)
STATIC_CENTER = (60, 24, 2, 0, 0)
STATIC_SCALE = (15, 5, 1, 1, 1)


class ConvTokenEncoder(nn.Module):
    def __init__(self, d_model: int = 128, n_tokens: int = 24) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, d_model, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(n_tokens),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.unsqueeze(1)).transpose(1, 2)


def _check_inputs(
    model: nn.Module,
    values: Iterable[torch.Tensor],
    names_and_tails: tuple[tuple[str, tuple[int, ...]], ...],
) -> list[torch.Tensor]:
    parameter = next(model.parameters())
    device = parameter.device
    dtype = parameter.dtype
    checked: list[torch.Tensor] = []
    batch_size: int | None = None

    for value, (name, expected_tail) in zip(values, names_and_tails, strict=True):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != len(expected_tail) + 1 or tuple(value.shape[1:]) != expected_tail:
            expected = ",".join(str(size) for size in expected_tail)
            raise ValueError(f"{name} expected shape [B,{expected}], got {tuple(value.shape)}")
        if batch_size is None:
            batch_size = value.shape[0]
        elif value.shape[0] != batch_size:
            raise ValueError(f"{name} batch size {value.shape[0]} does not match {batch_size}")
        if value.device != device:
            raise ValueError(f"{name} is on {value.device}; model is on {device}")
        checked.append(value.to(dtype=dtype))

    return checked


class HypoFormer(nn.Module):
    """UC04 three-horizon model: 5, 10, and 15 minute raw logits."""

    _INPUTS = (
        ("art", (3000,)),
        ("ecg", (7500,)),
        ("ppg", (3000,)),
        ("capno", (1875,)),
        ("numeric", (300, 7)),
        ("numeric_mask", (300, 7)),
        ("static", (5,)),
        ("modality_mask", (4,)),
    )

    def __init__(self, cfg: SimpleNamespace) -> None:
        super().__init__()
        d = cfg.D_MODEL
        nt = cfg.WAVE_TOKENS
        self.wave_tokens = nt
        self.num_tokens = cfg.NUM_TOKENS

        self.art_enc = ConvTokenEncoder(d, nt)
        self.ecg_enc = ConvTokenEncoder(d, nt)
        self.ppg_enc = ConvTokenEncoder(d, nt)
        self.capno_enc = ConvTokenEncoder(d, nt)

        self.num_proj = nn.Linear(len(NUM_KEYS) * 2, d)
        self.num_pool = nn.AdaptiveAvgPool1d(cfg.NUM_TOKENS)
        self.static_mlp = nn.Sequential(
            nn.Linear(len(STATIC_KEYS) * 2, 64),
            nn.GELU(),
            nn.Linear(64, d),
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.type_emb = nn.Embedding(7, d)
        total_tokens = 1 + 4 * nt + cfg.NUM_TOKENS + 1
        self.pos_emb = nn.Parameter(torch.zeros(1, total_tokens, d))

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.NHEAD,
            dim_feedforward=cfg.FF_DIM,
            dropout=cfg.DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.N_LAYERS)
        self.norm = nn.LayerNorm(d)
        self.heads = nn.ModuleList([nn.Linear(d, 1) for _ in range(3)])

        self.register_buffer("num_center", torch.tensor(NUM_CENTER, dtype=torch.float32))
        self.register_buffer("num_scale", torch.tensor(NUM_SCALE, dtype=torch.float32))
        self.register_buffer("static_center", torch.tensor(STATIC_CENTER, dtype=torch.float32))
        self.register_buffer("static_scale", torch.tensor(STATIC_SCALE, dtype=torch.float32))

        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def _numeric_tokens(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = (x - self.num_center.view(1, 1, -1)) / self.num_scale.view(1, 1, -1)
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).clamp(-6, 6)
        tokens = self.num_proj(torch.cat([z, mask], dim=-1))
        return self.num_pool(tokens.transpose(1, 2)).transpose(1, 2)

    def _static_token(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.isfinite(x).to(dtype=x.dtype)
        z = (x - self.static_center.view(1, -1)) / self.static_scale.view(1, -1)
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).clamp(-6, 6)
        return self.static_mlp(torch.cat([z, mask], dim=-1)).unsqueeze(1)

    def forward(
        self,
        art: torch.Tensor,
        ecg: torch.Tensor,
        ppg: torch.Tensor,
        capno: torch.Tensor,
        numeric: torch.Tensor,
        numeric_mask: torch.Tensor,
        static: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> torch.Tensor:
        values = _check_inputs(
            self,
            (art, ecg, ppg, capno, numeric, numeric_mask, static, modality_mask),
            self._INPUTS,
        )
        art, ecg, ppg, capno, numeric, numeric_mask, static, modality_mask = values
        batch = art.shape[0]

        tokens: list[torch.Tensor] = [self.cls.expand(batch, -1, -1)]
        masks: list[torch.Tensor] = [
            torch.zeros(batch, 1, dtype=torch.bool, device=art.device)
        ]
        type_ids = [0]

        for type_id, encoder, wave, present in (
            (1, self.art_enc, art, modality_mask[:, 0]),
            (2, self.ecg_enc, ecg, modality_mask[:, 1]),
            (3, self.ppg_enc, ppg, modality_mask[:, 2]),
            (4, self.capno_enc, capno, modality_mask[:, 3]),
        ):
            tokens.append(encoder(wave))
            masks.append((present < 0.5).unsqueeze(1).expand(-1, self.wave_tokens))
            type_ids += [type_id] * self.wave_tokens

        tokens.append(self._numeric_tokens(numeric, numeric_mask))
        masks.append(torch.zeros(batch, self.num_tokens, dtype=torch.bool, device=art.device))
        type_ids += [5] * self.num_tokens
        tokens.append(self._static_token(static))
        masks.append(torch.zeros(batch, 1, dtype=torch.bool, device=art.device))
        type_ids += [6]

        joined = torch.cat(tokens, dim=1)
        key_padding = torch.cat(masks, dim=1)
        types = torch.tensor(type_ids, dtype=torch.long, device=joined.device)
        joined = joined + self.type_emb(types).unsqueeze(0) + self.pos_emb[:, : joined.shape[1]]
        encoded = self.transformer(joined, src_key_padding_mask=key_padding)
        cls_output = self.norm(encoded[:, 0])
        return torch.cat([head(cls_output) for head in self.heads], dim=1)


def _read_artifact(artifact: Path) -> tuple[dict, object]:
    if artifact.is_dir():
        return (
            json.loads((artifact / "config.json").read_text(encoding="utf-8")),
            artifact / "model.pt",
        )
    if zipfile.is_zipfile(artifact):
        with zipfile.ZipFile(artifact) as archive:
            config = json.loads(archive.read("config.json").decode("utf-8"))
            state = io.BytesIO(archive.read("model.pt"))
        return config, state
    raise ValueError(f"Expected a UC04 package directory or ZIP archive: {artifact}")


def load_uc04_model(
    artifact_path: str | Path,
    device: str | torch.device = "cpu",
) -> HypoFormer:
    """Load the UC04 ZIP/package state dict safely and return an eval model.

    The package's smoke-test provenance is preserved in its config. Loading
    weights does not establish full-cohort or clinical validity.
    """
    artifact = Path(artifact_path)
    config_data, state_source = _read_artifact(artifact)
    required = ("D_MODEL", "WAVE_TOKENS", "NUM_TOKENS", "NHEAD", "N_LAYERS", "FF_DIM", "DROPOUT")
    missing = [key for key in required if key not in config_data]
    if missing:
        raise ValueError(f"UC04 model config is missing architecture fields: {missing}")

    config = SimpleNamespace(**config_data)
    model = HypoFormer(config).to(device)
    state = torch.load(state_source, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval()
