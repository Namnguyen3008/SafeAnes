"""UC05 RespFormer artifact loader and tensor inference contract.

Inputs must follow the package signal schema and notebook preprocessing. The
model returns raw hypoxemia logits plus an unvalidated weak-label ventilation
logit; this module does not calibrate or convert them into clinical alerts.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


NUM_FEATURES = (
    "SPO2", "ETCO2", "RR_CO2", "TV", "MV", "PIP", "PEEP", "PPLAT", "MAWP",
    "FIO2", "COMPLIANCE", "VENT_LEAK", "SET_FIO2", "SET_TV", "SET_PIP",
    "SET_RR", "HR", "MAC",
)
STATIC_FEATURES = ("age", "bmi", "asa", "sex_male", "weight", "height", "emop")
NUM_CENTER = (98, 35, 14, 450, 6, 18, 5, 15, 10, 45, 50, 0, 45, 450, 18, 12, 75, 1)
NUM_SCALE = (3, 10, 6, 200, 3, 8, 4, 8, 6, 20, 20, 200, 20, 200, 8, 6, 20, 0.5)
STATIC_CENTER = (60, 24, 2, 0, 65, 165, 0)
STATIC_SCALE = (15, 5, 1, 1, 15, 10, 1)


def _check_inputs(
    model: nn.Module,
    values: tuple[torch.Tensor, ...],
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


class ConvNeXt1DBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 4, kernel: int = 7) -> None:
        super().__init__()
        self.dw = nn.Conv1d(
            channels, channels, kernel, padding=kernel // 2, groups=channels
        )
        self.norm = nn.GroupNorm(1, channels)
        self.pw1 = nn.Conv1d(channels, channels * expansion, 1)
        self.pw2 = nn.Conv1d(channels * expansion, channels, 1)
        self.gamma = nn.Parameter(torch.ones(channels) * 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        y = self.norm(y)
        y = F.gelu(self.pw1(y))
        y = self.pw2(y)
        return x + y * self.gamma.view(1, -1, 1)


class WaveEncoder(nn.Module):
    def __init__(self, d: int = 256, tokens: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, 9, 4, 4),
            nn.GroupNorm(1, 64),
            nn.GELU(),
            nn.Conv1d(64, 128, 5, 2, 2),
            nn.GroupNorm(1, 128),
            nn.GELU(),
            ConvNeXt1DBlock(128),
            ConvNeXt1DBlock(128),
            ConvNeXt1DBlock(128),
            nn.Conv1d(128, d, 3, 2, 1),
            nn.GroupNorm(1, d),
            nn.GELU(),
            ConvNeXt1DBlock(d),
            ConvNeXt1DBlock(d),
            ConvNeXt1DBlock(d),
            nn.AdaptiveAvgPool1d(tokens),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.unsqueeze(1)).transpose(1, 2)


class RespFormerBackbone(nn.Module):
    def __init__(self, cfg: SimpleNamespace) -> None:
        super().__init__()
        d = cfg.D_MODEL
        self.cfg = cfg
        self.wave_tokens = cfg.WAVE_TOKENS
        self.encs = nn.ModuleList([WaveEncoder(d, cfg.WAVE_TOKENS) for _ in range(5)])
        self.num_proj = nn.Linear(len(NUM_FEATURES) * 2, d)
        self.num_pool = nn.AdaptiveAvgPool1d(cfg.NUM_TOKENS)
        self.static_mlp = nn.Sequential(
            nn.Linear(len(STATIC_FEATURES) * 2, 128),
            nn.GELU(),
            nn.Linear(128, d),
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.type_emb = nn.Embedding(8, d)
        total = 1 + 5 * cfg.WAVE_TOKENS + cfg.NUM_TOKENS + 1
        self.pos_emb = nn.Parameter(torch.zeros(1, total, d))

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.N_HEADS,
            dim_feedforward=cfg.FF_DIM,
            dropout=cfg.DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(layer, cfg.N_LAYERS)
        self.norm = nn.LayerNorm(d)
        self.register_buffer("num_center", torch.tensor(NUM_CENTER, dtype=torch.float32))
        self.register_buffer("num_scale", torch.tensor(NUM_SCALE, dtype=torch.float32))
        self.register_buffer("static_center", torch.tensor(STATIC_CENTER, dtype=torch.float32))
        self.register_buffer("static_scale", torch.tensor(STATIC_SCALE, dtype=torch.float32))

        types = [0]
        for index in range(5):
            types += [index + 1] * cfg.WAVE_TOKENS
        types += [6] * cfg.NUM_TOKENS
        types += [7]
        self.register_buffer("type_indices", torch.tensor(types, dtype=torch.long))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(
        self,
        capno: torch.Tensor,
        awp: torch.Tensor,
        ppg: torch.Tensor,
        flow: torch.Tensor,
        resp: torch.Tensor,
        numeric: torch.Tensor,
        numeric_mask: torch.Tensor,
        static: torch.Tensor,
        wave_mask: torch.Tensor,
    ) -> torch.Tensor:
        waves = [capno, awp, ppg, flow, resp]
        batch = capno.shape[0]
        tokens = [self.cls.expand(batch, -1, -1)]
        padding = [torch.zeros(batch, 1, dtype=torch.bool, device=capno.device)]

        for index, (encoder, wave) in enumerate(zip(self.encs, waves, strict=True)):
            encoded = encoder(wave)
            tokens.append(encoded)
            padding.append(
                (wave_mask[:, index] < 0.5).unsqueeze(1).expand(-1, self.wave_tokens)
            )

        numeric_z = (numeric - self.num_center.view(1, 1, -1)) / self.num_scale.view(1, 1, -1)
        numeric_z = torch.nan_to_num(numeric_z, nan=0, posinf=0, neginf=0).clamp(-8, 8)
        numeric_tokens = self.num_proj(torch.cat([numeric_z, numeric_mask], -1))
        numeric_tokens = self.num_pool(numeric_tokens.transpose(1, 2)).transpose(1, 2)
        tokens.append(numeric_tokens)
        padding.append(
            torch.zeros(batch, self.cfg.NUM_TOKENS, dtype=torch.bool, device=capno.device)
        )

        static_mask = torch.isfinite(static).to(dtype=static.dtype)
        static_z = (static - self.static_center.view(1, -1)) / self.static_scale.view(1, -1)
        static_z = torch.nan_to_num(static_z, nan=0, posinf=0, neginf=0).clamp(-8, 8)
        static_token = self.static_mlp(torch.cat([static_z, static_mask], -1)).unsqueeze(1)
        tokens.append(static_token)
        padding.append(torch.zeros(batch, 1, dtype=torch.bool, device=capno.device))

        joined = torch.cat(tokens, 1)
        key_padding = torch.cat(padding, 1)
        joined = joined + self.type_emb(self.type_indices).unsqueeze(0)
        joined = joined + self.pos_emb[:, : joined.shape[1]]
        encoded = self.tr(joined, src_key_padding_mask=key_padding)
        return self.norm(encoded[:, 0])


class RespFormer(nn.Module):
    """UC05 hypoxemia heads plus its unvalidated weak ventilation-pattern head."""

    _INPUTS = (
        ("capno", (3750,)),
        ("awp", (3750,)),
        ("ppg", (7500,)),
        ("flow", (3750,)),
        ("resp", (3750,)),
        ("numeric", (600, 18)),
        ("numeric_mask", (600, 18)),
        ("static", (7,)),
        ("wave_mask", (5,)),
    )

    def __init__(self, cfg: SimpleNamespace) -> None:
        super().__init__()
        self.backbone = RespFormerBackbone(cfg)
        self.hypox = nn.Linear(cfg.D_MODEL, 3)
        self.vent = nn.Linear(cfg.D_MODEL, 1)

    def forward(
        self,
        capno: torch.Tensor,
        awp: torch.Tensor,
        ppg: torch.Tensor,
        flow: torch.Tensor,
        resp: torch.Tensor,
        numeric: torch.Tensor,
        numeric_mask: torch.Tensor,
        static: torch.Tensor,
        wave_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = _check_inputs(
            self,
            (capno, awp, ppg, flow, resp, numeric, numeric_mask, static, wave_mask),
            self._INPUTS,
        )
        capno, awp, ppg, flow, resp, numeric, numeric_mask, static, wave_mask = values
        encoded = self.backbone(
            capno, awp, ppg, flow, resp, numeric, numeric_mask, static, wave_mask
        )
        return self.hypox(encoded), self.vent(encoded).squeeze(-1)


def _read_artifact(artifact: Path) -> tuple[dict, object]:
    if artifact.is_dir():
        return (
            json.loads((artifact / "config.json").read_text(encoding="utf-8")),
            artifact / "respformer.pt",
        )
    if zipfile.is_zipfile(artifact):
        with zipfile.ZipFile(artifact) as archive:
            config = json.loads(archive.read("config.json").decode("utf-8"))
            state = io.BytesIO(archive.read("respformer.pt"))
        return config, state
    raise ValueError(f"Expected a UC05 package directory or ZIP archive: {artifact}")


def load_uc05_model(
    artifact_path: str | Path,
    device: str | torch.device = "cpu",
) -> RespFormer:
    """Load the UC05 package state dict strictly and return an eval model."""
    artifact = Path(artifact_path)
    config_data, state_source = _read_artifact(artifact)
    required = ("D_MODEL", "WAVE_TOKENS", "NUM_TOKENS", "N_HEADS", "N_LAYERS", "FF_DIM", "DROPOUT")
    missing = [key for key in required if key not in config_data]
    if missing:
        raise ValueError(f"UC05 model config is missing architecture fields: {missing}")

    config = SimpleNamespace(**config_data)
    model = RespFormer(config).to(device)
    state = torch.load(state_source, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval()
