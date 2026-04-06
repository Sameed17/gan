"""Load trained GANs and run paired inference for the Streamlit demo."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import numpy as np
import torch

from gan_models import Critic, Discriminator, Generator

Z_DIM = 256
FEATURES = 64
CHANNELS = 3


def _epoch_from_name(path: Path) -> int:
    m = re.search(r"_epoch_(\d+)\.pt$", path.name)
    return int(m.group(1)) if m else -1


def list_generator_epochs(ckpt_dir: Path, pattern: str) -> List[int]:
    if not ckpt_dir.is_dir():
        return []
    epochs = [_epoch_from_name(p) for p in ckpt_dir.glob(pattern)]
    return sorted({e for e in epochs if e >= 0})


def generator_path(ckpt_dir: Path, prefix: str, epoch: int) -> Path:
    return ckpt_dir / f"{prefix}_generator_epoch_{epoch:03d}.pt"


def discriminator_path(ckpt_dir: Path, prefix: str, epoch: int) -> Path:
    if prefix == "dcgan":
        return ckpt_dir / f"dcgan_discriminator_epoch_{epoch:03d}.pt"
    return ckpt_dir / f"wgan_gp_critic_epoch_{epoch:03d}.pt"


def load_generator(ckpt_path: Path, device: torch.device) -> Generator:
    g = Generator(z_dim=Z_DIM, channels=CHANNELS, features=FEATURES).to(device)
    state = torch.load(ckpt_path, map_location=device)
    g.load_state_dict(state)
    g.eval()
    return g


def load_discriminator(ckpt_path: Path, device: torch.device) -> Discriminator:
    d = Discriminator(channels=CHANNELS, features=FEATURES).to(device)
    state = torch.load(ckpt_path, map_location=device)
    d.load_state_dict(state)
    d.eval()
    return d


def load_critic(ckpt_path: Path, device: torch.device) -> Critic:
    c = Critic(channels=CHANNELS, features=FEATURES).to(device)
    state = torch.load(ckpt_path, map_location=device)
    c.load_state_dict(state)
    c.eval()
    return c


def make_noise(batch: int, seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.randn(batch, Z_DIM, 1, 1, generator=gen).to(device)


def generate(g: Generator, z: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return g(z)


def tensors_to_rgb_uint8(fake: torch.Tensor) -> np.ndarray:
    """(N,3,H,W) in [-1,1] -> numpy (N,H,W,3) uint8."""
    x = fake.detach().cpu().float().numpy()
    x = np.transpose(x, (0, 2, 3, 1))
    x = (x + 1.0) * 0.5
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


@torch.no_grad()
def mean_discriminator_score(d: Discriminator, fake: torch.Tensor) -> float:
    s = d(fake)
    return float(s.mean().cpu())


@torch.no_grad()
def mean_critic_score(c: Critic, fake: torch.Tensor) -> float:
    s = c(fake)
    return float(s.mean().cpu())


def shared_epochs(dcgan_ckpt: Path, wgan_ckpt: Path) -> List[int]:
    a = set(list_generator_epochs(dcgan_ckpt, "dcgan_generator_epoch_*.pt"))
    b = set(list_generator_epochs(wgan_ckpt, "wgan_gp_generator_epoch_*.pt"))
    return sorted(a & b)
