"""Streamlit app: compare DCGAN vs WGAN-GP, run paired inference, view training logs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from gan_models import Critic, Discriminator, Generator

import re
from typing import List
import numpy as np

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


st.set_page_config(
    page_title="GAN Lab — DCGAN vs WGAN-GP",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.25rem;
    }
    .hero h1 { color: #e8eef7; font-size: 1.65rem; margin: 0 0 0.35rem 0; }
    .hero p { color: #a8c0e8; margin: 0; font-size: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def _cached_generator(ckpt_path: str, device_str: str) -> Generator:
    return load_generator(Path(ckpt_path), torch.device(device_str))


@st.cache_resource
def _cached_discriminator(ckpt_path: str, device_str: str) -> Discriminator:
    return load_discriminator(Path(ckpt_path), torch.device(device_str))


@st.cache_resource
def _cached_critic(ckpt_path: str, device_str: str) -> Critic:
    return load_critic(Path(ckpt_path), torch.device(device_str))


st.markdown(
    """
    <div class="hero">
        <h1>DCGAN vs WGAN-GP</h1>
        <p>Same latent noise, side-by-side samples, and training metrics — compare stability and
        sample quality.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Settings")
    out_root = Path(
        st.text_input(
            "Output directory",
            value="outputs",
            help="Same root as `train_gans.py` (contains dcgan/ and wgan_gp/).",
        )
    )
    dcgan_dir = out_root / "dcgan" / "checkpoints"
    wgan_dir = out_root / "wgan_gp" / "checkpoints"

    dcgan_epochs = list_generator_epochs(dcgan_dir, "dcgan_generator_epoch_*.pt")
    wgan_epochs = list_generator_epochs(wgan_dir, "wgan_gp_generator_epoch_*.pt")
    shared = shared_epochs(dcgan_dir, wgan_dir)

    dev_choice = st.selectbox(
        "Device",
        options=["auto", "cpu", "cuda"],
        index=0,
        help="auto uses CUDA when available.",
    )
    if dev_choice == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = dev_choice
    if dev_choice == "cuda" and not torch.cuda.is_available():
        st.warning("CUDA not available; using CPU.")

    use_same_epoch = st.checkbox(
        "Use same epoch for both models",
        value=bool(shared),
        disabled=not shared,
        help="Fair comparison when both runs saved checkpoints at that epoch.",
    )
    if not shared:
        st.caption("No overlapping epochs — choose an epoch per model below.")

    if use_same_epoch and shared:
        epoch_both = st.selectbox("Training epoch", options=shared, index=len(shared) - 1)
        dc_epoch = epoch_w_epoch = epoch_both
    else:
        dc_epoch = (
            st.selectbox("DCGAN epoch", options=dcgan_epochs, index=len(dcgan_epochs) - 1)
            if dcgan_epochs
            else None
        )
        epoch_w_epoch = (
            st.selectbox("WGAN-GP epoch", options=wgan_epochs, index=len(wgan_epochs) - 1)
            if wgan_epochs
            else None
        )

    seed = st.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=5, step=1)
    n_samples = st.slider("Samples per model", min_value=4, max_value=32, value=8, step=4)

    run = st.button("Generate comparison", type="primary")

    st.divider()
    st.caption("Run locally: `streamlit run app.py`")

tab_gen, tab_logs, tab_grids = st.tabs(
    ["Generate & compare", "Training curves", "Saved sample grids"]
)

with tab_gen:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
**DCGAN** — Non-saturating loss with a **Discriminator** (sigmoid → “fake vs real” probability).
Fast training, but can suffer from mode collapse and fragile balance between D and G.
            """
        )
    with c2:
        st.markdown(
            """
**WGAN-GP** — **Wasserstein** distance with a **Critic** (no sigmoid) and **gradient penalty**
on interpolations. More stable gradients; critic scores are unbounded but comparable across batches.
            """
        )

    st.divider()

    dc_g_path = generator_path(dcgan_dir, "dcgan", dc_epoch) if dc_epoch is not None else None
    w_g_path = (
        generator_path(wgan_dir, "wgan_gp", epoch_w_epoch) if epoch_w_epoch is not None else None
    )

    if not dcgan_epochs:
        st.error(f"No DCGAN checkpoints under `{dcgan_dir}`. Train with `train_gans.py` first.")
    if not wgan_epochs:
        st.error(f"No WGAN-GP checkpoints under `{wgan_dir}`. Train with `train_gans.py` first.")

    if run and dc_epoch is not None and epoch_w_epoch is not None and dc_g_path and w_g_path:
        if not dc_g_path.is_file():
            st.error(f"Missing file: {dc_g_path}")
        elif not w_g_path.is_file():
            st.error(f"Missing file: {w_g_path}")
        else:
            g_dc = _cached_generator(str(dc_g_path), device_str)
            g_wg = _cached_generator(str(w_g_path), device_str)
            device = torch.device(device_str)
            z = make_noise(n_samples, int(seed), device)

            fake_dc = generate(g_dc, z)
            fake_wg = generate(g_wg, z)
            imgs_dc = tensors_to_rgb_uint8(fake_dc)
            imgs_wg = tensors_to_rgb_uint8(fake_wg)

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("DCGAN")
                st.caption(f"Epoch {dc_epoch} · `z_dim={Z_DIM}` · seed {int(seed)}")
                st.image(list(imgs_dc), width=150)
            with col_b:
                st.subheader("WGAN-GP")
                st.caption(f"Epoch {epoch_w_epoch} · same noise as DCGAN")
                st.image(list(imgs_wg), width=150)

            st.subheader("Scores on these generated samples")
            st.caption(
                "DCGAN: mean discriminator output (sigmoid → roughly in [0, 1]). "
                "WGAN-GP: mean critic output (unbounded; higher often correlates with more “real” features)."
            )
            m1, m2, m3 = st.columns(3)

            d_path = discriminator_path(dcgan_dir, "dcgan", dc_epoch)
            c_path = discriminator_path(wgan_dir, "wgan_gp", epoch_w_epoch)

            if d_path.is_file():
                d_net = _cached_discriminator(str(d_path), device_str)
                score_d = mean_discriminator_score(d_net, fake_dc)
                m1.metric("DCGAN · mean D(fake)", f"{score_d:.4f}")
            else:
                m1.warning("Missing D checkpoint for this epoch.")

            if c_path.is_file():
                c_net = _cached_critic(str(c_path), device_str)
                score_c = mean_critic_score(c_net, fake_wg)
                m2.metric("WGAN-GP · mean C(fake)", f"{score_c:.4f}")
            else:
                m2.warning("Missing critic checkpoint for this epoch.")

            m3.metric("Latent batch size", str(n_samples))

    elif run:
        st.warning("Select valid epochs for both models.")

with tab_logs:
    p_dc = out_root / "dcgan" / "dcgan_log.csv"
    p_wg = out_root / "wgan_gp" / "wgan_gp_log.csv"

    lc1, lc2 = st.columns(2)
    with lc1:
        st.markdown("**DCGAN** — `loss_d`, `loss_g`")
        if p_dc.is_file():
            df = pd.read_csv(p_dc)
            if not df.empty and "epoch" in df.columns:
                st.line_chart(df.set_index("epoch")[["loss_d", "loss_g"]])
            else:
                st.info("Log file is empty or missing columns.")
        else:
            st.info(f"No file: `{p_dc}`")

    with lc2:
        st.markdown("**WGAN-GP** — critic loss, generator loss, Wasserstein estimate")
        if p_wg.is_file():
            dfw = pd.read_csv(p_wg)
            if not dfw.empty and "epoch" in dfw.columns:
                cols = [c for c in ("loss_c", "loss_g", "wasserstein") if c in dfw.columns]
                if cols:
                    st.line_chart(dfw.set_index("epoch")[cols])
            else:
                st.info("Log file is empty or missing columns.")
        else:
            st.info(f"No file: `{p_wg}`")

with tab_grids:
    samples_dc = out_root / "dcgan" / "samples"
    samples_wg = out_root / "wgan_gp" / "samples"

    gc1, gc2 = st.columns(2)
    with gc1:
        st.markdown("**DCGAN** — latest saved grid")
        if samples_dc.is_dir():
            pngs = sorted(samples_dc.glob("dcgan_epoch_*.png"))
            if pngs:
                st.image(str(pngs[-1]), use_container_width=True)
                st.caption(pngs[-1].name)
            else:
                st.info("No `dcgan_epoch_*.png` in samples folder.")
        else:
            st.info(f"No folder `{samples_dc}`")

    with gc2:
        st.markdown("**WGAN-GP** — latest saved grid")
        if samples_wg.is_dir():
            pngs = sorted(samples_wg.glob("wgan_gp_epoch_*.png"))
            if pngs:
                st.image(str(pngs[-1]), use_container_width=True)
                st.caption(pngs[-1].name)
            else:
                st.info("No `wgan_gp_epoch_*.png` in samples folder.")
        else:
            st.info(f"No folder `{samples_wg}`")
