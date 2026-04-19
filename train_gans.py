import csv
import re
from pathlib import Path
from typing import List, Tuple

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, utils
from tqdm import tqdm

from gan_models import Critic, Discriminator, Generator

data_dir = "data"
output_dir = "outputs"
model = "both"  # one of: "dcgan", "wgan-gp", "both"
image_size = 64
batch_size = 128
num_workers = 2
z_dim = 256
lr = 2e-4
epochs_dcgan = 30
epochs_wgangp = 30
lambda_gp = 10.0
critic_steps = 5
sample_every = 5
checkpoint_every = 10
resume = True


class FlatImageDataset(Dataset):
    def __init__(self, root: Path, transform: transforms.Compose):
        self.root = root
        self.transform = transform
        self.files: List[Path] = []
        self.files.extend(root.glob("*.png"))
        if not self.files:
            raise ValueError(f"No images found in: {root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        img = Image.open(self.files[index]).convert("RGB")
        return self.transform(img)


def resolve_dataset(data_root: Path, image_size: int) -> Dataset:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    return FlatImageDataset(root=data_root, transform=transform)


def gradient_penalty(critic: nn.Module, real: torch.Tensor, fake: torch.Tensor, device: torch.device):
    batch_size, channels, height, width = real.shape
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = real * epsilon + fake * (1.0 - epsilon)
    interpolated.requires_grad_(True)

    mixed_scores = critic(interpolated)
    grad = autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(mixed_scores),
    )[0]
    grad = grad.view(grad.shape[0], -1)
    return ((grad.norm(2, dim=1) - 1.0) ** 2).mean()


def save_grid(generator: nn.Module, fixed_noise: torch.Tensor, epoch: int, tag: str, out_dir: Path):
    generator.eval()
    with torch.no_grad():
        fake = generator(fixed_noise)
    out_dir.mkdir(parents=True, exist_ok=True)
    utils.save_image(
        fake,
        out_dir / f"{tag}_epoch_{epoch:03d}.png",
        normalize=True,
        value_range=(-1, 1),
        nrow=8,
    )
    generator.train()


def _extract_epoch(path: Path) -> int:
    match = re.search(r"_epoch_(\d+)\.pt$", path.name)
    if not match:
        return -1
    return int(match.group(1))


def _find_latest_pair(ckpt_dir: Path, gen_pattern: str, other_pattern: str) -> Tuple[int, Path, Path]:
    gen_files = sorted(ckpt_dir.glob(gen_pattern), key=_extract_epoch)
    other_files = sorted(ckpt_dir.glob(other_pattern), key=_extract_epoch)
    if not gen_files or not other_files:
        return 0, Path(), Path()

    gen_by_epoch = {_extract_epoch(p): p for p in gen_files}
    other_by_epoch = {_extract_epoch(p): p for p in other_files}
    shared_epochs = sorted(set(gen_by_epoch.keys()).intersection(other_by_epoch.keys()))
    if not shared_epochs:
        return 0, Path(), Path()

    latest_epoch = shared_epochs[-1]
    return latest_epoch, gen_by_epoch[latest_epoch], other_by_epoch[latest_epoch]


def train_dcgan(loader: DataLoader, device: torch.device, out_root: Path) -> None:
    g = Generator(z_dim=z_dim).to(device)
    d = Discriminator().to(device)

    criterion = nn.BCELoss()
    opt_g = optim.Adam(g.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(d.parameters(), lr=lr, betas=(0.5, 0.999))
    fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)

    sample_dir = out_root / "samples"
    ckpt_dir = out_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "dcgan_log.csv"
    start_epoch = 1
    if resume:
        last_epoch, g_ckpt, d_ckpt = _find_latest_pair(
            ckpt_dir, "dcgan_generator_epoch_*.pt", "dcgan_discriminator_epoch_*.pt"
        )
        if last_epoch > 0:
            g.load_state_dict(torch.load(g_ckpt, map_location=device))
            d.load_state_dict(torch.load(d_ckpt, map_location=device))
            start_epoch = last_epoch + 1
            print(f"[DCGAN] Resumed from epoch {last_epoch}")
        else:
            print("[DCGAN] Resume requested but no matching checkpoints found. Starting from scratch.")

    if start_epoch == 1:
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss_d", "loss_g"])
    elif not log_path.exists():
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss_d", "loss_g"])

    print("\n[Start] DCGAN training")
    if start_epoch > epochs_dcgan:
        print(f"[DCGAN] Latest checkpoint is epoch {start_epoch - 1}, nothing to train.")
        return

    for epoch in range(start_epoch, epochs_dcgan + 1):
        d_running = 0.0
        g_running = 0.0
        progress = tqdm(loader, desc=f"DCGAN Epoch {epoch}/{epochs_dcgan}", leave=False)
        for real in progress:
            real = real.to(device)
            bsz = real.shape[0]

            opt_d.zero_grad(set_to_none=True)
            d_real = d(real)
            loss_real = criterion(d_real, torch.ones_like(d_real))
            z = torch.randn(bsz, z_dim, 1, 1, device=device)
            fake = g(z)
            d_fake = d(fake.detach())
            loss_fake = criterion(d_fake, torch.zeros_like(d_fake))
            loss_d = 0.5 * (loss_real + loss_fake)
            loss_d.backward()
            opt_d.step()

            opt_g.zero_grad(set_to_none=True)
            d_fake_for_g = d(fake)
            loss_g = criterion(d_fake_for_g, torch.ones_like(d_fake_for_g))
            loss_g.backward()
            opt_g.step()

            d_running += loss_d.item()
            g_running += loss_g.item()
            progress.set_postfix(loss_d=f"{loss_d.item():.4f}", loss_g=f"{loss_g.item():.4f}")

        d_epoch = d_running / len(loader)
        g_epoch = g_running / len(loader)
        print(f"[DCGAN] Epoch {epoch}/{epochs_dcgan} | D: {d_epoch:.4f} | G: {g_epoch:.4f}")
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, f"{d_epoch:.6f}", f"{g_epoch:.6f}"])

        if epoch % sample_every == 0 or epoch == 1 or epoch == epochs_dcgan:
            save_grid(g, fixed_noise, epoch, "dcgan", sample_dir)
        if epoch % checkpoint_every == 0 or epoch == epochs_dcgan:
            torch.save(g.state_dict(), ckpt_dir / f"dcgan_generator_epoch_{epoch:03d}.pt")
            torch.save(d.state_dict(), ckpt_dir / f"dcgan_discriminator_epoch_{epoch:03d}.pt")


def train_wgan_gp(loader: DataLoader, device: torch.device, out_root: Path) -> None:
    g = Generator(z_dim=z_dim).to(device)
    c = Critic().to(device)

    opt_g = optim.Adam(g.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_c = optim.Adam(c.parameters(), lr=lr, betas=(0.5, 0.999))
    fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)

    sample_dir = out_root / "samples"
    ckpt_dir = out_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "wgan_gp_log.csv"
    start_epoch = 1
    if resume:
        last_epoch, g_ckpt, c_ckpt = _find_latest_pair(
            ckpt_dir, "wgan_gp_generator_epoch_*.pt", "wgan_gp_critic_epoch_*.pt"
        )
        if last_epoch > 0:
            g.load_state_dict(torch.load(g_ckpt, map_location=device))
            c.load_state_dict(torch.load(c_ckpt, map_location=device))
            start_epoch = last_epoch + 1
            print(f"[WGAN-GP] Resumed from epoch {last_epoch}")
        else:
            print("[WGAN-GP] Resume requested but no matching checkpoints found. Starting from scratch.")

    if start_epoch == 1:
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss_c", "loss_g", "wasserstein"])
    elif not log_path.exists():
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss_c", "loss_g", "wasserstein"])

    print("\n[Start] WGAN-GP training")
    if start_epoch > epochs_wgangp:
        print(f"[WGAN-GP] Latest checkpoint is epoch {start_epoch - 1}, nothing to train.")
        return

    for epoch in range(start_epoch, epochs_wgangp + 1):
        c_running = 0.0
        g_running = 0.0
        w_running = 0.0
        progress = tqdm(loader, desc=f"WGAN-GP Epoch {epoch}/{epochs_wgangp}", leave=False)
        for real in progress:
            real = real.to(device)
            bsz = real.shape[0]

            for _ in range(critic_steps):
                z = torch.randn(bsz, z_dim, 1, 1, device=device)
                fake = g(z)
                c_real = c(real).mean()
                c_fake = c(fake.detach()).mean()
                gp = gradient_penalty(c, real, fake.detach(), device)
                wasserstein = c_real - c_fake
                loss_c = -wasserstein + lambda_gp * gp

                opt_c.zero_grad(set_to_none=True)
                loss_c.backward()
                opt_c.step()

            z = torch.randn(bsz, z_dim, 1, 1, device=device)
            fake = g(z)
            loss_g = -c(fake).mean()

            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            c_running += loss_c.item()
            g_running += loss_g.item()
            w_running += wasserstein.item()
            progress.set_postfix(loss_c=f"{loss_c.item():.4f}", loss_g=f"{loss_g.item():.4f}")

        c_epoch = c_running / len(loader)
        g_epoch = g_running / len(loader)
        w_epoch = w_running / len(loader)
        print(
            f"[WGAN-GP] Epoch {epoch}/{epochs_wgangp} | C: {c_epoch:.4f} | "
            f"G: {g_epoch:.4f} | W-dist: {w_epoch:.4f}"
        )
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, f"{c_epoch:.6f}", f"{g_epoch:.6f}", f"{w_epoch:.6f}"])

        if epoch % sample_every == 0 or epoch == 1 or epoch == epochs_wgangp:
            save_grid(g, fixed_noise, epoch, "wgan_gp", sample_dir)
        if epoch % checkpoint_every == 0 or epoch == epochs_wgangp:
            torch.save(g.state_dict(), ckpt_dir / f"wgan_gp_generator_epoch_{epoch:03d}.pt")
            torch.save(c.state_dict(), ckpt_dir / f"wgan_gp_critic_epoch_{epoch:03d}.pt")


def main() -> None:
    data_root = Path(data_dir)
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = resolve_dataset(data_root, image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Loaded {len(dataset)} images from: {data_root}")

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if model in ("dcgan", "both"):
        train_dcgan(loader, device, out_root / "dcgan")
    if model in ("wgan-gp", "both"):
        train_wgan_gp(loader, device, out_root / "wgan_gp")

    print("Done. Compare sample grids and log files from both models.")


if __name__ == "__main__":
    main()
