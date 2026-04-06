from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def get_latest_sample(samples_dir: Path, model_prefix: str) -> Path:
    images = sorted(samples_dir.glob(f"{model_prefix}_epoch_*.png"))
    if not images:
        raise FileNotFoundError(f"No sample grids found for '{model_prefix}' in: {samples_dir}")
    return images[-1]


def extract_tiles_from_grid(grid_path: Path, n_samples: int) -> List[np.ndarray]:
    img = Image.open(grid_path).convert("RGB")
    arr = np.array(img)
    height, width, _ = arr.shape

    # Grids are saved with nrow=8 in training.
    cols = 8
    tile_h = height // cols
    tile_w = width // cols

    tiles: List[np.ndarray] = []
    for r in range(cols):
        for c in range(cols):
            y1, y2 = r * tile_h, (r + 1) * tile_h
            x1, x2 = c * tile_w, (c + 1) * tile_w
            tiles.append(arr[y1:y2, x1:x2])
            if len(tiles) == n_samples:
                return tiles
    return tiles


def plot_samples(samples: List[np.ndarray], title: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(samples), figsize=(2 * len(samples), 2.5))
    if len(samples) == 1:
        axes = [axes]
    for idx, ax in enumerate(axes):
        ax.imshow(samples[idx])
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(
    dcgan_samples: List[np.ndarray], wgan_samples: List[np.ndarray], out_path: Path
) -> None:
    n = min(len(dcgan_samples), len(wgan_samples))
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 5))
    for i in range(n):
        axes[0, i].imshow(dcgan_samples[i])
        axes[0, i].axis("off")
        axes[1, i].imshow(wgan_samples[i])
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("DCGAN", fontsize=12)
    axes[1, 0].set_ylabel("WGAN-GP", fontsize=12)
    fig.suptitle("DCGAN vs WGAN-GP (Generated Samples)", fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def visualize_results(
    output_dir: Path, n_samples: int, show: bool
) -> Tuple[Path, Path, Path]:
    dcgan_grid = get_latest_sample(output_dir / "dcgan" / "samples", "dcgan")
    wgan_grid = get_latest_sample(output_dir / "wgan_gp" / "samples", "wgan_gp")

    dcgan_samples = extract_tiles_from_grid(dcgan_grid, n_samples)
    wgan_samples = extract_tiles_from_grid(wgan_grid, n_samples)

    vis_dir = output_dir / "visualizations"
    dcgan_out = vis_dir / "dcgan_samples.png"
    wgan_out = vis_dir / "wgan_gp_samples.png"
    comp_out = vis_dir / "dcgan_vs_wgan_gp.png"

    plot_samples(dcgan_samples, f"DCGAN Generated Samples ({len(dcgan_samples)})", dcgan_out)
    plot_samples(wgan_samples, f"WGAN-GP Generated Samples ({len(wgan_samples)})", wgan_out)
    plot_comparison(dcgan_samples, wgan_samples, comp_out)

    if show:
        for path in (dcgan_out, wgan_out, comp_out):
            img = Image.open(path)
            plt.figure(figsize=(12, 4 if "vs" not in path.name else 6))
            plt.imshow(np.array(img))
            plt.axis("off")
            plt.title(path.name)
            plt.show()

    return dcgan_out, wgan_out, comp_out


if __name__ == "__main__":
    output_dir = Path("outputs")
    num_samples = 10
    show = False

    if num_samples < 5 or num_samples > 10:
        raise ValueError("num_samples should be between 5 and 10 for assignment requirement.")

    outputs = visualize_results(output_dir, num_samples, show)
    print("Saved visualizations:")
    for out in outputs:
        print(f"- {out}")
