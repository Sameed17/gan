# GAN Assignment Reattempt (DCGAN + WGAN-GP)

This reattempt is built for your assignment requirements:

- Baseline `DCGAN`
- Improved `WGAN-GP` (gradient penalty `lambda=10`, critic steps `5`)
- Image resolution `64x64`
- Latent noise dimension `z=100`
- Optimizer `Adam(lr=0.0002, betas=(0.5, 0.999))`

## Files

- `train_gans.py` - training pipeline for both models
- `gan_models.py` - Generator, Discriminator, Critic architecture

## Dataset

Place Anime Faces images inside `data/`.

Supported structures:

1. Flat images directly in `data/` (e.g., `data/1.png`, `data/2.png`)
2. ImageFolder style class subfolders (e.g., `data/anime/1.png`)

## Run

Install dependencies:

```bash
pip install torch torchvision pillow
```

Train both models (DCGAN first, then WGAN-GP):

```bash
python train_gans.py --model both --data-dir data --output-dir assignment_outputs
```

Train only one model:

```bash
python train_gans.py --model dcgan --data-dir data
python train_gans.py --model wgan-gp --data-dir data
```

## Output

Generated files are saved in `assignment_outputs/`:

- `dcgan/samples/*.png`
- `dcgan/dcgan_log.csv`
- `wgan_gp/samples/*.png`
- `wgan_gp/wgan_gp_log.csv`
- model checkpoints in each model's `checkpoints/` folder

## Visualization Module

This utility creates:

- generated samples from DCGAN
- generated samples from WGAN-GP
- side-by-side comparison between both models

It extracts **5-10 samples per model** from the latest saved sample grids and saves visualization images.

Run:

```bash
python visualize_results.py --output-dir assignment_outputs --num-samples 10
```

Saved files:

- `assignment_outputs/visualizations/dcgan_samples.png`
- `assignment_outputs/visualizations/wgan_gp_samples.png`
- `assignment_outputs/visualizations/dcgan_vs_wgan_gp.png`

## Suggested comparison for your report

- **Visual quality:** compare image grids from same epoch ranges
- **Diversity:** check repeated faces/poses/backgrounds across generated grids
- **Stability:** compare `dcgan_log.csv` vs `wgan_gp_log.csv` trends
  - DCGAN: generator/discriminator BCE losses
  - WGAN-GP: critic/generator losses and Wasserstein distance
