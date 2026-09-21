<h1 align="center">Heavytail High-dimensional Precipitation Benchmark</h1>

<p align="center">
  <b>A heavy-tailed, precipitation-centered benchmark for generative models.</b><br>
  Shared datasets, strong baselines, and extreme-value metrics for high-dimensional spatial rainfall.
</p>

<p align="center">
  <img alt="Python"    src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch"   src="https://img.shields.io/badge/PyTorch-2.8-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Lightning" src="https://img.shields.io/badge/Lightning-2.x-792EE5?logo=lightning&logoColor=white">
  <img alt="Status"    src="https://img.shields.io/badge/paper-under%20review%20%40%20ICLR%202027-b31b1b">
</p>

---

## Table of contents

- [What is this?](#what-is-this)
- [Requirements](#requirements)
- [The repository, file by file](#the-repository-file-by-file)
  - [The reference datasets scripts](#the-reference-datasets-scripts--run-once-shared-by-all-models)
  - [Top-level orchestrators](#top-level-orchestrators--dispatch-on-type_model)
  - [`src/data/`](#srcdata--datamodules-and-the-synthetic-generator)
  - [`src/models/`](#srcmodels--one-network--one-lightning-module-per-baseline)
  - [`src/sampling/`](#srcsampling--one-data-space-resumable-sampler-per-baseline)
  - [`src/metrics/`](#srcmetrics--dispatched-by-evaluatepy)
  - [Configs](#configs)
  - [SLURM launchers](#slurm-launchers-jean-zay)
  - [Outputs & assets](#outputs--assets)
- [Models](#models)
- [Metrics](#metrics)
- [Datasets](#datasets)
- [Pipeline and reference in the paper](#pipeline-and-reference-in-the-paper)
- [The four-stage pipeline](#the-four-stage-pipeline)
- [Comephore Data](#Comephore-Data)
- [Citation](#citation)
- [Author](#author)
  

---

# What is this?

Simulating extreme events in high-dimensional, heavy-tailed spatial fields — precipitation being the
leading example — is a central challenge in the geosciences. Deep generative models are a promising
alternative to classical spatial statistics, but the literature is fragmented: methods are evaluated on
different data and metrics, and rarely against a common statistical baseline.

The paper and its repository closes this gap with a reproducible benchmark built around precipitation:

- **two complementary datasets** — a *real* one from Météo-France COMEPHORE radar reanalysis, and a
  *synthetic* one with a **known, controllable tail index** and a **known dependence structure**;
- **seven generative baselines** spanning diffusion, flows, GANs and VAEs, plus a classical
  spatial-statistics model;
- **extreme-value metrics** that score not just overall fidelity but tail behaviour, the geometry of
  extremes, and asymptotic (co-exceedance) dependence — each read against a reference-vs-reference *noise floor*.

> 📄 **Paper:** *A Heavy-tailed and Precipitation Benchmark for Generative Models* - Work in progress.

<p align="center">
  <img src="hero.png" alt="Sample rainfall fields from the COMEPHORE and Synthetic benchmark datasets" width="820">
  <br>
  <sub><i>Real hourly rainfall fields from the COMEPHORE Massif-Central crop (256&times;256, 1&nbsp;km) and the Synthetic Dataset. Shared colour scale in mm.</i></sub>
</p>

---

## Requirements

```bash
# Python 3.10+, CUDA GPU recommended (256×256 fields are heavy).
pip install -r requirements.txt
```

On SLURM, submit the matching batch script (config path is `$1`), e.g. `sbatch train_ddp.sh configs_real_data/edm.yaml`.

---

# The repository, file by file

The whole codebase is organized around **one idea**: three orchestrators (`train`, `generate`,
`evaluate`) each read a single YAML and dispatch on its `type_model` key. Adding a method means adding
a branch and a config — never touching the control flow. The sections below explain what every file does.

### The reference datasets scripts — run once, shared by all models

This scripts are necessary build and preprocess the Comephore and Synthetic dataset to benchmark the models.

The first two are necessary to create the synthetic model, the third one just orchestrate the separation of train/val/test data for the Real (Comephore) and the Synthetic dataset.

| File | What it does |
| --- | --- |
| `fit_synthetic_data.py`                             | Fits the position- and tail-aware trans-Gaussian model to COMEPHORE; saves `synthetic_model_and_data/rain_field.pt`. |
| `generate_synthetic_asymptotic_dependency_data.py` | Samples the synthetic 256×256 stack from the fitted model, **injecting a controlled asymptotic-dependence region**. Resumable. |
| `split_train_val_data.py`                          | Builds the seeded train / val / test splits for both the real and the synthetic stacks. |


### Top-level orchestrators — dispatch on `type_model`

| File | Stage | What it does |
| --- | --- | --- |
| `train.py`    | 1 · train    | Reads one YAML, builds the selected model + datamodule, trains with PyTorch-Lightning (single-GPU or DDP). Writes `checkpoints/<dataset>/<model>/seed42/last.ckpt`. |
| `generate.py` | 2 · generate |  Reads one YAML, loads a trained checkpoint and the preprocessed data, streams `N` samples to a `.npy` memmap **in physical units (mm)**. Resumable. Requires `train.py` first. |
| `evaluate.py` | 3 · evaluate | Discovers every generated tensor under `generated/<dataset>/`, computes all metric families **plus the reference-vs-reference noise floor**, and writes long-form CSVs under `results/<dataset>/`. |



### `src/data/` — datamodules and the synthetic generator

| File | What it does |
| --- | --- |
| `data_module.py`               | PyTorch-Lightning DataModules (real + synthetic): loads the prepared tensors and serves batches to `train.py`. |
| `compact_processing.py`        | This dataset implements classes and functions necessary for the preprocessing of data for creating the synthetic dataset and implement the Comet Flow approach |
| `rain_field_synthetic_data.py` | The synthetic rain-field model itself — trans-Gaussian latent field, per-pixel tail-aware marginals, and the injection profile. This script creates the big class that describes the entire working of the synthetic model.  |

### `src/models/` — one network + one Lightning module per baseline

| File | Role |
| --- | --- |
| `edm_unet.py`            | EDM2 magnitude-preserving U-Net (shared by `edm` and `t_edm`). |
| `edm_tedm_diffusion.py`  | Lightning module for **EDM** and **t-EDM** (Gaussian vs Student-*t* noise + preconditioning). |
| `ddpm_unet.py`           | Improved-DDPM U-Net used by **Lévy** diffusion. |
| `levy_diffusion.py`      | Lightning module for **Lévy / α-stable** (DLPM) diffusion. |
| `pareto_gan_net.py`      | SNGAN-ResNet generator extended to 256×256 (piecewise-linear, unbounded output). |
| `pareto_gan.py`          | **Pareto GAN** module — generalized-Pareto latent + per-pixel power transform, energy-distance training. |
| `comet_coupling_net.py`  | Glow (coupling-based) normalizing-flow network for **COMET Flow**. |
| `comet_coupling.py`      | **COMET Flow** module — EVT marginal transform + coupling flow. |
| `vae_net.py`             | Encoder/decoder networks for the radius and angular VAEs. |
| `vae.py`                 | **Heavy-tailed (polar) VAE** module. |
<!-- 
| `tarflow_net.py`         | TarFlow (autoregressive Transformer flow) network. |
| `tarflow.py`             | TarFlow module — the flow component of `init_aware`. |
-->


### `src/sampling/` — one data-space, resumable sampler per baseline

| File | Role |
| --- | --- |
| `edm_tedm_sampling.py`      | Heun deterministic sampler for EDM / t-EDM. |
| `levy_diffusion_sampling.py`| DLIM deterministic sampler for Lévy diffusion. |
| `pareto_gan_sampling.py`    | Single-pass GAN sampling (Pareto latent → generator → power transform). |
| `comet_coupling_sampling.py`| COMET Flow sampling (flow inverse → Φ → inverse marginal). |
| `flow_comet_sampling.py`    | Shared flow→marginal inverse-transform utilities for the COMET pipelines. |
| `vae_sampling.py`           | Polar VAE sampling (Bernoulli dry gate → radius VAE → angular VAE → radius×angle). |
<!-- 
| `init_aware_sampling.py`    | Initialization-aware sampling: TarFlow draws the diffusion init at σ≈0.47, then the EDM denoiser finishes the descent. |
-->

### `src/metrics/` — dispatched by `evaluate.py`

| File | Metric family |
| --- | --- |
| `wasserstein_metrics.py`           | Sliced / max-sliced Wasserstein (+ signed-log) and sliced quantile errors (abs / rel). |
| `tail_metrics.py`                  | Expected-shortfall discrepancy and log–log tail-area (WMSLE). |
| `minkowski_metrics.py`             | Minkowski functionals of excursion sets (area, perimeter, Euler). |
| `asymptotic_dependency_metrics.py` | χ tail-dependence (direct and binned discrepancy). |
| `dry_metrics.py`                   | Dry fraction at fixed rainfall thresholds. |

### Configs

| Folder | Contents |
| --- | --- |
| `configs_archetypes/`     | Heavily-commented template configs — **start here to add a model**. |
| `configs_real_data/`      | One YAML per model for COMEPHORE, plus `evaluate_256.yaml`. |
| `configs_synthetic_data/` | One YAML per model for the synthetic data, plus `evaluate_256.yaml`.|

### SLURM launchers (Jean Zay)

| Script | Resources | Used for |
| --- | --- | --- |
| `train.sh`     | 1× A100        | single-GPU training |
| `train_2.sh`   | 1× H100 (12 h) | single-GPU training |
| `train_ddp.sh` | 4× A100, DDP   | multi-GPU training |
| `train_ddp_2.sh`| 4× H100, DDP  | multi-GPU training |
| `generate.sh`  | 1× H100        | sampling |
| `evaluate.sh`  | 1× H100        | metric computation |
| `fit_synthetic_data.sh`, `generate_synthetic_asymptotic_dependency_data.sh`, `split_train_val_data.sh` | — | data preparation |

### Outputs & assets

`gen_images/real_images/` and `gen_images/synthetic_images/` hold the qualitative galleries: one
subfolder per model, named by its key hyperparameter (`edm__seed42`, `t_edm__nu3.0`, `levy__alpha1.7`,
`pareto_gan__xi0.2`, `comet_coupling__seed42`, 
`vae__seed42`)
<!--`init_aware__sigma0.469979058`,--> , each with
30 PNG samples, a shared `colorbar.png`, and a `reference/` folder. `requirements.txt` pins the extras;
`generate.txt` is a run log.



---

## Models

Seven baselines, each a `type_model` key mapping to the files above. All are trained **unconditionally**
on the same 256×256 fields.

| `type_model`            | Family                              | Tail mechanism / key knob        | net · module · sampler                                                   |
| ----------------------- | ----------------------------------- | -------------------------------- | ------------------------------------------------------------------------ |
| `edm`                   | Score-based diffusion (EDM)         | Gaussian noise (baseline)        | `edm_unet` · `edm_tedm_diffusion` · `edm_tedm_sampling`                  |
| `t_edm`                 | Heavy-tailed diffusion              | Student-*t* noise, d.o.f. `ν`    | `edm_unet` · `edm_tedm_diffusion` · `edm_tedm_sampling`                  |
| `levy`                  | Lévy / α-stable diffusion           | α-stable noise, `α ∈ (1,2)`      | `ddpm_unet` · `levy_diffusion` · `levy_diffusion_sampling`               |
| `pareto_gan`            | GAN with an EVT tail                | generalized-Pareto tail, `ξ`     | `pareto_gan_net` · `pareto_gan` · `pareto_gan_sampling`                  |
| `comet_coupling`        | Normalizing flow (COMET + coupling) | EVT-bulk PIT marginal + coupling | `comet_coupling_net` · `comet_coupling` · `comet_coupling_sampling`      |
| `vae`                   | Heavy-tail VAE                      | polar (radius + angle) decoder   | `vae_net` · `vae` · `vae_sampling`                                       |
<!--
| `init_aware_comet_part` | Flow-warm-started diffusion         | flow init `σ` + EDM              | `tarflow_net` (+ `edm_unet`) · `tarflow` (+ EDM denoiser) · `init_aware_sampling` |
-->

The classical **trans-Gaussian** model (which builds the synthetic dataset) doubles as a non-deep baseline.

---

## Metrics

Each family targets a different failure mode of heavy-tailed spatial generation. All are computed by
`evaluate.py` into long-form CSVs, and every one is reported alongside a **reference-vs-reference noise
floor** (`--phase 2`): a model gap below the floor is indistinguishable from sampling error.

| Family                            | What it captures                                    |
| --------------------------------- | --------------------------------------------------- |
| Sliced Wasserstein (+ signed-log) | overall distributional fidelity                     |
| Sliced quantile error (abs / rel) | marginal accuracy far into the tail                 |
| Expected shortfall · log–log area | tail heaviness                                      |
| Minkowski functionals             | geometry of excursion sets (area, perimeter, Euler) |
| χ tail dependence (direct/binned) | asymptotic co-exceedance dependence                 |
| Dry fraction                      | zero-inflation at fixed rainfall thresholds         |

Evaluation runs at any resolution dividing 256 (area-pooling the source), with tail thresholds derived
once from the reference so every model answers the same question.

---

## Datasets

Both datasets are stacks of `[N, 1, 256, 256]` single-channel rainfall images in millimetres.

**Real — COMEPHORE.** Hourly precipitation reanalysis from Météo-France (radar network *ARAMIS* merged
with rain gauges), a 256×256 / 1 km crop over the Massif Central (2006–2026, `N = 147 324` wet-hour
frames). Distributed on [data.gouv.fr](https://www.data.gouv.fr/datasets/reanalyses-comephore) under the
*Etalab Open Licence 2.0*.

```bash
python split_train_val_data.py    # expects the raw stack in comephore_tensor/
```

**Synthetic.** A stationary trans-Gaussian rainfall model (after Benoit et al., 2026), made position-
and tail-aware, fitted to COMEPHORE, then sampled with an **artificially injected asymptotic-dependence
region**. Because the generator is known by construction, the tail index and the dependence structure
give an exact ground truth the real data cannot offer.

```bash
python fit_synthetic_data.py
python generate_synthetic_asymptotic_dependency_data.py
python split_train_val_data.py
```

---



## Pipeline and reference in the paper

Every config carries a fixed `seed` (42) for initialisation, shuffling and the split; generation is
seeded per batch (`SeedSequence`) so runs are resumable and bit-reproducible. Training used the IDRIS
**Jean Zay** cluster (A100 / H100, single-GPU and DDP).

| Paper                                 | Code |
| ------------------------------------- | ---- |
| App. A — Data                         | `split_train_val_data.py`, `fit_synthetic_data.py`, `generate_synthetic_asymptotic_dependency_data.py`, `src/data/` |
| App. B.1–B.7 — Models                 | `configs_*/<model>.yaml` + `src/models/` + `src/sampling/` |
| App. C.2–C.6 — Metrics                | `src/metrics/` (dispatched by `evaluate.py`) |
| App. C.5 — ground-truth χ             | `asymptotic_dependency_evaluation.py` |

---

# The four-stage pipeline

Now that the files are mapped, the flow is simple. Stages 1–3 are driven entirely by the per-model
YAML; stage 0 prepares the shared data once. **Each stage's output is the next stage's input.**

```
stage 0  data prep   split_train_val_data.py                            real + synthetic  ->  train / val / test
         (synth)      fit_synthetic_data.py                             fit trans-Gaussian model to COMEPHORE
                      generate_synthetic_asymptotic_dependency_data.py  sample with injected tail dependence

stage 1  train        train.py    --config <model>.yaml   ->  checkpoints/.../last.ckpt
stage 2  generate     generate.py --config <model>.yaml   ->  generated/<dataset>/<model>.npy  (mm, resumable)
stage 3  evaluate     evaluate.py <evaluate_*.yaml>       ->  results/<dataset>/*.csv          (+ noise floor)
```

Because every sampler writes in **physical units (mm)**, all models are directly comparable and
`evaluate.py` simply scores whatever it discovers under `generated/<dataset>/`.
---

## Comephore Data

Experiments use **COMEPHORE**, the hourly radar–rain-gauge precipitation
reanalysis over metropolitan France (Météo-France), with hourly reanalyses
starting from 1997. The raw data is publicly available as open data on
[meteo.data.gouv.fr](https://meteo.data.gouv.fr).

Turning the raw COMEPHORE archive into the tensors we used here requires a
non-trivial preprocessing pipeline (currently in R). This pipeline and the
processed dataset are **not yet released**.



