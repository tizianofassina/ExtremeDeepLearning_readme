## What is this?

Simulating extreme events in high-dimensional, heavy-tailed spatial fields — precipitation being
the leading example — is a central challenge in the geosciences. Deep generative models are a
promising alternative to classical spatial statistics, but the literature is fragmented: methods
are evaluated on different data and metrics, and rarely against a common statistical baseline.

This project builds a **reproducible benchmark centered on precipitation**, designed to compare
generative models specifically on their ability to reproduce *tail* behaviour, not just overall
fidelity. Its contribution is twofold — a **shared evaluation suite** and a **common protocol**
across model families:

- **Datasets.** Two complementary sources: a *real* one derived from Météo-France COMEPHORE radar
  reanalysis, and a *synthetic* one with controllable tail behaviour and dependence structure,
  providing an exact ground truth the real data cannot offer.

- **Models.** A unified pipeline implementing and comparing baselines across the main generative
  families — score-based diffusion, normalizing flows, GANs and VAEs — each adapted to heavy-tailed
  targets, together with a classical spatial-statistics baseline.

- **Metrics.** A dedicated set of extreme-value metrics — going beyond standard distributional
  fidelity to score tail heaviness, the geometry of extremes, and asymptotic (co-exceedance)
  dependence — each read against a reference-based noise floor. Assembling metrics that actually
  discriminate tail behaviour in high dimension is a core part of the contribution.
