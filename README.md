# Continuous-Time Deep Learning for ICU Mortality Prediction

Neural ODEs and Neural CDEs applied to irregularly-sampled, partially-observed clinical time series, benchmarked against classical baselines on the PhysioNet/CinC Challenge 2012 dataset.

## Motivation

Standard sequence models (RNNs, Transformers) assume data arrives on a regular grid. Clinical time series rarely do: in an ICU, heart rate might be checked every 15 minutes while a lab value is drawn twice a day, entire hours can pass with no measurement at all, and the frequency of measurement is itself informative — a clinician ordering more frequent tests is a signal about how sick a patient is.

This repo implements and compares a family of models built specifically for that setting, where time is a continuous variable inside the network rather than a discrete sequence index:

- **Neural ODEs** (Chen, Rubanova, Bettencourt & Duvenaud, *NeurIPS 2018*, Best Paper Award) parameterize the derivative of a hidden state with a neural network, `dz/dt = f(z, t)`, and obtain `z` at any time via a numerical ODE solver.
- **Latent ODEs** (Rubanova, Chen & Duvenaud, *NeurIPS 2019*) apply this to irregular time series: an RNN encodes a patient's sparse observations into a distribution over an initial latent state `z0`; an ODE solver integrates it forward continuously; a decoder reads off predictions at arbitrary times, including ones never observed.
- **Neural CDEs** (Kidger, Morrill, Foster & Lyons, *NeurIPS 2020*) address a real limitation of the above: an ODE's trajectory is fully determined by `z0`, with no mechanism to correct course as new data arrives. A Controlled Differential Equation instead drives the hidden state directly off an interpolation of the incoming data, `dz/dt = f(z) dX/dt`.

## An important caveat this repo takes seriously

**Klötergens et al., "Physiome-ODE: A Benchmark for Irregularly Sampled Multivariate Time-Series Forecasting Based on Biological ODEs" (ICLR 2025)** showed that on the datasets most commonly used to evaluate this class of models — including PhysioNet-2012 — a trivial baseline that predicts a constant value is often competitive with, or beats, full neural-ODE models.

For that reason, `src/models/baselines.py` includes a `PersistenceBaseline` with no learned dynamics at all (just summary statistics — last value, mean, count — into an MLP), and it is reported in every experiment alongside the other models. Whether the added complexity of Latent-ODE / Neural-CDE earns its keep on a given task is treated as an empirical question with an honest answer, not assumed.

## Dataset

**PhysioNet/CinC Challenge 2012** — "Predicting Mortality of ICU Patients." 12,000 adult ICU stays, up to 42 variables recorded over the first 48 hours of admission, in-hospital mortality as the binary label. Listed as **Open Access** on PhysioNet: unlike MIMIC-III/IV, no credentialing or data-use agreement is required.

```bash
bash data/download_physionet2012.sh
```

This downloads `set-a` (4,000 records with outcome labels — the only split with labels available outside the original Challenge, and the one used throughout this repo) into `data/raw/`.

A synthetic data generator (`src/data/synthetic.py`) produces structurally identical fake records, so the full pipeline can be exercised without the download:

```bash
pip install -r requirements.txt
python -m src.training.train --synthetic --epochs 5
```

Once the real data is downloaded:

```bash
python -m src.training.train --data-dir data/raw --epochs 30
```

## Repository structure

```
src/data/physionet.py        raw-file parser (Time,Parameter,Value -> PatientRecord)
src/data/preprocessing.py    bins records onto a shared hourly grid, builds batch tensors
src/data/synthetic.py        synthetic data generator matching the real data's structure
src/models/baselines.py      PersistenceBaseline, GRU, GRU-D (Che et al. 2018)
src/models/latent_ode.py     Latent ODE (Rubanova et al. 2019), via torchdiffeq
src/models/neural_cde.py     Neural CDE (Kidger et al. 2020), via torchcde
src/training/train.py        trains and evaluates every model, prints a comparison table
src/training/metrics.py      AUROC / AUPRC
tests/                       data-pipeline tests and forward/backward-pass tests per model
configs/default.yaml         reference hyperparameters for a full run
```

Run `pytest tests/ -v` to check the pipeline and every model's forward and backward pass.

## A design choice worth understanding

`src/data/preprocessing.py` bins each patient's first 48 hours onto a shared hourly grid (`t = 0, 1, ..., 48`) rather than using each patient's true, continuous observation times. This is a standard simplification in the ICU time-series literature: it lets the Latent ODE integrate an entire batch in a single `odeint` call instead of requiring a union-of-timestamps-plus-gather scheme. It does not remove the irregularity that matters — most (patient, hour, variable) cells remain unobserved, so missingness patterns and observation density are fully preserved. What is lost is sub-hour timing precision.

A natural extension is to drop the binning and integrate on the true union of observation times per batch; none of the model code in `src/models/` needs to change for this, only `preprocessing.py` and the `times` field of `Batch`.

## Roadmap

- [x] Data pipeline (parser, hourly binning, normalization, delta-time)
- [x] Baselines: persistence, GRU, GRU-D
- [x] Latent ODE and Neural CDE implementations, validated end-to-end on synthetic data
- [ ] Full training run on real PhysioNet-2012 data, results table below filled in
- [ ] Continuous-time (non-binned) variant of the pipeline
- [ ] Missingness-robustness ablation (progressively mask observations, compare degradation across models)
- [ ] Trajectory-forecasting task (extrapolation), in addition to classification
- [ ] Trajectory visualizations (reconstructed continuous curve vs. sparse raw observations)

## Results

*To be filled in after a full run on real data (`python -m src.training.train --data-dir data/raw --epochs 30`).*

| Model | AUROC | AUPRC |
|---|---|---|
| Persistence baseline | | |
| GRU | | |
| GRU-D | | |
| Latent ODE | | |
| Neural CDE | | |

## Known simplifications

- **Hourly binning** — see above.
- **GRU-D's decay target** is the post-normalization mean, i.e. 0, which is correct but makes that term less visually distinctive when inspected on its own.
- **Same-hour observations are averaged** in `Physionet2012Dataset._bin_one`; for a variable trending quickly within an hour, this discards information that the continuous-time variant would recover.
- The Latent-ODE reconstruction loss is weighted at `0.1x` against the classification loss; this ratio has not been tuned.

## References

- Chen, Rubanova, Bettencourt & Duvenaud. ["Neural Ordinary Differential Equations."](https://arxiv.org/abs/1806.07366) NeurIPS 2018 (Best Paper Award).
- Rubanova, Chen & Duvenaud. ["Latent Ordinary Differential Equations for Irregularly-Sampled Time Series."](https://arxiv.org/abs/1907.03907) NeurIPS 2019.
- Kidger, Morrill, Foster & Lyons. ["Neural Controlled Differential Equations for Irregular Time Series."](https://arxiv.org/abs/2005.08926) NeurIPS 2020.
- Che, Purushotham, Cho, Sontag & Liu. ["Recurrent Neural Networks for Multivariate Time Series with Missing Values."](https://www.nature.com/articles/s41598-018-24271-9) Scientific Reports, 2018.
- Klötergens, Yalavarthi, Scholz, Stubbemann, Born & Schmidt-Thieme. ["Physiome-ODE: A Benchmark for Irregularly Sampled Multivariate Time-Series Forecasting Based on Biological ODEs."](https://arxiv.org/abs/2502.07489) ICLR 2025.
- Silva, Moody, Scott, Celi & Mark. ["Predicting In-Hospital Mortality of ICU Patients: The PhysioNet/Computing in Cardiology Challenge 2012."](https://physionet.org/content/challenge-2012/1.0.0/)

Built on [torchdiffeq](https://github.com/rtqichen/torchdiffeq) and [torchcde](https://github.com/patrick-kidger/torchcde).

## License

MIT (see `LICENSE`). The PhysioNet 2012 dataset has its own terms of use; see the [dataset page](https://physionet.org/content/challenge-2012/1.0.0/) for details.
