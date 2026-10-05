# Continuous-Time Deep Learning for ICU Mortality Prediction

A research-oriented comparison of discrete and continuous-time models for in-hospital mortality prediction on the PhysioNet/CinC Challenge 2012 dataset.

The project evaluates summary-statistic, GRU, GRU-D, Latent ODE and Neural CDE models under a common development protocol, then exports the selected model as a reproducible inference artifact.

## Main result

The standard GRU is the strongest model on the retrospective Set C benchmark:

| Model | AUROC | AUPRC | Event 1 | Event 2 |
|---|---:|---:|---:|---:|
| Persistence baseline | 0.8396 | 0.4982 | 0.3641 | 395.5216 |
| **GRU** | **0.8547** | **0.5446** | **0.5197** | 224.5535 |
| GRU-D | 0.8538 | 0.5312 | 0.4640 | **57.1411** |
| Latent ODE | 0.8451 | 0.5125 | 0.4120 | 358.9472 |
| Neural CDE | 0.7516 | 0.3448 | 0.3625 | 122.0344 |

Higher is better for AUROC, AUPRC and Event 1; lower is better for Event 2.

The GRU is therefore used as the final inference model because it has the best discrimination and Event-1 score. GRU-D is substantially better on Event 2, which highlights a calibration trade-off: the deployed GRU sigmoid output is treated as a **risk score**, not as a calibrated clinical probability.

These are retrospective results on the now-public Challenge sets. They should not be interpreted as a new official leaderboard submission or rank.

## Dataset and protocol

The dataset is the **PhysioNet/CinC Challenge 2012** mortality task: 12,000 adult ICU stays split into Sets A, B and C, with measurements from the first 48 hours of admission.

The main experiments use a fixed protocol:

1. Split Set A into 80% development-train and 20% validation with stratification and seed 42.
2. Fit vocabulary and normalization statistics on the development-train partition only.
3. Select the best epoch by validation AUPRC with patience 6.
4. Select the Event-1 decision threshold on the same Set-A validation partition.
5. Refit preprocessing on all Set A and retrain for the selected number of epochs.
6. Generate Set B/C predictions before reading their outcomes.
7. Read the now-public B/C outcomes only for retrospective scoring.

The full protocol, selected epochs and thresholds are documented in [`docs/experimental_protocol.md`](docs/experimental_protocol.md).

## Time representation

The main benchmark maps each stay onto a shared hourly grid, `t = 0, 1, ..., 48`. Missingness and time-since-last-observation remain explicit model inputs, while sub-hour timing is discarded.

Notebook 05 performs a controlled Latent-ODE ablation in which the architecture, objective and numerical integration setup are held fixed while only the time representation changes. On Set C, the hourly controlled model reaches AUROC `0.8474` and AUPRC `0.4974`, while the exact-time version reaches AUROC `0.7863` and AUPRC `0.3834`. In this configuration, preserving exact timestamps does not improve mortality discrimination.

## Models

- **Persistence baseline** — last value, mean and observation count per variable, plus static covariates, followed by an MLP.
- **GRU** — forward-filled values, observation mask and `tanh(delta_t / 24)` at each hourly step.
- **GRU-D** — learned input and hidden-state decay driven by time since the previous observation.
- **Latent ODE** — backward recurrent encoder, variational initial latent state, learned ODE dynamics and masked reconstruction objective.
- **Neural CDE** — continuous hidden dynamics driven by a Hermite-cubic control path containing time, measurements and missingness.

The mathematical details are in [`docs/theory.md`](docs/theory.md).

## Repository structure

```text
src/
├── data/
│   ├── physionet.py          # raw PhysioNet parser
│   ├── preprocessing.py      # hourly representation and normalization
│   └── synthetic.py          # synthetic records for tests
├── models/
│   ├── baselines.py          # Persistence, GRU and GRU-D research models
│   ├── gru.py                # final GRU architecture
│   ├── latent_ode.py
│   └── neural_cde.py
├── training/
│   ├── metrics.py            # AUROC, AUPRC, Event 1 and Event 2
│   ├── train.py              # Set-A development comparison
│   └── finalize_gru.py       # final all-A fit and bundle export
├── inference.py
└── model_bundle.py

scripts/
└── validate_final_model.py

tests/
└── test_inference.py

configs/
└── default.yaml
```

The notebooks contain the research experiments; the Python package contains the reusable model, preprocessing, training and inference code.

## Setup

```bash
pip install -r requirements.txt
bash data/download_physionet2012.sh
```

A fast synthetic development run does not require the real dataset:

```bash
python -m src.training.train --synthetic --epochs 3
```

A Set-A development comparison can be run with:

```bash
python -m src.training.train --data-dir data/raw
```

## Final model artifact

The selected GRU is retrained on all Set A using the epoch count and threshold frozen during Set-A development:

```bash
python -m src.training.finalize_gru
```

This creates:

```text
artifacts/gru_physionet2012.pt
```

The bundle contains the trained state dictionary, variable vocabulary, static-variable order, normalization statistics, model configuration and Event-1 threshold.

Validate the exported artifact on label-free Set B records with:

```bash
python -m scripts.validate_final_model
```

Programmatic inference:

```python
from pathlib import Path

from src.data.physionet import load_split
from src.inference import predict_records

records = load_split(Path("data/raw"), split="set-b")

output = predict_records(
    records,
    "artifacts/gru_physionet2012.pt",
    device="cpu",
)
```

`output["score"]` contains the GRU risk score and `output["prediction"]` applies the frozen threshold.

## Tests

Run the complete test suite from the repository root:

```bash
python -m pytest -q
```

The CI workflow runs the same synthetic-data test suite and does not require the PhysioNet dataset.

## Limitations

The main benchmark discretizes observations to an hourly grid, and the exact-time experiment is an ablation rather than the deployed preprocessing path. The Latent ODE mortality classifier uses the posterior mean `q(z0)` together with static covariates, so its ODE trajectory affects classification indirectly through the joint reconstruction/KL objective. The reported benchmark uses one fixed development split and seed rather than repeated-seed uncertainty estimates. Finally, no post-hoc probability calibration is applied to the selected GRU.

## References

- Chen, Rubanova, Bettencourt & Duvenaud. [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366). NeurIPS 2018.
- Rubanova, Chen & Duvenaud. [Latent Ordinary Differential Equations for Irregularly-Sampled Time Series](https://arxiv.org/abs/1907.03907). NeurIPS 2019.
- Kidger, Morrill, Foster & Lyons. [Neural Controlled Differential Equations for Irregular Time Series](https://arxiv.org/abs/2005.08926). NeurIPS 2020.
- Che, Purushotham, Cho, Sontag & Liu. [Recurrent Neural Networks for Multivariate Time Series with Missing Values](https://www.nature.com/articles/s41598-018-24271-9). Scientific Reports, 2018.
- Silva, Moody, Scott, Celi & Mark. [Predicting In-Hospital Mortality of ICU Patients: The PhysioNet/Computing in Cardiology Challenge 2012](https://physionet.org/content/challenge-2012/1.0.0/).

## License

MIT. The PhysioNet 2012 dataset has its own terms of use; see the dataset page for details.
