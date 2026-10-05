# Mathematical Background

This document summarizes the models implemented in `src/models/` and the time representations used by the experiments.

## 1. Problem formulation

For one ICU stay, let

- \(s \in \mathbb{R}^{S}\) denote static covariates;
- \(x(t) \in \mathbb{R}^{D}\) denote clinical variables;
- \(m(t) \in \{0,1\}^{D}\) denote which variables are observed;
- \(y \in \{0,1\}\) denote in-hospital mortality.

The objective is to estimate a mortality score from the first 48 hours of an irregularly sampled, partially observed multivariate time series.

The main benchmark uses an hourly representation. Raw observations falling in the same hour are averaged, while the mask and time since the previous observation retain information about missingness and sampling frequency.

## 2. GRU

The GRU receives, at each hourly step,

\[
u_t =
\left[
\tilde{x}_t,\;
m_t,\;
\tanh(\Delta_t / 24)
\right],
\]

where \(\tilde{x}_t\) is the forward-filled normalized measurement vector and \(\Delta_t\) is the per-variable time since the previous observation.

The recurrent state evolves as

\[
h_t = \mathrm{GRU}(u_t, h_{t-1}),
\]

and the mortality logit is produced from the final recurrent state concatenated with normalized static covariates.

This is the final model selected by the benchmark.

## 3. GRU-D

GRU-D introduces learned decay toward a reference input value and decay of the hidden state. For one variable,

\[
\gamma_x = \exp(-\max(0, W_x \Delta_t + b_x)),
\]

and an unobserved value is imputed as a mixture of the most recent value and the population mean. The hidden state uses an analogous decay term.

Because the dynamic variables are normalized on the development data, their population reference mean is zero in the implemented representation.

## 4. Neural ODEs

A Neural ODE defines a continuous hidden state by

\[
\frac{dz(t)}{dt} = f_\theta(z(t), t).
\]

Given \(z(t_0)\), a numerical ODE solver evaluates

\[
z(t_1)
=
z(t_0)
+
\int_{t_0}^{t_1} f_\theta(z(t), t)\,dt.
\]

The implementation uses `torchdiffeq`. Gradients are computed through the solver by default; the model can optionally use the adjoint implementation exposed by `torchdiffeq`.

## 5. Latent ODE

The Latent ODE uses a backward recurrent encoder to infer a distribution over the initial latent state,

\[
q_\phi(z_0 \mid x_{1:T})
=
\mathcal{N}
\left(
\mu_\phi,\;
\mathrm{diag}(\sigma_\phi^2)
\right).
\]

A sample is obtained with the reparameterization trick,

\[
z_0 = \mu_\phi + \sigma_\phi \odot \epsilon,
\qquad
\epsilon \sim \mathcal{N}(0,I).
\]

The latent state is then integrated through

\[
\frac{dz}{dt} = f_\theta(z),
\]

and a decoder reconstructs the observed variables along the latent trajectory.

The training loss is

\[
\mathcal{L}
=
\mathcal{L}_{\mathrm{BCE}}
+
0.1\,\mathcal{L}_{\mathrm{recon}}
+
10^{-3}\,\mathcal{L}_{\mathrm{KL}}.
\]

The reconstruction term is a masked MSE, so only observed values contribute.

For mortality classification, the implemented head uses the posterior mean \(\mu_\phi = q(z_0)\) concatenated with static covariates:

\[
\hat y = \sigma(g_\psi(\mu_\phi, s)).
\]

The classifier therefore does **not** read the terminal latent state \(z(T)\) directly. The ODE trajectory influences classification indirectly through the shared encoder and the reconstruction/KL objective.

## 6. Neural CDE

A Neural Controlled Differential Equation evolves a hidden state under a continuous control path \(X(t)\):

\[
z_t
=
z_0
+
\int_0^t f_\theta(z_s)\,dX_s.
\]

For a differentiable control path,

\[
\frac{dz_t}{dt}
=
f_\theta(z_t)\frac{dX_t}{dt}.
\]

The implementation builds \(X\) from

\[
[\;t,\;x_1,\ldots,x_D,\;m_1,\ldots,m_D\;]
\]

and uses `torchcde.hermite_cubic_coefficients_with_backward_differences`, i.e. Hermite cubic interpolation with backward differences. The terminal hidden state is concatenated with static covariates for mortality classification.

## 7. Hourly and exact-time representations

The production and main benchmark path uses a shared hourly grid because it gives all patients the same sequence length and permits efficient batched training.

The controlled exact-time ablation in Notebook 05 uses patient-specific raw observation times. The two ablation arms share the same encoder, latent dynamics, classifier, loss and fixed-step RK4 integration setup; only the temporal representation changes.

The experiment therefore addresses a narrow question: whether preserving sub-hour timestamps improves this Latent-ODE configuration. It does not establish that exact timestamps are generally unhelpful for continuous-time models.

## 8. Model selection and calibration

Development choices are made from Set A only. AUPRC selects the epoch and the Event-1 threshold is selected on the Set-A validation partition.

Challenge Event 1 is

\[
\min(\mathrm{sensitivity}, \mathrm{PPV}),
\]

so it depends on a binary threshold.

Event 2 evaluates calibration using the Challenge's range-normalized Hosmer-Lemeshow statistic. Lower values are better.

The selected GRU has the strongest discrimination and Event-1 score, but GRU-D has a substantially lower Event-2 score. For this reason, the exported GRU sigmoid output is documented as a risk score rather than a calibrated probability.

## 9. Scope

The repository includes an exact-time research ablation but the exported inference pipeline uses hourly preprocessing. It does not currently include post-hoc probability calibration, repeated-seed confidence intervals, conformal prediction, or a production exact-time serving path.

See [`experimental_protocol.md`](experimental_protocol.md) for the evaluation protocol and results.
