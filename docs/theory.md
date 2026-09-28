# Mathematical Background

This document derives the models implemented in `src/models/`. It assumes familiarity with standard deep learning but not with differential equations. For usage instructions see the main [README](../README.md).

## 1. Problem formalization

A patient's ICU stay is represented as:

- **Static covariates** $s \in \mathbb{R}^{S}$ (age, gender, height, ICU type, weight).
- **A set of irregularly-timed, partially-observed multivariate observations** $\{(t_i, \mathbf{x}_i, \mathbf{m}_i)\}_{i=1}^{N}$, where $t_i \in [0, 48]$ hours, $\mathbf{x}_i \in \mathbb{R}^D$ is the vector of $D$ clinical variables at time $t_i$, and $\mathbf{m}_i \in \{0,1\}^D$ marks which entries of $\mathbf{x}_i$ were actually measured.
- **A binary label** $y \in \{0,1\}$: in-hospital mortality.

The task is to learn $p_\theta(y \mid s, \{(t_i, \mathbf{x}_i, \mathbf{m}_i)\})$. Three properties of this data make it a poor fit for a standard RNN or Transformer operating on a fixed step index: the *time between* observations is irregular and informative, different variables are missing at different times ("informative missingness" — a clinician orders a test *because* they suspect a problem), and the model may need to make a prediction, or reason about the patient's state, at a time that was never observed.

## 2. Neural ODEs

A residual network computes $h_{t+1} = h_t + f_\theta(h_t)$, a discrete Euler step. Chen, Rubanova, Bettencourt & Duvenaud (NeurIPS 2018) take the continuum limit: treat depth (or time) as continuous and define the hidden state's *derivative* with a network,

$$\frac{d\mathbf{z}(t)}{dt} = f_\theta(\mathbf{z}(t), t).$$

Given an initial state $\mathbf{z}(t_0)$, the state at any later time follows by solving this initial value problem numerically:

$$\mathbf{z}(t_1) = \mathbf{z}(t_0) + \int_{t_0}^{t_1} f_\theta(\mathbf{z}(t), t)\,dt =: \mathrm{ODESolve}(\mathbf{z}(t_0), f_\theta, t_0, t_1).$$

Gradients can be obtained either by literally backpropagating through the solver's internal steps (this repo's default), or with the **adjoint sensitivity method** the original paper introduces: define $\mathbf{a}(t) = \partial \mathcal{L}/\partial \mathbf{z}(t)$ and solve a second ODE *backward in time*,

$$\frac{d\mathbf{a}(t)}{dt} = -\mathbf{a}(t)^\top \frac{\partial f_\theta}{\partial \mathbf{z}},$$

which yields exact gradients in $O(1)$ memory regardless of solver step count, at the cost of one extra solve. `torchdiffeq.odeint_adjoint` implements this; `LatentODE(..., use_adjoint=True)` switches to it.

## 3. Latent ODE (`src/models/latent_ode.py`)

Rubanova, Chen & Duvenaud (NeurIPS 2019) wrap the Neural ODE in a variational-autoencoder-style model built for exactly this setting.

**Encoder** — an RNN reads $(t_i, \mathbf{x}_i, \mathbf{m}_i)$ *backward in time* and outputs an amortized approximate posterior over the initial latent state,

$$q_\phi(\mathbf{z}_0 \mid \mathbf{x}_{1:N}) = \mathcal{N}\big(\mathbf{z}_0;\ \mu_\phi,\ \mathrm{diag}(\sigma_\phi^2)\big),$$

sampled via the reparameterization trick, $\mathbf{z}_0 = \mu_\phi + \sigma_\phi \odot \epsilon,\ \epsilon \sim \mathcal{N}(0, I)$ (`Encoder` in the code).

**Latent dynamics** — $\mathbf{z}_0$ is integrated forward through a Neural ODE (`ODEFunc`) to obtain the trajectory $\mathbf{z}(t)$ at every time of interest — including times never observed.

**Decoder** — $p_\theta(\mathbf{x}_i \mid \mathbf{z}(t_i))$ reconstructs an observation from the latent state at that time (`Decoder`).

**Training objective** — the evidence lower bound (ELBO):

$$\mathcal{L}_{\mathrm{ELBO}} = \mathbb{E}_{q_\phi(\mathbf{z}_0)}\Big[\textstyle\sum_{i=1}^{N} \log p_\theta(\mathbf{x}_i \mid \mathbf{z}(t_i))\Big] - \mathrm{KL}\big(q_\phi(\mathbf{z}_0)\,\|\,p(\mathbf{z}_0)\big), \qquad p(\mathbf{z}_0)=\mathcal{N}(0,I).$$

In code, the reconstruction term is a **masked** MSE (`masked_mse`) — only observed entries of $\mathbf{x}_i$ contribute, since most are missing — and the KL term has the standard closed form for two Gaussians (`kl_standard_normal`):

$$\mathrm{KL}\big(\mathcal{N}(\mu,\sigma^2)\,\|\,\mathcal{N}(0,1)\big) = -\tfrac{1}{2}\sum_j \big(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\big).$$

A classifier head $p_\psi(y \mid \mu_\phi, s)$ predicts mortality from the (mean of the) inferred initial state and the static covariates. The total loss actually minimized in `train.py` is

$$\mathcal{L} = \mathcal{L}_{\mathrm{BCE}}(y,\hat y) \;+\; \lambda_1\,\mathcal{L}_{\mathrm{recon}} \;+\; \lambda_2\,\mathrm{KL}, \qquad \lambda_1 = 0.1,\ \lambda_2 = 10^{-3},$$

deliberately weighted toward the classification task, which is this repo's primary target — see the README's "known simplifications" for why that ratio is a real hyperparameter, not a constant to trust.

## 4. Neural CDE (`src/models/neural_cde.py`)

The Latent ODE has a real weakness: once $\mathbf{z}_0$ is sampled, the trajectory for $t > t_0$ is **entirely determined by the ODE** — a new observation arriving later can only influence the model by being re-fed through the encoder, not by nudging the trajectory already in flight.

Kidger, Morrill, Foster & Lyons (NeurIPS 2020) fix this by making the *data itself* the driving signal. First, interpolate the (possibly missing) observations into a continuous control path $X:[0,T]\to\mathbb{R}^{C}$ — this repo uses natural cubic splines with backward differences (`torchcde.hermite_cubic_coefficients_with_backward_differences`), with $C = 1 + 2D$ channels: time, the $D$ variables, and their own missingness mask, so the model can tell not just the last value but *how stale* it is ("informative missingness" as an explicit input, not just an implicit gap). The hidden state is then defined as the solution of the controlled differential equation

$$\mathbf{z}_t = \mathbf{z}_0 + \int_0^t f_\theta(\mathbf{z}_s)\,dX_s,$$

a Riemann–Stieltjes integral against $X$. Since $X$ is piecewise differentiable, this is equivalent to the ODE

$$\frac{d\mathbf{z}_t}{dt} = f_\theta(\mathbf{z}_t)\,\frac{dX_t}{dt},$$

which is exactly what `torchcde.cdeint` (built on `torchdiffeq.odeint` internally) solves. Here $f_\theta: \mathbb{R}^H \to \mathbb{R}^{H\times C}$ is matrix-valued (`CDEFunc`), so $f_\theta(\mathbf{z}_t)\,\dot X_t$ is a matrix-vector product mapping the $C$-dimensional data derivative into an $H$-dimensional hidden-state update — the continuous-time analogue of a GRU/LSTM update, with the data itself playing the role of the discrete input at each step. This is why a Neural CDE can react to new information as it arrives, in a way a plain Latent ODE cannot.

For classification, only the terminal state $\mathbf{z}_T$ is needed: $\hat y = \sigma(g_\psi(\mathbf{z}_T, s))$, trained with binary cross-entropy.

## 5. Comparison

| | State at $t$ determined by | Update mechanism | Reacts to new data mid-sequence |
|---|---|---|---|
| GRU / GRU-D | discrete recurrence | $h_t = \mathrm{GRUCell}(x_t, h_{t-1})$ | yes, at each discrete step |
| Latent ODE | $\mathbf{z}_0$ only | continuous flow $\dot{\mathbf{z}} = f_\theta(\mathbf{z}, t)$ | only by re-running the encoder |
| Neural CDE | $\mathbf{z}_0$ **and** the path $X$ | continuous, data-driven flow $\dot{\mathbf{z}} = f_\theta(\mathbf{z})\,\dot X_t$ | yes, continuously |

## 6. What this repo does *not* implement

- The fully continuous (non-binned) time grid — see the README's "design choice worth understanding." The equations above are exact for continuous $t$; the current data pipeline evaluates them on an hourly grid.
- Second-order / higher-order solvers beyond what `torchdiffeq`/`torchcde` expose via the `method=` argument.
- Uncertainty calibration beyond the Latent ODE's built-in $q_\phi(\mathbf{z}_0)$ — e.g. no ensembling, no conformal prediction.

## References

See the main [README](../README.md#references).
