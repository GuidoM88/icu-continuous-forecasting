"""
Neural Controlled Differential Equation for irregularly-sampled, partially-
observed clinical time series.

Kidger, Morrill, Foster & Lyons, "Neural Controlled Differential Equations
for Irregular Time Series", NeurIPS 2020.

Unlike a plain (latent) ODE -- whose trajectory is fully determined by its
initial condition z0, with no mechanism for correcting course as new data
arrives -- a CDE's hidden state is driven directly by an interpolation of
the observed data itself: dz/dt = f(z) dX/dt. We build the control path X
with Hermite cubic splines with backward differences, torchcde's
recommended interpolation when the raw data contains missing values, and,
following the "informative missingness" recipe in torchcde's own
example/irregular_data.py, append each channel's own missingness mask
(plus time itself) as extra channels -- so the model can tell not just
*what* the last reading was but *how stale* it is.

We only need the terminal hidden state z_T for classification, so (as in
the official example) we solve directly for that rather than for the full
trajectory.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchcde


class CDEFunc(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, hidden: int = 64):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.net = nn.Sequential(
            nn.Linear(hidden_channels, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden_channels * input_channels),
        )

    def forward(self, t, z):
        out = self.net(z).tanh()
        return out.view(*z.shape[:-1], self.hidden_channels, self.input_channels)


class NeuralCDE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        static_dim: int,
        hidden_channels: int = 32,
        hidden: int = 64,
        interpolation: str = "cubic",
        solver: str = "rk4",
        step_size: float = 1.0,
    ):
        super().__init__()
        # channels fed to the CDE: [time, value_1..D, mask_1..D]
        self.input_channels = 1 + input_dim * 2
        self.hidden_channels = hidden_channels
        self.interpolation = interpolation
        self.solver = solver
        self.step_size = step_size

        self.initial = nn.Linear(self.input_channels, hidden_channels)
        self.func = CDEFunc(self.input_channels, hidden_channels, hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels + static_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def build_path(self, batch):
        B, T, D = batch.values.shape
        t = batch.times.view(1, T, 1).expand(B, T, 1).to(batch.values.device)
        x = torch.where(batch.mask > 0, batch.values, torch.full_like(batch.values, float("nan")))
        x = torch.cat([t, x, batch.mask], dim=-1)
        if self.interpolation == "cubic":
            coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)
            return torchcde.CubicSpline(coeffs)
        coeffs = torchcde.linear_interpolation_coeffs(x)
        return torchcde.LinearInterpolation(coeffs)

    def forward(self, batch):
        X = self.build_path(batch)
        z0 = self.initial(X.evaluate(X.interval[0]))
        zt = torchcde.cdeint(
            X=X,
            func=self.func,
            z0=z0,
            t=X.interval,
            method=self.solver,
            options={"step_size": self.step_size} if self.solver in ("rk4", "euler") else {},
        )
        zT = zt[..., -1, :]
        logits = self.classifier(
            torch.cat([zT, torch.nan_to_num(batch.static, nan=0.0)], dim=-1)
        ).squeeze(-1)
        return logits
