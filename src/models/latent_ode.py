"""
Latent ODE for irregularly-sampled, partially-observed clinical time series.

Rubanova, Chen & Duvenaud, "Latent Ordinary Differential Equations for
Irregularly-Sampled Time Series", NeurIPS 2019.
Built on Chen, Rubanova, Bettencourt & Duvenaud, "Neural Ordinary
Differential Equations", NeurIPS 2019 (Best Paper Award).

Architecture, adapted from torchdiffeq/examples/latent_ode.py for the
masked multivariate clinical setting:
  1. An RNN encoder reads (forward-filled value, mask, delta_t) *backward*
     in time and outputs q(z0) = N(mu, sigma^2) -- an amortised posterior
     over the initial latent state.
  2. z0 ~ q(z0) is integrated forward through a learned ODE dz/dt = f(z)
     with torchdiffeq.odeint, on the single time grid shared by the whole
     batch (see src/data/preprocessing.py for why that's a fixed vector
     here rather than a per-patient one).
  3. A decoder reconstructs the (normalised) observations from the latent
     trajectory, giving a reconstruction term to train the ODE/encoder with
     (in addition to a KL term against a standard-normal prior on z0); a
     separate classifier head predicts in-hospital mortality from z0.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchdiffeq import odeint


class ODEFunc(nn.Module):
    def __init__(self, latent_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, latent_dim),
        )
        self.nfe = 0

    def forward(self, t, z):
        self.nfe += 1
        return self.net(z)


class Encoder(nn.Module):
    """Reads (filled_value, mask, delta_t) backward in time -> q(z0)."""

    def __init__(self, input_dim: int, latent_dim: int, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRUCell(input_size=input_dim * 3, hidden_size=hidden)
        self.hidden = hidden
        self.to_stats = nn.Linear(hidden, latent_dim * 2)

    def forward(self, filled_values, mask, delta_t):
        B, T, D = filled_values.shape
        h = torch.zeros(B, self.hidden, device=filled_values.device)
        for t in reversed(range(T)):
            x = torch.cat(
                [filled_values[:, t], mask[:, t], torch.tanh(delta_t[:, t] / 24.0)], dim=-1
            )
            h = self.gru(x, h)
        mean, logvar = self.to_stats(h).chunk(2, dim=-1)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(), nn.Linear(hidden, output_dim)
        )

    def forward(self, z):
        return self.net(z)


class LatentODE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        static_dim: int,
        latent_dim: int = 16,
        hidden: int = 64,
        solver: str = "dopri5",
        use_adjoint: bool = False,
    ):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim, hidden)
        self.odefunc = ODEFunc(latent_dim, hidden)
        self.decoder = Decoder(latent_dim, input_dim, hidden)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim + static_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.latent_dim = latent_dim
        self.solver = solver
        self._odeint = odeint
        if use_adjoint:
            from torchdiffeq import odeint_adjoint

            self._odeint = odeint_adjoint

    def forward(self, batch, filled_values, sample: bool = True):
        mean, logvar = self.encoder(filled_values, batch.mask, batch.delta_t)
        if sample:
            eps = torch.randn_like(mean)
            z0 = mean + eps * torch.exp(0.5 * logvar)
        else:
            z0 = mean

        # batch.times: [T] -- a single grid shared by every patient in the
        # dataset (see preprocessing.Physionet2012Dataset), so a plain
        # odeint call integrates the whole batch of z0's at once.
        latent_traj = self._odeint(self.odefunc, z0, batch.times, method=self.solver)
        latent_traj = latent_traj.permute(1, 0, 2)  # [B, T, latent_dim]
        recon = self.decoder(latent_traj)

        logits = self.classifier(
            torch.cat([mean, torch.nan_to_num(batch.static, nan=0.0)], dim=-1)
        ).squeeze(-1)
        return logits, recon, mean, logvar


def kl_standard_normal(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1)


def masked_mse(recon: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    target = torch.nan_to_num(values, nan=0.0)
    sq_err = (recon - target) ** 2 * mask
    return sq_err.sum(dim=(-1, -2)) / mask.sum(dim=(-1, -2)).clamp(min=1)
