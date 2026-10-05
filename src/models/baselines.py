"""
Baseline models for in-hospital mortality prediction.

PersistenceBaseline summarizes each dynamic variable with its last observed
value, mean and observation count, then combines those features with static
covariates in a small MLP.

GRUBaseline processes forward-filled values together with observation masks
and time-since-last-observation features.

GRUD implements the decay-based recurrent formulation of Che et al. (2018),
with learned input and hidden-state decay driven by elapsed time.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PersistenceBaseline(nn.Module):
    def __init__(self, input_dim: int, static_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 3 + static_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch, filled_values=None):
        values, mask, static = batch.values, batch.mask, batch.static
        obs = torch.nan_to_num(values, nan=0.0) * mask
        count = mask.sum(dim=1)
        mean = obs.sum(dim=1) / count.clamp(min=1)

        B, T, D = values.shape
        last = torch.zeros(B, D, device=values.device)
        have = torch.zeros(B, D, dtype=torch.bool, device=values.device)
        for t in reversed(range(T)):
            m = mask[:, t] > 0
            take = m & (~have)
            last = torch.where(take, obs[:, t], last)
            have = have | m

        feats = torch.cat([last, mean, count, torch.nan_to_num(static, nan=0.0)], dim=-1)
        return self.net(feats).squeeze(-1)


class GRUBaseline(nn.Module):
    def __init__(self, input_dim: int, static_dim: int, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim * 3, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + static_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, batch, filled_values):
        x = torch.cat([filled_values, batch.mask, torch.tanh(batch.delta_t / 24.0)], dim=-1)
        _, h = self.gru(x)
        h = h.squeeze(0)
        feats = torch.cat([h, torch.nan_to_num(batch.static, nan=0.0)], dim=-1)
        return self.head(feats).squeeze(-1)


class GRUDCell(nn.Module):
    """One step of Che et al. (2018) GRU-D."""

    def __init__(self, input_dim: int, hidden: int):
        super().__init__()
        self.gamma_x = nn.Linear(input_dim, input_dim)
        self.gamma_h = nn.Linear(input_dim, hidden)
        self.gru_cell = nn.GRUCell(input_size=input_dim * 2, hidden_size=hidden)

    def forward(self, x_t, m_t, delta_t, x_last, x_mean, h_prev):
        gamma_x = torch.exp(-torch.relu(self.gamma_x(delta_t)))
        gamma_h = torch.exp(-torch.relu(self.gamma_h(delta_t)))

        x_hat = m_t * x_t + (1 - m_t) * (gamma_x * x_last + (1 - gamma_x) * x_mean)
        h_decayed = gamma_h * h_prev

        gru_in = torch.cat([x_hat, m_t], dim=-1)
        h_new = self.gru_cell(gru_in, h_decayed)
        return h_new, x_hat


class GRUD(nn.Module):
    def __init__(self, input_dim: int, static_dim: int, hidden: int = 64):
        super().__init__()
        self.cell = GRUDCell(input_dim, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden + static_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.hidden = hidden

    def forward(self, batch, x_mean):
        """x_mean: [D] population mean per variable (0 after normalisation, but
        kept as an explicit arg so the decay target is never hard-coded)."""
        values, mask, delta_t = batch.values, batch.mask, batch.delta_t
        B, T, D = values.shape
        x = torch.nan_to_num(values, nan=0.0)

        h = torch.zeros(B, self.hidden, device=values.device)
        x_last = x_mean.to(values.device).expand(B, D).clone()
        x_mean_b = x_mean.to(values.device).expand(B, D)
        for t in range(T):
            h, x_hat = self.cell(x[:, t], mask[:, t], delta_t[:, t], x_last, x_mean_b, h)
            x_last = torch.where(mask[:, t] > 0, x[:, t], x_last)

        feats = torch.cat([h, torch.nan_to_num(batch.static, nan=0.0)], dim=-1)
        return self.head(feats).squeeze(-1)
