"""Final GRU architecture for ICU mortality classification."""
from __future__ import annotations

import torch
import torch.nn as nn


class GRUMortalityModel(nn.Module):
    def __init__(self, input_dim: int, static_dim: int, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim * 3,
            hidden_size=hidden,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + static_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch, filled_values):
        x = torch.cat(
            [
                filled_values,
                batch.mask,
                torch.tanh(batch.delta_t / 24.0),
            ],
            dim=-1,
        )
        _, h = self.gru(x)
        h = h.squeeze(0)

        features = torch.cat(
            [
                h,
                torch.nan_to_num(batch.static, nan=0.0),
            ],
            dim=-1,
        )
        return self.head(features).squeeze(-1)
