from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class ActorCriticConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int = 64
    n_layers: int = 2
    init_log_std: float = -0.5


def build_mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int) -> nn.Module:
    layers = []
    d = in_dim

    for _ in range(max(1, n_layers)):
        layers.append(nn.Linear(d, hidden_dim))
        layers.append(nn.Tanh())
        d = hidden_dim

    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """
    PPO 的 Actor-Critic 网络。

    输入:
        state: 市场状态 + 信号状态 + 图状态 + 组合状态

    输出:
        actor_mean: 连续动作分布均值
        log_std:    连续动作分布标准差的 log
        value:      状态价值 V(s)
    """

    def __init__(self, cfg: ActorCriticConfig):
        super().__init__()

        self.cfg = cfg

        self.actor = build_mlp(
            in_dim=cfg.state_dim,
            hidden_dim=cfg.hidden_dim,
            out_dim=cfg.action_dim,
            n_layers=cfg.n_layers,
        )

        self.critic = build_mlp(
            in_dim=cfg.state_dim,
            hidden_dim=cfg.hidden_dim,
            out_dim=1,
            n_layers=cfg.n_layers,
        )

        self.log_std = nn.Parameter(
            torch.ones(cfg.action_dim, dtype=torch.float32) * float(cfg.init_log_std)
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.actor(state)
        std = torch.exp(self.log_std).expand_as(mean)
        value = self.critic(state).squeeze(-1)
        return mean, std, value

    def distribution(self, state: torch.Tensor) -> torch.distributions.Normal:
        mean, std, _ = self.forward(state)
        return torch.distributions.Normal(mean, std)

    def value(self, state: torch.Tensor) -> torch.Tensor:
        _, _, v = self.forward(state)
        return v