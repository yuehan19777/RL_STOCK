from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F

from controller_ppo.networks import ActorCritic, ActorCriticConfig


@dataclass
class PPOTrainResult:
    model: ActorCritic
    history: list[dict]


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_nonterminal = 1.0 - dones[t]
            next_value = 0.0
        else:
            next_nonterminal = 1.0 - dones[t]
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        adv[t] = last_gae

    ret = adv + values
    return adv.astype(np.float32), ret.astype(np.float32)


@torch.no_grad()
def collect_rollout(env, model: ActorCritic, device: torch.device):
    states = []
    actions = []
    logps = []
    rewards = []
    values = []
    dones = []
    infos = []

    state = env.reset()
    done = False

    while not done:
        st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        dist = model.distribution(st)
        value = model.value(st)

        action = dist.sample()
        logp = dist.log_prob(action).sum(dim=-1)

        action_np = action.squeeze(0).detach().cpu().numpy()

        next_state, reward, done, info = env.step(action_np)

        states.append(state)
        actions.append(action_np)
        logps.append(float(logp.item()))
        rewards.append(float(reward))
        values.append(float(value.item()))
        dones.append(float(done))
        infos.append(info)

        state = next_state

    data = {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "logps": np.asarray(logps, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "values": np.asarray(values, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.float32),
        "infos": infos,
    }

    return data


def train_ppo(env, cfg: Dict[str, Any], device: str | None = None) -> PPOTrainResult:
    ppo_cfg = cfg.get("ppo", {})

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dev = torch.device(device)

    state_dim = env.state_dim
    action_dim = env.action_dim

    net_cfg = ppo_cfg.get("network", {})

    model = ActorCritic(
        ActorCriticConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=int(net_cfg.get("hidden_dim", 64)),
            n_layers=int(net_cfg.get("n_layers", 2)),
            init_log_std=float(net_cfg.get("init_log_std", -0.5)),
        )
    ).to(dev)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(ppo_cfg.get("lr", 3e-4)),
    )

    gamma = float(ppo_cfg.get("gamma", 0.99))
    gae_lambda = float(ppo_cfg.get("gae_lambda", 0.95))
    clip_ratio = float(ppo_cfg.get("clip_ratio", 0.2))
    epochs = int(ppo_cfg.get("epochs", 8))
    minibatch_size = int(ppo_cfg.get("minibatch_size", 128))
    ent_coef = float(ppo_cfg.get("ent_coef", 0.005))
    vf_coef = float(ppo_cfg.get("vf_coef", 0.5))
    max_grad_norm = float(ppo_cfg.get("max_grad_norm", 1.0))
    total_iterations = int(ppo_cfg.get("total_iterations", 50))

    history: list[dict] = []

    for it in range(total_iterations):
        rollout = collect_rollout(env, model, dev)

        adv, ret = compute_gae(
            rewards=rollout["rewards"],
            values=rollout["values"],
            dones=rollout["dones"],
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        states = torch.tensor(rollout["states"], dtype=torch.float32, device=dev)
        actions = torch.tensor(rollout["actions"], dtype=torch.float32, device=dev)
        old_logps = torch.tensor(rollout["logps"], dtype=torch.float32, device=dev)
        returns = torch.tensor(ret, dtype=torch.float32, device=dev)
        advantages = torch.tensor(adv, dtype=torch.float32, device=dev)

        n = states.shape[0]
        idx = np.arange(n)

        pi_losses = []
        v_losses = []
        entropies = []

        for _ in range(epochs):
            np.random.shuffle(idx)

            for start in range(0, n, minibatch_size):
                mb = idx[start:start + minibatch_size]

                st = states[mb]
                ac = actions[mb]
                old_lp = old_logps[mb]
                rt = returns[mb]
                ad = advantages[mb]

                dist = model.distribution(st)
                new_logp = dist.log_prob(ac).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                value = model.value(st)

                ratio = torch.exp(new_logp - old_lp)

                unclipped = ratio * ad
                clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * ad

                pi_loss = -torch.min(unclipped, clipped).mean()
                v_loss = F.mse_loss(value, rt)

                loss = pi_loss + vf_coef * v_loss - ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                pi_losses.append(float(pi_loss.item()))
                v_losses.append(float(v_loss.item()))
                entropies.append(float(entropy.item()))

        ep_reward = float(np.sum(rollout["rewards"]))
        ep_mean_reward = float(np.mean(rollout["rewards"]))
        final_equity = float(rollout["infos"][-1].get("equity", 1.0))

        row = {
            "iter": it + 1,
            "episode_reward": ep_reward,
            "mean_reward": ep_mean_reward,
            "final_equity_train_episode": final_equity,
            "pi_loss": float(np.mean(pi_losses)) if pi_losses else 0.0,
            "v_loss": float(np.mean(v_losses)) if v_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
        }

        history.append(row)

        print(
            f"[PPO] iter={it + 1}/{total_iterations} "
            f"reward={ep_reward:.6f} "
            f"mean_reward={ep_mean_reward:.6f} "
            f"equity={final_equity:.4f} "
            f"pi_loss={row['pi_loss']:.6f} "
            f"v_loss={row['v_loss']:.6f}"
        )

    return PPOTrainResult(model=model, history=history)