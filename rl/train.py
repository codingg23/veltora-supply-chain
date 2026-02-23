"""
PPO training script for the SupplyChainEnv.
Uses stable-baselines3 with a custom callback for episode metrics logging.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from simulation.env import SupplyChainEnv

CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR        = Path("logs")


class SupplyChainCallback(BaseCallback):
    """Logs per-episode supply-chain KPIs to a JSON file."""

    def __init__(self, log_path: str = "logs/episode_metrics.jsonl", verbose: int = 0):
        super().__init__(verbose)
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                record = {
                    "episode": self.n_calls,
                    "total_reward": info["episode"]["r"],
                    "episode_length": info["episode"]["l"],
                    "stock_outs": info.get("stock_outs", []),
                    "budget_spent": info.get("budget_spent", 0),
                }
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
        return True


def train(
    total_timesteps: int = 500_000,
    n_envs: int = 4,
    seed: int = 42,
    save_path: str = "checkpoints/ppo_supply_chain",
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 256,
    n_epochs: int = 10,
    gamma: float = 0.99,
    ent_coef: float = 0.01,
    disruption_prob: float = 0.05,
) -> "PPO":
    """
    Train a PPO agent on the SupplyChainEnv.

    Key hyperparameters match the published benchmark results:
    - 500k timesteps → ~500 episodes of 90 days each (×4 envs)
    - Entropy bonus 0.01 encourages exploration of order strategies
    - VecNormalize for observation and reward scaling
    """
    if not SB3_AVAILABLE:
        raise ImportError(
            "stable-baselines3 is required: pip install stable-baselines3"
        )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    def make_env(rank: int):
        def _init():
            env = SupplyChainEnv(seed=seed + rank, disruption_prob=disruption_prob)
            env = Monitor(env, filename=str(LOG_DIR / f"env_{rank}"))
            return env
        return _init

    vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = VecNormalize(
        SubprocVecEnv([make_env(100)]),
        norm_obs=True, norm_reward=False, clip_obs=10.0, training=False
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        ent_coef=ent_coef,
        verbose=1,
        seed=seed,
        tensorboard_log=str(LOG_DIR),
        policy_kwargs=dict(net_arch=[256, 256, 128]),
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(CHECKPOINT_DIR),
        log_path=str(LOG_DIR),
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
    )

    supply_chain_cb = SupplyChainCallback(log_path=str(LOG_DIR / "episode_metrics.jsonl"))

    print(f"Training PPO on SupplyChainEnv for {total_timesteps:,} timesteps "
          f"across {n_envs} parallel envs...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, supply_chain_cb],
        progress_bar=True,
    )

    model.save(save_path)
    vec_env.save(str(Path(save_path).parent / "vec_normalize.pkl"))
    print(f"Model saved to {save_path}")
    return model


def evaluate_policy(
    model_path: str = "checkpoints/best_model",
    n_episodes: int = 20,
    seed: int = 999,
    disruption_prob: float = 0.05,
    verbose: bool = True,
) -> dict[str, float]:
    """
    Evaluate a trained policy and return summary metrics:
    - mean_reward, stock_out_rate, mean_budget_utilisation
    """
    if not SB3_AVAILABLE:
        raise ImportError("stable-baselines3 is required")

    model = PPO.load(model_path)
    env = SupplyChainEnv(seed=seed, disruption_prob=disruption_prob)

    rewards: list[float] = []
    stock_out_counts: list[int] = []
    budget_utils: list[float] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        ep_stock_outs = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_stock_outs += len(info.get("stock_outs", []))
            done = terminated or truncated

        budget_util = info.get("budget_spent", 0) / 5_500_000
        rewards.append(ep_reward)
        stock_out_counts.append(ep_stock_outs)
        budget_utils.append(budget_util)

        if verbose:
            print(f"  Episode {ep+1:2d}: reward={ep_reward:8.1f}  "
                  f"stock_outs={ep_stock_outs}  budget={budget_util:.1%}")

    metrics = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_stock_outs_per_episode": float(np.mean(stock_out_counts)),
        "stock_out_rate": float(np.mean([c > 0 for c in stock_out_counts])),
        "mean_budget_utilisation": float(np.mean(budget_utils)),
    }
    if verbose:
        print(f"\nEvaluation summary ({n_episodes} episodes):")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    return metrics


def baseline_rule_based(n_episodes: int = 20, seed: int = 999) -> dict[str, float]:
    """
    Rule-based baseline: always order EOQ when on_hand < reorder_point.
    Used to compute improvement headroom for the paper.
    """
    from simulation.env import DEMAND_PER_DAY, LEAD_TIMES_BASE, EOQ_UNITS, N

    env = SupplyChainEnv(seed=seed, disruption_prob=0.05)
    rewards, stock_outs = [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        ep_so = 0
        done = False
        while not done:
            # Rule: order EOQ if on_hand < 1.5 × lead_time × demand
            action = np.zeros(N, dtype=int)
            for i in range(N):
                on_hand = env._state[i].on_hand
                rop = 1.5 * LEAD_TIMES_BASE[i] * DEMAND_PER_DAY[i]
                if on_hand < rop and env._state[i].on_order == 0:
                    action[i] = 1   # order EOQ
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_so += len(info.get("stock_outs", []))
            done = terminated or truncated
        rewards.append(ep_reward)
        stock_outs.append(ep_so)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_stock_outs_per_episode": float(np.mean(stock_outs)),
        "stock_out_rate": float(np.mean([c > 0 for c in stock_outs])),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train / evaluate supply-chain RL agent")
    parser.add_argument("--mode", choices=["train", "eval", "baseline"], default="train")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", default="checkpoints/best_model")
    args = parser.parse_args()

    if args.mode == "train":
        train(total_timesteps=args.timesteps, n_envs=args.n_envs, seed=args.seed)
    elif args.mode == "eval":
        evaluate_policy(model_path=args.model_path, seed=args.seed)
    elif args.mode == "baseline":
        metrics = baseline_rule_based(seed=args.seed)
        print("Rule-based baseline:", json.dumps(metrics, indent=2))
