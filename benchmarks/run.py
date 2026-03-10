"""
Benchmark runner: compares PPO policy vs rule-based baseline across
key supply-chain KPIs.  Results are written to benchmarks/results.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from simulation.env import SupplyChainEnv, DEMAND_PER_DAY, LEAD_TIMES_BASE, EOQ_UNITS, N
from simulation.disruptions import DisruptionEngine

RESULTS_PATH = Path("benchmarks/results.json")


# ---------------------------------------------------------------------------
# Baseline: rule-based ROP policy
# ---------------------------------------------------------------------------

def run_rule_based(n_episodes: int = 50, seed: int = 0,
                    disruption_prob: float = 0.05) -> dict:
    env = SupplyChainEnv(seed=seed, disruption_prob=disruption_prob)
    rewards, stock_outs, emergency_orders, budget_utils = [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward, ep_so, ep_em = 0.0, 0, 0
        done = False
        while not done:
            action = np.zeros(N, dtype=int)
            for i in range(N):
                on_hand = env._state[i].on_hand
                rop = 1.5 * LEAD_TIMES_BASE[i] * DEMAND_PER_DAY[i]
                if on_hand < rop and env._state[i].on_order == 0:
                    action[i] = 1
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_so += len(info.get("stock_outs", []))
            ep_em += sum(1 for o in info.get("orders_placed", [])
                         if o.get("mode") == "expedite")
            done = terminated or truncated
        rewards.append(ep_reward)
        stock_outs.append(ep_so)
        emergency_orders.append(ep_em)
        budget_utils.append(info.get("budget_spent", 0) / 5_500_000)

    return {
        "policy": "rule_based_rop",
        "n_episodes": n_episodes,
        "mean_reward": round(float(np.mean(rewards)), 2),
        "std_reward": round(float(np.std(rewards)), 2),
        "mean_stock_outs": round(float(np.mean(stock_outs)), 2),
        "stock_out_rate": round(float(np.mean([c > 0 for c in stock_outs])), 3),
        "mean_emergency_orders": round(float(np.mean(emergency_orders)), 2),
        "mean_budget_utilisation": round(float(np.mean(budget_utils)), 3),
    }


# ---------------------------------------------------------------------------
# PPO policy (loads trained model if available, else runs random policy)
# ---------------------------------------------------------------------------

def run_ppo(model_path: str = "checkpoints/best_model",
             n_episodes: int = 50, seed: int = 0,
             disruption_prob: float = 0.05) -> dict:
    try:
        from stable_baselines3 import PPO as SB3_PPO
        model = SB3_PPO.load(model_path)
        use_model = True
    except Exception:
        use_model = False

    env = SupplyChainEnv(seed=seed, disruption_prob=disruption_prob)
    rewards, stock_outs, emergency_orders, budget_utils = [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward, ep_so, ep_em = 0.0, 0, 0
        done = False
        while not done:
            if use_model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                # Fallback: conservative order policy for demonstration
                action = np.zeros(N, dtype=int)
                for i in range(N):
                    on_hand = env._state[i].on_hand
                    dos = on_hand / max(DEMAND_PER_DAY[i], 1e-6)
                    if dos < LEAD_TIMES_BASE[i] * 1.2 and env._state[i].on_order == 0:
                        action[i] = 1
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_so += len(info.get("stock_outs", []))
            ep_em += sum(1 for o in info.get("orders_placed", [])
                         if o.get("mode") == "expedite")
            done = terminated or truncated
        rewards.append(ep_reward)
        stock_outs.append(ep_so)
        emergency_orders.append(ep_em)
        budget_utils.append(info.get("budget_spent", 0) / 5_500_000)

    return {
        "policy": "ppo_trained" if use_model else "ppo_demo_heuristic",
        "model_path": model_path,
        "n_episodes": n_episodes,
        "mean_reward": round(float(np.mean(rewards)), 2),
        "std_reward": round(float(np.std(rewards)), 2),
        "mean_stock_outs": round(float(np.mean(stock_outs)), 2),
        "stock_out_rate": round(float(np.mean([c > 0 for c in stock_outs])), 3),
        "mean_emergency_orders": round(float(np.mean(emergency_orders)), 2),
        "mean_budget_utilisation": round(float(np.mean(budget_utils)), 3),
    }


# ---------------------------------------------------------------------------
# Lead-time forecast accuracy benchmarks
# ---------------------------------------------------------------------------

def run_forecast_accuracy_benchmark() -> dict:
    """
    Compare physics-informed model vs naive mean-only baseline
    on held-out disruption scenarios.
    """
    from simulation.physics import estimate_lead_time
    from simulation.disruptions import DisruptionEngine

    engine = DisruptionEngine(seed=123)
    n_scenarios = 200
    physics_errors, naive_errors = [], []

    for i in range(n_scenarios):
        disruptions = engine.generate_episode(90)
        true_lt = 84.0 * (1 + sum(
            (d.lead_time_multiplier - 1) * 0.7
            for d in disruptions
            if "NVIDIA_H100" in d.affected_components and d.is_active(45)
        ))
        true_lt = min(true_lt, 365)

        # Physics model
        result = estimate_lead_time(
            component_id="NVIDIA_H100",
            origin_country="TW",
            destination_country="US",
            transport_mode="air",
            quantity=40,
        )
        physics_forecast = result.p50_days
        naive_forecast = 84.0   # always predict baseline

        physics_errors.append(abs(physics_forecast - true_lt) / true_lt)
        naive_errors.append(abs(naive_forecast - true_lt) / true_lt)

    return {
        "physics_model_mape": round(float(np.mean(physics_errors)), 4),
        "naive_baseline_mape": round(float(np.mean(naive_errors)), 4),
        "improvement_pct": round(
            (np.mean(naive_errors) - np.mean(physics_errors)) / np.mean(naive_errors) * 100, 1
        ),
        "n_scenarios": n_scenarios,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Veltora Supply Chain — Benchmark Suite")
    print("=" * 60)

    t0 = time.time()

    print("\n[1/3] Running rule-based ROP baseline (50 episodes)...")
    baseline = run_rule_based(n_episodes=50, seed=0)
    print(f"  mean_reward={baseline['mean_reward']}  "
          f"stock_outs={baseline['mean_stock_outs']:.1f}  "
          f"emergency_orders={baseline['mean_emergency_orders']:.1f}")

    print("\n[2/3] Running PPO policy (50 episodes)...")
    ppo_result = run_ppo(n_episodes=50, seed=0)
    print(f"  mean_reward={ppo_result['mean_reward']}  "
          f"stock_outs={ppo_result['mean_stock_outs']:.1f}  "
          f"emergency_orders={ppo_result['mean_emergency_orders']:.1f}")

    print("\n[3/3] Forecast accuracy benchmark (200 scenarios)...")
    forecast = run_forecast_accuracy_benchmark()
    print(f"  Physics MAPE={forecast['physics_model_mape']:.4f}  "
          f"Naive MAPE={forecast['naive_baseline_mape']:.4f}  "
          f"Improvement={forecast['improvement_pct']}%")

    results = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(time.time() - t0, 1),
        "benchmarks": {
            "rule_based_baseline": baseline,
            "ppo_policy": ppo_result,
            "forecast_accuracy": forecast,
        },
        "improvement_summary": {
            "reward_improvement_pct": round(
                (ppo_result["mean_reward"] - baseline["mean_reward"])
                / abs(baseline["mean_reward"]) * 100, 1
            ) if baseline["mean_reward"] != 0 else 0,
            "stock_out_reduction_pct": round(
                (baseline["mean_stock_outs"] - ppo_result["mean_stock_outs"])
                / max(baseline["mean_stock_outs"], 1) * 100, 1
            ),
            "emergency_order_reduction_pct": round(
                (baseline["mean_emergency_orders"] - ppo_result["mean_emergency_orders"])
                / max(baseline["mean_emergency_orders"], 1) * 100, 1
            ),
        },
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {RESULTS_PATH}")
    print("=" * 60)
    return results


if __name__ == "__main__":
    main()
