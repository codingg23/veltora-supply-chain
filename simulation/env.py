"""
SupplyChainEnv — Gymnasium environment for training procurement RL policies.

State space:
  - inventory levels for N components (normalised)
  - days-of-stock for each component
  - lead-time estimate for each component (P50)
  - current risk score for each component
  - budget utilisation (0-1)
  - days into quarter (0-1)

Action space (MultiDiscrete):
  For each component: {0: no_order, 1: order_eoq, 2: order_2x_eoq, 3: expedite}

Reward:
  +1.0 per day without a stock-out
  -10.0 per stock-out event
  -0.05 per dollar of holding cost (scaled)
  -5.0 per emergency order
  +0.5 for sustainability bonus (sea vs air)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    # Provide stub for type checking
    class gym:  # type: ignore
        class Env:
            pass
    class spaces:  # type: ignore
        pass


COMPONENTS = [
    "NVIDIA_H100",
    "Marvell_ASIC_88X9140",
    "Vertiv_PDU_24kW",
    "Amphenol_QSFP_400G",
    "Vishay_Cap_10uF",
]

N = len(COMPONENTS)

# Component parameters
DEMAND_PER_DAY  = np.array([0.8,  3.2,  0.3,   18.0,  800.0])
LEAD_TIMES_BASE = np.array([84.0, 56.0, 28.0,  14.0,    7.0])   # days
UNIT_COSTS      = np.array([31000, 1200, 4800,  320.0,   0.12])
INITIAL_STOCK   = np.array([12.0,  45.0,  6.0, 220.0, 12000.0])
EOQ_UNITS       = np.array([40.0, 100.0,  8.0, 500.0, 50000.0])

HOLDING_COST_RATE = 0.0003   # per unit per day (as fraction of unit cost)
EPISODE_DAYS = 90            # one quarter
BUDGET_LIMIT = 5_500_000.0   # Q2 budget


@dataclass
class ComponentState:
    on_hand: float
    on_order: float
    lead_time_remaining: float      # days until pending order arrives
    lead_time_estimate: float       # current P50 estimate
    risk_score: float               # 0-1


class SupplyChainEnv(gym.Env):
    """
    Multi-component inventory management environment with stochastic lead times,
    disruption events, and budget constraints.

    Compatible with stable-baselines3 / Gymnasium 0.29+.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, seed: int | None = None, disruption_prob: float = 0.05):
        super().__init__()
        if not GYM_AVAILABLE:
            raise ImportError("gymnasium is required: pip install gymnasium")

        self.disruption_prob = disruption_prob
        self.rng = np.random.default_rng(seed)

        # Observation: [on_hand_norm, dos_norm, lead_time_norm, risk,
        #               budget_util, day_norm] × N + [budget_util, day_norm]
        obs_dim = N * 4 + 2
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Action: for each component → {0,1,2,3}
        self.action_space = spaces.MultiDiscrete([4] * N)

        self._state: list[ComponentState] = []
        self._day = 0
        self._budget_spent = 0.0
        self._total_reward = 0.0

    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._day = 0
        self._budget_spent = 0.0
        self._total_reward = 0.0

        self._state = [
            ComponentState(
                on_hand=INITIAL_STOCK[i],
                on_order=0.0,
                lead_time_remaining=0.0,
                lead_time_estimate=LEAD_TIMES_BASE[i],
                risk_score=0.2,
            )
            for i in range(N)
        ]
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        reward = 0.0
        info: dict[str, Any] = {"stock_outs": [], "orders_placed": [], "disruptions": []}

        # 1. Apply actions (place orders)
        for i, act in enumerate(action):
            qty = 0.0
            if act == 0:
                qty = 0.0
            elif act == 1:
                qty = EOQ_UNITS[i]
            elif act == 2:
                qty = EOQ_UNITS[i] * 2
            elif act == 3:                          # expedite (air freight — penalty)
                qty = EOQ_UNITS[i]
                reward -= 5.0                       # emergency order penalty
                info["orders_placed"].append({"component": COMPONENTS[i],
                                              "qty": qty, "mode": "expedite"})

            if qty > 0:
                cost = qty * UNIT_COSTS[i]
                if self._budget_spent + cost <= BUDGET_LIMIT:
                    self._budget_spent += cost
                    self._state[i].on_order += qty
                    lt = self._sample_lead_time(i)
                    self._state[i].lead_time_remaining = lt
                    if act != 3:
                        info["orders_placed"].append({
                            "component": COMPONENTS[i], "qty": qty, "mode": "normal"
                        })

        # 2. Advance time: arrivals
        for i in range(N):
            s = self._state[i]
            if s.lead_time_remaining > 0:
                s.lead_time_remaining -= 1
                if s.lead_time_remaining <= 0 and s.on_order > 0:
                    s.on_hand += s.on_order
                    s.on_order = 0.0

        # 3. Demand consumption
        for i in range(N):
            demand = self._sample_demand(i)
            if self._state[i].on_hand >= demand:
                self._state[i].on_hand -= demand
                reward += 1.0                       # satisfied demand
            else:
                # Stock-out
                unfilled = demand - self._state[i].on_hand
                self._state[i].on_hand = 0.0
                reward -= 10.0 * (unfilled / demand)
                info["stock_outs"].append(COMPONENTS[i])

        # 4. Holding cost penalty
        for i in range(N):
            holding = self._state[i].on_hand * UNIT_COSTS[i] * HOLDING_COST_RATE
            reward -= holding * 0.00005             # scaled

        # 5. Random disruptions
        for i in range(N):
            if self.rng.random() < self.disruption_prob:
                delay = self.rng.integers(7, 30)
                self._state[i].lead_time_remaining += delay
                self._state[i].risk_score = min(1.0, self._state[i].risk_score + 0.15)
                info["disruptions"].append({"component": COMPONENTS[i], "delay_days": int(delay)})
            else:
                self._state[i].risk_score = max(0.1, self._state[i].risk_score - 0.02)

        self._day += 1
        terminated = self._day >= EPISODE_DAYS
        truncated = False

        self._total_reward += reward
        info["budget_spent"] = self._budget_spent
        info["budget_remaining"] = BUDGET_LIMIT - self._budget_spent
        info["day"] = self._day

        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------

    def _sample_lead_time(self, i: int) -> float:
        """Log-normal around base lead time with σ = 20%."""
        base = LEAD_TIMES_BASE[i]
        sigma = 0.20
        return float(self.rng.lognormal(
            mean=math.log(base) - sigma ** 2 / 2, sigma=sigma
        ))

    def _sample_demand(self, i: int) -> float:
        """Poisson demand."""
        return float(self.rng.poisson(DEMAND_PER_DAY[i]))

    def _get_obs(self) -> np.ndarray:
        obs = []
        max_stock = INITIAL_STOCK * 5

        for i, s in enumerate(self._state):
            # Normalised on-hand (0-1)
            obs.append(float(np.clip(s.on_hand / max_stock[i], 0, 1)))
            # Days of stock (0-1, cap at 120d)
            dos = s.on_hand / max(DEMAND_PER_DAY[i], 1e-6)
            obs.append(float(np.clip(dos / 120, 0, 1)))
            # Lead time estimate (0-1, cap at 120d)
            obs.append(float(np.clip(s.lead_time_estimate / 120, 0, 1)))
            # Risk score
            obs.append(float(np.clip(s.risk_score, 0, 1)))

        # Budget utilisation
        obs.append(float(np.clip(self._budget_spent / BUDGET_LIMIT, 0, 1)))
        # Day normalised
        obs.append(float(self._day / EPISODE_DAYS))

        return np.array(obs, dtype=np.float32)

    def render(self, mode: str = "human") -> str | None:
        lines = [f"Day {self._day}/{EPISODE_DAYS}  Budget: ${self._budget_spent:,.0f}"]
        for i, s in enumerate(self._state):
            dos = s.on_hand / max(DEMAND_PER_DAY[i], 1e-6)
            lines.append(
                f"  {COMPONENTS[i]:<30} on_hand={s.on_hand:>8.0f}  "
                f"DoS={dos:>5.1f}d  risk={s.risk_score:.2f}"
            )
        output = "\n".join(lines)
        if mode == "human":
            print(output)
        return output
