"""
DisruptionEngine — Generates realistic stochastic supply-chain disruption scenarios
for simulation, stress-testing, and RL environment augmentation.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Generator

import numpy as np


class DisruptionType(Enum):
    FAB_CAPACITY_CRUNCH   = auto()
    GEOPOLITICAL_TENSION  = auto()
    NATURAL_DISASTER      = auto()
    PORT_CONGESTION       = auto()
    LOGISTICS_STRIKE      = auto()
    QUALITY_RECALL        = auto()
    DEMAND_SPIKE          = auto()
    CURRENCY_SHOCK        = auto()


@dataclass
class Disruption:
    disruption_id: str
    disruption_type: DisruptionType
    affected_components: list[str]
    severity: float            # 0-1
    onset_day: int
    duration_days: int
    lead_time_multiplier: float
    demand_multiplier: float
    price_multiplier: float
    description: str

    @property
    def end_day(self) -> int:
        return self.onset_day + self.duration_days

    def is_active(self, day: int) -> bool:
        return self.onset_day <= day < self.end_day

    def lead_time_impact(self, day: int) -> float:
        """Return lead time multiplier with ramp-up and ramp-down."""
        if not self.is_active(day):
            return 1.0
        progress = (day - self.onset_day) / max(self.duration_days, 1)
        # Gaussian envelope: peaks at midpoint
        envelope = math.exp(-8 * (progress - 0.5) ** 2)
        return 1.0 + (self.lead_time_multiplier - 1.0) * envelope


# ---------------------------------------------------------------------------
# Disruption generators
# ---------------------------------------------------------------------------

_DISRUPTION_TEMPLATES: dict[DisruptionType, dict] = {
    DisruptionType.FAB_CAPACITY_CRUNCH: {
        "affected": ["NVIDIA_H100", "Marvell_ASIC_88X9140"],
        "severity_range": (0.4, 0.9),
        "duration_range": (60, 180),
        "lt_mult_range": (1.5, 3.5),
        "demand_mult": 1.0,
        "price_mult_range": (1.1, 1.8),
        "prob_per_quarter": 0.15,
    },
    DisruptionType.GEOPOLITICAL_TENSION: {
        "affected": ["NVIDIA_H100", "Marvell_ASIC_88X9140", "Vishay_Cap_10uF"],
        "severity_range": (0.3, 1.0),
        "duration_range": (30, 365),
        "lt_mult_range": (1.2, 4.0),
        "demand_mult": 1.0,
        "price_mult_range": (1.1, 2.5),
        "prob_per_quarter": 0.20,
    },
    DisruptionType.NATURAL_DISASTER: {
        "affected": ["NVIDIA_H100", "Amphenol_QSFP_400G"],
        "severity_range": (0.5, 1.0),
        "duration_range": (14, 90),
        "lt_mult_range": (1.5, 5.0),
        "demand_mult": 0.9,
        "price_mult_range": (1.2, 2.0),
        "prob_per_quarter": 0.05,
    },
    DisruptionType.PORT_CONGESTION: {
        "affected": ["Amphenol_QSFP_400G", "Vertiv_PDU_24kW", "Vishay_Cap_10uF"],
        "severity_range": (0.2, 0.7),
        "duration_range": (7, 45),
        "lt_mult_range": (1.1, 2.0),
        "demand_mult": 1.0,
        "price_mult_range": (1.0, 1.3),
        "prob_per_quarter": 0.30,
    },
    DisruptionType.LOGISTICS_STRIKE: {
        "affected": ["Vertiv_PDU_24kW", "Amphenol_QSFP_400G"],
        "severity_range": (0.3, 0.8),
        "duration_range": (7, 21),
        "lt_mult_range": (1.3, 2.5),
        "demand_mult": 1.0,
        "price_mult_range": (1.0, 1.2),
        "prob_per_quarter": 0.10,
    },
    DisruptionType.QUALITY_RECALL: {
        "affected": ["Vishay_Cap_10uF", "Amphenol_QSFP_400G"],
        "severity_range": (0.4, 0.9),
        "duration_range": (14, 60),
        "lt_mult_range": (1.0, 1.0),
        "demand_mult": 0.0,                         # supply stops entirely
        "price_mult_range": (0.7, 0.9),             # spot price crashes then spikes
        "prob_per_quarter": 0.04,
    },
    DisruptionType.DEMAND_SPIKE: {
        "affected": ["NVIDIA_H100", "Marvell_ASIC_88X9140"],
        "severity_range": (0.3, 0.8),
        "duration_range": (30, 90),
        "lt_mult_range": (1.1, 1.5),
        "demand_mult": 1.5,
        "price_mult_range": (1.1, 1.6),
        "prob_per_quarter": 0.25,
    },
    DisruptionType.CURRENCY_SHOCK: {
        "affected": ["NVIDIA_H100", "Marvell_ASIC_88X9140", "Samsung_HBM3"],
        "severity_range": (0.1, 0.6),
        "duration_range": (30, 120),
        "lt_mult_range": (1.0, 1.0),
        "demand_mult": 1.0,
        "price_mult_range": (0.85, 1.25),
        "prob_per_quarter": 0.20,
    },
}


class DisruptionEngine:
    """
    Generates stochastic disruption sequences for a simulation episode.

    Usage:
        engine = DisruptionEngine(seed=42)
        episode_disruptions = engine.generate_episode(episode_days=90)
        for d in episode_disruptions:
            if d.is_active(current_day):
                lead_time *= d.lead_time_impact(current_day)
    """

    def __init__(self, seed: int | None = None,
                 intensity_multiplier: float = 1.0):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.intensity = intensity_multiplier
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"DIS-{self._counter:04d}"

    def generate_episode(self, episode_days: int = 90) -> list[Disruption]:
        """Sample all disruptions that occur during a training episode."""
        disruptions = []
        for dtype, template in _DISRUPTION_TEMPLATES.items():
            prob = template["prob_per_quarter"] * self.intensity
            if self.rng.random() < prob:
                onset = self.rng.randint(0, episode_days - 1)
                dur_min, dur_max = template["duration_range"]
                duration = self.rng.randint(dur_min, min(dur_max, episode_days - onset))
                sev_min, sev_max = template["severity_range"]
                severity = self.rng.uniform(sev_min, sev_max)
                lt_min, lt_max = template["lt_mult_range"]
                lt_mult = self.rng.uniform(lt_min, lt_max)
                pm_min, pm_max = template["price_mult_range"]
                price_mult = self.rng.uniform(pm_min, pm_max)
                disruptions.append(Disruption(
                    disruption_id=self._next_id(),
                    disruption_type=dtype,
                    affected_components=template["affected"],
                    severity=round(severity, 3),
                    onset_day=onset,
                    duration_days=duration,
                    lead_time_multiplier=round(lt_mult, 3),
                    demand_multiplier=template["demand_mult"],
                    price_multiplier=round(price_mult, 3),
                    description=f"{dtype.name.replace('_', ' ').title()} "
                                f"(severity={severity:.2f}, duration={duration}d)",
                ))
        disruptions.sort(key=lambda d: d.onset_day)
        return disruptions

    def stream(self, episode_days: int = 90) -> Generator[Disruption, None, None]:
        """Yield disruptions sorted by onset day."""
        for d in self.generate_episode(episode_days):
            yield d

    def monte_carlo_lead_time(self, base_lt: float, component: str,
                               disruptions: list[Disruption],
                               day: int, n_samples: int = 1000) -> dict:
        """
        Monte Carlo integration of disruption impacts on lead time.
        Returns P10/P50/P90 of the disturbed lead time distribution.
        """
        active = [d for d in disruptions
                  if d.is_active(day) and component in d.affected_components]

        samples = []
        for _ in range(n_samples):
            lt = self.np_rng.lognormal(
                mean=math.log(base_lt) - 0.04,   # σ=0.2 → variance term
                sigma=0.20,
            )
            for d in active:
                # Stochastic severity realisation
                sev = self.np_rng.uniform(0.5, 1.0) * d.severity
                lt_impact = 1.0 + (d.lead_time_multiplier - 1.0) * sev
                lt *= lt_impact
            samples.append(lt)

        samples_arr = np.array(samples)
        return {
            "component": component,
            "day": day,
            "n_active_disruptions": len(active),
            "base_lead_time_days": base_lt,
            "p10_days": round(float(np.percentile(samples_arr, 10)), 1),
            "p50_days": round(float(np.percentile(samples_arr, 50)), 1),
            "p90_days": round(float(np.percentile(samples_arr, 90)), 1),
            "mean_days": round(float(samples_arr.mean()), 1),
            "active_disruption_ids": [d.disruption_id for d in active],
        }
