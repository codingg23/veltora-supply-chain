"""
physics.py

Physics-informed lead time model.

The key insight: manufacturing and logistics lead times are not arbitrary numbers.
They're constrained by physical production rates, transport physics, and queue
dynamics. Pure ML models trained on historical data can predict plausible but
physically impossible outcomes. Physics constraints prevent this.

Model:
    lead_time_total = t_queue + t_manufacture + t_test + t_transport + t_customs

Each component is modelled separately. The physics give us hard lower bounds
that ML residuals are added to for uncertainty quantification.

References:
- Hopp & Spearman, "Factory Physics" (queue theory for manufacturing)
- IATA cargo transit time models
- WTO customs clearance studies by HS code
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional


# Transport mode speeds (km/day, realistic including handling/loading)
TRANSPORT_SPEEDS = {
    "air":      5_800,   # ~0.8 x commercial aircraft speed with handling
    "sea":        480,   # ~20 knots average container ship
    "rail":       700,   # intermodal rail
    "road":       500,   # long-haul trucking
}

# Customs clearance baseline days by trade lane (median historical)
CUSTOMS_BASELINE = {
    ("CN", "US"): 4.2,
    ("CN", "EU"): 3.8,
    ("CN", "AU"): 3.1,
    ("TW", "US"): 3.5,
    ("US", "EU"): 1.8,
    ("EU", "US"): 1.9,
    ("JP", "US"): 2.8,
    ("KR", "US"): 3.0,
}
CUSTOMS_DEFAULT = 3.5  # days, if trade lane not in table


@dataclass
class ComponentSpec:
    component_id: str
    category: str           # "semiconductor", "cable", "mechanical", "cooling"
    origin_country: str
    destination_country: str

    # Manufacturing parameters
    lot_size: int = 1
    production_rate_per_day: float = 1.0   # units per day at the fab
    yield_rate: float = 0.95               # fraction of units passing QA
    fab_utilisation: float = 0.75          # current fab load (0-1)

    # Transport
    transport_mode: str = "sea"
    origin_city: str = "Shenzhen"
    destination_city: str = "Dallas"
    distance_km: float = 14_000.0

    # Testing requirements
    requires_burn_in: bool = False
    burn_in_days: float = 0.0
    acceptance_test_days: float = 0.5


@dataclass
class LeadTimeEstimate:
    component_id: str
    t_queue_days: float
    t_manufacture_days: float
    t_test_days: float
    t_transport_days: float
    t_customs_days: float
    total_p50_days: float    # median estimate
    total_p90_days: float    # 90th percentile (for risk planning)
    total_p10_days: float    # 10th percentile (best case)
    physics_minimum_days: float  # hard lower bound from physics


def queue_wait_days(utilisation: float, service_rate_per_day: float) -> float:
    """
    M/M/1 queue approximation for manufacturing wait time.

    At low utilisation, queue is short. As utilisation approaches 1.0,
    wait time grows explosively. This is why fabs at 95% utilisation
    have wildly unpredictable lead times.

    W = rho / (mu * (1 - rho))
    where rho = utilisation, mu = service rate
    """
    rho = float(np.clip(utilisation, 0.0, 0.98))
    if rho < 0.05:
        return 0.5  # negligible queue
    return rho / (service_rate_per_day * (1.0 - rho))


def manufacture_days(
    lot_size: int,
    production_rate: float,
    yield_rate: float,
) -> float:
    """
    Time to manufacture lot_size good units.

    Need to produce ceil(lot_size / yield_rate) raw units to get
    lot_size passing units. Production rate is in good units per day
    before yield adjustment.
    """
    units_to_produce = math.ceil(lot_size / max(yield_rate, 0.01))
    return units_to_produce / max(production_rate, 0.001)


def transport_days(
    distance_km: float,
    mode: str,
    congestion_factor: float = 1.0,
) -> float:
    """
    Transit time from physics of the transport mode.

    congestion_factor: multiplier > 1.0 for port congestion, weather, etc.
    The physics give us a hard floor - air freight from Shanghai to LA
    cannot take less than ~1.5 days regardless of conditions.
    """
    speed = TRANSPORT_SPEEDS.get(mode, TRANSPORT_SPEEDS["sea"])
    base_days = distance_km / speed
    return base_days * max(congestion_factor, 1.0)


def customs_days(
    origin: str,
    destination: str,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Customs clearance time. Stochastic - sampled from log-normal
    fitted to historical clearing times for the trade lane.
    """
    if rng is None:
        rng = np.random.default_rng()

    baseline = CUSTOMS_BASELINE.get((origin, destination), CUSTOMS_DEFAULT)
    # log-normal with sigma=0.4 gives realistic tail for delays
    sample = rng.lognormal(mean=math.log(baseline), sigma=0.4)
    return float(np.clip(sample, 0.5, baseline * 8))


def estimate_lead_time(
    spec: ComponentSpec,
    congestion_factor: float = 1.0,
    n_monte_carlo: int = 1000,
    seed: Optional[int] = None,
) -> LeadTimeEstimate:
    """
    Full lead time estimate for a component.

    Uses Monte Carlo over the stochastic elements (customs, yield variation,
    queue wait) while keeping the deterministic physics floors.
    """
    rng = np.random.default_rng(seed)

    # deterministic physics floors
    t_queue_base = queue_wait_days(spec.fab_utilisation, spec.production_rate_per_day)
    t_mfg_base = manufacture_days(spec.lot_size, spec.production_rate_per_day, spec.yield_rate)
    t_transport_base = transport_days(spec.distance_km, spec.transport_mode, congestion_factor)
    t_test_base = spec.acceptance_test_days + spec.burn_in_days

    physics_minimum = t_mfg_base + t_transport_base + t_test_base

    # Monte Carlo for uncertainty
    totals = []
    for _ in range(n_monte_carlo):
        # queue wait has high variance - exponential conditional on M/M/1 mean
        t_q = rng.exponential(max(t_queue_base, 0.1))

        # manufacture time varies with yield (binomial yield sampling)
        actual_yield = rng.beta(
            spec.yield_rate * 20, (1 - spec.yield_rate) * 20
        )  # beta concentrates near spec yield_rate
        t_m = manufacture_days(spec.lot_size, spec.production_rate_per_day, actual_yield)

        # transport with weather/congestion noise
        cong = congestion_factor * rng.lognormal(0, 0.2)
        t_tr = transport_days(spec.distance_km, spec.transport_mode, cong)

        # customs
        t_cu = customs_days(spec.origin_country, spec.destination_country, rng)

        totals.append(t_q + t_m + t_test_base + t_tr + t_cu)

    totals = np.array(totals)

    return LeadTimeEstimate(
        component_id=spec.component_id,
        t_queue_days=round(t_queue_base, 1),
        t_manufacture_days=round(t_mfg_base, 1),
        t_test_days=round(t_test_base, 1),
        t_transport_days=round(t_transport_base, 1),
        t_customs_days=round(float(np.median([customs_days(spec.origin_country, spec.destination_country, rng) for _ in range(100)])), 1),
        total_p50_days=round(float(np.percentile(totals, 50)), 1),
        total_p90_days=round(float(np.percentile(totals, 90)), 1),
        total_p10_days=round(float(np.percentile(totals, 10)), 1),
        physics_minimum_days=round(physics_minimum, 1),
    )


# ---------------------------------------------------------------------------
# Disruption impact models
# ---------------------------------------------------------------------------

def port_congestion_factor(ships_waiting: int, berths: int) -> float:
    """
    Estimate port congestion multiplier from queue length.
    Uses M/D/c queue approximation (deterministic service, c berths).
    """
    rho = min(ships_waiting / max(berths * 10, 1), 0.99)
    return 1.0 + rho ** 2 * 3.0   # up to 4x at extreme congestion


def fab_utilisation_from_orders(
    outstanding_orders: int,
    fab_capacity_per_month: int,
    backlog_months: float = 0.0,
) -> float:
    """
    Estimate fab utilisation from order pipeline and known backlog.
    Saturates at 0.97 - fabs hold some headroom for priority orders.
    """
    load = (outstanding_orders / max(fab_capacity_per_month, 1)) + (backlog_months / 6.0)
    return float(np.clip(load, 0.0, 0.97))


def geopolitical_delay_days(
    risk_score: float,  # 0-1, 0=no risk, 1=active conflict/embargo
    baseline_transit_days: float,
) -> float:
    """
    Additional days from geopolitical disruption to a trade lane.
    A risk_score of 0.5 (elevated tension, no embargo) adds ~20% to transit.
    A risk_score of 0.9 (near-embargo, rerouting required) can double transit.
    """
    if risk_score < 0.1:
        return 0.0
    multiplier = 1.0 + risk_score ** 1.5 * 2.5
    return baseline_transit_days * (multiplier - 1.0)
