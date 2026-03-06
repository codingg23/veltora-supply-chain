"""Tests for physics-informed lead time model."""
from __future__ import annotations

import math
import pytest

from simulation.physics import (
    queue_wait_days,
    manufacture_days,
    transport_days,
    customs_days,
    estimate_lead_time,
    port_congestion_factor,
)


class TestQueueWait:
    def test_zero_utilisation(self):
        assert queue_wait_days(arrival_rate=0, service_rate=1) == 0.0

    def test_low_utilisation(self):
        # rho = 0.5 → W = 0.5 / (1 * 0.5) = 1 day
        result = queue_wait_days(arrival_rate=0.5, service_rate=1.0)
        assert abs(result - 1.0) < 0.01

    def test_high_utilisation_blows_up(self):
        # rho approaching 1 → very large wait
        result = queue_wait_days(arrival_rate=0.99, service_rate=1.0)
        assert result > 50

    def test_saturated_capped(self):
        # rho >= 1 → capped at 180 days
        result = queue_wait_days(arrival_rate=1.5, service_rate=1.0)
        assert result == 180.0


class TestManufactureDays:
    def test_perfect_yield(self):
        result = manufacture_days(units=100, yield_rate=1.0, throughput_per_day=100)
        assert result == 1.0

    def test_low_yield_increases_time(self):
        low_yield = manufacture_days(units=100, yield_rate=0.5, throughput_per_day=100)
        high_yield = manufacture_days(units=100, yield_rate=1.0, throughput_per_day=100)
        assert low_yield > high_yield

    def test_proportional_to_units(self):
        r1 = manufacture_days(units=100, yield_rate=0.9, throughput_per_day=50)
        r2 = manufacture_days(units=200, yield_rate=0.9, throughput_per_day=50)
        assert abs(r2 / r1 - 2.0) < 0.01


class TestTransportDays:
    def test_air_faster_than_sea(self):
        air = transport_days(distance_km=10_000, mode="air")
        sea = transport_days(distance_km=10_000, mode="sea")
        assert air < sea

    def test_speed_constants(self):
        # Air speed ~5800 km/day → 10000km ≈ 1.72d
        result = transport_days(distance_km=5800, mode="air")
        assert abs(result - 1.0) < 0.1

    def test_unknown_mode_raises(self):
        with pytest.raises((KeyError, ValueError)):
            transport_days(distance_km=1000, mode="teleport")


class TestCustomsDays:
    def test_low_risk_lane(self):
        # DE→US should be lower than CN→US
        result_de = customs_days(origin="DE", destination="US", n_samples=200)
        result_cn = customs_days(origin="CN", destination="US", n_samples=200)
        assert result_de < result_cn

    def test_positive_duration(self):
        result = customs_days(origin="TW", destination="US", n_samples=100)
        assert result > 0

    def test_returns_float(self):
        result = customs_days(origin="KR", destination="GB", n_samples=50)
        assert isinstance(result, float)


class TestEstimateLeadTime:
    def test_returns_three_percentiles(self):
        result = estimate_lead_time(
            component_id="NVIDIA_H100",
            origin_country="TW",
            destination_country="US",
            transport_mode="air",
            quantity=10,
        )
        assert result.p10_days < result.p50_days < result.p90_days

    def test_air_p50_less_than_sea_p50(self):
        air = estimate_lead_time("NVIDIA_H100", "TW", "US", "air", 10)
        sea = estimate_lead_time("NVIDIA_H100", "TW", "US", "sea", 10)
        assert air.p50_days < sea.p50_days

    def test_high_utilisation_fab_increases_lead_time(self):
        """TSMC at 91% util → longer queue than a hypothetical 50% util fab."""
        result = estimate_lead_time("NVIDIA_H100", "TW", "US", "air", 10)
        # P50 should be well above 30 days due to fab queue
        assert result.p50_days > 30

    def test_uncertainty_quantified(self):
        result = estimate_lead_time("NVIDIA_H100", "TW", "US", "sea", 50)
        # P90 - P10 should be meaningful (>5 days)
        assert result.p90_days - result.p10_days > 5


class TestPortCongestion:
    def test_congested_port_above_one(self):
        factor = port_congestion_factor("Shanghai")
        assert factor > 1.0

    def test_uncongested_port_near_one(self):
        factor = port_congestion_factor("Hamburg")
        assert 0.95 <= factor <= 1.2

    def test_unknown_port_defaults(self):
        factor = port_congestion_factor("Atlantis")
        assert factor >= 1.0
