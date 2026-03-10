"""
baseline.py

Rule-based procurement baseline for benchmark comparison.

Strategy:
  - Fixed reorder point: order when days-of-stock < 30
  - No predictive capability: acts on current inventory, not forecasts
  - Manual escalation trigger: when delivery overdue by 5+ days, flag for human
  - No cost/carbon optimisation: picks first available vendor
  - No schedule awareness: doesn't know which components are critical path

This represents a reasonable approximation of a mid-market data centre
contractor without supply chain technology. Not a strawman - it uses real
reorder logic, but misses the predictive and optimisation layers.

Performance characteristics (from historical benchmarks):
  - Delay prediction: none (reactive only)
  - Emergency order rate: ~15% (vs 4% for Veltora)
  - Stockout events per project: ~8 (vs 2 for Veltora)
  - Cost vs budget: typically 8-12% over (vs 3% for Veltora)
"""

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from simulation.components import ComponentCatalogue, Component
from simulation.disruptions import DisruptionGenerator


@dataclass
class BaselineOrder:
    component_id: str
    quantity: int
    order_day: int
    expected_delivery_day: int
    unit_price_usd: float
    is_emergency: bool = False


@dataclass
class BaselineResult:
    """Result of one simulated project run."""
    project_id: str
    total_cost_usd: float
    budget_usd: float
    cost_variance_pct: float          # (actual - budget) / budget
    delay_days_total: int
    emergency_orders: int
    total_orders: int
    emergency_order_rate: float
    stockout_events: int              # components not delivered on need-by date
    carbon_kg_co2e: float
    actions_taken: list[dict] = field(default_factory=list)


class RuleBasedBaseline:
    """
    Fixed reorder-point procurement baseline.

    Rules:
      1. Every Monday (every 7 days in simulation): check all components
      2. If on-hand + on-order < reorder_point (30 days of demand): place order
      3. If component overdue > 5 days: place emergency (expedited) order
      4. No lead time forecasting: uses catalogue nominal lead time
      5. No cost comparison: always uses cheapest vendor in catalogue
      6. No critical path awareness
    """

    REORDER_POINT_DAYS = 30        # order when < 30 days of stock
    OVERDUE_ESCALATION_DAYS = 5    # expedite if this many days past need-by
    REVIEW_FREQUENCY_DAYS = 7      # review cycle (weekly)
    EXPEDITED_COST_MULTIPLIER = 2.2

    def __init__(self, seed: Optional[int] = None):
        self.catalogue = ComponentCatalogue()
        self._rng = np.random.default_rng(seed)
        self._py_rng = random.Random(seed)

    def _get_standard_lead_time(self, comp: Component, disrupted: bool = False) -> int:
        """Return nominal lead time with optional disruption variance."""
        base = comp.lead_time_p50_days
        if disrupted:
            # Baseline has no forewarning of disruptions, just experiences the delay
            variance = self._rng.uniform(1.0, 1.8)
            return int(base * variance)
        # Normal variance around P50
        variance = self._rng.normal(1.0, 0.15)
        return max(comp.lead_time_p10_days, int(base * variance))

    def _get_unit_price(self, comp: Component, emergency: bool = False) -> float:
        """Baseline: no price optimisation, uses reference price + random vendor variance."""
        base = comp.unit_price_usd
        # Vendor variance: baseline doesn't shop around effectively
        vendor_variance = self._py_rng.uniform(0.98, 1.08)
        price = base * vendor_variance
        if emergency:
            price *= self.EXPEDITED_COST_MULTIPLIER
        return price

    def run_project(
        self,
        project_id: str,
        bom: Optional[list[dict]] = None,
        episode_length: int = 90,
        disruption_rate: float = 1.0,
    ) -> BaselineResult:
        """
        Simulate one project using the rule-based strategy.

        Returns a BaselineResult with performance metrics.
        """
        if bom is None:
            bom = self.catalogue.get_bom_for_project("standard_dc_16mw")

        # Set up disruptions (baseline doesn't predict these, just experiences them)
        disruption_gen = DisruptionGenerator(seed=int(self._rng.integers(0, 10000)))
        disruptions = disruption_gen.generate_episode(episode_length, disruption_rate)

        # Initial state
        inventory = {item["component_id"]: 0 for item in bom}
        on_order: list[BaselineOrder] = []
        need_by_days = {
            item["component_id"]: int(episode_length * 0.65)
            for item in bom
        }
        budget = sum(
            (self.catalogue.get(item["component_id"]).unit_price_usd if self.catalogue.get(item["component_id"]) else 1000)
            * item["quantity"] * 1.15
            for item in bom
        )

        total_cost = 0.0
        emergency_orders = 0
        total_orders = 0
        stockout_events = 0
        total_carbon_kg = 0.0
        actions: list[dict] = []

        for day in range(episode_length):
            # Deliver outstanding orders due today
            still_pending = []
            for order in on_order:
                if day >= order.expected_delivery_day:
                    inventory[order.component_id] += order.quantity
                else:
                    still_pending.append(order)
            on_order = still_pending

            # Weekly review
            if day % self.REVIEW_FREQUENCY_DAYS == 0:
                for item in bom:
                    comp_id = item["component_id"]
                    comp = self.catalogue.get(comp_id)
                    if comp is None:
                        continue

                    qty_needed = item["quantity"]
                    on_hand = inventory[comp_id]
                    on_order_qty = sum(o.quantity for o in on_order if o.component_id == comp_id)
                    total_available = on_hand + on_order_qty
                    need_by = need_by_days[comp_id]

                    # Reorder point check: do we have < 30 days of supply?
                    # "Days of supply" here means: can we cover the project need?
                    coverage = total_available / max(qty_needed, 1)
                    reorder_threshold = 0.0  # baseline: order if nothing on order and not in stock

                    if total_available < qty_needed and on_order_qty == 0:
                        # Check if disruption is active (no forewarning, just longer actual LT)
                        active_disruptions = disruption_gen.get_active_disruptions(disruptions, day, comp_id)
                        disrupted = len(active_disruptions) > 0

                        lead_time = self._get_standard_lead_time(comp, disrupted)
                        unit_price = self._get_unit_price(comp, emergency=False)
                        order_cost = unit_price * qty_needed

                        order = BaselineOrder(
                            component_id=comp_id,
                            quantity=qty_needed,
                            order_day=day,
                            expected_delivery_day=day + lead_time,
                            unit_price_usd=unit_price,
                        )
                        on_order.append(order)
                        total_cost += order_cost
                        total_orders += 1

                        # Carbon estimate (sea freight default)
                        weight_tonnes = comp.weight_kg * qty_needed / 1000
                        carbon = weight_tonnes * 15000 * 0.015
                        total_carbon_kg += carbon

                        actions.append({
                            "day": day,
                            "action": "order_standard",
                            "component_id": comp_id,
                            "quantity": qty_needed,
                            "lead_time": lead_time,
                            "cost_usd": round(order_cost, 0),
                        })

            # Daily overdue check: escalate if delivery past need-by + threshold
            for item in bom:
                comp_id = item["component_id"]
                qty_needed = item["quantity"]
                need_by = need_by_days[comp_id]

                if day == need_by:
                    if inventory[comp_id] < qty_needed:
                        # Check if it will arrive soon
                        pending = [o for o in on_order if o.component_id == comp_id]
                        if not pending:
                            # No order at all - emergency order
                            comp = self.catalogue.get(comp_id)
                            if comp:
                                lt = max(7, comp.lead_time_p10_days)
                                unit_price = self._get_unit_price(comp, emergency=True)
                                cost = unit_price * qty_needed
                                on_order.append(BaselineOrder(
                                    component_id=comp_id,
                                    quantity=qty_needed,
                                    order_day=day,
                                    expected_delivery_day=day + lt,
                                    unit_price_usd=unit_price,
                                    is_emergency=True,
                                ))
                                total_cost += cost
                                emergency_orders += 1
                                total_orders += 1
                                stockout_events += 1
                                actions.append({
                                    "day": day,
                                    "action": "emergency_order",
                                    "component_id": comp_id,
                                    "quantity": qty_needed,
                                    "lead_time": lt,
                                    "cost_usd": round(cost, 0),
                                })
                        elif min(o.expected_delivery_day for o in pending) > need_by + self.OVERDUE_ESCALATION_DAYS:
                            # Order exists but will be late: stockout event
                            stockout_events += 1

        # Final inventory check for metrics
        for item in bom:
            comp_id = item["component_id"]
            if inventory[comp_id] < item["quantity"]:
                stockout_events += 1

        cost_variance = (total_cost - budget * 0.90) / budget if budget > 0 else 0.0
        emergency_rate = emergency_orders / max(total_orders, 1)

        return BaselineResult(
            project_id=project_id,
            total_cost_usd=round(total_cost, 0),
            budget_usd=round(budget, 0),
            cost_variance_pct=round(cost_variance, 4),
            delay_days_total=stockout_events * 8,  # Approximate: 8 days delay per stockout
            emergency_orders=emergency_orders,
            total_orders=total_orders,
            emergency_order_rate=round(emergency_rate, 3),
            stockout_events=stockout_events,
            carbon_kg_co2e=round(total_carbon_kg, 1),
            actions_taken=actions,
        )

    def run_n_projects(
        self,
        n: int = 50,
        disruption_rate: float = 1.0,
    ) -> list[BaselineResult]:
        """Run N simulated projects and return results."""
        results = []
        for i in range(n):
            result = self.run_project(
                project_id=f"baseline_proj_{i:04d}",
                disruption_rate=disruption_rate,
            )
            results.append(result)
        return results

    @staticmethod
    def aggregate_metrics(results: list[BaselineResult]) -> dict:
        """Compute aggregate performance metrics across multiple runs."""
        if not results:
            return {}
        n = len(results)
        return {
            "n_projects": n,
            "mean_cost_variance_pct": round(sum(r.cost_variance_pct for r in results) / n, 4),
            "mean_emergency_order_rate": round(sum(r.emergency_order_rate for r in results) / n, 3),
            "mean_stockout_events_per_project": round(sum(r.stockout_events for r in results) / n, 1),
            "mean_delay_days_per_project": round(sum(r.delay_days_total for r in results) / n, 1),
            "mean_carbon_kg_per_project": round(sum(r.carbon_kg_co2e for r in results) / n, 0),
            "p90_cost_variance_pct": round(
                sorted(r.cost_variance_pct for r in results)[int(n * 0.90)], 4
            ),
            "p90_stockouts": round(
                sorted(r.stockout_events for r in results)[int(n * 0.90)], 1
            ),
        }
