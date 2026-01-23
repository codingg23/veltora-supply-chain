"""
ProcurementOptimizer — Agent 2
Decides *when* and *how much* to buy, using multi-echelon inventory theory
and a learned policy from the Gymnasium simulation environment.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Any

import anthropic

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5")

# ---------------------------------------------------------------------------
# Inventory theory helpers
# ---------------------------------------------------------------------------

def economic_order_quantity(demand_per_day: float, order_cost: float,
                             holding_cost_per_unit_day: float) -> float:
    """Classic EOQ: sqrt(2 * D * K / h)."""
    if holding_cost_per_unit_day <= 0 or demand_per_day <= 0:
        return 0.0
    return math.sqrt(2 * demand_per_day * order_cost / holding_cost_per_unit_day)


def reorder_point(demand_per_day: float, lead_time_days: float,
                  service_level: float = 0.95) -> float:
    """
    ROP = mu_lead * demand + z * sigma.
    Uses normal approximation; sigma estimated as 20% of mean demand.
    """
    import scipy.stats as stats  # type: ignore
    z = stats.norm.ppf(service_level)
    mean_demand_during_lt = demand_per_day * lead_time_days
    sigma = 0.20 * mean_demand_during_lt
    return mean_demand_during_lt + z * sigma


def days_of_stock(on_hand: float, demand_per_day: float) -> float:
    if demand_per_day <= 0:
        return 999.0
    return on_hand / demand_per_day


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

# Simulated current inventory positions (units)
INVENTORY_STATE: dict[str, dict] = {
    "NVIDIA_H100": {"on_hand": 12, "on_order": 40, "demand_per_day": 0.8, "unit_cost": 31000},
    "Marvell_ASIC_88X9140": {"on_hand": 45, "on_order": 100, "demand_per_day": 3.2, "unit_cost": 1200},
    "Vertiv_PDU_24kW": {"on_hand": 6, "on_order": 8, "demand_per_day": 0.3, "unit_cost": 4800},
    "Amphenol_QSFP_400G": {"on_hand": 220, "on_order": 500, "demand_per_day": 18.0, "unit_cost": 320},
    "Vishay_Cap_10uF": {"on_hand": 12000, "on_order": 50000, "demand_per_day": 800, "unit_cost": 0.12},
}

SUPPLIER_LEAD_TIMES: dict[str, float] = {
    "NVIDIA_H100": 84,
    "Marvell_ASIC_88X9140": 56,
    "Vertiv_PDU_24kW": 28,
    "Amphenol_QSFP_400G": 14,
    "Vishay_Cap_10uF": 7,
}


def _get_inventory_position(component_id: str) -> dict:
    state = INVENTORY_STATE.get(component_id)
    if not state:
        return {"error": f"Unknown component: {component_id}"}
    lt = SUPPLIER_LEAD_TIMES.get(component_id, 30)
    dos = days_of_stock(state["on_hand"] + state["on_order"], state["demand_per_day"])
    rop = reorder_point(state["demand_per_day"], lt)
    return {
        "component_id": component_id,
        "on_hand": state["on_hand"],
        "on_order": state["on_order"],
        "demand_per_day": state["demand_per_day"],
        "days_of_stock": round(dos, 1),
        "reorder_point": round(rop, 1),
        "lead_time_days": lt,
        "unit_cost_usd": state["unit_cost"],
        "status": "below_rop" if (state["on_hand"] + state["on_order"]) < rop else "ok",
    }


def _calculate_optimal_order(component_id: str, scenario: str = "base") -> dict:
    state = INVENTORY_STATE.get(component_id)
    if not state:
        return {"error": f"Unknown component: {component_id}"}

    # Scenario multipliers
    scenario_demand_mult = {"base": 1.0, "upside": 1.3, "downside": 0.7}.get(scenario, 1.0)
    demand = state["demand_per_day"] * scenario_demand_mult
    lt = SUPPLIER_LEAD_TIMES.get(component_id, 30)

    order_cost = max(500, state["unit_cost"] * 0.05)   # 5% of unit cost or $500
    holding_cost = state["unit_cost"] * 0.0003         # 0.03% per day (~10% p.a.)

    eoq = economic_order_quantity(demand, order_cost, holding_cost)
    rop = reorder_point(demand, lt, service_level=0.95)
    current_pos = state["on_hand"] + state["on_order"]
    order_now = max(0, eoq) if current_pos < rop else 0

    return {
        "component_id": component_id,
        "scenario": scenario,
        "eoq_units": round(eoq),
        "reorder_point": round(rop, 1),
        "current_inventory_position": current_pos,
        "recommended_order_qty": round(order_now),
        "order_value_usd": round(order_now * state["unit_cost"], 2),
        "rationale": (
            "below reorder point — place order now" if order_now > 0
            else "inventory position adequate — no order required"
        ),
    }


def _get_budget_utilisation(period: str = "Q2-2025") -> dict:
    # Simulated budget figures
    budgets = {
        "Q1-2025": {"allocated_usd": 4_200_000, "spent_usd": 3_980_000},
        "Q2-2025": {"allocated_usd": 5_500_000, "spent_usd": 2_140_000},
        "Q3-2025": {"allocated_usd": 5_000_000, "spent_usd": 0},
    }
    b = budgets.get(period, budgets["Q2-2025"])
    remaining = b["allocated_usd"] - b["spent_usd"]
    utilisation = b["spent_usd"] / b["allocated_usd"]
    return {
        "period": period,
        "allocated_usd": b["allocated_usd"],
        "spent_usd": b["spent_usd"],
        "remaining_usd": remaining,
        "utilisation_pct": round(utilisation * 100, 1),
        "forecast_overrun": utilisation > 0.85,
    }


def _evaluate_supplier_quotes(component_id: str, quotes: list[dict]) -> dict:
    """Score quotes on price, lead time, quality rating, and payment terms."""
    if not quotes:
        return {"error": "no quotes provided"}
    scored = []
    for q in quotes:
        price_score = 1 / (q.get("unit_price", 9999) + 1e-6)
        lt_score = 1 / (q.get("lead_time_days", 999) + 1e-6)
        quality = q.get("quality_rating", 0.5)
        payment_days = q.get("payment_days", 30)
        payment_score = payment_days / 90  # longer payment terms → better cash flow
        composite = (0.40 * price_score / 0.001 +
                     0.30 * lt_score / 0.05 +
                     0.20 * quality +
                     0.10 * payment_score)
        scored.append({**q, "composite_score": round(composite, 4)})
    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    return {
        "component_id": component_id,
        "recommended_supplier": scored[0].get("supplier"),
        "ranking": scored,
        "selection_rationale": (
            f"Best composite score ({scored[0]['composite_score']:.4f}) "
            f"balancing price, lead time, quality and payment terms"
        ),
    }


def _get_demand_forecast(component_id: str, horizon_days: int = 90) -> dict:
    state = INVENTORY_STATE.get(component_id)
    if not state:
        return {"error": f"Unknown component: {component_id}"}
    d = state["demand_per_day"]
    # Simple trend + noise simulation
    return {
        "component_id": component_id,
        "horizon_days": horizon_days,
        "mean_daily_demand": d,
        "total_forecast_units": round(d * horizon_days),
        "p10_total": round(d * horizon_days * 0.78),
        "p90_total": round(d * horizon_days * 1.28),
        "trend": "flat",
        "seasonality_note": "Q4 DC build-outs typically +25% demand spike",
    }


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "get_inventory_position",
        "description": (
            "Return current on-hand, on-order, days-of-stock and reorder-point "
            "status for a component."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "component_id": {"type": "string", "description": "Component identifier"},
            },
            "required": ["component_id"],
        },
    },
    {
        "name": "calculate_optimal_order",
        "description": (
            "Use EOQ and reorder-point theory to calculate the optimal order "
            "quantity under a named demand scenario (base / upside / downside)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "component_id": {"type": "string"},
                "scenario": {"type": "string", "enum": ["base", "upside", "downside"]},
            },
            "required": ["component_id"],
        },
    },
    {
        "name": "get_budget_utilisation",
        "description": "Return procurement budget allocation and spend for a quarter.",
        "input_schema": {
            "type": "object",
            "properties": {
                "period": {"type": "string", "description": "e.g. Q2-2025"},
            },
            "required": ["period"],
        },
    },
    {
        "name": "evaluate_supplier_quotes",
        "description": (
            "Score and rank a list of supplier quotes for a component on price, "
            "lead time, quality rating and payment terms."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "component_id": {"type": "string"},
                "quotes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "supplier": {"type": "string"},
                            "unit_price": {"type": "number"},
                            "lead_time_days": {"type": "number"},
                            "quality_rating": {"type": "number"},
                            "payment_days": {"type": "number"},
                        },
                    },
                },
            },
            "required": ["component_id", "quotes"],
        },
    },
    {
        "name": "get_demand_forecast",
        "description": "Return a probabilistic demand forecast for a component over a horizon.",
        "input_schema": {
            "type": "object",
            "properties": {
                "component_id": {"type": "string"},
                "horizon_days": {"type": "integer", "default": 90},
            },
            "required": ["component_id"],
        },
    },
]

DISPATCH = {
    "get_inventory_position": lambda inp: _get_inventory_position(**inp),
    "calculate_optimal_order": lambda inp: _calculate_optimal_order(**inp),
    "get_budget_utilisation": lambda inp: _get_budget_utilisation(**inp),
    "evaluate_supplier_quotes": lambda inp: _evaluate_supplier_quotes(**inp),
    "get_demand_forecast": lambda inp: _get_demand_forecast(**inp),
}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@dataclass
class ProcurementDecision:
    component_id: str
    action: str                  # "order_now" | "monitor" | "expedite" | "defer"
    quantity: int
    estimated_cost_usd: float
    urgency: str                 # "critical" | "high" | "normal" | "low"
    rationale: str
    raw_response: str


class ProcurementOptimizer:
    """
    Autonomous procurement agent.  Uses EOQ / ROP theory + agentic tool calls
    to recommend buy decisions across the component portfolio.
    """

    SYSTEM = """You are ProcurementOptimizer, a specialist supply-chain AI for
Veltora's data-centre infrastructure business.

Your mandate:
- Maintain inventory positions that avoid stock-outs while minimising working capital.
- Use economic-order-quantity theory, reorder-point analysis, and demand forecasts.
- Evaluate supplier quotes objectively using a multi-criteria score.
- Flag any component where days-of-stock falls below the lead-time buffer.

Always reason step-by-step, call tools to get data before making recommendations,
and conclude with a structured JSON procurement decision block.

Return your final answer as valid JSON in this exact shape:
{
  "component_id": "...",
  "action": "order_now|monitor|expedite|defer",
  "quantity": <int>,
  "estimated_cost_usd": <float>,
  "urgency": "critical|high|normal|low",
  "rationale": "..."
}"""

    def __init__(self, api_key: str | None = None):
        self.client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    def decide(self, component_id: str, context: str = "") -> ProcurementDecision:
        messages: list[dict] = [
            {
                "role": "user",
                "content": (
                    f"Analyse procurement requirements for component: {component_id}. "
                    f"{'Additional context: ' + context if context else ''} "
                    "Check inventory position, calculate optimal order quantity under "
                    "base and upside scenarios, review the demand forecast, and then "
                    "provide your procurement decision."
                ),
            }
        ]
        for _ in range(8):
            resp = self.client.messages.create(
                model=MODEL,
                max_tokens=2048,
                system=self.SYSTEM,
                tools=TOOLS,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": resp.content})

            if resp.stop_reason == "tool_use":
                tool_results = []
                for block in resp.content:
                    if block.type == "tool_use":
                        result = DISPATCH[block.name](block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        })
                messages.append({"role": "user", "content": tool_results})
            else:
                break

        raw = ""
        for block in resp.content:
            if hasattr(block, "text"):
                raw += block.text

        # Extract JSON block
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            data = json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            data = {
                "component_id": component_id,
                "action": "monitor",
                "quantity": 0,
                "estimated_cost_usd": 0.0,
                "urgency": "normal",
                "rationale": raw[:500],
            }

        return ProcurementDecision(
            component_id=data.get("component_id", component_id),
            action=data.get("action", "monitor"),
            quantity=int(data.get("quantity", 0)),
            estimated_cost_usd=float(data.get("estimated_cost_usd", 0)),
            urgency=data.get("urgency", "normal"),
            rationale=data.get("rationale", ""),
            raw_response=raw,
        )
