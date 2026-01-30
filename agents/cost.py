"""
CostOptimizer — Agent 4
Total-cost-of-ownership modelling, should-cost analysis, and
price negotiation recommendations.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import anthropic

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5")

# ---------------------------------------------------------------------------
# Market price data (USD, Q2 2025 spot)
# ---------------------------------------------------------------------------

MARKET_PRICES: dict[str, dict] = {
    "NVIDIA_H100": {
        "spot_usd": 31_000,
        "contract_usd": 28_500,
        "should_cost_usd": 22_000,      # wafer + packaging + margin estimate
        "price_trend_90d_pct": -4.2,
        "min_order_discount_tiers": [(10, 0.03), (50, 0.07), (100, 0.12)],
    },
    "Marvell_ASIC_88X9140": {
        "spot_usd": 1_200,
        "contract_usd": 1_050,
        "should_cost_usd": 780,
        "price_trend_90d_pct": 1.1,
        "min_order_discount_tiers": [(200, 0.05), (1000, 0.10)],
    },
    "Vertiv_PDU_24kW": {
        "spot_usd": 4_800,
        "contract_usd": 4_200,
        "should_cost_usd": 3_100,
        "price_trend_90d_pct": 2.8,
        "min_order_discount_tiers": [(20, 0.06), (50, 0.11)],
    },
    "Amphenol_QSFP_400G": {
        "spot_usd": 320,
        "contract_usd": 275,
        "should_cost_usd": 195,
        "price_trend_90d_pct": -8.5,    # commoditising fast
        "min_order_discount_tiers": [(1000, 0.08), (5000, 0.15)],
    },
    "Vishay_Cap_10uF": {
        "spot_usd": 0.12,
        "contract_usd": 0.09,
        "should_cost_usd": 0.06,
        "price_trend_90d_pct": -1.5,
        "min_order_discount_tiers": [(100_000, 0.10), (500_000, 0.18)],
    },
}

LOGISTICS_COST_PER_KG: dict[str, float] = {
    "air": 8.50,
    "sea": 0.45,
    "rail": 0.90,
    "road": 1.20,
}

COMPONENT_WEIGHTS_KG: dict[str, float] = {
    "NVIDIA_H100": 1.2,
    "Marvell_ASIC_88X9140": 0.08,
    "Vertiv_PDU_24kW": 18.0,
    "Amphenol_QSFP_400G": 0.05,
    "Vishay_Cap_10uF": 0.002,
}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _get_market_price(component_id: str) -> dict:
    p = MARKET_PRICES.get(component_id)
    if not p:
        return {"error": f"No pricing data for {component_id}"}
    markup_pct = round((p["spot_usd"] - p["should_cost_usd"]) / p["should_cost_usd"] * 100, 1)
    return {
        "component_id": component_id,
        "spot_usd": p["spot_usd"],
        "contract_usd": p["contract_usd"],
        "should_cost_usd": p["should_cost_usd"],
        "current_markup_over_should_cost_pct": markup_pct,
        "price_trend_90d_pct": p["price_trend_90d_pct"],
        "negotiation_headroom_pct": max(0, markup_pct * 0.4),
    }


def _calculate_volume_discount(component_id: str, quantity: int) -> dict:
    p = MARKET_PRICES.get(component_id)
    if not p:
        return {"error": f"No pricing data for {component_id}"}
    base = p["spot_usd"]
    discount = 0.0
    for min_qty, disc in sorted(p["min_order_discount_tiers"]):
        if quantity >= min_qty:
            discount = disc
    discounted = base * (1 - discount)
    return {
        "component_id": component_id,
        "quantity": quantity,
        "base_unit_price_usd": base,
        "discount_pct": round(discount * 100, 1),
        "discounted_unit_price_usd": round(discounted, 4),
        "total_cost_usd": round(discounted * quantity, 2),
        "savings_vs_spot_usd": round((base - discounted) * quantity, 2),
    }


def _calculate_tco(component_id: str, quantity: int,
                    transport_mode: str = "sea", lifetime_years: int = 5) -> dict:
    p = MARKET_PRICES.get(component_id, {})
    if not p:
        return {"error": f"No data for {component_id}"}

    unit_price = p.get("contract_usd", p.get("spot_usd", 1000))
    weight = COMPONENT_WEIGHTS_KG.get(component_id, 1.0)
    freight_per_kg = LOGISTICS_COST_PER_KG.get(transport_mode, 1.0)

    procurement_cost = unit_price * quantity
    freight_cost = weight * freight_per_kg * quantity
    import_duty = procurement_cost * 0.025       # 2.5% average
    inventory_holding = procurement_cost * 0.10  # 10% p.a. × 1 year cycle
    maintenance_pct = 0.03 * lifetime_years
    maintenance_cost = procurement_cost * maintenance_pct

    total_tco = (procurement_cost + freight_cost + import_duty +
                 inventory_holding + maintenance_cost)
    return {
        "component_id": component_id,
        "quantity": quantity,
        "transport_mode": transport_mode,
        "lifetime_years": lifetime_years,
        "procurement_cost_usd": round(procurement_cost, 2),
        "freight_cost_usd": round(freight_cost, 2),
        "import_duty_usd": round(import_duty, 2),
        "inventory_holding_usd": round(inventory_holding, 2),
        "maintenance_cost_usd": round(maintenance_cost, 2),
        "total_tco_usd": round(total_tco, 2),
        "cost_per_unit_tco_usd": round(total_tco / quantity, 4),
    }


def _get_price_forecast(component_id: str, horizon_days: int = 90) -> dict:
    p = MARKET_PRICES.get(component_id)
    if not p:
        return {"error": f"No pricing data for {component_id}"}
    daily_trend = p["price_trend_90d_pct"] / 90 / 100
    forecast_price = p["spot_usd"] * (1 + daily_trend) ** horizon_days
    recommendation = (
        "buy_now" if daily_trend > 0.0003 else
        "defer" if daily_trend < -0.0005 else
        "neutral"
    )
    return {
        "component_id": component_id,
        "current_spot_usd": p["spot_usd"],
        "forecast_price_usd": round(forecast_price, 2),
        "horizon_days": horizon_days,
        "price_change_pct": round((forecast_price / p["spot_usd"] - 1) * 100, 2),
        "timing_recommendation": recommendation,
        "model": "linear_extrapolation_90d_trend",
    }


def _compare_supplier_costs(component_id: str, quantity: int) -> dict:
    """Simulate three supplier options with different price/lead-time trade-offs."""
    p = MARKET_PRICES.get(component_id)
    if not p:
        return {"error": f"No pricing data for {component_id}"}
    spot = p["spot_usd"]
    options = [
        {"supplier": "preferred_contract", "unit_price": spot * 0.92,
         "lead_time_days": 60, "quality_rating": 0.94},
        {"supplier": "spot_market_broker", "unit_price": spot * 1.05,
         "lead_time_days": 14, "quality_rating": 0.78},
        {"supplier": "alternate_approved", "unit_price": spot * 0.97,
         "lead_time_days": 45, "quality_rating": 0.89},
    ]
    for o in options:
        o["total_cost_usd"] = round(o["unit_price"] * quantity, 2)
        o["cost_vs_spot_pct"] = round((o["unit_price"] / spot - 1) * 100, 1)
    return {
        "component_id": component_id,
        "quantity": quantity,
        "reference_spot_usd": spot,
        "options": options,
    }


TOOLS: list[dict] = [
    {
        "name": "get_market_price",
        "description": "Return spot, contract, and should-cost prices with negotiation headroom.",
        "input_schema": {
            "type": "object",
            "properties": {"component_id": {"type": "string"}},
            "required": ["component_id"],
        },
    },
    {
        "name": "calculate_volume_discount",
        "description": "Calculate tiered volume discount and total order cost for a given quantity.",
        "input_schema": {
            "type": "object",
            "properties": {
                "component_id": {"type": "string"},
                "quantity": {"type": "integer"},
            },
            "required": ["component_id", "quantity"],
        },
    },
    {
        "name": "calculate_tco",
        "description": "Compute total cost of ownership including freight, duty, holding, and maintenance.",
        "input_schema": {
            "type": "object",
            "properties": {
                "component_id": {"type": "string"},
                "quantity": {"type": "integer"},
                "transport_mode": {"type": "string", "enum": ["air", "sea", "rail", "road"]},
                "lifetime_years": {"type": "integer", "default": 5},
            },
            "required": ["component_id", "quantity"],
        },
    },
    {
        "name": "get_price_forecast",
        "description": "Forecast component price over a horizon and recommend buy-now vs defer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "component_id": {"type": "string"},
                "horizon_days": {"type": "integer", "default": 90},
            },
            "required": ["component_id"],
        },
    },
    {
        "name": "compare_supplier_costs",
        "description": "Compare cost, lead time and quality trade-offs across supplier options.",
        "input_schema": {
            "type": "object",
            "properties": {
                "component_id": {"type": "string"},
                "quantity": {"type": "integer"},
            },
            "required": ["component_id", "quantity"],
        },
    },
]

DISPATCH = {
    "get_market_price": lambda i: _get_market_price(**i),
    "calculate_volume_discount": lambda i: _calculate_volume_discount(**i),
    "calculate_tco": lambda i: _calculate_tco(**i),
    "get_price_forecast": lambda i: _get_price_forecast(**i),
    "compare_supplier_costs": lambda i: _compare_supplier_costs(**i),
}


@dataclass
class CostRecommendation:
    component_id: str
    recommended_action: str
    optimal_quantity: int
    estimated_unit_cost_usd: float
    total_cost_usd: float
    savings_opportunity_usd: float
    rationale: str


class CostOptimizer:
    """Minimises total procurement cost using should-cost analysis, TCO modelling,
    and price-trend timing."""

    SYSTEM = """You are CostOptimizer, a procurement cost specialist for Veltora.

Your goals:
- Minimise total cost of ownership, not just unit price.
- Use should-cost analysis to identify negotiation headroom.
- Recommend optimal order quantities to capture volume discounts.
- Time purchases based on price trend forecasts.
- Compare supplier options on a TCO basis, not just price.

Always call tools to gather data before making recommendations.
Return your final answer as JSON:
{
  "component_id": "...",
  "recommended_action": "...",
  "optimal_quantity": <int>,
  "estimated_unit_cost_usd": <float>,
  "total_cost_usd": <float>,
  "savings_opportunity_usd": <float>,
  "rationale": "..."
}"""

    def __init__(self, api_key: str | None = None):
        self.client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    def optimise(self, component_id: str, quantity: int) -> CostRecommendation:
        messages: list[dict] = [
            {
                "role": "user",
                "content": (
                    f"Optimise procurement cost for {component_id}, "
                    f"target quantity {quantity} units. "
                    "Analyse market price, should-cost, volume discounts, TCO, "
                    "price forecast, and supplier options. "
                    "Identify the biggest savings opportunity."
                ),
            }
        ]
        for _ in range(8):
            resp = self.client.messages.create(
                model=MODEL, max_tokens=2048, system=self.SYSTEM,
                tools=TOOLS, messages=messages,
            )
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason == "tool_use":
                results = []
                for block in resp.content:
                    if block.type == "tool_use":
                        results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(DISPATCH[block.name](block.input)),
                        })
                messages.append({"role": "user", "content": results})
            else:
                break

        raw = "".join(b.text for b in resp.content if hasattr(b, "text"))
        try:
            start, end = raw.index("{"), raw.rindex("}") + 1
            data = json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            data = {}

        return CostRecommendation(
            component_id=component_id,
            recommended_action=data.get("recommended_action", "negotiate"),
            optimal_quantity=int(data.get("optimal_quantity", quantity)),
            estimated_unit_cost_usd=float(data.get("estimated_unit_cost_usd", 0)),
            total_cost_usd=float(data.get("total_cost_usd", 0)),
            savings_opportunity_usd=float(data.get("savings_opportunity_usd", 0)),
            rationale=data.get("rationale", raw[:500]),
        )
