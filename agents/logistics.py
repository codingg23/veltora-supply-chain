"""
LogisticsCoordinator — Agent 9
Multi-modal route optimisation, customs clearance prediction, and carrier selection.
"""
from __future__ import annotations

import json
import os
import math
from dataclasses import dataclass

import anthropic

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5")

# Carrier contracts (rates in USD per kg, transit in days)
CARRIER_RATES: dict[str, list[dict]] = {
    "air": [
        {"carrier": "FedEx_Priority", "rate_usd_kg": 9.20, "transit_days": 2,
         "reliability_pct": 97.5, "max_kg": 500},
        {"carrier": "DHL_Express",    "rate_usd_kg": 8.80, "transit_days": 3,
         "reliability_pct": 96.8, "max_kg": 1000},
        {"carrier": "Cathay_Cargo",   "rate_usd_kg": 6.40, "transit_days": 5,
         "reliability_pct": 91.2, "max_kg": 10000},
    ],
    "sea": [
        {"carrier": "Maersk_FCL",   "rate_usd_kg": 0.52, "transit_days": 28,
         "reliability_pct": 88.0, "max_kg": 25000},
        {"carrier": "MSC_LCL",      "rate_usd_kg": 0.68, "transit_days": 35,
         "reliability_pct": 84.0, "max_kg": 5000},
        {"carrier": "COSCO_FCL",    "rate_usd_kg": 0.44, "transit_days": 32,
         "reliability_pct": 82.0, "max_kg": 25000},
    ],
    "rail": [
        {"carrier": "DB_Cargo",     "rate_usd_kg": 0.95, "transit_days": 18,
         "reliability_pct": 90.0, "max_kg": 20000},
    ],
}

# Customs clearance expected duration by origin-destination pair (hours)
CUSTOMS_DURATION_HOURS: dict[tuple[str, str], dict] = {
    ("CN", "US"): {"mean": 48, "sigma": 24, "risk": "high"},
    ("TW", "US"): {"mean": 36, "sigma": 18, "risk": "medium"},
    ("KR", "US"): {"mean": 24, "sigma": 12, "risk": "low"},
    ("DE", "US"): {"mean": 12, "sigma": 6,  "risk": "low"},
    ("CN", "GB"): {"mean": 36, "sigma": 18, "risk": "medium"},
    ("TW", "GB"): {"mean": 24, "sigma": 12, "risk": "low"},
    ("CN", "SG"): {"mean": 18, "sigma": 9,  "risk": "low"},
}

ACTIVE_SHIPMENTS: list[dict] = [
    {"shipment_id": "SHP-2025-0842", "component": "NVIDIA_H100",
     "qty": 40, "carrier": "Cathay_Cargo", "mode": "air",
     "origin": "TW", "destination": "US",
     "status": "in_transit", "eta": "2025-04-12",
     "current_location": "Taipei Taoyuan Airport"},
    {"shipment_id": "SHP-2025-0801", "component": "Vishay_Cap_10uF",
     "qty": 50000, "carrier": "COSCO_FCL", "mode": "sea",
     "origin": "CN", "destination": "GB",
     "status": "customs_hold", "eta": "2025-04-18",
     "current_location": "Felixstowe Port — customs inspection"},
    {"shipment_id": "SHP-2025-0789", "component": "Amphenol_QSFP_400G",
     "qty": 500, "carrier": "DHL_Express", "mode": "air",
     "origin": "CN", "destination": "US",
     "status": "delivered", "eta": "2025-03-30",
     "current_location": "Veltora DC-West warehouse"},
]


def _select_optimal_carrier(weight_kg: float, mode: str,
                              urgency: str = "normal") -> dict:
    carriers = CARRIER_RATES.get(mode, [])
    eligible = [c for c in carriers if c["max_kg"] >= weight_kg]
    if not eligible:
        return {"error": f"No carrier supports {weight_kg}kg via {mode}"}
    # Score: urgency=critical → weight reliability heavily, else minimise cost
    scored = []
    for c in eligible:
        cost_score = 1 / (c["rate_usd_kg"] * weight_kg + 1)
        reliability_score = c["reliability_pct"] / 100
        transit_score = 1 / (c["transit_days"] + 1)
        if urgency == "critical":
            composite = 0.15 * cost_score + 0.50 * reliability_score + 0.35 * transit_score
        elif urgency == "high":
            composite = 0.25 * cost_score + 0.40 * reliability_score + 0.35 * transit_score
        else:
            composite = 0.50 * cost_score + 0.30 * reliability_score + 0.20 * transit_score
        scored.append({**c, "total_cost_usd": round(c["rate_usd_kg"] * weight_kg, 2),
                        "composite_score": round(composite, 5)})
    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    return {
        "weight_kg": weight_kg,
        "mode": mode,
        "urgency": urgency,
        "recommended_carrier": scored[0]["carrier"],
        "estimated_cost_usd": scored[0]["total_cost_usd"],
        "transit_days": scored[0]["transit_days"],
        "all_options": scored,
    }


def _predict_customs_clearance(origin: str, destination: str, commodity: str) -> dict:
    key = (origin.upper(), destination.upper())
    data = CUSTOMS_DURATION_HOURS.get(key, {"mean": 36, "sigma": 18, "risk": "medium"})
    # Monte Carlo P50/P90 approximation
    import random
    samples = [max(0, data["mean"] + random.gauss(0, data["sigma"]))
               for _ in range(500)]
    samples.sort()
    p50 = samples[250]
    p90 = samples[450]
    return {
        "origin": origin, "destination": destination, "commodity": commodity,
        "mean_clearance_hours": data["mean"],
        "p50_hours": round(p50, 1), "p90_hours": round(p90, 1),
        "risk_tier": data["risk"],
        "recommendation": (
            "pre-clear with broker 72h before arrival" if data["risk"] == "high"
            else "standard clearance process"
        ),
    }


def _get_active_shipments(status_filter: str | None = None) -> dict:
    shipments = ACTIVE_SHIPMENTS
    if status_filter:
        shipments = [s for s in shipments if s["status"] == status_filter]
    alerts = [s for s in shipments if s["status"] in ("customs_hold", "delayed")]
    return {
        "total_active": len(shipments),
        "shipments": shipments,
        "alerts": alerts,
        "alert_count": len(alerts),
    }


def _optimise_consolidation(shipments: list[dict]) -> dict:
    """Group shipments by destination to find consolidation savings."""
    by_dest: dict[str, list] = {}
    for s in shipments:
        d = s.get("destination", "UNKNOWN")
        by_dest.setdefault(d, []).append(s)
    consolidations = []
    for dest, group in by_dest.items():
        if len(group) > 1:
            total_weight = sum(s.get("weight_kg", 10) for s in group)
            savings_est = total_weight * 0.8   # rough USD savings from FCL vs LCL
            consolidations.append({
                "destination": dest,
                "shipments": [s.get("shipment_id", "?") for s in group],
                "combined_weight_kg": total_weight,
                "estimated_savings_usd": round(savings_est, 2),
            })
    return {
        "consolidation_opportunities": consolidations,
        "total_estimated_savings_usd": round(
            sum(c["estimated_savings_usd"] for c in consolidations), 2
        ),
    }


def _calculate_landed_cost(component_id: str, unit_price_usd: float,
                             quantity: int, weight_kg_per_unit: float,
                             origin: str, destination: str,
                             transport_mode: str = "sea") -> dict:
    total_weight = weight_kg_per_unit * quantity
    carriers = CARRIER_RATES.get(transport_mode, [])
    freight_rate = carriers[0]["rate_usd_kg"] if carriers else 1.0
    freight = freight_rate * total_weight
    goods_value = unit_price_usd * quantity
    duty_rate = 0.025 if origin not in ("DE", "GB", "FR") else 0.0
    duty = goods_value * duty_rate
    insurance = goods_value * 0.001
    customs_data = CUSTOMS_DURATION_HOURS.get(
        (origin.upper(), destination.upper()), {"mean": 36})
    customs_brokerage = 150 + customs_data["mean"] * 2
    landed = goods_value + freight + duty + insurance + customs_brokerage
    return {
        "component_id": component_id,
        "quantity": quantity,
        "goods_value_usd": round(goods_value, 2),
        "freight_usd": round(freight, 2),
        "import_duty_usd": round(duty, 2),
        "insurance_usd": round(insurance, 2),
        "customs_brokerage_usd": round(customs_brokerage, 2),
        "total_landed_cost_usd": round(landed, 2),
        "landed_cost_per_unit_usd": round(landed / quantity, 4),
        "freight_as_pct_of_goods": round(freight / goods_value * 100, 2),
    }


TOOLS = [
    {"name": "select_optimal_carrier",
     "description": "Score and select the best carrier for a shipment weight, mode, and urgency.",
     "input_schema": {"type": "object",
                      "properties": {"weight_kg": {"type": "number"},
                                     "mode": {"type": "string"},
                                     "urgency": {"type": "string", "enum": ["normal", "high", "critical"]}},
                      "required": ["weight_kg", "mode"]}},
    {"name": "predict_customs_clearance",
     "description": "Predict P50/P90 customs clearance time for an origin-destination pair.",
     "input_schema": {"type": "object",
                      "properties": {"origin": {"type": "string"},
                                     "destination": {"type": "string"},
                                     "commodity": {"type": "string"}},
                      "required": ["origin", "destination", "commodity"]}},
    {"name": "get_active_shipments",
     "description": "List active shipments, optionally filtered by status.",
     "input_schema": {"type": "object",
                      "properties": {"status_filter": {"type": "string"}}}},
    {"name": "optimise_consolidation",
     "description": "Identify shipments that can be consolidated to save freight cost.",
     "input_schema": {"type": "object",
                      "properties": {"shipments": {
                          "type": "array", "items": {"type": "object"}}},
                      "required": ["shipments"]}},
    {"name": "calculate_landed_cost",
     "description": "Compute fully landed cost including freight, duty, insurance, and customs brokerage.",
     "input_schema": {"type": "object",
                      "properties": {"component_id": {"type": "string"},
                                     "unit_price_usd": {"type": "number"},
                                     "quantity": {"type": "integer"},
                                     "weight_kg_per_unit": {"type": "number"},
                                     "origin": {"type": "string"},
                                     "destination": {"type": "string"},
                                     "transport_mode": {"type": "string"}},
                      "required": ["component_id", "unit_price_usd", "quantity",
                                   "weight_kg_per_unit", "origin", "destination"]}},
]

DISPATCH = {
    "select_optimal_carrier": lambda i: _select_optimal_carrier(**i),
    "predict_customs_clearance": lambda i: _predict_customs_clearance(**i),
    "get_active_shipments": lambda i: _get_active_shipments(**i),
    "optimise_consolidation": lambda i: _optimise_consolidation(**i),
    "calculate_landed_cost": lambda i: _calculate_landed_cost(**i),
}


@dataclass
class Logisticsplan:
    recommended_mode: str
    recommended_carrier: str
    estimated_transit_days: int
    estimated_freight_cost_usd: float
    customs_risk: str
    active_alerts: list[str]
    rationale: str


class LogisticsCoordinator:
    """Optimises routing, carrier selection, customs clearance, and freight cost."""

    SYSTEM = """You are LogisticsCoordinator, Veltora's freight and customs specialist.

Tasks:
- Select the optimal carrier and mode for each shipment.
- Predict customs clearance risk and advise on pre-clearance.
- Monitor active shipments and flag delays or customs holds.
- Find consolidation savings across concurrent shipments.
- Calculate fully-landed costs for procurement decisions.

Return JSON:
{
  "recommended_mode": "air|sea|rail|road",
  "recommended_carrier": "...",
  "estimated_transit_days": <int>,
  "estimated_freight_cost_usd": <float>,
  "customs_risk": "low|medium|high",
  "active_alerts": ["..."]
}"""

    def __init__(self, api_key: str | None = None):
        self.client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    def plan(self, component_id: str, weight_kg: float,
              origin: str, destination: str, urgency: str = "normal") -> Logisticsplan:
        messages: list[dict] = [
            {"role": "user", "content": (
                f"Plan logistics for {component_id}: {weight_kg}kg from {origin} to {destination}, "
                f"urgency={urgency}. Check active shipments for alerts, select optimal carrier "
                "for both air and sea, predict customs clearance, and recommend the best option."
            )}
        ]
        for _ in range(8):
            resp = self.client.messages.create(
                model=MODEL, max_tokens=2048, system=self.SYSTEM,
                tools=TOOLS, messages=messages)
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason == "tool_use":
                results = []
                for block in resp.content:
                    if block.type == "tool_use":
                        results.append({"type": "tool_result", "tool_use_id": block.id,
                                        "content": json.dumps(DISPATCH[block.name](block.input))})
                messages.append({"role": "user", "content": results})
            else:
                break
        raw = "".join(b.text for b in resp.content if hasattr(b, "text"))
        try:
            s, e = raw.index("{"), raw.rindex("}") + 1
            data = json.loads(raw[s:e])
        except (ValueError, json.JSONDecodeError):
            data = {}
        return Logisticsplan(
            recommended_mode=data.get("recommended_mode", "sea"),
            recommended_carrier=data.get("recommended_carrier", ""),
            estimated_transit_days=int(data.get("estimated_transit_days", 30)),
            estimated_freight_cost_usd=float(data.get("estimated_freight_cost_usd", 0)),
            customs_risk=data.get("customs_risk", "medium"),
            active_alerts=data.get("active_alerts", []),
            rationale=raw[:500],
        )
