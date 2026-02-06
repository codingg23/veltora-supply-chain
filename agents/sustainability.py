"""
SustainabilityOptimizer — Agent 6
Carbon footprint, Scope 3 supplier emissions, and green procurement scoring.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import anthropic

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5")

# kg CO2e per unit
EMBODIED_CARBON: dict[str, float] = {
    "NVIDIA_H100": 1_450.0,    # GH100 die + packaging + HBM3
    "Marvell_ASIC_88X9140": 38.0,
    "Vertiv_PDU_24kW": 820.0,
    "Amphenol_QSFP_400G": 4.2,
    "Vishay_Cap_10uF": 0.008,
}

TRANSPORT_EMISSION_FACTORS: dict[str, float] = {   # kg CO2e per tonne-km
    "air": 0.602,
    "sea": 0.016,
    "rail": 0.022,
    "road": 0.096,
}

SUPPLIER_ESG_SCORES: dict[str, dict] = {
    "TSMC": {"score": 0.82, "renewable_pct": 14.0, "has_sbti": True, "tier1_verified": True},
    "Samsung": {"score": 0.76, "renewable_pct": 9.0, "has_sbti": True, "tier1_verified": True},
    "Vishay": {"score": 0.61, "renewable_pct": 22.0, "has_sbti": False, "tier1_verified": False},
    "Vertiv": {"score": 0.68, "renewable_pct": 31.0, "has_sbti": True, "tier1_verified": True},
    "Amphenol": {"score": 0.70, "renewable_pct": 18.0, "has_sbti": False, "tier1_verified": True},
}

def _get_embodied_carbon(component_id: str, quantity: int = 1) -> dict:
    ec = EMBODIED_CARBON.get(component_id)
    if ec is None:
        return {"error": f"No carbon data for {component_id}"}
    return {
        "component_id": component_id,
        "quantity": quantity,
        "embodied_carbon_kg_co2e_per_unit": ec,
        "total_kg_co2e": round(ec * quantity, 2),
        "total_tonnes_co2e": round(ec * quantity / 1000, 4),
    }

def _calculate_transport_emissions(component_id: str, quantity: int,
                                    origin_country: str, transport_mode: str,
                                    distance_km: float) -> dict:
    ef = TRANSPORT_EMISSION_FACTORS.get(transport_mode)
    if ef is None:
        return {"error": f"Unknown transport mode: {transport_mode}"}
    from simulation.physics import COMPONENT_WEIGHTS_KG_APPROX
    weight_kg = {
        "NVIDIA_H100": 1.2, "Marvell_ASIC_88X9140": 0.08,
        "Vertiv_PDU_24kW": 18.0, "Amphenol_QSFP_400G": 0.05, "Vishay_Cap_10uF": 0.002,
    }.get(component_id, 0.5)
    total_weight_tonnes = weight_kg * quantity / 1000
    emissions_kg = ef * total_weight_tonnes * distance_km
    return {
        "component_id": component_id,
        "transport_mode": transport_mode,
        "distance_km": distance_km,
        "total_weight_tonnes": round(total_weight_tonnes, 4),
        "transport_emissions_kg_co2e": round(emissions_kg, 2),
        "vs_air_multiplier": round(TRANSPORT_EMISSION_FACTORS["air"] / ef, 1),
    }

def _get_supplier_esg_score(supplier: str) -> dict:
    s = SUPPLIER_ESG_SCORES.get(supplier)
    if not s:
        return {"error": f"No ESG data for supplier: {supplier}"}
    return {"supplier": supplier, **s,
            "rating": "A" if s["score"] > 0.80 else "B" if s["score"] > 0.65 else "C"}

def _calculate_scope3_emissions(components: list[dict]) -> dict:
    total = 0.0
    breakdown = []
    for c in components:
        cid, qty = c["component_id"], c["quantity"]
        ec = EMBODIED_CARBON.get(cid, 50.0) * qty
        total += ec
        breakdown.append({"component_id": cid, "quantity": qty,
                           "scope3_kg_co2e": round(ec, 2)})
    return {
        "total_scope3_kg_co2e": round(total, 2),
        "total_scope3_tonnes_co2e": round(total / 1000, 3),
        "breakdown": breakdown,
        "equivalent_trees_planted": round(total / 21.77),  # avg tree ~21.77kg CO2/yr
    }

def _get_green_alternatives(component_id: str) -> dict:
    alts = {
        "NVIDIA_H100": [
            {"option": "NVIDIA_H100_NVL", "carbon_reduction_pct": 0,
             "note": "Same die, lower embodied per PFLOP due to higher utilisation"},
            {"option": "AMD_MI300X", "carbon_reduction_pct": 8.0,
             "note": "Slightly lower embodied carbon, different fab process"},
        ],
        "Vertiv_PDU_24kW": [
            {"option": "Vertiv_PDU_24kW_recycled_alu",
             "carbon_reduction_pct": 18.0, "note": "Recycled aluminium chassis"},
        ],
    }
    return {
        "component_id": component_id,
        "green_alternatives": alts.get(component_id, [
            {"option": "generic_greener_alt", "carbon_reduction_pct": 10.0,
             "note": "Seek suppliers with >30% renewable manufacturing energy"}
        ]),
    }

TOOLS: list[dict] = [
    {"name": "get_embodied_carbon",
     "description": "Return embodied carbon (kg CO2e) for a component.",
     "input_schema": {"type": "object",
                      "properties": {"component_id": {"type": "string"},
                                     "quantity": {"type": "integer"}},
                      "required": ["component_id"]}},
    {"name": "calculate_transport_emissions",
     "description": "Calculate CO2e from shipping components by a given transport mode.",
     "input_schema": {"type": "object",
                      "properties": {"component_id": {"type": "string"},
                                     "quantity": {"type": "integer"},
                                     "origin_country": {"type": "string"},
                                     "transport_mode": {"type": "string"},
                                     "distance_km": {"type": "number"}},
                      "required": ["component_id", "quantity", "origin_country",
                                   "transport_mode", "distance_km"]}},
    {"name": "get_supplier_esg_score",
     "description": "Return ESG rating, renewable energy %, and SBTi status for a supplier.",
     "input_schema": {"type": "object",
                      "properties": {"supplier": {"type": "string"}},
                      "required": ["supplier"]}},
    {"name": "calculate_scope3_emissions",
     "description": "Calculate aggregate Scope 3 emissions across a list of components.",
     "input_schema": {"type": "object",
                      "properties": {"components": {
                          "type": "array",
                          "items": {"type": "object",
                                    "properties": {"component_id": {"type": "string"},
                                                   "quantity": {"type": "integer"}}}}},
                      "required": ["components"]}},
    {"name": "get_green_alternatives",
     "description": "Suggest lower-carbon alternative components or suppliers.",
     "input_schema": {"type": "object",
                      "properties": {"component_id": {"type": "string"}},
                      "required": ["component_id"]}},
]

DISPATCH = {
    "get_embodied_carbon": lambda i: _get_embodied_carbon(**i),
    "calculate_transport_emissions": lambda i: _calculate_transport_emissions(**i),
    "get_supplier_esg_score": lambda i: _get_supplier_esg_score(**i),
    "calculate_scope3_emissions": lambda i: _calculate_scope3_emissions(**i),
    "get_green_alternatives": lambda i: _get_green_alternatives(**i),
}

@dataclass
class SustainabilityReport:
    total_scope3_tonnes_co2e: float
    highest_impact_component: str
    recommended_transport_mode: str
    green_score: float       # 0-1
    key_actions: list[str]
    rationale: str

class SustainabilityOptimizer:
    """Minimises Scope 3 supply-chain emissions while maintaining delivery commitments."""

    SYSTEM = """You are SustainabilityOptimizer, Veltora's green procurement specialist.

Goals:
- Quantify Scope 3 embodied carbon for procurement decisions.
- Prefer sea/rail over air freight where lead time permits.
- Score suppliers on ESG credentials and push for SBTi-aligned partners.
- Recommend lower-carbon alternatives where cost-neutral.

Return final answer as JSON:
{
  "total_scope3_tonnes_co2e": <float>,
  "highest_impact_component": "...",
  "recommended_transport_mode": "...",
  "green_score": <0-1>,
  "key_actions": ["..."]
}"""

    def __init__(self, api_key: str | None = None):
        self.client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    def evaluate(self, components: list[dict]) -> SustainabilityReport:
        messages: list[dict] = [
            {"role": "user", "content": (
                f"Evaluate sustainability impact of procuring: {json.dumps(components)}. "
                "Calculate Scope 3 emissions, assess supplier ESG scores, recommend "
                "the lowest-carbon transport mode, and identify green alternatives."
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
        return SustainabilityReport(
            total_scope3_tonnes_co2e=float(data.get("total_scope3_tonnes_co2e", 0)),
            highest_impact_component=data.get("highest_impact_component", ""),
            recommended_transport_mode=data.get("recommended_transport_mode", "sea"),
            green_score=float(data.get("green_score", 0.5)),
            key_actions=data.get("key_actions", []),
            rationale=raw[:500],
        )
