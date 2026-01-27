"""
RiskMitigation — Agent 3
Graph-based supply-chain dependency analysis + probabilistic disruption scoring.
Identifies single points of failure and recommends hedging strategies.
"""
from __future__ import annotations

import json
import os
import math
from dataclasses import dataclass
from typing import Any

import anthropic

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5")

# ---------------------------------------------------------------------------
# Dependency graph (nodes = components/suppliers, edges = depends-on)
# ---------------------------------------------------------------------------

# Adjacency list: component → list of (supplier, criticality 0-1)
DEPENDENCY_GRAPH: dict[str, list[dict]] = {
    "DC_Server_Node": [
        {"node": "NVIDIA_H100", "criticality": 0.95, "substitutable": False},
        {"node": "Marvell_ASIC_88X9140", "criticality": 0.80, "substitutable": False},
        {"node": "Samsung_HBM3", "criticality": 0.90, "substitutable": True},
        {"node": "Amphenol_QSFP_400G", "criticality": 0.70, "substitutable": True},
    ],
    "NVIDIA_H100": [
        {"node": "TSMC_N4P_fab", "criticality": 1.0, "substitutable": False},
        {"node": "HBM3_supply", "criticality": 0.90, "substitutable": False},
    ],
    "Marvell_ASIC_88X9140": [
        {"node": "TSMC_N5_fab", "criticality": 0.95, "substitutable": False},
    ],
    "TSMC_N4P_fab": [
        {"node": "ASML_EUV_lithography", "criticality": 1.0, "substitutable": False},
        {"node": "Taiwan_power_grid", "criticality": 0.85, "substitutable": False},
    ],
    "Samsung_HBM3": [
        {"node": "Samsung_fab_Korea", "criticality": 1.0, "substitutable": False},
    ],
}

# Risk event probabilities and impact magnitudes
RISK_EVENTS: dict[str, dict] = {
    "taiwan_strait_tension": {
        "probability_12m": 0.18,
        "affected_nodes": ["TSMC_N4P_fab", "TSMC_N5_fab", "NVIDIA_H100", "Marvell_ASIC_88X9140"],
        "lead_time_multiplier": 2.8,
        "price_multiplier": 1.6,
        "duration_days_p50": 180,
    },
    "us_china_tariff_escalation": {
        "probability_12m": 0.35,
        "affected_nodes": ["Vishay_Cap_10uF", "Samsung_HBM3"],
        "lead_time_multiplier": 1.3,
        "price_multiplier": 1.25,
        "duration_days_p50": 365,
    },
    "port_strike_us_west_coast": {
        "probability_12m": 0.12,
        "affected_nodes": ["Amphenol_QSFP_400G", "NVIDIA_H100"],
        "lead_time_multiplier": 1.5,
        "price_multiplier": 1.1,
        "duration_days_p50": 21,
    },
    "semiconductor_fab_fire": {
        "probability_12m": 0.04,
        "affected_nodes": ["TSMC_N4P_fab", "Samsung_fab_Korea"],
        "lead_time_multiplier": 4.0,
        "price_multiplier": 2.5,
        "duration_days_p50": 270,
    },
    "logistics_fuel_price_spike": {
        "probability_12m": 0.45,
        "affected_nodes": ["Amphenol_QSFP_400G", "Vertiv_PDU_24kW"],
        "lead_time_multiplier": 1.1,
        "price_multiplier": 1.15,
        "duration_days_p50": 60,
    },
}


# ---------------------------------------------------------------------------
# Graph algorithms
# ---------------------------------------------------------------------------

def _betweenness_centrality(graph: dict) -> dict[str, float]:
    """Simplified betweenness: count how many paths go through each node."""
    nodes = set(graph.keys())
    for deps in graph.values():
        for d in deps:
            nodes.add(d["node"])

    centrality: dict[str, float] = {n: 0.0 for n in nodes}
    all_nodes = list(nodes)

    for source in all_nodes:
        # BFS to find all reachable nodes and paths
        visited: dict[str, list[str]] = {source: [source]}
        queue = [source]
        while queue:
            current = queue.pop(0)
            for dep in graph.get(current, []):
                neighbour = dep["node"]
                if neighbour not in visited:
                    visited[neighbour] = visited[current] + [neighbour]
                    queue.append(neighbour)
        for target, path in visited.items():
            if target != source:
                for intermediate in path[1:-1]:
                    centrality[intermediate] = centrality.get(intermediate, 0) + 1

    # Normalise
    max_c = max(centrality.values()) if centrality else 1
    return {n: round(v / max(max_c, 1), 4) for n, v in centrality.items()}


def _get_dependency_chain(component_id: str, depth: int = 3) -> dict:
    """BFS up to `depth` levels."""
    result: dict[str, Any] = {component_id: []}
    queue = [(component_id, 0)]
    visited = {component_id}
    while queue:
        node, d = queue.pop(0)
        if d >= depth:
            continue
        deps = DEPENDENCY_GRAPH.get(node, [])
        result[node] = deps
        for dep in deps:
            child = dep["node"]
            if child not in visited:
                visited.add(child)
                queue.append((child, d + 1))
    return {"root": component_id, "depth": depth, "graph": result}


def _score_disruption_risk(component_id: str) -> dict:
    """Expected value of disruption impact = P(event) × impact × duration."""
    affected_by = []
    total_ev_days = 0.0
    for event_name, event in RISK_EVENTS.items():
        if component_id in event["affected_nodes"]:
            ev = (event["probability_12m"] *
                  event["lead_time_multiplier"] *
                  event["duration_days_p50"])
            total_ev_days += ev
            affected_by.append({
                "event": event_name,
                "probability_12m": event["probability_12m"],
                "lead_time_multiplier": event["lead_time_multiplier"],
                "ev_delay_days": round(ev, 1),
            })
    risk_score = min(1.0, total_ev_days / 200)
    return {
        "component_id": component_id,
        "risk_score": round(risk_score, 3),
        "risk_tier": (
            "critical" if risk_score > 0.6 else
            "high" if risk_score > 0.35 else
            "medium" if risk_score > 0.15 else "low"
        ),
        "expected_value_delay_days": round(total_ev_days, 1),
        "contributing_events": affected_by,
    }


def _identify_single_points_of_failure(scope: str = "full") -> dict:
    centrality = _betweenness_centrality(DEPENDENCY_GRAPH)
    spofs = [
        {"node": n, "centrality": c, "tier": "critical" if c > 0.5 else "high"}
        for n, c in centrality.items()
        if c > 0.3
    ]
    spofs.sort(key=lambda x: x["centrality"], reverse=True)
    return {
        "scope": scope,
        "single_points_of_failure": spofs[:10],
        "total_nodes_analysed": len(centrality),
        "methodology": "betweenness centrality on directed dependency graph",
    }


def _get_hedging_options(component_id: str) -> dict:
    hedges = {
        "NVIDIA_H100": [
            {"strategy": "dual_source", "alternate_supplier": "AMD_MI300X",
             "cost_premium_pct": 8.0, "lead_time_delta_days": -10,
             "implementation_weeks": 12},
            {"strategy": "buffer_stock", "safety_stock_units": 20,
             "holding_cost_usd_month": 620000, "risk_reduction_pct": 45},
        ],
        "TSMC_N4P_fab": [
            {"strategy": "fab_diversification", "alternate": "Samsung_3nm",
             "nre_cost_usd": 4_500_000, "timeline_weeks": 24,
             "risk_reduction_pct": 60},
        ],
        "Marvell_ASIC_88X9140": [
            {"strategy": "redesign_for_alternate_fab",
             "alternate_fab": "GlobalFoundries_12LP",
             "nre_cost_usd": 800_000, "timeline_weeks": 18,
             "risk_reduction_pct": 55},
            {"strategy": "increase_safety_stock", "units": 200,
             "holding_cost_usd_month": 24_000, "risk_reduction_pct": 30},
        ],
    }
    options = hedges.get(component_id, [
        {"strategy": "buffer_stock", "recommended_dos": 30,
         "cost_estimate": "low", "risk_reduction_pct": 20},
    ])
    return {"component_id": component_id, "hedging_options": options}


def _simulate_disruption_scenario(event_name: str, component_id: str) -> dict:
    event = RISK_EVENTS.get(event_name)
    if not event:
        return {"error": f"Unknown event: {event_name}"}
    affected = component_id in event["affected_nodes"]
    if not affected:
        return {
            "event": event_name,
            "component_id": component_id,
            "impact": "none",
            "note": "Component not in affected node list for this event",
        }
    lt_mult = event["lead_time_multiplier"]
    price_mult = event["price_multiplier"]
    duration = event["duration_days_p50"]
    return {
        "event": event_name,
        "component_id": component_id,
        "probability_12m": event["probability_12m"],
        "lead_time_multiplier": lt_mult,
        "price_multiplier": price_mult,
        "disruption_duration_days_p50": duration,
        "recovery_strategy": (
            "activate buffer stock + expedite alternate supplier" if lt_mult > 1.5
            else "monitor and extend safety stock"
        ),
    }


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "get_dependency_chain",
        "description": "Return the upstream dependency graph for a component up to N levels deep.",
        "input_schema": {
            "type": "object",
            "properties": {
                "component_id": {"type": "string"},
                "depth": {"type": "integer", "default": 3},
            },
            "required": ["component_id"],
        },
    },
    {
        "name": "score_disruption_risk",
        "description": "Calculate probabilistic risk score for a component based on known risk events.",
        "input_schema": {
            "type": "object",
            "properties": {"component_id": {"type": "string"}},
            "required": ["component_id"],
        },
    },
    {
        "name": "identify_single_points_of_failure",
        "description": "Use betweenness centrality to find single points of failure in the supply graph.",
        "input_schema": {
            "type": "object",
            "properties": {"scope": {"type": "string", "default": "full"}},
        },
    },
    {
        "name": "get_hedging_options",
        "description": "Return available risk-hedging strategies for a component (dual-source, buffer stock, redesign).",
        "input_schema": {
            "type": "object",
            "properties": {"component_id": {"type": "string"}},
            "required": ["component_id"],
        },
    },
    {
        "name": "simulate_disruption_scenario",
        "description": "Simulate the impact of a named disruption event on a specific component.",
        "input_schema": {
            "type": "object",
            "properties": {
                "event_name": {"type": "string"},
                "component_id": {"type": "string"},
            },
            "required": ["event_name", "component_id"],
        },
    },
]

DISPATCH = {
    "get_dependency_chain": lambda i: _get_dependency_chain(**i),
    "score_disruption_risk": lambda i: _score_disruption_risk(**i),
    "identify_single_points_of_failure": lambda i: _identify_single_points_of_failure(**i),
    "get_hedging_options": lambda i: _get_hedging_options(**i),
    "simulate_disruption_scenario": lambda i: _simulate_disruption_scenario(**i),
}


@dataclass
class RiskAssessment:
    component_id: str
    risk_score: float
    risk_tier: str
    top_threats: list[dict]
    recommended_hedges: list[dict]
    spof_flag: bool
    rationale: str


class RiskMitigation:
    """
    Autonomous risk agent. Builds a dependency graph, scores probabilistic
    disruption risk, flags single points of failure, and proposes hedges.
    """

    SYSTEM = """You are RiskMitigation, a supply-chain risk specialist for Veltora.

Your responsibilities:
- Map upstream dependency chains for critical components.
- Score disruption risk using event probabilities and impact magnitudes.
- Identify single points of failure via betweenness centrality analysis.
- Propose concrete hedging strategies (dual-source, buffer stock, redesign).
- Simulate specific risk scenarios to size their impact.

Think systematically: start broad (SPOF scan), then drill into the specific
component, then recommend actionable mitigations ranked by cost-effectiveness.

Conclude with a JSON risk summary:
{
  "component_id": "...",
  "risk_score": <0-1 float>,
  "risk_tier": "critical|high|medium|low",
  "top_threats": [{"event": "...", "ev_delay_days": ...}],
  "recommended_hedges": [{"strategy": "...", "priority": "..."}],
  "spof_flag": <bool>
}"""

    def __init__(self, api_key: str | None = None):
        self.client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    def assess(self, component_id: str) -> RiskAssessment:
        messages: list[dict] = [
            {
                "role": "user",
                "content": (
                    f"Perform a full risk assessment for component: {component_id}. "
                    "Identify its dependency chain, score its disruption risk, check "
                    "if it is a single point of failure, simulate the top two risk "
                    "scenarios, and recommend hedging strategies."
                ),
            }
        ]
        for _ in range(10):
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

        return RiskAssessment(
            component_id=component_id,
            risk_score=float(data.get("risk_score", 0.5)),
            risk_tier=data.get("risk_tier", "medium"),
            top_threats=data.get("top_threats", []),
            recommended_hedges=data.get("recommended_hedges", []),
            spof_flag=bool(data.get("spof_flag", False)),
            rationale=raw[:800],
        )
