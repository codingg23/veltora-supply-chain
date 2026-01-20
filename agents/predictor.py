"""
predictor.py

SupplyChainPredictor - forecasts component shortages and delivery risks
60-90 days ahead using the physics-informed lead time model.

This is the most technically differentiated agent. Instead of asking an LLM
to guess delivery dates, it:
  1. Pulls current fab utilisation, port congestion, and geopolitical signals
  2. Runs the physics model to get P10/P50/P90 lead time distributions
  3. Compares P90 against project critical path dates
  4. Flags components where P90 > deadline as "at risk"

The LLM's job is to synthesise these signals into a coherent risk narrative
and prioritise which risks need immediate action.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional
import anthropic

from simulation.physics import (
    ComponentSpec, LeadTimeEstimate,
    estimate_lead_time, fab_utilisation_from_orders,
    port_congestion_factor, geopolitical_delay_days,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the SupplyChainPredictor for a data centre construction project.

Your job is to forecast component delivery risks 60-90 days ahead. You have access to:
- Physics-informed lead time models (not guesses - actual manufacturing queue theory)
- Real-time fab utilisation signals
- Port congestion data
- Geopolitical risk scores by trade lane

When assessing risk:
- A component is AT RISK if its P90 lead time exceeds the project need-by date
- A component is CRITICAL if it's on the critical path AND at risk
- Flag when fab utilisation > 80% for key suppliers (lead times become unpredictable)
- Flag when P90 - P50 > 20 days (high uncertainty is itself a risk)

Be specific. "PDU transformers from Eaton's South Carolina facility have a 67-day P90
lead time against a 55-day need-by - recommend placing order immediately" is useful.
"There are some risks" is not.

Structure your response:
1. Critical risks (need action today)
2. Watch items (monitor, may need action in 2 weeks)
3. On track (for completeness)

End with: Confidence: [0-100]%"""


@dataclass
class ComponentRisk:
    component_id: str
    description: str
    need_by_date: str       # ISO date
    lead_time_p50: float    # days
    lead_time_p90: float    # days
    order_deadline: str     # ISO date - latest safe order date
    risk_level: str         # "critical", "at_risk", "watch", "on_track"
    risk_drivers: list[str] = field(default_factory=list)


class SupplyChainPredictor:
    """
    Physics-informed supply chain risk forecaster.

    Combines deterministic physics (manufacturing queues, transport times)
    with an LLM reasoning layer that synthesises signals into risk narratives.
    """

    def __init__(self, memory_client=None, model: str = "claude-sonnet-4-6"):
        self.name = "SupplyChainPredictor"
        self.memory = memory_client
        self.model = model
        self.client = anthropic.Anthropic()
        self._tools = self._build_tools()

    def _build_tools(self) -> list[dict]:
        return [
            {
                "name": "estimate_component_lead_time",
                "description": "Get physics-based P10/P50/P90 lead time estimate for a component.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "component_id": {"type": "string"},
                        "category": {"type": "string", "enum": ["semiconductor", "cable", "mechanical", "cooling", "electrical", "networking"]},
                        "origin_country": {"type": "string", "description": "ISO country code, e.g. CN, TW, US"},
                        "destination_country": {"type": "string"},
                        "fab_utilisation": {"type": "number", "description": "Current fab load 0-1"},
                        "transport_mode": {"type": "string", "enum": ["air", "sea", "rail", "road"], "default": "sea"},
                        "distance_km": {"type": "number"},
                        "lot_size": {"type": "integer", "default": 1},
                    },
                    "required": ["component_id", "category", "origin_country", "destination_country"],
                },
            },
            {
                "name": "get_fab_utilisation",
                "description": "Get current utilisation for a semiconductor fab or manufacturer.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "manufacturer": {"type": "string", "description": "e.g. TSMC, Samsung, Vishay"},
                        "component_type": {"type": "string"},
                    },
                    "required": ["manufacturer"],
                },
            },
            {
                "name": "get_port_congestion",
                "description": "Get current port congestion factor for a shipping route.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "origin_port": {"type": "string", "description": "e.g. Shenzhen, Shanghai, Rotterdam"},
                        "destination_port": {"type": "string"},
                    },
                    "required": ["origin_port", "destination_port"],
                },
            },
            {
                "name": "get_geopolitical_risk",
                "description": "Get geopolitical risk score for a trade lane (0=no risk, 1=active disruption).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "origin_country": {"type": "string"},
                        "destination_country": {"type": "string"},
                    },
                    "required": ["origin_country", "destination_country"],
                },
            },
            {
                "name": "check_project_critical_path",
                "description": "Get critical path dates for components in the current project.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "string"},
                        "component_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["project_id"],
                },
            },
        ]

    def execute_tool(self, tool_name: str, tool_input: dict) -> Any:
        if tool_name == "estimate_component_lead_time":
            return self._estimate_lead_time(**tool_input)
        if tool_name == "get_fab_utilisation":
            return self._get_fab_utilisation(**tool_input)
        if tool_name == "get_port_congestion":
            return self._get_port_congestion(**tool_input)
        if tool_name == "get_geopolitical_risk":
            return self._get_geopolitical_risk(**tool_input)
        if tool_name == "check_project_critical_path":
            return self._get_critical_path(**tool_input)
        raise ValueError(f"Unknown tool: {tool_name}")

    def _estimate_lead_time(
        self,
        component_id: str,
        category: str,
        origin_country: str,
        destination_country: str,
        fab_utilisation: float = 0.70,
        transport_mode: str = "sea",
        distance_km: float = 14000.0,
        lot_size: int = 1,
        **kwargs,
    ) -> dict:
        # category-specific defaults
        defaults = {
            "semiconductor": {"production_rate": 0.5, "yield_rate": 0.92, "burn_in_days": 2.0, "test_days": 1.5},
            "cooling":       {"production_rate": 2.0, "yield_rate": 0.97, "burn_in_days": 0.0, "test_days": 1.0},
            "cable":         {"production_rate": 10.0,"yield_rate": 0.99, "burn_in_days": 0.0, "test_days": 0.25},
            "mechanical":    {"production_rate": 3.0, "yield_rate": 0.98, "burn_in_days": 0.0, "test_days": 0.5},
            "electrical":    {"production_rate": 2.5, "yield_rate": 0.97, "burn_in_days": 1.0, "test_days": 0.5},
            "networking":    {"production_rate": 1.0, "yield_rate": 0.95, "burn_in_days": 1.0, "test_days": 2.0},
        }
        d = defaults.get(category, defaults["mechanical"])

        spec = ComponentSpec(
            component_id=component_id,
            category=category,
            origin_country=origin_country,
            destination_country=destination_country,
            lot_size=lot_size,
            production_rate_per_day=d["production_rate"],
            yield_rate=d["yield_rate"],
            fab_utilisation=fab_utilisation,
            transport_mode=transport_mode,
            origin_city="",
            destination_city="",
            distance_km=distance_km,
            requires_burn_in=d["burn_in_days"] > 0,
            burn_in_days=d["burn_in_days"],
            acceptance_test_days=d["test_days"],
        )

        est = estimate_lead_time(spec, n_monte_carlo=500)
        return {
            "component_id": component_id,
            "p10_days": est.total_p10_days,
            "p50_days": est.total_p50_days,
            "p90_days": est.total_p90_days,
            "physics_minimum_days": est.physics_minimum_days,
            "breakdown": {
                "queue_days": est.t_queue_days,
                "manufacture_days": est.t_manufacture_days,
                "test_days": est.t_test_days,
                "transport_days": est.t_transport_days,
                "customs_days": est.t_customs_days,
            },
            "uncertainty_band_days": round(est.total_p90_days - est.total_p10_days, 1),
        }

    def _get_fab_utilisation(self, manufacturer: str, component_type: str = "") -> dict:
        # In production this pulls from supplier API / procurement intelligence feeds
        # These are realistic estimates for illustration
        known = {
            "TSMC": 0.91, "Samsung": 0.87, "Intel": 0.79,
            "Vishay": 0.83, "Murata": 0.88, "TDK": 0.82,
            "Eaton": 0.74, "ABB": 0.71, "Schneider": 0.76,
            "Vertiv": 0.78, "Kohler": 0.69,
        }
        util = known.get(manufacturer, 0.75)
        return {
            "manufacturer": manufacturer,
            "utilisation": util,
            "status": "constrained" if util > 0.85 else "elevated" if util > 0.75 else "normal",
            "lead_time_risk": "high" if util > 0.88 else "medium" if util > 0.78 else "low",
        }

    def _get_port_congestion(self, origin_port: str, destination_port: str) -> dict:
        congestion_levels = {
            "Shanghai": 1.4, "Shenzhen": 1.3, "Rotterdam": 1.1,
            "Los Angeles": 1.35, "Long Beach": 1.35, "Singapore": 1.2,
            "Hamburg": 1.05, "Felixstowe": 1.15,
        }
        factor = congestion_levels.get(origin_port, 1.1)
        return {
            "origin_port": origin_port,
            "destination_port": destination_port,
            "congestion_factor": factor,
            "status": "severe" if factor > 1.4 else "elevated" if factor > 1.2 else "normal",
            "extra_days_estimate": round((factor - 1.0) * 7, 1),
        }

    def _get_geopolitical_risk(self, origin_country: str, destination_country: str) -> dict:
        risk_table = {
            ("CN", "US"): 0.42, ("TW", "US"): 0.38, ("CN", "EU"): 0.31,
            ("RU", "EU"): 0.85, ("BY", "EU"): 0.80, ("IR", "US"): 0.92,
        }
        risk = risk_table.get((origin_country, destination_country), 0.10)
        return {
            "origin": origin_country,
            "destination": destination_country,
            "risk_score": risk,
            "level": "critical" if risk > 0.7 else "elevated" if risk > 0.3 else "low",
        }

    def _get_critical_path(self, project_id: str, component_ids: list = None) -> dict:
        # Would pull from project management system in production
        return {
            "project_id": project_id,
            "note": "Critical path data requires connection to project scheduler agent",
            "component_ids": component_ids or [],
        }

    def forecast(self, project_id: str, component_list: list[dict], horizon_days: int = 90) -> dict:
        """
        Main entry point. Runs the full risk forecast for a project.

        component_list: list of dicts with component info
        Returns structured risk report.
        """
        task = (
            f"Run a {horizon_days}-day supply chain risk forecast for project {project_id}.\n\n"
            f"Components to assess:\n{json.dumps(component_list, indent=2)}\n\n"
            f"For each component:\n"
            f"1. Get the physics-based lead time estimate\n"
            f"2. Check fab utilisation for the manufacturer\n"
            f"3. Check port congestion on the shipping route\n"
            f"4. Check geopolitical risk on the trade lane\n"
            f"5. Flag any component where P90 lead time > {horizon_days * 0.7:.0f} days as at-risk\n\n"
            f"Prioritise by impact on project critical path."
        )

        messages = [{"role": "user", "content": task}]

        for _ in range(8):  # max tool rounds
            response = self.client.messages.create(
                model=self.model,
                max_tokens=3000,
                system=SYSTEM_PROMPT,
                tools=self._tools,
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                text = "\n".join(b.text for b in response.content if hasattr(b, "text"))
                return {"project_id": project_id, "horizon_days": horizon_days, "report": text}

            if response.stop_reason != "tool_use":
                break

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                logger.info(f"[Predictor] {block.name}({json.dumps(block.input)[:80]})")
                try:
                    result = self.execute_tool(block.name, block.input)
                    content = json.dumps(result)
                except Exception as e:
                    content = f"Error: {e}"
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": content})

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        return {"project_id": project_id, "error": "Could not complete forecast"}
