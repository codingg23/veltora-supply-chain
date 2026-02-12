"""
QualityAssurance — Agent 8
Incoming inspection, defect trend analysis, and supplier corrective action tracking.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass

import anthropic

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5")

# Inspection results (last 90 days, simulated)
INSPECTION_LOG: list[dict] = [
    {"lot_id": "LOT-2025-0341", "component": "NVIDIA_H100",      "supplier": "TSMC",
     "qty_received": 20,  "qty_defective": 0, "defect_types": [],
     "date": "2025-03-28", "disposition": "accept"},
    {"lot_id": "LOT-2025-0312", "component": "Vertiv_PDU_24kW",  "supplier": "Vertiv",
     "qty_received": 10,  "qty_defective": 2, "defect_types": ["loose_terminal", "cosmetic"],
     "date": "2025-03-15", "disposition": "accept_with_waiver"},
    {"lot_id": "LOT-2025-0298", "component": "Amphenol_QSFP_400G","supplier": "Amphenol",
     "qty_received": 500, "qty_defective": 9, "defect_types": ["insertion_loss_oor"],
     "date": "2025-03-08", "disposition": "accept_with_waiver"},
    {"lot_id": "LOT-2025-0271", "component": "Vishay_Cap_10uF",  "supplier": "Vishay",
     "qty_received": 50000,"qty_defective": 0,"defect_types": [],
     "date": "2025-02-20", "disposition": "accept"},
    {"lot_id": "LOT-2025-0244", "component": "Marvell_ASIC_88X9140","supplier": "TSMC",
     "qty_received": 200,  "qty_defective": 3,"defect_types": ["solder_bridging"],
     "date": "2025-02-05", "disposition": "reject_and_return"},
]

OPEN_CAPAS: list[dict] = [
    {"capa_id": "CAPA-2025-007", "supplier": "Vertiv", "root_cause": "torque_spec_deviation",
     "due_date": "2025-04-30", "status": "in_progress", "effectiveness_verified": False},
    {"capa_id": "CAPA-2025-003", "supplier": "Amphenol", "root_cause": "fibre_polishing_process",
     "due_date": "2025-03-31", "status": "overdue", "effectiveness_verified": False},
]


def _get_inspection_results(component_id: str | None = None) -> dict:
    logs = INSPECTION_LOG
    if component_id:
        logs = [l for l in logs if l["component"] == component_id]
    total_qty = sum(l["qty_received"] for l in logs)
    total_def = sum(l["qty_defective"] for l in logs)
    ppm = round(total_def / total_qty * 1_000_000, 1) if total_qty else 0
    return {
        "filter_component": component_id,
        "lots_inspected": len(logs),
        "total_units_inspected": total_qty,
        "total_defects": total_def,
        "defect_ppm": ppm,
        "recent_lots": logs[-5:],
    }


def _analyse_defect_trends(supplier: str) -> dict:
    supplier_lots = [l for l in INSPECTION_LOG if l["supplier"] == supplier]
    if not supplier_lots:
        return {"error": f"No inspection data for supplier: {supplier}"}
    defect_types: dict[str, int] = {}
    for lot in supplier_lots:
        for dt in lot["defect_types"]:
            defect_types[dt] = defect_types.get(dt, 0) + 1
    top_defects = sorted(defect_types.items(), key=lambda x: x[1], reverse=True)
    total_units = sum(l["qty_received"] for l in supplier_lots)
    total_defects = sum(l["qty_defective"] for l in supplier_lots)
    ppm = round(total_defects / total_units * 1_000_000, 1) if total_units else 0
    return {
        "supplier": supplier,
        "lots_inspected": len(supplier_lots),
        "defect_ppm": ppm,
        "top_defect_types": [{"type": t, "count": c} for t, c in top_defects[:5]],
        "trend": "improving" if ppm < 30 else "stable" if ppm < 80 else "worsening",
        "capa_needed": ppm > 50,
    }


def _get_open_capas(supplier: str | None = None) -> dict:
    capas = OPEN_CAPAS
    if supplier:
        capas = [c for c in capas if c["supplier"] == supplier]
    overdue = [c for c in capas if c["status"] == "overdue"]
    return {
        "filter_supplier": supplier,
        "total_open_capas": len(capas),
        "overdue_capas": len(overdue),
        "capas": capas,
        "escalation_needed": len(overdue) > 0,
    }


def _calculate_acceptance_sampling_plan(lot_size: int, aql: float = 0.65) -> dict:
    """MIL-STD-1916 style single sampling plan lookup."""
    # Simplified table: (lot_size_max, sample_size, accept_number)
    table = [
        (90,    5,   0),
        (280,   8,   0),
        (500,  13,   0),
        (1200, 20,   0),
        (3200, 32,   0),
        (10000,50,   1),
        (35000,80,   2),
        (float("inf"), 125, 3),
    ]
    for max_size, n, ac in table:
        if lot_size <= max_size:
            return {
                "lot_size": lot_size,
                "aql_pct": aql,
                "sample_size": n,
                "accept_number": ac,
                "reject_number": ac + 1,
                "plan_code": f"SS-{n}-{ac}",
                "standard": "MIL-STD-1916 approximation",
            }
    return {"error": "lot_size out of range"}


def _flag_critical_quality_issues() -> dict:
    high_ppm = [l for l in INSPECTION_LOG if l["qty_received"] > 0 and
                l["qty_defective"] / l["qty_received"] * 1e6 > 5000]
    rejected = [l for l in INSPECTION_LOG if l["disposition"] == "reject_and_return"]
    overdue_capas = [c for c in OPEN_CAPAS if c["status"] == "overdue"]
    critical = []
    for l in rejected:
        critical.append(f"REJECT: {l['lot_id']} {l['component']} from {l['supplier']}")
    for c in overdue_capas:
        critical.append(f"OVERDUE CAPA: {c['capa_id']} at {c['supplier']}")
    return {
        "critical_issues": critical,
        "issue_count": len(critical),
        "escalation_required": len(critical) > 0,
        "rejected_lots_90d": len(rejected),
        "overdue_capas": len(overdue_capas),
    }


TOOLS = [
    {"name": "get_inspection_results",
     "description": "Return incoming inspection results, defect PPM, and recent lots.",
     "input_schema": {"type": "object",
                      "properties": {"component_id": {"type": "string"}}}},
    {"name": "analyse_defect_trends",
     "description": "Analyse defect type trends for a specific supplier.",
     "input_schema": {"type": "object",
                      "properties": {"supplier": {"type": "string"}},
                      "required": ["supplier"]}},
    {"name": "get_open_capas",
     "description": "List open Corrective Action / Preventive Action items, optionally filtered by supplier.",
     "input_schema": {"type": "object",
                      "properties": {"supplier": {"type": "string"}}}},
    {"name": "calculate_acceptance_sampling_plan",
     "description": "Return MIL-STD sampling plan (sample size, accept number) for a lot size and AQL.",
     "input_schema": {"type": "object",
                      "properties": {"lot_size": {"type": "integer"},
                                     "aql": {"type": "number", "default": 0.65}},
                      "required": ["lot_size"]}},
    {"name": "flag_critical_quality_issues",
     "description": "Identify rejected lots, overdue CAPAs, and other critical quality events.",
     "input_schema": {"type": "object", "properties": {}}},
]

DISPATCH = {
    "get_inspection_results": lambda i: _get_inspection_results(**i),
    "analyse_defect_trends": lambda i: _analyse_defect_trends(**i),
    "get_open_capas": lambda i: _get_open_capas(**i),
    "calculate_acceptance_sampling_plan": lambda i: _calculate_acceptance_sampling_plan(**i),
    "flag_critical_quality_issues": lambda i: _flag_critical_quality_issues(),
}


@dataclass
class QualityStatus:
    overall_ppm: float
    critical_issues: list[str]
    suppliers_at_risk: list[str]
    recommended_actions: list[str]
    rationale: str


class QualityAssurance:
    """Monitors incoming quality, tracks CAPAs, and escalates critical defects."""

    SYSTEM = """You are QualityAssurance, Veltora's incoming quality specialist.

Responsibilities:
- Monitor defect PPM across all components and suppliers.
- Identify overdue CAPAs and escalate.
- Flag critical quality events (rejected lots, systematic defect patterns).
- Recommend sampling plan adjustments for high-risk lots.

Return JSON:
{
  "overall_ppm": <float>,
  "critical_issues": ["..."],
  "suppliers_at_risk": ["..."],
  "recommended_actions": ["..."]
}"""

    def __init__(self, api_key: str | None = None):
        self.client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    def audit(self) -> QualityStatus:
        messages: list[dict] = [
            {"role": "user", "content": (
                "Perform a quality audit. Flag any critical issues, analyse defect "
                "trends for Vertiv and Amphenol, check all open CAPAs, and recommend actions."
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
        return QualityStatus(
            overall_ppm=float(data.get("overall_ppm", 0)),
            critical_issues=data.get("critical_issues", []),
            suppliers_at_risk=data.get("suppliers_at_risk", []),
            recommended_actions=data.get("recommended_actions", []),
            rationale=raw[:500],
        )
