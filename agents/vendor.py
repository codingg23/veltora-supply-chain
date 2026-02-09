"""
VendorCoordinator — Agent 7
Manages supplier relationships, SLA compliance, and qualification pipelines.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass

import anthropic

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5")

VENDOR_DB: dict[str, dict] = {
    "TSMC": {
        "status": "approved", "tier": 1,
        "on_time_delivery_pct_12m": 88.2, "defect_ppm": 12, "credit_days": 45,
        "contract_expiry": "2026-06-30", "strategic": True,
        "contacts": [{"name": "Li Wei", "role": "account_manager", "region": "APAC"}],
    },
    "Samsung_Components": {
        "status": "approved", "tier": 1,
        "on_time_delivery_pct_12m": 91.5, "defect_ppm": 8, "credit_days": 30,
        "contract_expiry": "2025-12-31", "strategic": True,
        "contacts": [{"name": "Park Jun", "role": "account_manager", "region": "APAC"}],
    },
    "Vishay": {
        "status": "approved", "tier": 2,
        "on_time_delivery_pct_12m": 95.1, "defect_ppm": 4, "credit_days": 60,
        "contract_expiry": "2026-03-31", "strategic": False,
        "contacts": [{"name": "Mark Chen", "role": "sales", "region": "NA"}],
    },
    "Vertiv": {
        "status": "approved", "tier": 1,
        "on_time_delivery_pct_12m": 82.3, "defect_ppm": 45, "credit_days": 30,
        "contract_expiry": "2025-09-30", "strategic": True,
        "contacts": [{"name": "Sarah Mills", "role": "account_director", "region": "EMEA"}],
    },
    "Amphenol": {
        "status": "approved", "tier": 2,
        "on_time_delivery_pct_12m": 93.7, "defect_ppm": 18, "credit_days": 45,
        "contract_expiry": "2026-01-31", "strategic": False,
        "contacts": [{"name": "Tom Hayes", "role": "sales", "region": "NA"}],
    },
    "New_Cooling_Vendor": {
        "status": "qualification", "tier": 3,
        "on_time_delivery_pct_12m": None, "defect_ppm": None, "credit_days": 30,
        "contract_expiry": None, "strategic": False,
        "contacts": [{"name": "Alex Dumont", "role": "sales", "region": "EMEA"}],
        "qualification_stage": "sample_testing",
        "qualification_start": "2025-01-15",
    },
}

SLA_THRESHOLDS = {"on_time_delivery_pct_min": 90.0, "defect_ppm_max": 50}


def _get_vendor_profile(vendor_id: str) -> dict:
    v = VENDOR_DB.get(vendor_id)
    if not v:
        return {"error": f"Unknown vendor: {vendor_id}"}
    otd = v.get("on_time_delivery_pct_12m")
    defect = v.get("defect_ppm")
    sla_breach = (
        (otd is not None and otd < SLA_THRESHOLDS["on_time_delivery_pct_min"]) or
        (defect is not None and defect > SLA_THRESHOLDS["defect_ppm_max"])
    )
    return {"vendor_id": vendor_id, **v, "sla_breach": sla_breach}


def _check_sla_compliance(vendor_id: str) -> dict:
    v = VENDOR_DB.get(vendor_id)
    if not v:
        return {"error": f"Unknown vendor: {vendor_id}"}
    otd = v.get("on_time_delivery_pct_12m")
    defect = v.get("defect_ppm")
    issues = []
    if otd is not None and otd < SLA_THRESHOLDS["on_time_delivery_pct_min"]:
        issues.append(f"OTD {otd}% below threshold {SLA_THRESHOLDS['on_time_delivery_pct_min']}%")
    if defect is not None and defect > SLA_THRESHOLDS["defect_ppm_max"]:
        issues.append(f"Defect PPM {defect} above threshold {SLA_THRESHOLDS['defect_ppm_max']}")
    return {
        "vendor_id": vendor_id,
        "compliant": len(issues) == 0,
        "issues": issues,
        "recommended_action": (
            "issue_corrective_action_plan" if issues else "no_action_required"
        ),
    }


def _get_qualification_pipeline() -> dict:
    in_qualification = [
        {"vendor_id": vid, "stage": v.get("qualification_stage"),
         "start_date": v.get("qualification_start")}
        for vid, v in VENDOR_DB.items()
        if v.get("status") == "qualification"
    ]
    return {
        "vendors_in_qualification": in_qualification,
        "count": len(in_qualification),
        "stages": ["rfi", "audit", "sample_testing", "trial_order", "approved"],
    }


def _get_contracts_expiring(days_ahead: int = 90) -> dict:
    from datetime import date, timedelta
    cutoff = date.today() + timedelta(days=days_ahead)
    expiring = []
    for vid, v in VENDOR_DB.items():
        exp = v.get("contract_expiry")
        if exp:
            exp_date = date.fromisoformat(exp)
            if exp_date <= cutoff:
                days_left = (exp_date - date.today()).days
                expiring.append({
                    "vendor_id": vid,
                    "contract_expiry": exp,
                    "days_until_expiry": days_left,
                    "strategic": v.get("strategic", False),
                    "urgency": "critical" if days_left < 30 else "high" if days_left < 60 else "medium",
                })
    expiring.sort(key=lambda x: x["days_until_expiry"])
    return {"expiring_contracts": expiring, "horizon_days": days_ahead}


def _score_vendor_health(vendor_id: str) -> dict:
    v = VENDOR_DB.get(vendor_id)
    if not v:
        return {"error": f"Unknown vendor: {vendor_id}"}
    otd = v.get("on_time_delivery_pct_12m", 0) or 0
    defect = v.get("defect_ppm", 100) or 100
    otd_score = otd / 100
    defect_score = max(0, 1 - defect / 100)
    tier_score = {1: 1.0, 2: 0.7, 3: 0.4}.get(v.get("tier", 3), 0.4)
    health = 0.50 * otd_score + 0.35 * defect_score + 0.15 * tier_score
    return {
        "vendor_id": vendor_id,
        "health_score": round(health, 3),
        "health_tier": "A" if health > 0.85 else "B" if health > 0.70 else "C",
        "otd_score": round(otd_score, 3),
        "defect_score": round(defect_score, 3),
        "tier_score": tier_score,
    }


TOOLS = [
    {"name": "get_vendor_profile",
     "description": "Retrieve full vendor profile including SLA status.",
     "input_schema": {"type": "object",
                      "properties": {"vendor_id": {"type": "string"}},
                      "required": ["vendor_id"]}},
    {"name": "check_sla_compliance",
     "description": "Check whether a vendor is meeting OTD and defect SLAs.",
     "input_schema": {"type": "object",
                      "properties": {"vendor_id": {"type": "string"}},
                      "required": ["vendor_id"]}},
    {"name": "get_qualification_pipeline",
     "description": "List all vendors currently in the qualification process.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_contracts_expiring",
     "description": "Find vendor contracts expiring within a given number of days.",
     "input_schema": {"type": "object",
                      "properties": {"days_ahead": {"type": "integer", "default": 90}}}},
    {"name": "score_vendor_health",
     "description": "Compute a composite vendor health score (OTD, defects, tier).",
     "input_schema": {"type": "object",
                      "properties": {"vendor_id": {"type": "string"}},
                      "required": ["vendor_id"]}},
]

DISPATCH = {
    "get_vendor_profile": lambda i: _get_vendor_profile(**i),
    "check_sla_compliance": lambda i: _check_sla_compliance(**i),
    "get_qualification_pipeline": lambda i: _get_qualification_pipeline(),
    "get_contracts_expiring": lambda i: _get_contracts_expiring(**i),
    "score_vendor_health": lambda i: _score_vendor_health(**i),
}


@dataclass
class VendorReport:
    sla_breaches: list[str]
    expiring_contracts: list[str]
    qualification_updates: list[str]
    recommended_actions: list[str]
    rationale: str


class VendorCoordinator:
    """Manages supplier relationships, flags SLA breaches and contract renewals."""

    SYSTEM = """You are VendorCoordinator, Veltora's supplier relationship manager.

Tasks:
- Audit vendor SLA compliance for key suppliers.
- Flag contracts expiring within 90 days.
- Check the qualification pipeline for new suppliers.
- Score overall vendor health.
- Recommend corrective actions and escalations.

Return final JSON:
{
  "sla_breaches": ["vendor: issue"],
  "expiring_contracts": ["vendor: date"],
  "qualification_updates": ["vendor: stage"],
  "recommended_actions": ["..."]
}"""

    def __init__(self, api_key: str | None = None):
        self.client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    def review(self) -> VendorReport:
        messages: list[dict] = [
            {"role": "user", "content": (
                "Perform a full vendor health review. Check SLA compliance for all "
                "tier-1 suppliers (TSMC, Samsung_Components, Vertiv), find contracts "
                "expiring in the next 90 days, and check the qualification pipeline."
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
        return VendorReport(
            sla_breaches=data.get("sla_breaches", []),
            expiring_contracts=data.get("expiring_contracts", []),
            qualification_updates=data.get("qualification_updates", []),
            recommended_actions=data.get("recommended_actions", []),
            rationale=raw[:500],
        )
