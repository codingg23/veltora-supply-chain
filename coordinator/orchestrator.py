"""
SupplyChainOrchestrator — Master coordinator.
Fans out to all 9 specialist agents, resolves conflicts, and synthesises
a unified supply-chain action plan.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

import anthropic

from agents.predictor import SupplyChainPredictor
from agents.procurement import ProcurementOptimizer
from agents.risk import RiskMitigation
from agents.cost import CostOptimizer
from agents.scheduler import ProjectScheduler
from agents.sustainability import SustainabilityOptimizer
from agents.vendor import VendorCoordinator
from agents.quality import QualityAssurance
from agents.logistics import LogisticsCoordinator

MODEL = os.getenv("COORDINATOR_MODEL", "claude-opus-4-5")

COMPONENTS_UNDER_REVIEW = [
    "NVIDIA_H100",
    "Marvell_ASIC_88X9140",
    "Vertiv_PDU_24kW",
    "Amphenol_QSFP_400G",
    "Vishay_Cap_10uF",
]


@dataclass
class AgentResult:
    agent_name: str
    status: str          # "ok" | "error"
    data: Any
    elapsed_ms: int


@dataclass
class OrchestratorOutput:
    timestamp: str
    horizon_days: int
    agent_results: list[AgentResult]
    conflict_resolutions: list[dict]
    action_plan: list[dict]
    risk_level: str
    executive_summary: str


class SupplyChainOrchestrator:
    """
    Coordinates all 9 supply-chain agents.  Runs specialist agents in parallel,
    detects inter-agent conflicts (e.g. procurement wants to order more while
    sustainability wants to reduce air freight emissions), and uses a meta-agent
    (Claude claude-opus-4-5) to resolve them and produce a unified action plan.
    """

    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ["ANTHROPIC_API_KEY"]
        self.client = anthropic.Anthropic(api_key=key)

        # Instantiate all agents
        self.predictor      = SupplyChainPredictor(api_key=key)
        self.procurement    = ProcurementOptimizer(api_key=key)
        self.risk           = RiskMitigation(api_key=key)
        self.cost           = CostOptimizer(api_key=key)
        self.scheduler      = ProjectScheduler(api_key=key)
        self.sustainability = SustainabilityOptimizer(api_key=key)
        self.vendor         = VendorCoordinator(api_key=key)
        self.quality        = QualityAssurance(api_key=key)
        self.logistics      = LogisticsCoordinator(api_key=key)

    # ------------------------------------------------------------------
    # Parallel agent execution
    # ------------------------------------------------------------------

    def _run_predictor(self) -> AgentResult:
        t0 = time.monotonic()
        try:
            # Run for the highest-risk component
            result = self.predictor.forecast("NVIDIA_H100")
            return AgentResult("SupplyChainPredictor", "ok",
                               {"horizon": result.horizon_days,
                                "p50_lead_time": result.p50_days,
                                "confidence": result.confidence,
                                "top_risk": result.top_risk_factor},
                               int((time.monotonic() - t0) * 1000))
        except Exception as exc:
            return AgentResult("SupplyChainPredictor", "error", str(exc),
                               int((time.monotonic() - t0) * 1000))

    def _run_procurement(self) -> AgentResult:
        t0 = time.monotonic()
        results = []
        for cid in ["NVIDIA_H100", "Marvell_ASIC_88X9140"]:
            try:
                d = self.procurement.decide(cid)
                results.append({"component": cid, "action": d.action,
                                 "quantity": d.quantity, "urgency": d.urgency})
            except Exception as exc:
                results.append({"component": cid, "error": str(exc)})
        return AgentResult("ProcurementOptimizer", "ok", results,
                           int((time.monotonic() - t0) * 1000))

    def _run_risk(self) -> AgentResult:
        t0 = time.monotonic()
        try:
            assessment = self.risk.assess("NVIDIA_H100")
            return AgentResult("RiskMitigation", "ok",
                               {"risk_score": assessment.risk_score,
                                "risk_tier": assessment.risk_tier,
                                "spof": assessment.spof_flag,
                                "top_threats": assessment.top_threats[:3]},
                               int((time.monotonic() - t0) * 1000))
        except Exception as exc:
            return AgentResult("RiskMitigation", "error", str(exc),
                               int((time.monotonic() - t0) * 1000))

    def _run_cost(self) -> AgentResult:
        t0 = time.monotonic()
        try:
            rec = self.cost.optimise("NVIDIA_H100", 50)
            return AgentResult("CostOptimizer", "ok",
                               {"action": rec.recommended_action,
                                "optimal_qty": rec.optimal_quantity,
                                "savings_usd": rec.savings_opportunity_usd},
                               int((time.monotonic() - t0) * 1000))
        except Exception as exc:
            return AgentResult("CostOptimizer", "error", str(exc),
                               int((time.monotonic() - t0) * 1000))

    def _run_scheduler(self) -> AgentResult:
        t0 = time.monotonic()
        try:
            analysis = self.scheduler.analyse()
            return AgentResult("ProjectScheduler", "ok",
                               {"duration_days": analysis.project_duration_days,
                                "critical_path": analysis.critical_path,
                                "schedule_risk": analysis.schedule_risk},
                               int((time.monotonic() - t0) * 1000))
        except Exception as exc:
            return AgentResult("ProjectScheduler", "error", str(exc),
                               int((time.monotonic() - t0) * 1000))

    def _run_sustainability(self) -> AgentResult:
        t0 = time.monotonic()
        components = [{"component_id": c, "quantity": 10}
                      for c in COMPONENTS_UNDER_REVIEW]
        try:
            report = self.sustainability.evaluate(components)
            return AgentResult("SustainabilityOptimizer", "ok",
                               {"scope3_tonnes": report.total_scope3_tonnes_co2e,
                                "green_score": report.green_score,
                                "transport_rec": report.recommended_transport_mode},
                               int((time.monotonic() - t0) * 1000))
        except Exception as exc:
            return AgentResult("SustainabilityOptimizer", "error", str(exc),
                               int((time.monotonic() - t0) * 1000))

    def _run_vendor(self) -> AgentResult:
        t0 = time.monotonic()
        try:
            report = self.vendor.review()
            return AgentResult("VendorCoordinator", "ok",
                               {"sla_breaches": report.sla_breaches,
                                "expiring_contracts": report.expiring_contracts,
                                "actions": report.recommended_actions},
                               int((time.monotonic() - t0) * 1000))
        except Exception as exc:
            return AgentResult("VendorCoordinator", "error", str(exc),
                               int((time.monotonic() - t0) * 1000))

    def _run_quality(self) -> AgentResult:
        t0 = time.monotonic()
        try:
            status = self.quality.audit()
            return AgentResult("QualityAssurance", "ok",
                               {"overall_ppm": status.overall_ppm,
                                "critical_issues": status.critical_issues,
                                "suppliers_at_risk": status.suppliers_at_risk},
                               int((time.monotonic() - t0) * 1000))
        except Exception as exc:
            return AgentResult("QualityAssurance", "error", str(exc),
                               int((time.monotonic() - t0) * 1000))

    def _run_logistics(self) -> AgentResult:
        t0 = time.monotonic()
        try:
            plan = self.logistics.plan(
                "NVIDIA_H100", weight_kg=48.0,
                origin="TW", destination="US", urgency="high"
            )
            return AgentResult("LogisticsCoordinator", "ok",
                               {"mode": plan.recommended_mode,
                                "carrier": plan.recommended_carrier,
                                "transit_days": plan.estimated_transit_days,
                                "freight_cost": plan.estimated_freight_cost_usd,
                                "customs_risk": plan.customs_risk,
                                "alerts": plan.active_alerts},
                               int((time.monotonic() - t0) * 1000))
        except Exception as exc:
            return AgentResult("LogisticsCoordinator", "error", str(exc),
                               int((time.monotonic() - t0) * 1000))

    # ------------------------------------------------------------------
    # Conflict detection & meta-agent resolution
    # ------------------------------------------------------------------

    def _detect_conflicts(self, results: list[AgentResult]) -> list[dict]:
        """Heuristic conflict detection between agent recommendations."""
        conflicts = []
        result_map = {r.agent_name: r for r in results if r.status == "ok"}

        # Conflict 1: Procurement wants air-freight speed, Sustainability wants sea
        proc = result_map.get("ProcurementOptimizer")
        sust = result_map.get("SustainabilityOptimizer")
        logi = result_map.get("LogisticsCoordinator")
        if proc and sust and logi:
            proc_urgent = any(
                item.get("urgency") in ("critical", "high")
                for item in (proc.data if isinstance(proc.data, list) else [])
            )
            sust_mode = sust.data.get("transport_rec", "sea") if isinstance(sust.data, dict) else "sea"
            logi_mode = logi.data.get("mode", "sea") if isinstance(logi.data, dict) else "sea"
            if proc_urgent and sust_mode == "sea" and logi_mode == "air":
                conflicts.append({
                    "conflict_id": "C001",
                    "type": "transport_mode",
                    "agents": ["ProcurementOptimizer", "SustainabilityOptimizer", "LogisticsCoordinator"],
                    "description": "Procurement urgency requires air freight; sustainability recommends sea.",
                    "resolution": None,
                })

        # Conflict 2: Risk wants buffer stock increase, Cost wants to minimise inventory
        risk = result_map.get("RiskMitigation")
        cost = result_map.get("CostOptimizer")
        if risk and cost:
            high_risk = isinstance(risk.data, dict) and risk.data.get("risk_tier") in ("critical", "high")
            cost_action = cost.data.get("action", "") if isinstance(cost.data, dict) else ""
            if high_risk and "defer" in cost_action:
                conflicts.append({
                    "conflict_id": "C002",
                    "type": "inventory_level",
                    "agents": ["RiskMitigation", "CostOptimizer"],
                    "description": "Risk agent recommends safety stock increase; cost agent recommends defer.",
                    "resolution": None,
                })

        return conflicts

    def _resolve_conflicts(self, conflicts: list[dict],
                            agent_results: list[AgentResult]) -> list[dict]:
        if not conflicts:
            return []

        summary = json.dumps([
            {"agent": r.agent_name, "data": r.data}
            for r in agent_results if r.status == "ok"
        ], indent=2)

        conflict_desc = json.dumps(conflicts, indent=2)

        prompt = f"""You are the master orchestrator for a 9-agent supply chain system.

The following inter-agent conflicts have been detected:
{conflict_desc}

Here is the full context from all agents:
{summary}

For each conflict, reason through the trade-offs and provide a resolution.
Consider: business risk vs cost vs sustainability vs delivery schedule.
Output a JSON list of resolutions:
[{{"conflict_id": "C001", "resolution": "...", "rationale": "...", "winning_agent": "..."}}]"""

        resp = self.client.messages.create(
            model=MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text if resp.content else "[]"
        try:
            start = raw.index("[")
            end = raw.rindex("]") + 1
            resolutions = json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            resolutions = [{"conflict_id": c["conflict_id"],
                            "resolution": "default to risk-conservative approach",
                            "rationale": "parse error in meta-agent response"}
                           for c in conflicts]

        # Merge resolutions back into conflicts
        res_map = {r["conflict_id"]: r for r in resolutions}
        for c in conflicts:
            c["resolution"] = res_map.get(c["conflict_id"], {}).get("resolution", "unresolved")
        return conflicts

    def _synthesise_action_plan(self, agent_results: list[AgentResult],
                                 resolved_conflicts: list[dict]) -> tuple[list[dict], str]:
        summary = json.dumps([
            {"agent": r.agent_name, "status": r.status, "data": r.data}
            for r in agent_results
        ], indent=2)

        conflicts_text = json.dumps(resolved_conflicts, indent=2) if resolved_conflicts else "None"

        prompt = f"""You are the supply-chain master orchestrator for Veltora, a data-centre
infrastructure company. You have received intelligence from 9 specialist agents.

Agent outputs:
{summary}

Resolved conflicts:
{conflicts_text}

Synthesise a unified 30-60-90 day action plan. Prioritise by business impact.
Output:
1. A JSON action_plan list: [{{"priority": 1, "action": "...", "owner_agent": "...",
   "timeline": "30d|60d|90d", "impact": "...", "effort": "low|medium|high"}}]
2. An executive_summary string (2-3 sentences for a C-level audience).
3. An overall risk_level: "low" | "medium" | "high" | "critical"

Format your response as:
ACTION_PLAN:
<json list>
EXECUTIVE_SUMMARY:
<text>
RISK_LEVEL:
<level>"""

        resp = self.client.messages.create(
            model=MODEL,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text if resp.content else ""

        # Parse action plan
        action_plan = []
        try:
            ap_start = raw.index("ACTION_PLAN:") + len("ACTION_PLAN:")
            ap_end = raw.index("EXECUTIVE_SUMMARY:")
            ap_json = raw[ap_start:ap_end].strip()
            action_plan = json.loads(ap_json)
        except (ValueError, json.JSONDecodeError):
            action_plan = [{"priority": 1, "action": "Review agent outputs manually",
                            "owner_agent": "human", "timeline": "30d",
                            "impact": "high", "effort": "medium"}]

        # Parse summary
        exec_summary = ""
        try:
            es_start = raw.index("EXECUTIVE_SUMMARY:") + len("EXECUTIVE_SUMMARY:")
            es_end = raw.index("RISK_LEVEL:")
            exec_summary = raw[es_start:es_end].strip()
        except ValueError:
            exec_summary = "Supply chain analysis complete. See action plan for details."

        # Parse risk level
        risk_level = "medium"
        try:
            rl_start = raw.index("RISK_LEVEL:") + len("RISK_LEVEL:")
            risk_level = raw[rl_start:].strip().split()[0].lower()
        except (ValueError, IndexError):
            pass

        return action_plan, exec_summary, risk_level

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, horizon_days: int = 90) -> OrchestratorOutput:
        """Run all 9 agents in parallel, resolve conflicts, and synthesise plan."""
        print(f"[Orchestrator] Starting supply-chain analysis — horizon {horizon_days}d")

        runners = [
            self._run_predictor,
            self._run_procurement,
            self._run_risk,
            self._run_cost,
            self._run_scheduler,
            self._run_sustainability,
            self._run_vendor,
            self._run_quality,
            self._run_logistics,
        ]

        agent_results: list[AgentResult] = []
        with ThreadPoolExecutor(max_workers=9) as executor:
            futures = {executor.submit(fn): fn.__name__ for fn in runners}
            for future in as_completed(futures):
                result = future.result()
                agent_results.append(result)
                print(f"  [{result.agent_name}] {result.status} ({result.elapsed_ms}ms)")

        # Sort by agent name for deterministic output
        agent_results.sort(key=lambda r: r.agent_name)

        # Conflict detection & resolution
        conflicts = self._detect_conflicts(agent_results)
        print(f"[Orchestrator] Detected {len(conflicts)} inter-agent conflict(s)")
        resolved = self._resolve_conflicts(conflicts, agent_results)

        # Synthesis
        action_plan, exec_summary, risk_level = self._synthesise_action_plan(
            agent_results, resolved
        )

        return OrchestratorOutput(
            timestamp=datetime.utcnow().isoformat() + "Z",
            horizon_days=horizon_days,
            agent_results=agent_results,
            conflict_resolutions=resolved,
            action_plan=action_plan,
            risk_level=risk_level,
            executive_summary=exec_summary,
        )
