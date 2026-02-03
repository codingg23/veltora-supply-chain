"""
ProjectScheduler — Agent 5
CPM / critical-path scheduling for data-centre build programmes.
Identifies float, bottlenecks, and schedule compression opportunities.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import anthropic

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5")

# ---------------------------------------------------------------------------
# Project task data (simplified DC build programme)
# ---------------------------------------------------------------------------

TASKS: dict[str, dict] = {
    "T01": {"name": "Site civil works",          "duration_days": 45, "predecessors": [],          "resources": ["civil_team"]},
    "T02": {"name": "Power infrastructure",      "duration_days": 30, "predecessors": ["T01"],      "resources": ["electrical_team", "Vertiv_PDU_24kW"]},
    "T03": {"name": "Cooling system install",    "duration_days": 25, "predecessors": ["T01"],      "resources": ["hvac_team"]},
    "T04": {"name": "Network spine cabling",     "duration_days": 15, "predecessors": ["T02"],      "resources": ["network_team", "Amphenol_QSFP_400G"]},
    "T05": {"name": "Server rack installation",  "duration_days": 20, "predecessors": ["T02","T03"],"resources": ["dc_ops", "NVIDIA_H100"]},
    "T06": {"name": "GPU node commissioning",    "duration_days": 14, "predecessors": ["T04","T05"],"resources": ["dc_ops", "Marvell_ASIC_88X9140"]},
    "T07": {"name": "Network fabric testing",    "duration_days": 10, "predecessors": ["T04"],      "resources": ["network_team"]},
    "T08": {"name": "Software stack deployment", "duration_days": 12, "predecessors": ["T06","T07"],"resources": ["platform_team"]},
    "T09": {"name": "Load testing & burn-in",    "duration_days": 7,  "predecessors": ["T08"],      "resources": ["dc_ops"]},
    "T10": {"name": "Customer acceptance",       "duration_days": 3,  "predecessors": ["T09"],      "resources": ["account_team"]},
}

RESOURCE_COSTS_PER_DAY: dict[str, float] = {
    "civil_team": 8_500,
    "electrical_team": 6_200,
    "hvac_team": 5_800,
    "network_team": 4_500,
    "dc_ops": 3_800,
    "platform_team": 5_200,
    "account_team": 2_400,
}


# ---------------------------------------------------------------------------
# CPM algorithm
# ---------------------------------------------------------------------------

def _cpm(tasks: dict) -> dict[str, dict]:
    """Forward + backward pass to compute ES, EF, LS, LF, float, critical."""
    # Forward pass
    ef: dict[str, int] = {}
    es: dict[str, int] = {}
    for tid, task in tasks.items():
        preds = task["predecessors"]
        es[tid] = max((ef[p] for p in preds), default=0)
        ef[tid] = es[tid] + task["duration_days"]

    project_duration = max(ef.values())

    # Backward pass
    ls: dict[str, int] = {}
    lf: dict[str, int] = {}
    for tid in reversed(list(tasks.keys())):
        successors = [t for t, d in tasks.items() if tid in d["predecessors"]]
        lf[tid] = min((ls[s] for s in successors), default=project_duration)
        ls[tid] = lf[tid] - tasks[tid]["duration_days"]

    result = {}
    for tid in tasks:
        total_float = ls[tid] - es[tid]
        result[tid] = {
            "name": tasks[tid]["name"],
            "duration_days": tasks[tid]["duration_days"],
            "early_start": es[tid],
            "early_finish": ef[tid],
            "late_start": ls[tid],
            "late_finish": lf[tid],
            "total_float_days": total_float,
            "critical": total_float == 0,
            "predecessors": tasks[tid]["predecessors"],
        }
    return result


def _get_critical_path() -> dict:
    schedule = _cpm(TASKS)
    critical = [tid for tid, t in schedule.items() if t["critical"]]
    project_duration = max(t["early_finish"] for t in schedule.values())
    return {
        "project_duration_days": project_duration,
        "critical_path_tasks": critical,
        "critical_path_names": [schedule[t]["name"] for t in critical],
        "full_schedule": schedule,
    }


def _get_task_float(task_id: str) -> dict:
    schedule = _cpm(TASKS)
    task = schedule.get(task_id)
    if not task:
        return {"error": f"Unknown task: {task_id}"}
    return {"task_id": task_id, **task}


def _analyse_resource_bottlenecks() -> dict:
    schedule = _cpm(TASKS)
    # Count resource-days on critical path
    resource_critical_days: dict[str, int] = {}
    for tid, data in schedule.items():
        if data["critical"]:
            for r in TASKS[tid]["resources"]:
                if r in RESOURCE_COSTS_PER_DAY:
                    resource_critical_days[r] = (
                        resource_critical_days.get(r, 0) + data["duration_days"]
                    )
    bottlenecks = sorted(
        [{"resource": r, "critical_days": d, "daily_cost_usd": RESOURCE_COSTS_PER_DAY.get(r, 0)}
         for r, d in resource_critical_days.items()],
        key=lambda x: x["critical_days"], reverse=True,
    )
    return {
        "bottleneck_resources": bottlenecks[:5],
        "recommendation": (
            f"Focus crashing budget on {bottlenecks[0]['resource']} "
            f"({bottlenecks[0]['critical_days']} critical-path days)"
            if bottlenecks else "No resource bottlenecks on critical path"
        ),
    }


def _simulate_delay(task_id: str, delay_days: int) -> dict:
    if task_id not in TASKS:
        return {"error": f"Unknown task: {task_id}"}
    modified = {k: dict(v) for k, v in TASKS.items()}
    modified[task_id]["duration_days"] += delay_days
    original_schedule = _cpm(TASKS)
    new_schedule = _cpm(modified)
    orig_dur = max(t["early_finish"] for t in original_schedule.values())
    new_dur = max(t["early_finish"] for t in new_schedule.values())
    return {
        "delayed_task": task_id,
        "task_name": TASKS[task_id]["name"],
        "delay_days": delay_days,
        "original_project_duration": orig_dur,
        "new_project_duration": new_dur,
        "project_slip_days": new_dur - orig_dur,
        "on_critical_path": original_schedule[task_id]["critical"],
    }


def _get_crash_options(target_reduction_days: int) -> dict:
    schedule = _cpm(TASKS)
    critical_tasks = [tid for tid, t in schedule.items() if t["critical"]]
    options = []
    for tid in critical_tasks:
        task = TASKS[tid]
        # Assume 20% crash possible at 50% cost premium
        crashable = max(1, int(task["duration_days"] * 0.20))
        resource_cost = sum(
            RESOURCE_COSTS_PER_DAY.get(r, 0) for r in task["resources"]
        )
        crash_cost = resource_cost * 0.5 * crashable
        options.append({
            "task_id": tid,
            "task_name": task["name"],
            "max_crash_days": crashable,
            "crash_cost_usd": round(crash_cost),
            "cost_per_day_saved_usd": round(crash_cost / crashable) if crashable else 0,
        })
    options.sort(key=lambda x: x["cost_per_day_saved_usd"])
    # Greedy selection
    selected, total_reduction, total_cost = [], 0, 0
    for o in options:
        if total_reduction >= target_reduction_days:
            break
        take = min(o["max_crash_days"], target_reduction_days - total_reduction)
        total_reduction += take
        total_cost += o["cost_per_day_saved_usd"] * take
        selected.append({**o, "days_crashed": take})
    return {
        "target_reduction_days": target_reduction_days,
        "achievable_reduction_days": total_reduction,
        "total_crash_cost_usd": round(total_cost),
        "selected_tasks": selected,
    }


TOOLS: list[dict] = [
    {
        "name": "get_critical_path",
        "description": "Run CPM to find the critical path and project duration.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_task_float",
        "description": "Return float (schedule slack) for a specific task.",
        "input_schema": {
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
        },
    },
    {
        "name": "analyse_resource_bottlenecks",
        "description": "Identify which resources have the most critical-path days.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "simulate_delay",
        "description": "Simulate the knock-on effect of a task delay on the overall project end date.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "delay_days": {"type": "integer"},
            },
            "required": ["task_id", "delay_days"],
        },
    },
    {
        "name": "get_crash_options",
        "description": "Get cost-optimal schedule compression (crashing) plan to hit a target reduction.",
        "input_schema": {
            "type": "object",
            "properties": {"target_reduction_days": {"type": "integer"}},
            "required": ["target_reduction_days"],
        },
    },
]

DISPATCH = {
    "get_critical_path": lambda i: _get_critical_path(),
    "get_task_float": lambda i: _get_task_float(**i),
    "analyse_resource_bottlenecks": lambda i: _analyse_resource_bottlenecks(),
    "simulate_delay": lambda i: _simulate_delay(**i),
    "get_crash_options": lambda i: _get_crash_options(**i),
}


@dataclass
class ScheduleAnalysis:
    project_duration_days: int
    critical_path: list[str]
    bottleneck_resource: str
    schedule_risk: str
    crash_recommendation: str
    rationale: str


class ProjectScheduler:
    """CPM-based scheduling agent that identifies critical path, float, and
    compression options for DC build programmes."""

    SYSTEM = """You are ProjectScheduler, a project management specialist for Veltora.

Your responsibilities:
- Run CPM to identify the critical path and project duration.
- Identify tasks with zero float (schedule risk).
- Spot resource bottlenecks on the critical path.
- Simulate the impact of supply-chain delays on project completion.
- Recommend cost-optimal crashing strategies.

Call tools to gather schedule data before recommending.
Return your final answer as JSON:
{
  "project_duration_days": <int>,
  "critical_path": ["T01", ...],
  "bottleneck_resource": "...",
  "schedule_risk": "low|medium|high|critical",
  "crash_recommendation": "...",
  "rationale": "..."
}"""

    def __init__(self, api_key: str | None = None):
        self.client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    def analyse(self, context: str = "") -> ScheduleAnalysis:
        messages: list[dict] = [
            {
                "role": "user",
                "content": (
                    "Perform a full schedule analysis for the current DC build programme. "
                    "Find the critical path, identify the top resource bottleneck, "
                    "simulate a 14-day delay on the most critical task, and recommend "
                    "whether crashing is cost-effective. "
                    + (f"Context: {context}" if context else "")
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

        return ScheduleAnalysis(
            project_duration_days=int(data.get("project_duration_days", 0)),
            critical_path=data.get("critical_path", []),
            bottleneck_resource=data.get("bottleneck_resource", "unknown"),
            schedule_risk=data.get("schedule_risk", "medium"),
            crash_recommendation=data.get("crash_recommendation", ""),
            rationale=raw[:600],
        )
