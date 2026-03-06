"""
Unit tests for supply-chain agents using mock Anthropic client.
No real API calls are made.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers to build mock Anthropic responses
# ---------------------------------------------------------------------------

def _mock_tool_response(tool_name: str, tool_input: dict, tool_id: str = "tool_1"):
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.input = tool_input
    block.id = tool_id
    return block

def _mock_text_response(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block

def _mock_api_response(content: list, stop_reason: str = "end_turn"):
    resp = MagicMock()
    resp.content = content
    resp.stop_reason = stop_reason
    return resp


# ---------------------------------------------------------------------------
# ProcurementOptimizer
# ---------------------------------------------------------------------------

class TestProcurementOptimizer:
    @patch("agents.procurement.anthropic.Anthropic")
    def test_decide_returns_dataclass(self, MockAnthropic):
        decision_json = json.dumps({
            "component_id": "NVIDIA_H100",
            "action": "order_now",
            "quantity": 40,
            "estimated_cost_usd": 1_240_000,
            "urgency": "high",
            "rationale": "Below reorder point with 84-day lead time.",
        })
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = _mock_api_response(
            [_mock_text_response(decision_json)]
        )

        from agents.procurement import ProcurementOptimizer
        agent = ProcurementOptimizer(api_key="test-key")
        result = agent.decide("NVIDIA_H100")

        assert result.action == "order_now"
        assert result.quantity == 40
        assert result.urgency == "high"
        assert result.estimated_cost_usd == 1_240_000

    @patch("agents.procurement.anthropic.Anthropic")
    def test_tool_call_loop(self, MockAnthropic):
        """Verify agent processes a tool_use response before final answer."""
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client

        tool_resp = _mock_api_response(
            [_mock_tool_response("get_inventory_position", {"component_id": "NVIDIA_H100"})],
            stop_reason="tool_use"
        )
        final_resp = _mock_api_response(
            [_mock_text_response(json.dumps({
                "component_id": "NVIDIA_H100",
                "action": "monitor",
                "quantity": 0,
                "estimated_cost_usd": 0,
                "urgency": "low",
                "rationale": "Inventory OK",
            }))]
        )
        mock_client.messages.create.side_effect = [tool_resp, final_resp]

        from agents.procurement import ProcurementOptimizer
        agent = ProcurementOptimizer(api_key="test-key")
        result = agent.decide("NVIDIA_H100")
        assert result.action == "monitor"
        assert mock_client.messages.create.call_count == 2


# ---------------------------------------------------------------------------
# RiskMitigation
# ---------------------------------------------------------------------------

class TestRiskMitigation:
    @patch("agents.risk.anthropic.Anthropic")
    def test_assess_returns_dataclass(self, MockAnthropic):
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = _mock_api_response(
            [_mock_text_response(json.dumps({
                "component_id": "NVIDIA_H100",
                "risk_score": 0.72,
                "risk_tier": "critical",
                "top_threats": [{"event": "taiwan_strait_tension", "ev_delay_days": 32}],
                "recommended_hedges": [{"strategy": "dual_source", "priority": "high"}],
                "spof_flag": True,
            }))]
        )

        from agents.risk import RiskMitigation
        agent = RiskMitigation(api_key="test-key")
        result = agent.assess("NVIDIA_H100")

        assert result.risk_score == pytest.approx(0.72)
        assert result.risk_tier == "critical"
        assert result.spof_flag is True
        assert len(result.top_threats) == 1


# ---------------------------------------------------------------------------
# CostOptimizer
# ---------------------------------------------------------------------------

class TestCostOptimizer:
    @patch("agents.cost.anthropic.Anthropic")
    def test_optimise_calculates_savings(self, MockAnthropic):
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = _mock_api_response(
            [_mock_text_response(json.dumps({
                "component_id": "NVIDIA_H100",
                "recommended_action": "negotiate",
                "optimal_quantity": 50,
                "estimated_unit_cost_usd": 28_500,
                "total_cost_usd": 1_425_000,
                "savings_opportunity_usd": 112_500,
                "rationale": "Contract price 8% below spot; volume discount at 50 units.",
            }))]
        )

        from agents.cost import CostOptimizer
        agent = CostOptimizer(api_key="test-key")
        result = agent.optimise("NVIDIA_H100", 50)

        assert result.recommended_action == "negotiate"
        assert result.savings_opportunity_usd == pytest.approx(112_500)
        assert result.optimal_quantity == 50


# ---------------------------------------------------------------------------
# ProjectScheduler
# ---------------------------------------------------------------------------

class TestProjectScheduler:
    @patch("agents.scheduler.anthropic.Anthropic")
    def test_analyse_returns_schedule(self, MockAnthropic):
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = _mock_api_response(
            [_mock_text_response(json.dumps({
                "project_duration_days": 109,
                "critical_path": ["T01", "T02", "T04", "T05", "T06", "T08", "T09", "T10"],
                "bottleneck_resource": "dc_ops",
                "schedule_risk": "medium",
                "crash_recommendation": "Crash T05 to recover 4 days for $68k",
                "rationale": "CPM analysis complete.",
            }))]
        )

        from agents.scheduler import ProjectScheduler
        agent = ProjectScheduler(api_key="test-key")
        result = agent.analyse()

        assert result.project_duration_days == 109
        assert "T01" in result.critical_path
        assert result.schedule_risk == "medium"


# ---------------------------------------------------------------------------
# Physics tool functions (unit tests without mocking)
# ---------------------------------------------------------------------------

class TestInventoryTheory:
    def test_eoq_positive(self):
        from agents.procurement import economic_order_quantity
        eoq = economic_order_quantity(
            demand_per_day=0.8, order_cost=1550, holding_cost_per_unit_day=9.3
        )
        assert eoq > 0

    def test_reorder_point_above_mean(self):
        from agents.procurement import reorder_point
        rop = reorder_point(demand_per_day=0.8, lead_time_days=84, service_level=0.95)
        mean_demand = 0.8 * 84
        assert rop > mean_demand   # safety stock adds buffer

    def test_betweenness_centrality_non_negative(self):
        from agents.risk import _betweenness_centrality, DEPENDENCY_GRAPH
        centrality = _betweenness_centrality(DEPENDENCY_GRAPH)
        assert all(v >= 0 for v in centrality.values())

    def test_disruption_risk_score_bounded(self):
        from agents.risk import _score_disruption_risk
        result = _score_disruption_risk("NVIDIA_H100")
        assert 0.0 <= result["risk_score"] <= 1.0

    def test_cpm_identifies_critical_tasks(self):
        from agents.scheduler import _cpm, TASKS
        schedule = _cpm(TASKS)
        critical = [t for t, d in schedule.items() if d["critical"]]
        assert len(critical) > 0
        # T01 should always be critical (everything depends on it)
        assert "T01" in critical
