"""
conflict.py

ConflictResolver - scores competing agent recommendations and resolves
disagreements using a priority matrix.

Priority matrix (highest wins when agents disagree):
  1. Safety (Class A quality deviation → mandatory hold)
  2. Schedule (critical path impact → must act)
  3. Cost (optimise within safety/schedule constraints)
  4. Sustainability (optimise within cost constraints)

Conflict detection:
  Agents are in conflict when their recommended actions are mutually exclusive:
    - Cost says "wait 3 weeks for price to drop"
    - Risk says "order now, 65% probability of 30-day supply disruption"
    → These are conflicting. Risk wins by priority matrix.

  Agents are compatible when recommendations can be combined:
    - Procurement says "use Eaton for PDUs"
    - Sustainability says "Eaton has CDP-A rating, good choice"
    → No conflict.

Scoring model:
  Each recommendation is scored on 4 dimensions:
    urgency_score:   how urgently does this need to happen? (0-1)
    cost_impact:     absolute dollar impact (positive = cost saving, negative = cost)
    risk_exposure:   probability-weighted delay days * cost_per_day
    sustainability:  tCO2e impact (negative = more carbon)

  Resolution priority:
    safety_flag → always wins (deterministic)
    urgency > 0.8 → second tier
    score = urgency * 0.40 + risk_reduction * 0.35 + cost_saving * 0.15 + sustainability * 0.10
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

COST_PER_DELAY_DAY_USD = 50_000  # $/day project delay (from README)


@dataclass
class AgentRecommendation:
    """Structured recommendation from a single agent."""
    agent_name: str
    action: str                    # Short description of what the agent recommends
    rationale: str                 # Why
    urgency: float                 # 0-1, how time-sensitive
    cost_impact_usd: float         # Positive = saving, negative = additional cost
    delay_risk_days: float         # Expected project delay days if NOT acted on
    delay_probability: float       # Probability the delay materialises (0-1)
    carbon_impact_tco2e: float     # Positive = reduction, negative = increase
    safety_flag: bool = False      # True = mandatory, overrides all scoring
    schedule_flag: bool = False    # True = critical path impact
    requires_human_approval: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class Resolution:
    """Output of the conflict resolver."""
    winning_recommendation: AgentRecommendation
    conflicting_recommendations: list[AgentRecommendation]
    resolution_reason: str
    priority_applied: str          # "safety", "schedule", "cost", "sustainability", "scoring"
    winning_score: float
    scores: dict[str, float]       # agent_name -> score
    compatible_recommendations: list[AgentRecommendation] = field(default_factory=list)
    human_approval_required: bool = False


class ConflictResolver:
    """
    Resolves conflicts between agent recommendations using the priority matrix.

    Usage:
        resolver = ConflictResolver()
        resolution = resolver.resolve(recommendations)
    """

    def __init__(
        self,
        cost_per_delay_day_usd: float = COST_PER_DELAY_DAY_USD,
        priority_weights: Optional[dict] = None,
    ):
        self.cost_per_delay_day = cost_per_delay_day_usd
        # Scoring weights for non-deterministic resolution
        self.weights = priority_weights or {
            "urgency": 0.40,
            "risk_reduction": 0.35,
            "cost_saving": 0.15,
            "sustainability": 0.10,
        }

    def resolve(self, recommendations: list[AgentRecommendation]) -> Resolution:
        """
        Resolve a set of (potentially conflicting) agent recommendations.

        Returns the winning recommendation and full resolution reasoning.
        """
        if not recommendations:
            raise ValueError("No recommendations to resolve")

        if len(recommendations) == 1:
            rec = recommendations[0]
            return Resolution(
                winning_recommendation=rec,
                conflicting_recommendations=[],
                resolution_reason="Single recommendation, no conflict",
                priority_applied="none",
                winning_score=1.0,
                scores={rec.agent_name: 1.0},
                human_approval_required=rec.requires_human_approval,
            )

        # Tier 1: Safety flags are mandatory
        safety_recs = [r for r in recommendations if r.safety_flag]
        if safety_recs:
            winner = safety_recs[0]
            conflicts = [r for r in recommendations if r is not winner and self._is_conflicting(winner, r)]
            return Resolution(
                winning_recommendation=winner,
                conflicting_recommendations=conflicts,
                resolution_reason=f"Safety flag from {winner.agent_name}: mandatory override. {winner.rationale}",
                priority_applied="safety",
                winning_score=1.0,
                scores={r.agent_name: (1.0 if r.safety_flag else 0.0) for r in recommendations},
                human_approval_required=True,
            )

        # Tier 2: Schedule flags (critical path)
        schedule_recs = [r for r in recommendations if r.schedule_flag]
        if schedule_recs:
            # Among schedule-flagged, pick highest urgency
            winner = max(schedule_recs, key=lambda r: r.urgency)
            conflicts = [r for r in recommendations if r is not winner and self._is_conflicting(winner, r)]
            compatible = [r for r in recommendations if r is not winner and not self._is_conflicting(winner, r)]
            return Resolution(
                winning_recommendation=winner,
                conflicting_recommendations=conflicts,
                resolution_reason=(
                    f"Critical path impact from {winner.agent_name} (urgency={winner.urgency:.2f}). "
                    f"{winner.rationale}"
                ),
                priority_applied="schedule",
                winning_score=winner.urgency,
                scores={r.agent_name: r.urgency for r in recommendations},
                compatible_recommendations=compatible,
                human_approval_required=winner.requires_human_approval,
            )

        # Tier 3+: Scoring
        scores = self._score_all(recommendations)
        winner = max(recommendations, key=lambda r: scores[r.agent_name])
        conflicts = [r for r in recommendations if r is not winner and self._is_conflicting(winner, r)]
        compatible = [r for r in recommendations if r is not winner and not self._is_conflicting(winner, r)]

        winning_score = scores[winner.agent_name]
        priority = self._infer_priority(winner, scores)

        reason_parts = [f"{winner.agent_name} wins with score {winning_score:.3f}."]
        if conflicts:
            reason_parts.append(
                f"Overrides: {', '.join(r.agent_name for r in conflicts)}."
            )
        reason_parts.append(winner.rationale)

        return Resolution(
            winning_recommendation=winner,
            conflicting_recommendations=conflicts,
            resolution_reason=" ".join(reason_parts),
            priority_applied=priority,
            winning_score=winning_score,
            scores=scores,
            compatible_recommendations=compatible,
            human_approval_required=winner.requires_human_approval,
        )

    def _score_all(self, recommendations: list[AgentRecommendation]) -> dict[str, float]:
        """Compute composite score for each recommendation."""
        # Normalise each dimension across the recommendation set
        urgencies = [r.urgency for r in recommendations]
        risk_values = [r.delay_risk_days * r.delay_probability * self.cost_per_delay_day for r in recommendations]
        cost_savings = [r.cost_impact_usd for r in recommendations]
        carbon_savings = [r.carbon_impact_tco2e for r in recommendations]

        def _norm(values: list[float]) -> list[float]:
            rng = max(values) - min(values)
            if rng == 0:
                return [0.5] * len(values)
            return [(v - min(values)) / rng for v in values]

        norm_urgency = _norm(urgencies)
        norm_risk = _norm(risk_values)
        norm_cost = _norm(cost_savings)
        norm_carbon = _norm(carbon_savings)

        scores = {}
        for i, rec in enumerate(recommendations):
            score = (
                self.weights["urgency"] * norm_urgency[i]
                + self.weights["risk_reduction"] * norm_risk[i]
                + self.weights["cost_saving"] * norm_cost[i]
                + self.weights["sustainability"] * norm_carbon[i]
            )
            scores[rec.agent_name] = round(score, 4)

        return scores

    def _is_conflicting(self, rec_a: AgentRecommendation, rec_b: AgentRecommendation) -> bool:
        """
        Simple conflict detection: recommendations conflict if they advocate opposite actions.
        In production, this would use structured action types rather than string matching.
        """
        action_a = rec_a.action.lower()
        action_b = rec_b.action.lower()

        conflict_pairs = [
            ({"buy", "order", "expedite", "place_order"}, {"wait", "delay", "hold", "defer"}),
            ({"hold_shipment"}, {"release", "approve", "pass"}),
            ({"switch_supplier"}, {"keep", "maintain", "continue"}),
        ]
        for set_a, set_b in conflict_pairs:
            a_in_a = any(kw in action_a for kw in set_a)
            a_in_b = any(kw in action_b for kw in set_a)
            b_in_a = any(kw in action_a for kw in set_b)
            b_in_b = any(kw in action_b for kw in set_b)
            if (a_in_a and b_in_b) or (b_in_a and a_in_b):
                return True

        # If urgency delta is very large, treat as conflicting
        if abs(rec_a.urgency - rec_b.urgency) > 0.5 and rec_a.cost_impact_usd * rec_b.cost_impact_usd < 0:
            return True

        return False

    def _infer_priority(self, winner: AgentRecommendation, scores: dict[str, float]) -> str:
        """Infer which priority tier drove the decision."""
        if winner.safety_flag:
            return "safety"
        if winner.schedule_flag:
            return "schedule"
        if winner.urgency > 0.7:
            return "urgency"
        if winner.cost_impact_usd > 50_000:
            return "cost"
        if winner.carbon_impact_tco2e > 5:
            return "sustainability"
        return "scoring"

    def merge_compatible(
        self,
        base: AgentRecommendation,
        compatible: list[AgentRecommendation],
    ) -> dict:
        """
        Merge compatible recommendations into a combined action plan.
        Compatible recommendations supplement the winner rather than conflict.
        """
        combined_cost_saving = base.cost_impact_usd + sum(r.cost_impact_usd for r in compatible)
        combined_carbon = base.carbon_impact_tco2e + sum(r.carbon_impact_tco2e for r in compatible)

        return {
            "primary_action": base.action,
            "primary_agent": base.agent_name,
            "supplementary_actions": [
                {"agent": r.agent_name, "action": r.action}
                for r in compatible
            ],
            "combined_cost_impact_usd": round(combined_cost_saving, 0),
            "combined_carbon_impact_tco2e": round(combined_carbon, 2),
            "all_agents_contributing": [base.agent_name] + [r.agent_name for r in compatible],
        }
