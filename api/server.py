"""
FastAPI server — REST interface to the supply-chain orchestrator.
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from coordinator.orchestrator import SupplyChainOrchestrator, OrchestratorOutput
from memory.working import WorkingMemory


# ---------------------------------------------------------------------------
# Rate limiting (simple in-memory token bucket)
# ---------------------------------------------------------------------------

class TokenBucket:
    def __init__(self, rate: float, capacity: float):
        self.rate = rate
        self.capacity = capacity
        self._tokens: dict[str, tuple[float, float]] = {}

    def consume(self, key: str, tokens: float = 1.0) -> bool:
        now = time.monotonic()
        last_tokens, last_time = self._tokens.get(key, (self.capacity, now))
        refill = (now - last_time) * self.rate
        current = min(self.capacity, last_tokens + refill)
        if current >= tokens:
            self._tokens[key] = (current - tokens, now)
            return True
        self._tokens[key] = (current, now)
        return False


_bucket = TokenBucket(rate=1.0, capacity=60)   # 60 req/min per IP

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

_orchestrator: SupplyChainOrchestrator | None = None
_memory: WorkingMemory | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator, _memory
    _orchestrator = SupplyChainOrchestrator()
    _memory = WorkingMemory()
    yield
    _orchestrator = None
    _memory = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Veltora Supply Chain API",
    description="9-agent autonomous supply chain intelligence for data-centre operations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

bearer = HTTPBearer()
API_KEY = os.getenv("SC_API_KEY", "veltora-dev-key")


def _auth(credentials: HTTPAuthorizationCredentials = Depends(bearer)) -> str:
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key")
    return credentials.credentials


def _rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    if not _bucket.consume(ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded: 60 requests/minute",
        )


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class AnalyseRequest(BaseModel):
    horizon_days: int = 90
    focus_component: str | None = None

class AgentRequest(BaseModel):
    agent_name: str
    params: dict[str, Any] = {}

class HealthResponse(BaseModel):
    status: str
    version: str
    agents_ready: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    return HealthResponse(
        status="ok",
        version="1.0.0",
        agents_ready=_orchestrator is not None,
    )


@app.post("/analyse", tags=["supply-chain"])
def analyse(
    body: AnalyseRequest,
    _auth: str = Depends(_auth),
    _rl: None = Depends(_rate_limit),
) -> dict:
    """
    Run the full 9-agent supply-chain analysis.
    Returns action plan, conflict resolutions, and executive summary.
    """
    if not _orchestrator:
        raise HTTPException(503, "Orchestrator not initialised")

    result: OrchestratorOutput = _orchestrator.run(horizon_days=body.horizon_days)

    # Cache in working memory
    if _memory:
        _memory.set("last_analysis", {
            "timestamp": result.timestamp,
            "risk_level": result.risk_level,
            "action_count": len(result.action_plan),
        })

    return {
        "timestamp": result.timestamp,
        "horizon_days": result.horizon_days,
        "risk_level": result.risk_level,
        "executive_summary": result.executive_summary,
        "action_plan": result.action_plan,
        "conflict_resolutions": result.conflict_resolutions,
        "agent_statuses": [
            {"agent": r.agent_name, "status": r.status, "elapsed_ms": r.elapsed_ms}
            for r in result.agent_results
        ],
    }


@app.post("/agents/{agent_name}", tags=["agents"])
def run_single_agent(
    agent_name: str,
    body: AgentRequest,
    _auth: str = Depends(_auth),
    _rl: None = Depends(_rate_limit),
) -> dict:
    """Run a single named agent with custom parameters."""
    if not _orchestrator:
        raise HTTPException(503, "Orchestrator not initialised")

    AGENT_MAP = {
        "predictor":      lambda: _orchestrator.predictor.forecast(
                              body.params.get("component_id", "NVIDIA_H100")),
        "procurement":    lambda: _orchestrator.procurement.decide(
                              body.params.get("component_id", "NVIDIA_H100")),
        "risk":           lambda: _orchestrator.risk.assess(
                              body.params.get("component_id", "NVIDIA_H100")),
        "cost":           lambda: _orchestrator.cost.optimise(
                              body.params.get("component_id", "NVIDIA_H100"),
                              body.params.get("quantity", 50)),
        "scheduler":      lambda: _orchestrator.scheduler.analyse(),
        "sustainability": lambda: _orchestrator.sustainability.evaluate(
                              body.params.get("components", [{"component_id": "NVIDIA_H100", "quantity": 10}])),
        "vendor":         lambda: _orchestrator.vendor.review(),
        "quality":        lambda: _orchestrator.quality.audit(),
        "logistics":      lambda: _orchestrator.logistics.plan(
                              body.params.get("component_id", "NVIDIA_H100"),
                              body.params.get("weight_kg", 48.0),
                              body.params.get("origin", "TW"),
                              body.params.get("destination", "US"),
                              body.params.get("urgency", "normal")),
    }

    runner = AGENT_MAP.get(agent_name.lower())
    if not runner:
        raise HTTPException(404, f"Unknown agent: {agent_name}. "
                            f"Available: {list(AGENT_MAP.keys())}")
    try:
        result = runner()
        # Dataclasses → dict
        import dataclasses
        if dataclasses.is_dataclass(result):
            result = dataclasses.asdict(result)
        return {"agent": agent_name, "result": result}
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/memory", tags=["ops"])
def get_memory_state(
    _auth: str = Depends(_auth),
    _rl: None = Depends(_rate_limit),
) -> dict:
    """Return current working memory state."""
    if not _memory:
        return {"keys": [], "last_analysis": None}
    keys = _memory.scan("*")
    last = _memory.get("last_analysis")
    return {"keys": keys, "last_analysis": last}
