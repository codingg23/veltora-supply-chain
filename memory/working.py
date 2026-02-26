"""
WorkingMemory — Redis-backed short-term memory for cross-agent state sharing.
Agents write their latest outputs here so the orchestrator and other agents
can read recent context without duplicating API calls.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class WorkingMemory:
    """
    Thread-safe key-value store with TTL.
    Falls back to an in-process dict if Redis is unavailable.
    """

    DEFAULT_TTL = 3600   # 1 hour

    def __init__(self, url: str | None = None, ttl: int = DEFAULT_TTL):
        self.ttl = ttl
        self._local: dict[str, tuple[Any, float]] = {}   # fallback store

        if REDIS_AVAILABLE:
            redis_url = url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
            try:
                self.client: redis.Redis | None = redis.from_url(
                    redis_url, decode_responses=True, socket_connect_timeout=2
                )
                self.client.ping()
            except Exception:
                self.client = None
        else:
            self.client = None

    # ------------------------------------------------------------------

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        payload = json.dumps(value, default=str)
        effective_ttl = ttl or self.ttl
        if self.client:
            self.client.setex(f"sc:{key}", effective_ttl, payload)
        else:
            self._local[key] = (value, time.time() + effective_ttl)

    def get(self, key: str) -> Any | None:
        if self.client:
            raw = self.client.get(f"sc:{key}")
            return json.loads(raw) if raw else None
        entry = self._local.get(key)
        if entry and time.time() < entry[1]:
            return entry[0]
        self._local.pop(key, None)
        return None

    def delete(self, key: str) -> None:
        if self.client:
            self.client.delete(f"sc:{key}")
        else:
            self._local.pop(key, None)

    def scan(self, pattern: str = "*") -> list[str]:
        """List keys matching a pattern."""
        if self.client:
            keys = self.client.keys(f"sc:{pattern}")
            return [k.removeprefix("sc:") for k in keys]
        now = time.time()
        return [k for k, (_, exp) in self._local.items() if now < exp]

    # ------------------------------------------------------------------
    # Agent-specific helpers

    def write_agent_output(self, agent_name: str, data: Any) -> None:
        """Persist an agent's latest output with a timestamp."""
        self.set(f"agent:{agent_name}:latest", {
            "agent": agent_name,
            "timestamp": time.time(),
            "data": data,
        })

    def read_agent_output(self, agent_name: str) -> Any | None:
        record = self.get(f"agent:{agent_name}:latest")
        return record["data"] if record else None

    def heartbeat(self, agent_name: str) -> None:
        self.set(f"heartbeat:{agent_name}", time.time(), ttl=120)

    def get_live_agents(self) -> list[str]:
        now = time.time()
        keys = self.scan("heartbeat:*")
        live = []
        for k in keys:
            ts = self.get(k)
            if ts and now - float(ts) < 120:
                live.append(k.removeprefix("heartbeat:"))
        return live
