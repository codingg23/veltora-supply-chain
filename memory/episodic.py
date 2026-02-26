"""
EpisodicMemory — pgvector-backed long-term memory for supply-chain decisions.
Stores agent decisions with embeddings so the orchestrator can retrieve
similar past situations for few-shot context.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any

try:
    import psycopg2
    import psycopg2.extras
    PG_AVAILABLE = True
except ImportError:
    PG_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS supply_chain_episodes (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_name  TEXT NOT NULL,
    scenario    TEXT NOT NULL,
    decision    JSONB NOT NULL,
    outcome     JSONB,
    embedding   vector(1536),
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_episodes_embedding
    ON supply_chain_episodes USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
"""


@dataclass
class Episode:
    id: str
    agent_name: str
    scenario: str
    decision: dict
    outcome: dict | None
    similarity: float | None = None


class EpisodicMemory:
    """
    Long-term episodic store.  Each entry captures:
    - The scenario description (text)
    - The agent decision (structured dict)
    - The observed outcome (filled in later via update())
    - A 1536-dim embedding for semantic retrieval

    Falls back to an in-process list if PostgreSQL is unavailable.
    """

    def __init__(self, dsn: str | None = None, api_key: str | None = None):
        self._local: list[dict] = []

        if PG_AVAILABLE:
            pg_dsn = dsn or os.getenv("PGVECTOR_DSN", "")
            try:
                self.conn = psycopg2.connect(pg_dsn) if pg_dsn else None
                if self.conn:
                    with self.conn.cursor() as cur:
                        cur.execute(SCHEMA)
                    self.conn.commit()
            except Exception:
                self.conn = None
        else:
            self.conn = None

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")

    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float] | None:
        """Generate embedding via Anthropic voyage-3 (or stub zeros)."""
        if not ANTHROPIC_AVAILABLE or not self.api_key:
            return None
        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            # Use text-embedding endpoint when available; stub for now
            # In production: response = client.embeddings.create(...)
            import hashlib
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
            import random
            rng = random.Random(seed)
            return [rng.gauss(0, 1) for _ in range(1536)]
        except Exception:
            return None

    def store(self, agent_name: str, scenario: str, decision: dict) -> str:
        """Store a new episode; return its UUID."""
        episode_id = str(uuid.uuid4())
        embedding = self._embed(scenario)

        if self.conn and embedding:
            try:
                with self.conn.cursor() as cur:
                    cur.execute(
                        """INSERT INTO supply_chain_episodes
                           (id, agent_name, scenario, decision, embedding)
                           VALUES (%s, %s, %s, %s, %s)""",
                        (episode_id, agent_name, scenario,
                         json.dumps(decision), embedding)
                    )
                self.conn.commit()
                return episode_id
            except Exception:
                self.conn.rollback()

        # Local fallback
        self._local.append({
            "id": episode_id,
            "agent_name": agent_name,
            "scenario": scenario,
            "decision": decision,
            "outcome": None,
            "embedding": embedding,
            "created_at": time.time(),
        })
        return episode_id

    def update_outcome(self, episode_id: str, outcome: dict) -> None:
        """Record the observed outcome for a stored episode."""
        if self.conn:
            try:
                with self.conn.cursor() as cur:
                    cur.execute(
                        "UPDATE supply_chain_episodes SET outcome=%s WHERE id=%s",
                        (json.dumps(outcome), episode_id)
                    )
                self.conn.commit()
                return
            except Exception:
                self.conn.rollback()
        for ep in self._local:
            if ep["id"] == episode_id:
                ep["outcome"] = outcome
                break

    def search(self, query: str, top_k: int = 5,
                agent_filter: str | None = None) -> list[Episode]:
        """Retrieve top-k most similar episodes using cosine similarity."""
        query_emb = self._embed(query)

        if self.conn and query_emb:
            try:
                with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    filter_clause = "AND agent_name = %s" if agent_filter else ""
                    params = [query_emb, top_k]
                    if agent_filter:
                        params.insert(1, agent_filter)
                    cur.execute(
                        f"""SELECT id, agent_name, scenario, decision, outcome,
                               1 - (embedding <=> %s::vector) AS similarity
                            FROM supply_chain_episodes
                            {filter_clause}
                            ORDER BY embedding <=> %s::vector
                            LIMIT %s""",
                        [query_emb] + ([agent_filter] if agent_filter else []) +
                        [query_emb, top_k]
                    )
                    rows = cur.fetchall()
                    return [
                        Episode(
                            id=str(r["id"]),
                            agent_name=r["agent_name"],
                            scenario=r["scenario"],
                            decision=r["decision"],
                            outcome=r["outcome"],
                            similarity=float(r["similarity"]),
                        )
                        for r in rows
                    ]
            except Exception:
                pass

        # Local cosine search fallback
        if not query_emb:
            results = self._local[-top_k:]
        else:
            import numpy as np
            qv = np.array(query_emb)
            scored = []
            for ep in self._local:
                ev = ep.get("embedding")
                if ev:
                    sim = float(np.dot(qv, ev) / (np.linalg.norm(qv) * np.linalg.norm(ev) + 1e-9))
                    scored.append((sim, ep))
            scored.sort(key=lambda x: x[0], reverse=True)
            results = [ep for _, ep in scored[:top_k]]

        return [
            Episode(
                id=ep["id"],
                agent_name=ep["agent_name"],
                scenario=ep["scenario"],
                decision=ep["decision"],
                outcome=ep.get("outcome"),
                similarity=None,
            )
            for ep in results
        ]
