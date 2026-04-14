"""SQLite persistence for completed cases.

Implements the schema from docs/specs/memory-context.md §3.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from sme_causal.orchestrator.state import CaseState


class CaseStore:
    """Thin wrapper around SQLite for case audit trail and cooldown checks."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        # check_same_thread=False: Streamlit reruns/ThreadPoolExecutor may
        # touch the connection from different threads. Writes remain serial
        # (Pipeline.run is synchronous), so this is safe.
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _ensure_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """\
            CREATE TABLE IF NOT EXISTS cases (
                case_id       TEXT PRIMARY KEY,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                client_id     TEXT NOT NULL,
                raw_query     TEXT,
                request_json  TEXT NOT NULL,
                context_json  TEXT NOT NULL,
                result_json   TEXT,
                status        TEXT NOT NULL CHECK (status IN ('done', 'aborted', 'degraded')),
                abort_reason  TEXT,
                requires_human_review BOOLEAN DEFAULT FALSE,
                review_reason TEXT,
                trace_id      TEXT,
                latency_ms    INTEGER,
                prompt_versions_json TEXT,
                experiment_variant TEXT,
                rag_iterations INTEGER DEFAULT 0,
                llm_critic_issues_json TEXT,
                updated_at    TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_cases_client_id ON cases(client_id);
            CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status);
            CREATE INDEX IF NOT EXISTS idx_cases_created_at ON cases(created_at);
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def save_case(self, state: CaseState) -> None:
        """Persist a completed CaseState to SQLite."""
        critic = state.get("critic_result", {})
        llm_issues = critic.get("llm_issues", [])
        try:
            self._conn.execute(
                """\
                INSERT OR REPLACE INTO cases (
                    case_id, client_id, raw_query, request_json, context_json,
                    result_json, status, abort_reason, requires_human_review,
                    review_reason, trace_id, latency_ms, prompt_versions_json,
                    experiment_variant, rag_iterations, llm_critic_issues_json,
                    updated_at
                ) VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    CURRENT_TIMESTAMP
                )
                """,
                (
                    state["case_id"],
                    state["client_id"],
                    state.get("raw_query"),
                    json.dumps(state.get("intervention_delta", {}), ensure_ascii=False),
                    json.dumps(state.get("client_context", {}), ensure_ascii=False),
                    json.dumps(state.get("explanation", {}), ensure_ascii=False) or None,
                    state.get("status", "done"),
                    state.get("abort_reason"),
                    state.get("requires_human_review", False),
                    state.get("review_reason"),
                    state.get("trace_id"),
                    state.get("latency_ms"),
                    json.dumps(state.get("prompt_versions", {}), ensure_ascii=False),
                    state.get("variant"),
                    state.get("rag_iterations", 0),
                    json.dumps(llm_issues, ensure_ascii=False) if llm_issues else None,
                ),
            )
            self._conn.commit()
            logger.info(
                "case_id={} status={} persisted to SQLite",
                state["case_id"][:8],
                state.get("status"),
            )
        except Exception:
            logger.exception("Failed to persist case {}", state.get("case_id", "?"))

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def get_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM cases WHERE case_id = ?", (case_id,)
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def get_cases_by_client(self, client_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM cases WHERE client_id = ? ORDER BY created_at DESC LIMIT ?",
            (client_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Cooldown
    # ------------------------------------------------------------------
    def check_cooldown(
        self,
        client_id: str,
        intervention_delta: Dict[str, Any],
        days: int = 30,
    ) -> bool:
        """Return True if a similar intervention was done recently (cooldown active).

        Uses LIKE match on the first key of intervention_delta in request_json.
        """
        return self.get_recent_done_case(client_id, intervention_delta, days) is not None

    def get_recent_done_case(
        self,
        client_id: str,
        intervention_delta: Dict[str, Any],
        days: int = 30,
    ) -> Optional[Dict[str, Any]]:
        """Return the most recent completed case matching cooldown criteria.

        Returns None if cooldown does not apply (empty delta, no matching case).
        """
        if not intervention_delta:
            return None
        first_key = next(iter(intervention_delta))
        pattern = f'%"{first_key}"%'
        row = self._conn.execute(
            """\
            SELECT * FROM cases
            WHERE client_id = ? AND status = 'done'
              AND request_json LIKE ?
              AND created_at > datetime('now', ? || ' days')
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (client_id, pattern, f"-{days}"),
        ).fetchone()
        return dict(row) if row is not None else None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def close(self) -> None:
        self._conn.close()
