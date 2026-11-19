"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store the request and its start time; return its correlation key."""
        key = request_id or user_id or uuid4().hex
        started = perf_counter()
        self._open[key] = {"started": started, "index": len(self.logs)}
        self.logs.append(
            {
                "request_id": key,
                "user_id": user_id,
                "input": text,
                "started_at": utc_now_iso(),
                "output": None,
                "blocked": None,
                "layer": None,
                "latency_ms": None,
            }
        )
        return key

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Complete the matching request record with output and latency."""
        key = request_id or user_id
        opened = self._open.pop(key, None)
        if opened is None:
            key = request_id or uuid4().hex
            self.logs.append(
                {
                    "request_id": key,
                    "user_id": user_id,
                    "input": None,
                    "started_at": None,
                    "output": text,
                    "blocked": blocked,
                    "layer": layer,
                    "latency_ms": None,
                    "completed_at": utc_now_iso(),
                }
            )
            return

        record = self.logs[opened["index"]]
        record.update(
            {
                "output": text,
                "blocked": blocked,
                "layer": layer,
                "latency_ms": round((perf_counter() - opened["started"]) * 1000, 3),
                "completed_at": utc_now_iso(),
            }
        )

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
