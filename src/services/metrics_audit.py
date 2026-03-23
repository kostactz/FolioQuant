"""
Metrics audit logging utilities.

Phase 0 objective:
- Persist a lightweight, append-only JSONL audit stream for diagnostics.
- Provide traceability for message_id / sequence / timestamps / core metrics.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class MetricsAuditLogger:
    """Thread-safe append-only JSONL logger for metric snapshots."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, record: Dict[str, Any]) -> None:
        """Append one JSON record as a single line."""
        try:
            payload = json.dumps(record, separators=(",", ":"), default=str)
            with self._lock:
                with self.file_path.open("a", encoding="utf-8") as f:
                    f.write(payload + "\n")
        except Exception as exc:
            logger.debug("Failed to append metrics audit record: %s", exc)
