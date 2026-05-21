"""
EventLogger — Structured JSON analytics + anomaly logger.

Writes two types of files to outputs/:
  snapshot_{ts}.json   — periodic traffic analytics (density, speed, counts)
  anomaly_{ts}.json    — immediate write on every anomaly alert

Both formats are designed to feed the REST API / dashboard directly.

Usage:
    logger = EventLogger(cfg)

    # Each frame:
    logger.maybe_log(analytics_dict)

    # On anomaly:
    for event in anomaly_events:
        if event.alert:
            logger.log_anomaly(event)
"""

import json
import os
import time
import logging
from typing import List

logger = logging.getLogger(__name__)


class EventLogger:

    def __init__(self, cfg: dict):
        acfg = cfg.get("analytics", {})
        self.output_dir = acfg.get("output_dir", "outputs/")
        self.interval   = float(acfg.get("snapshot_interval_seconds", 5.0))

        os.makedirs(self.output_dir, exist_ok=True)

        # Separate subdirectory for anomaly records
        self.anomaly_dir = os.path.join(self.output_dir, "anomalies")
        os.makedirs(self.anomaly_dir, exist_ok=True)

        self._last_snapshot = time.time()
        self._anomaly_count = 0

        logger.info(
            "EventLogger ready | snapshots → %s | anomalies → %s | interval=%.0fs",
            self.output_dir, self.anomaly_dir, self.interval,
        )

    # ── Traffic snapshot (periodic) ───────────────────────────────

    def maybe_log(self, analytics_dict: dict) -> None:
        """Write a traffic snapshot if the interval has elapsed."""
        now = time.time()
        if now - self._last_snapshot >= self.interval:
            ts   = int(now)
            path = os.path.join(self.output_dir, f"snapshot_{ts}.json")
            self._write(path, analytics_dict)
            logger.debug("Snapshot saved: %s", path)
            self._last_snapshot = now

    # ── Anomaly event (immediate) ─────────────────────────────────

    def log_anomaly(self, event) -> None:
        """
        Immediately write an anomaly event to disk.
        event: AnomalyEvent from anomaly_engine.py (has .to_dict() method).
        """
        self._anomaly_count += 1
        ts   = int(time.time() * 1000)   # ms precision for deduplication
        path = os.path.join(
            self.anomaly_dir,
            f"anomaly_{event.event_type}_{ts}.json",
        )
        payload = event.to_dict() if hasattr(event, "to_dict") else vars(event)
        self._write(path, payload)
        logger.warning(
            "Anomaly logged [%d] | type=%s | track=%d | conf=%.0f | file=%s",
            self._anomaly_count,
            event.event_type,
            event.track_id,
            event.confidence,
            os.path.basename(path),
        )

    def log_anomaly_batch(self, events: List) -> None:
        """Convenience: log a list of AnomalyEvents, only those with alert=True."""
        for ev in events:
            if getattr(ev, "alert", False):
                self.log_anomaly(ev)

    # ── Internal ──────────────────────────────────────────────────

    @staticmethod
    def _write(path: str, data: dict) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except OSError as e:
            logger.error("EventLogger write failed: %s | %s", path, e)