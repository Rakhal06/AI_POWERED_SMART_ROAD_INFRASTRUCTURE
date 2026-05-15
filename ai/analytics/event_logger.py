"""
EventLogger — Structured JSON analytics logger.
Writes periodic snapshots to disk.
Designed to feed a dashboard or REST API later.
"""

import json
import os
import time
import logging

logger = logging.getLogger(__name__)


class EventLogger:
    def __init__(self, cfg: dict):
        self.output_dir = cfg["analytics"]["output_dir"]
        self.interval = cfg["analytics"]["snapshot_interval_seconds"]
        self.last_snapshot = time.time()
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"EventLogger ready. Snapshots -> {self.output_dir}")

    def maybe_log(self, analytics_dict: dict):
        """Write snapshot if interval elapsed."""
        now = time.time()
        if now - self.last_snapshot >= self.interval:
            ts = int(now)
            path = os.path.join(self.output_dir, f"snapshot_{ts}.json")
            with open(path, "w") as f:
                json.dump(analytics_dict, f, indent=2)
            logger.info(f"Snapshot saved: {path}")
            self.last_snapshot = now