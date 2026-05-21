"""
MockCamera — Generates synthetic traffic frames for testing
when no real camera or video is available.
Simulates vehicles as colored moving rectangles with IDs.

Usage:
    cap = MockCamera(width=1280, height=720, num_vehicles=12)
    while True:
        ret, frame = cap.read()
        # frame is a BGR numpy array — drop-in for cv2.VideoCapture
"""

import cv2
import numpy as np
import random
import time
import math


class MockVehicle:
    COLORS = {
        "car":        (0, 180, 255),
        "motorcycle": (0, 255, 120),
        "truck":      (60, 60, 255),
        "bus":        (255, 140, 0),
        "auto":       (180, 0, 255),   # Indian three-wheeler
    }
    SIZES = {
        "car":        (80, 45),
        "motorcycle": (40, 30),
        "truck":      (110, 60),
        "bus":        (120, 65),
        "auto":       (60, 40),
    }

    def __init__(self, frame_w: int, frame_h: int):
        self.type   = random.choices(
            ["car", "car", "car", "motorcycle", "truck", "bus", "auto"],
            weights=[40, 40, 40, 25, 10, 8, 15],
        )[0]
        self.w, self.h = self.SIZES[self.type]
        self.x      = float(random.randint(-self.w, frame_w))
        self.y      = float(random.randint(frame_h // 3, frame_h - self.h - 20))
        self.speed  = random.uniform(2.5, 7.0)   # pixels per frame
        self.color  = self.COLORS[self.type]
        self.fw     = frame_w
        self.id     = random.randint(10, 999)

    def move(self):
        self.x += self.speed
        # Slight sinusoidal vertical drift for realism
        self.y += math.sin(self.x / 80) * 0.4

    def is_offscreen(self) -> bool:
        return self.x > self.fw + self.w

    def draw(self, frame: np.ndarray) -> None:
        x1 = int(self.x)
        y1 = int(self.y)
        x2 = x1 + self.w
        y2 = y1 + self.h

        # Vehicle body
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, -1)
        # Windscreen
        cv2.rectangle(
            frame,
            (x1 + 8, y1 + 5),
            (x2 - 8, y1 + 18),
            (200, 230, 255), -1,
        )
        # Wheels
        for wx in [x1 + 12, x2 - 12]:
            cv2.circle(frame, (wx, y2), 8, (30, 30, 30), -1)
        # Label
        cv2.putText(
            frame,
            f"{self.type[:3].upper()} #{self.id}",
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.color, 1,
        )


class MockCamera:
    """
    Drop-in replacement for cv2.VideoCapture for offline simulation.

    Mimics the full cv2.VideoCapture API:
        read() → (True, frame)
        isOpened() → True
        release()
        set(prop_id, value)
        get(prop_id)
    """

    def __init__(
        self,
        width:        int = 1280,
        height:       int = 720,
        num_vehicles: int = 10,
        fps:          int = 20,
    ):
        self.width    = width
        self.height   = height
        self.fps      = fps
        self.vehicles = [MockVehicle(width, height) for _ in range(num_vehicles)]
        self._target_vehicles = num_vehicles
        self._frame_delay = 1.0 / fps
        self._last_frame  = time.time()

        # Background colours
        self.road_color = (60, 60, 60)
        self.sky_color  = (135, 180, 220)
        self.lane_color = (200, 200, 200)

    # ── cv2 API compatibility ─────────────────────────────────────

    def read(self):
        elapsed = time.time() - self._last_frame
        if elapsed < self._frame_delay:
            time.sleep(self._frame_delay - elapsed)
        self._last_frame = time.time()
        return True, self._render()

    def isOpened(self) -> bool:
        return True

    def release(self) -> None:
        pass

    def set(self, prop_id, value) -> None:
        pass

    def get(self, prop_id) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop_id == cv2.CAP_PROP_FPS:
            return float(self.fps)
        return 0.0

    # ── Rendering ─────────────────────────────────────────────────

    def _render(self) -> np.ndarray:
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Sky
        frame[: self.height // 3] = self.sky_color
        # Road
        frame[self.height // 3 :] = self.road_color

        # Horizon line
        cv2.line(
            frame,
            (0, self.height // 3),
            (self.width, self.height // 3),
            (255, 255, 255), 2,
        )
        # Centre divider
        cv2.line(
            frame,
            (self.width // 2, self.height // 3),
            (self.width // 2, self.height),
            self.lane_color, 2,
        )
        # Dashed lane lines
        for x_frac in [0.25, 0.75]:
            x = int(self.width * x_frac)
            for y in range(self.height // 3, self.height, 40):
                cv2.line(frame, (x, y), (x, y + 20), (150, 150, 150), 1)

        # Header
        cv2.putText(
            frame,
            "Smart RSU — Simulation Mode  [NH-44]",
            (self.width // 2 - 200, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 100), 2,
        )

        # Move + draw vehicles
        for v in self.vehicles:
            v.move()
            v.draw(frame)

        # Replace off-screen vehicles
        self.vehicles = [v for v in self.vehicles if not v.is_offscreen()]
        while len(self.vehicles) < self._target_vehicles:
            v = MockVehicle(self.width, self.height)
            v.x = -v.w
            self.vehicles.append(v)

        # Timestamp overlay
        ts = time.strftime("%H:%M:%S")
        cv2.putText(
            frame, ts,
            (self.width - 110, self.height - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
        )
        return frame