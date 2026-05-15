"""
MockCamera — Generates synthetic traffic frames for testing
when no real camera or video is available.
Simulates vehicles as colored moving rectangles with IDs.
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
    }

    def __init__(self, frame_w, frame_h):
        self.type   = random.choice(["car", "car", "car", "motorcycle", "truck", "bus"])
        self.w      = {"car": 80, "motorcycle": 40, "truck": 110, "bus": 120}[self.type]
        self.h      = {"car": 45, "motorcycle": 30, "truck": 60,  "bus": 65}[self.type]
        self.x      = float(random.randint(-self.w, frame_w))
        self.y      = float(random.randint(frame_h // 3, frame_h - self.h - 20))
        self.speed  = random.uniform(2.5, 7.0)   # pixels per frame
        self.color  = self.COLORS[self.type]
        self.fw     = frame_w
        self.id     = random.randint(10, 999)

    def move(self):
        self.x += self.speed
        # Slight vertical drift for realism
        self.y += math.sin(self.x / 80) * 0.4

    def is_offscreen(self):
        return self.x > self.fw + self.w

    def draw(self, frame):
        x1 = int(self.x)
        y1 = int(self.y)
        x2 = x1 + self.w
        y2 = y1 + self.h
        # Vehicle body
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, -1)
        # Windscreen
        cv2.rectangle(frame, (x1 + 8, y1 + 5), (x2 - 8, y1 + 18), (200, 230, 255), -1)
        # Wheels
        for wx in [x1 + 12, x2 - 12]:
            cv2.circle(frame, (wx, y2), 8, (30, 30, 30), -1)
        # Label
        cv2.putText(frame, f"{self.type[:3].upper()} #{self.id}",
                    (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.color, 1)


class MockCamera:
    """
    Drop-in replacement for cv2.VideoCapture for simulation testing.

    Usage:
        cap = MockCamera(width=1280, height=720, num_vehicles=12)
        while True:
            ret, frame = cap.read()
    """

    def __init__(self, width=1280, height=720, num_vehicles=10, fps=20):
        self.width    = width
        self.height   = height
        self.fps      = fps
        self.vehicles = [MockVehicle(width, height) for _ in range(num_vehicles)]
        self._frame_delay = 1.0 / fps
        self._last_frame  = time.time()

        # Road background colors
        self.road_color  = (60, 60, 60)
        self.lane_color  = (200, 200, 200)
        self.sky_color   = (135, 180, 220)

    def read(self):
        # Throttle to target FPS
        elapsed = time.time() - self._last_frame
        if elapsed < self._frame_delay:
            time.sleep(self._frame_delay - elapsed)
        self._last_frame = time.time()

        frame = self._render()
        return True, frame

    def isOpened(self):
        return True

    def release(self):
        pass

    def set(self, prop_id, value):
        pass  # compatibility with cv2 API

    def get(self, prop_id):
        import cv2
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:  return self.width
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT: return self.height
        return 0

    def _render(self):
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Sky
        frame[:self.height // 3] = self.sky_color

        # Road surface
        frame[self.height // 3:] = self.road_color

        # Lane dividers
        mid_y = self.height // 3
        cv2.line(frame, (0, mid_y), (self.width, mid_y), (255, 255, 255), 2)
        cv2.line(frame, (self.width // 2, mid_y),
                 (self.width // 2, self.height), self.lane_color, 2)

        # Lane labels background
        cv2.putText(frame, "NH-44 Simulation Mode",
                    (self.width // 2 - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 100), 2)

        # Move + draw vehicles
        for v in self.vehicles:
            v.move()
            v.draw(frame)

        # Replace off-screen vehicles
        self.vehicles = [v for v in self.vehicles if not v.is_offscreen()]
        while len(self.vehicles) < 10:
            v = MockVehicle(self.width, self.height)
            v.x = -v.w  # enter from left
            self.vehicles.append(v)

        # Timestamp
        ts = time.strftime("%H:%M:%S")
        cv2.putText(frame, ts, (self.width - 110, self.height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        return frame