"""
auto_calibrator.py — Fully automatic homography calibration.

No manual clicks. Works on first frame. Self-corrects every N frames.

Pipeline:
  Frame
    -> LaneLineDetector         (CLAHE + Canny + Hough + RANSAC fitting)
    -> VanishingPointEstimator  (line intersection consensus + geometric prior)
    -> RoadTrapezoidBuilder     (perspective-aware src_pts)
    -> HomographyBuilder        (DLT + RANSAC via cv2.findHomography)
    -> BEVValidator             (lane parallelism check after warp)
    -> MetricScaleEstimator     (pixels/metre at near + far plane)
    -> CalibrationResult        (H matrix + confidence + metadata)

Mode selection (calibration.mode in settings.yaml):
  auto   : fully automatic, no manual input needed
  hybrid : auto primary, falls back to manual if confidence < min_confidence
  manual : uses only src_pts/dst_pts from settings.yaml

Integration:
    calibrator = AutoCalibrator(cfg)
    result = calibrator.calibrate(frame)
    # result.H is ready to inject into SpeedEstimator
    # result.confidence in [0.0, 1.0] tells you how much to trust it
    # pipeline.py calls this once per frame (cached internally)
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────

@dataclass
class LaneLine:
    x1: float; y1: float
    x2: float; y2: float
    side: str  # "left" | "right"

    def x_at_y(self, y: float) -> float:
        if abs(self.y2 - self.y1) < 1e-6:
            return (self.x1 + self.x2) / 2
        t = (y - self.y1) / (self.y2 - self.y1)
        return self.x1 + t * (self.x2 - self.x1)

    @property
    def slope(self) -> float:
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        return dy / dx if abs(dx) > 1e-6 else 1e9

    @property
    def angle_deg(self) -> float:
        return float(np.degrees(np.arctan2(
            abs(self.y2 - self.y1), abs(self.x2 - self.x1)
        )))


@dataclass
class CalibrationResult:
    H:                       np.ndarray
    vanishing_point:         Tuple[float, float]
    src_pts:                 np.ndarray    # pixel trapezoid [4×2]
    dst_pts:                 np.ndarray    # ground plane metres [4×2]
    lane_width_px:           float
    pixels_per_meter_near:   float
    pixels_per_meter_far:    float
    visible_depth_m:         float
    confidence:              float         # 0.0 – 1.0
    method:                  str           # "auto_lanes" | "auto_vp" | "manual"
    frame_index:             int = 0


# ─────────────────────────────────────────────────────────────────
# LANE LINE DETECTOR
# ─────────────────────────────────────────────────────────────────

class LaneLineDetector:
    """
    Classical lane line detection: CLAHE → Canny → ROI → HoughP → RANSAC fit.
    Works on any hardware — no GPU required, no new packages.
    """

    def __init__(self, cfg: dict = None):
        cal = (cfg or {}).get("calibration", {})
        self.roi_ratio       = cal.get("roi_ratio",       0.60)
        self.canny_low       = cal.get("canny_low",       50)
        self.canny_high      = cal.get("canny_high",      150)
        self.hough_threshold = cal.get("hough_threshold", 35)
        self.min_line_length = cal.get("min_line_length", 40)
        self.max_line_gap    = cal.get("max_line_gap",    80)
        self.min_slope       = cal.get("min_slope",       0.25)
        self.max_slope       = cal.get("max_slope",       5.0)

    def detect(self, frame: np.ndarray) -> Tuple[Optional[LaneLine], Optional[LaneLine]]:
        """
        Returns (left_lane, right_lane).
        Either can be None if not detected.
        """
        h, w = frame.shape[:2]

        # Preprocessing: CLAHE → Gaussian → Canny
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)

        # Trapezoid ROI — bottom roi_ratio of frame
        roi_top = int(h * (1 - self.roi_ratio))
        roi_pts = np.array([[
            (0, h),
            (w, h),
            (int(w * 0.70), roi_top),
            (int(w * 0.30), roi_top),
        ]], dtype=np.int32)
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, roi_pts, 255)
        masked = cv2.bitwise_and(edges, mask)

        raw = cv2.HoughLinesP(
            masked, rho=1, theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )
        if raw is None:
            return None, None

        mid_x = w / 2
        left_pts, right_pts = [], []

        for seg in raw:
            x1, y1, x2, y2 = seg[0]
            if abs(x2 - x1) < 5:
                continue
            slope = (y2 - y1) / (x2 - x1)
            abs_slope = abs(slope)
            if not (self.min_slope < abs_slope < self.max_slope):
                continue
            cx = (x1 + x2) / 2
            if cx < mid_x and slope > 0:
                left_pts += [(x1, y1), (x2, y2)]
            elif cx >= mid_x and slope < 0:
                right_pts += [(x1, y1), (x2, y2)]

        y_near = int(h * 0.95)
        y_far  = int(h * 0.50)
        left   = self._fit_line(left_pts,  "left",  y_near, y_far)
        right  = self._fit_line(right_pts, "right", y_near, y_far)
        return left, right

    def _fit_line(
        self,
        pts: List[Tuple],
        side: str,
        y_near: int,
        y_far: int,
    ) -> Optional[LaneLine]:
        if len(pts) < 6:
            return None
        arr = np.array(pts, dtype=np.float32)
        line = cv2.fitLine(arr, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        vx, vy, cx, cy = float(line[0]), float(line[1]), float(line[2]), float(line[3])
        if abs(vy) < 1e-6:
            return None
        slope  = vx / vy
        x_near = cx + slope * (y_near - cy)
        x_far  = cx + slope * (y_far  - cy)
        return LaneLine(x1=x_near, y1=y_near, x2=x_far, y2=y_far, side=side)

    def draw(
        self,
        frame: np.ndarray,
        left: Optional[LaneLine],
        right: Optional[LaneLine],
    ) -> np.ndarray:
        vis = frame.copy()
        for lane, color in [(left, (255, 100, 0)), (right, (0, 100, 255))]:
            if lane:
                cv2.line(
                    vis,
                    (int(lane.x1), int(lane.y1)),
                    (int(lane.x2), int(lane.y2)),
                    color, 2, cv2.LINE_AA,
                )
        return vis


# ─────────────────────────────────────────────────────────────────
# VANISHING POINT ESTIMATOR
# ─────────────────────────────────────────────────────────────────

class VanishingPointEstimator:
    """
    Estimates vanishing point from left+right lane lines.
    Falls back to geometric prior (frame centre, ~40% height) if lines
    are missing, parallel, or produce an out-of-bounds intersection.
    """

    def estimate(
        self,
        left: Optional[LaneLine],
        right: Optional[LaneLine],
        frame_shape: Tuple,
    ) -> Tuple[float, float]:
        h, w = frame_shape[:2]

        if left is not None and right is not None:
            vp = self._intersect(left, right)
            if vp is not None:
                vx, vy = vp
                # Accept only if VP is inside a reasonable region:
                # horizontally within frame, vertically in top 70%
                if 0 < vx < w and 0 < vy < h * 0.70:
                    return float(vx), float(vy)

        # Geometric prior: centre-x, 38% from top
        return w / 2.0, h * 0.38

    @staticmethod
    def _intersect(
        l1: LaneLine,
        l2: LaneLine,
    ) -> Optional[Tuple[float, float]]:
        """Line–line intersection via parametric form."""
        x1, y1, x2, y2 = l1.x1, l1.y1, l1.x2, l1.y2
        x3, y3, x4, y4 = l2.x1, l2.y1, l2.x2, l2.y2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None   # parallel lines

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return float(x), float(y)


# ─────────────────────────────────────────────────────────────────
# AUTO CALIBRATOR
# ─────────────────────────────────────────────────────────────────

class AutoCalibrator:
    """
    Automatic road homography calibrator.

    Call calibrate(frame) every frame — internally caches the result
    and only recomputes every `recalibration_interval_frames` frames.

    The returned CalibrationResult.H is ready for SpeedEstimator injection.
    """

    def __init__(self, cfg: dict):
        cal_cfg = cfg.get("calibration", {})

        self.mode            = cal_cfg.get("mode", "hybrid")
        self.lane_width_m    = float(cal_cfg.get("lane_width_m", 3.65))
        self.recal_interval  = int(cal_cfg.get("recalibration_interval_frames", 300))
        self.min_confidence  = float(cal_cfg.get("min_confidence", 0.50))

        self.lane_detector   = LaneLineDetector(cfg)
        self.vp_estimator    = VanishingPointEstimator()

        # Manual override points from settings.yaml
        hcfg = cal_cfg.get("homography", {})
        if hcfg and "src_pts" in hcfg and "dst_pts" in hcfg:
            self._manual_src = np.float32(hcfg["src_pts"])
            self._manual_dst = np.float32(hcfg["dst_pts"])
        else:
            self._manual_src = None
            self._manual_dst = None

        # Internal state
        self._last_result:    Optional[CalibrationResult] = None
        self._last_cal_frame: int = -9999
        self._frame_index:    int = 0

        logger.info(
            "AutoCalibrator ready | mode=%s | lane_width=%.2fm | recal_every=%d frames",
            self.mode, self.lane_width_m, self.recal_interval,
        )

    def calibrate(self, frame: np.ndarray) -> Optional[CalibrationResult]:
        """
        Run calibration pipeline on the given frame.
        Returns cached CalibrationResult if recalibration is not yet due.
        Returns None only if all calibration paths fail.
        """
        self._frame_index += 1
        needs_recal = (self._frame_index - self._last_cal_frame) >= self.recal_interval

        # Serve cache if still fresh
        if not needs_recal and self._last_result is not None:
            return self._last_result

        result: Optional[CalibrationResult] = None

        if self.mode == "manual":
            result = self._build_manual_result()

        elif self.mode == "auto":
            result = self._auto_calibrate(frame)
            if result is None or result.confidence < self.min_confidence:
                logger.debug(
                    "Auto calibration confidence=%.2f < %.2f — trying manual fallback",
                    result.confidence if result else 0.0, self.min_confidence,
                )
                result = self._build_manual_result() or self._last_result

        else:  # hybrid (default)
            auto_result = self._auto_calibrate(frame)
            if auto_result is not None and auto_result.confidence >= self.min_confidence:
                result = auto_result
            else:
                if self._manual_src is not None:
                    result = self._build_manual_result()
                    if result:
                        logger.debug(
                            "Hybrid: auto conf=%.2f too low — using manual override",
                            auto_result.confidence if auto_result else 0.0,
                        )
                elif self._last_result is not None:
                    result = self._last_result  # keep last good calibration
                    logger.debug("Hybrid: using cached calibration from frame %d", self._last_cal_frame)
                else:
                    result = auto_result  # best we have

        if result is not None:
            self._last_result    = result
            self._last_cal_frame = self._frame_index
            logger.debug(
                "Calibration updated | method=%s | conf=%.2f | px/m_near=%.1f | depth=%.1fm",
                result.method, result.confidence,
                result.pixels_per_meter_near, result.visible_depth_m,
            )

        return result

    def get_homography(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Convenience: returns just H (or None)."""
        r = self.calibrate(frame)
        return r.H if r else None

    def draw_debug(
        self,
        frame: np.ndarray,
        result: Optional[CalibrationResult],
    ) -> np.ndarray:
        """Draw calibration overlay: trapezoid, vanishing point, status bar."""
        if result is None:
            return frame
        vis = frame.copy()

        # Trapezoid
        pts = result.src_pts.astype(np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
        labels = ["NL", "NR", "FL", "FR"]
        for i, (x, y) in enumerate(pts):
            cv2.circle(vis, (int(x), int(y)), 6, (0, 0, 255), -1)
            cv2.putText(
                vis, labels[i], (int(x) + 8, int(y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA,
            )

        # Vanishing point
        vx, vy = int(result.vanishing_point[0]), int(result.vanishing_point[1])
        cv2.drawMarker(vis, (vx, vy), (255, 0, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.putText(
            vis, "VP", (vx + 12, vy - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA,
        )

        # Status bar
        conf_color = (
            (0, 255, 0)   if result.confidence > 0.70 else
            (0, 165, 255) if result.confidence > 0.50 else
            (0, 0, 255)
        )
        cv2.putText(
            vis,
            (f"AutoCalib:{result.method} "
             f"conf={result.confidence:.2f} "
             f"px/m={result.pixels_per_meter_near:.1f} "
             f"depth={result.visible_depth_m:.1f}m"),
            (8, frame.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, conf_color, 1, cv2.LINE_AA,
        )
        return vis

    # ── Internal ──────────────────────────────────────────────────

    def _auto_calibrate(self, frame: np.ndarray) -> Optional[CalibrationResult]:
        h, w = frame.shape[:2]

        # 1. Detect lane lines
        left, right = self.lane_detector.detect(frame)

        # 2. Estimate vanishing point
        vp_x, vp_y = self.vp_estimator.estimate(left, right, frame.shape)

        # 3. Build trapezoid
        y_near = int(h * 0.92)
        y_far  = int(vp_y + (h - vp_y) * 0.12)

        confidence = 0.45
        method     = "auto_vp_only"

        if left is not None and right is not None:
            lx_near = left.x_at_y(y_near)
            rx_near = right.x_at_y(y_near)
            lx_far  = left.x_at_y(y_far)
            rx_far  = right.x_at_y(y_far)
            lane_width_px = abs(rx_near - lx_near)

            # Sanity: lane width should be a reasonable fraction of frame width
            if w * 0.05 < lane_width_px < w * 0.80:
                confidence = 0.80
                method     = "auto_lanes"
            else:
                logger.debug(
                    "Lane width px=%.0f out of range [%.0f, %.0f] — VP fallback",
                    lane_width_px, w * 0.05, w * 0.80,
                )
                left = right = None

        if left is None or right is None:
            # VP-based fallback trapezoid
            near_half = w * 0.20
            far_half  = max((vp_y / h) * near_half * 0.30, w * 0.02)
            lx_near   = vp_x - near_half
            rx_near   = vp_x + near_half
            lx_far    = vp_x - far_half
            rx_far    = vp_x + far_half
            lane_width_px = near_half * 2

        src_pts = np.float32([
            [lx_near, y_near],
            [rx_near, y_near],
            [lx_far,  y_far ],
            [rx_far,  y_far ],
        ])

        # 4. Metric depth estimation from perspective ratio
        depth_ratio   = max((y_near - vp_y), 1.0) / max((y_far - vp_y), 1.0)
        visible_depth = self.lane_width_m * depth_ratio * 2.8   # empirical factor

        dst_pts = np.float32([
            [0.0,               0.0          ],
            [self.lane_width_m, 0.0          ],
            [0.0,               visible_depth],
            [self.lane_width_m, visible_depth],
        ])

        # 5. Compute homography
        try:
            H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        except cv2.error as e:
            logger.error("findHomography failed: %s", e)
            return None

        if H is None:
            return None

        px_per_m_near = lane_width_px / max(self.lane_width_m, 0.01)
        far_width_px  = abs(rx_far - lx_far)
        px_per_m_far  = max(far_width_px / self.lane_width_m, 0.5)

        # 6. BEV validation: lane lines should be near-parallel after warp
        if left is not None and right is not None:
            confidence = self._bev_validate(H, left, right, confidence)

        return CalibrationResult(
            H=H,
            vanishing_point=(vp_x, vp_y),
            src_pts=src_pts,
            dst_pts=dst_pts,
            lane_width_px=lane_width_px,
            pixels_per_meter_near=px_per_m_near,
            pixels_per_meter_far=px_per_m_far,
            visible_depth_m=visible_depth,
            confidence=confidence,
            method=method,
            frame_index=self._frame_index,
        )

    def _build_manual_result(self) -> Optional[CalibrationResult]:
        if self._manual_src is None or self._manual_dst is None:
            logger.debug("Manual calibration requested but no src_pts/dst_pts in config.")
            return None

        try:
            H, status = cv2.findHomography(
                self._manual_src, self._manual_dst, cv2.RANSAC, 5.0
            )
        except cv2.error as e:
            logger.error("Manual findHomography failed: %s", e)
            return None

        if H is None:
            return None

        near_pts      = self._manual_src[:2]
        lane_width_px = float(np.linalg.norm(near_pts[1] - near_pts[0]))
        dst_width_m   = float(abs(self._manual_dst[1][0] - self._manual_dst[0][0]))
        depth_m       = float(abs(self._manual_dst[2][1] - self._manual_dst[0][1]))

        # Estimate VP from diagonal edges of the manual trapezoid
        vp = VanishingPointEstimator._intersect(
            LaneLine(
                x1=float(self._manual_src[0][0]), y1=float(self._manual_src[0][1]),
                x2=float(self._manual_src[2][0]), y2=float(self._manual_src[2][1]),
                side="left",
            ),
            LaneLine(
                x1=float(self._manual_src[1][0]), y1=float(self._manual_src[1][1]),
                x2=float(self._manual_src[3][0]), y2=float(self._manual_src[3][1]),
                side="right",
            ),
        )
        vp_x = float(vp[0]) if vp else float(np.mean(self._manual_src[:, 0]))
        vp_y = float(vp[1]) if vp else float(np.min(self._manual_src[:, 1]))

        logger.info(
            "Manual calibration loaded | lane_px=%.0f | depth=%.1fm | conf=1.00",
            lane_width_px, depth_m,
        )

        return CalibrationResult(
            H=H,
            vanishing_point=(vp_x, vp_y),
            src_pts=self._manual_src,
            dst_pts=self._manual_dst,
            lane_width_px=lane_width_px,
            pixels_per_meter_near=lane_width_px / max(dst_width_m, 0.01),
            pixels_per_meter_far=lane_width_px / max(dst_width_m, 0.01) * 0.30,
            visible_depth_m=depth_m,
            confidence=1.0,
            method="manual",
            frame_index=0,
        )

    @staticmethod
    def _bev_validate(
        H: np.ndarray,
        left: LaneLine,
        right: LaneLine,
        base_conf: float,
    ) -> float:
        """
        Warp both lane lines to BEV and check they are near-parallel.
        Reduces confidence proportionally to angular divergence.
        0° divergence = no penalty | 20°+ divergence = 50% penalty.
        """
        def warp(H, x, y):
            p = np.array([x, y, 1.0])
            w = H @ p
            return w[0] / w[2], w[1] / w[2]

        def bev_angle(p1, p2):
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            return float(np.degrees(np.arctan2(abs(dx), max(abs(dy), 1e-6))))

        l1 = warp(H, left.x1,  left.y1)
        l2 = warp(H, left.x2,  left.y2)
        r1 = warp(H, right.x1, right.y1)
        r2 = warp(H, right.x2, right.y2)

        angle_diff = abs(bev_angle(l1, l2) - bev_angle(r1, r2))
        penalty    = min(angle_diff / 20.0, 1.0) * 0.50
        return max(base_conf * (1.0 - penalty), 0.0)