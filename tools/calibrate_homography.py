"""
calibrate_homography.py — One-time homography calibration tool.

Run this ONCE on your traffic video to find the correct src_pts/dst_pts
for your specific camera angle and position.

Usage:
    python calibrate_homography.py --video datasets/traffic.mp4
    python calibrate_homography.py --video datasets/traffic.mp4 --frame 50

Instructions:
    1. A frame from your video will open.
    2. Click EXACTLY 4 points on the road that form a rectangle
       whose real-world dimensions you know (e.g. lane markings).
    3. The tool prints the src_pts to paste into settings.yaml.
    4. Press R to reset and re-click. Press Q to quit.

What points to pick (UK motorway):
    - Lane line dashes are 4 m long, gaps are 6 m  (10 m cycle)
    - Lane width = 3.65 m
    - Pick two parallel lane lines, two dashes apart:
        Point 1: start of a near dash, left lane line
        Point 2: start of a near dash, right lane line
        Point 3: start of dash 10 m ahead, left lane line
        Point 4: start of dash 10 m ahead, right lane line
    Then dst_pts = [[0,0],[3.65,0],[0,10],[3.65,10]]
"""

import cv2
import numpy as np
import argparse
import sys
import json


# ── Real-world coords to match your 4 clicked points ────────────
# EDIT THESE to match whatever road features you click.
# Default: single lane, 3.65 m wide, 10 m deep rectangle.
REAL_WORLD_PTS = np.float32([
    [0.00,  0.0],   # Point 1: near-left
    [3.65,  0.0],   # Point 2: near-right
    [0.00, 10.0],   # Point 3: far-left   (10 m ahead)
    [3.65, 10.0],   # Point 4: far-right  (10 m ahead)
])

# ── Validation: known speed test ────────────────────────────────
# If you know a vehicle crosses a certain pixel range in N frames at FPS,
# set these for auto-validation after calibration.
TEST_SPEED_KMH   = 80.0   # expected speed of test vehicle
TEST_FPS         = 30.0
# ────────────────────────────────────────────────────────────────


clicked_pts = []
frame_display = None
frame_clean = None


def mouse_callback(event, x, y, flags, param):
    global clicked_pts, frame_display
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_pts) < 4:
            clicked_pts.append([x, y])
            # Draw dot + index
            cv2.circle(frame_display, (x, y), 7, (0, 0, 255), -1)
            cv2.putText(frame_display, str(len(clicked_pts)),
                        (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)
            if len(clicked_pts) == 4:
                # Draw the quadrilateral
                pts = np.array(clicked_pts, dtype=np.int32)
                cv2.polylines(frame_display, [pts], True, (0, 255, 0), 2)
                _show_result()


def _show_result():
    global clicked_pts
    src = np.float32(clicked_pts)
    dst = REAL_WORLD_PTS

    # Compute homography
    H, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        print("ERROR: Homography failed. Try different points.")
        return

    print("\n" + "="*60)
    print("CALIBRATION RESULT")
    print("="*60)
    print("\nPaste this into your settings.yaml under analytics.homography:\n")

    yaml_block = f"""  homography:
    src_pts:
      - [{clicked_pts[0][0]}, {clicked_pts[0][1]}]
      - [{clicked_pts[1][0]}, {clicked_pts[1][1]}]
      - [{clicked_pts[2][0]}, {clicked_pts[2][1]}]
      - [{clicked_pts[3][0]}, {clicked_pts[3][1]}]
    dst_pts:
      - [{REAL_WORLD_PTS[0][0]}, {REAL_WORLD_PTS[0][1]}]
      - [{REAL_WORLD_PTS[1][0]}, {REAL_WORLD_PTS[1][1]}]
      - [{REAL_WORLD_PTS[2][0]}, {REAL_WORLD_PTS[2][1]}]
      - [{REAL_WORLD_PTS[3][0]}, {REAL_WORLD_PTS[3][1]}]"""

    print(yaml_block)

    # Also print as Python lists for direct paste into traffic_analyzer.py
    print("\nOR paste into traffic_analyzer.py _DEFAULT_SRC_PTS / _DEFAULT_DST_PTS:\n")
    print(f"_DEFAULT_SRC_PTS = np.float32({clicked_pts})")
    print(f"_DEFAULT_DST_PTS = np.float32({REAL_WORLD_PTS.tolist()})")

    # Validate: transform all 4 points and check they match dst
    print("\n--- Validation ---")
    for i, (px, py) in enumerate(clicked_pts):
        pt = np.array([[[px, py]]], dtype=np.float32)
        world = cv2.perspectiveTransform(pt, H)[0][0]
        expected = REAL_WORLD_PTS[i]
        err = np.linalg.norm(world - expected)
        print(f"  Point {i+1}: pixel({px},{py}) -> world({world[0]:.2f},{world[1]:.2f})"
              f"  expected({expected[0]},{expected[1]})  err={err:.3f}m")

    # Speed sanity check
    # A vehicle moving 1 lane width (3.65 m) in 1 second = 13.1 km/h
    print(f"\n--- Speed sanity ---")
    print(f"  1 m/s = 3.6 km/h")
    print(f"  At {TEST_FPS} FPS, a {TEST_SPEED_KMH} km/h vehicle moves "
          f"{TEST_SPEED_KMH/3.6:.2f} m/s = "
          f"{TEST_SPEED_KMH/3.6/TEST_FPS:.3f} m/frame")

    # Save to file
    result = {
        "src_pts": clicked_pts,
        "dst_pts": REAL_WORLD_PTS.tolist(),
        "H": H.tolist(),
    }
    with open("homography_calibration.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nCalibration saved to homography_calibration.json")
    print("="*60)


def main():
    global clicked_pts, frame_display, frame_clean

    parser = argparse.ArgumentParser(description="Homography calibration tool")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--frame", type=int, default=30,
                        help="Which frame number to use (default: 30)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {args.video}")
        sys.exit(1)

    # Seek to requested frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"ERROR: Could not read frame {args.frame}")
        sys.exit(1)

    frame_clean = frame.copy()
    frame_display = frame.copy()

    h, w = frame.shape[:2]
    print(f"\nVideo frame size: {w}x{h}")
    print(f"\nINSTRUCTIONS:")
    print(f"  Click 4 road points in this order:")
    print(f"  1 = near-left   (bottom-left of your rectangle)")
    print(f"  2 = near-right  (bottom-right of your rectangle)")
    print(f"  3 = far-left    (top-left of your rectangle)")
    print(f"  4 = far-right   (top-right of your rectangle)")
    print(f"")
    print(f"  The rectangle should be a REAL road area you know the size of.")
    print(f"  UK motorway: lane width=3.65m, dashes repeat every 10m.")
    print(f"")
    print(f"  Real-world pts configured as: {REAL_WORLD_PTS.tolist()}")
    print(f"  (Edit REAL_WORLD_PTS at top of script if using different distances)")
    print(f"")
    print(f"  R = reset clicks   Q = quit")

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration", min(w, 1280), min(h, 720))
    cv2.setMouseCallback("Calibration", mouse_callback)

    # Draw guide grid
    _draw_guide(frame_display, w, h)

    while True:
        cv2.imshow("Calibration", frame_display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            clicked_pts = []
            frame_display = frame_clean.copy()
            _draw_guide(frame_display, w, h)
            print("\nReset — click 4 points again.")

    cv2.destroyAllWindows()


def _draw_guide(img, w, h):
    """Draw faint thirds-grid to help locate lane edges."""
    color = (60, 60, 60)
    for i in range(1, 4):
        x = w * i // 4
        cv2.line(img, (x, 0), (x, h), color, 1)
    for i in range(1, 4):
        y = h * i // 4
        cv2.line(img, (0, y), (w, y), color, 1)
    cv2.putText(img, "Click 4 road points: 1=near-left  2=near-right  3=far-left  4=far-right",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "R=reset  Q=quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


if __name__ == "__main__":
    main()