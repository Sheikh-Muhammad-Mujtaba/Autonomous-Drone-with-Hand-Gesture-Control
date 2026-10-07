"""
Reusable 2D top-down map renderer for Tello position tracking.

Draws an auto-scaling, auto-panning grid with real-distance labels,
path history, and yaw-direction arrows for diagnostics.

The map uses a conventional coordinate system: +x right, +y up.
Supports an optional second yaw value per point so dead-reckoning
heading and onboard-sensor heading can be overlaid and compared.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

import cv2
import numpy as np


class PositionMapRenderer:
    """Renders a scrollable, auto-scaled 2D map of drone position history.

    Grid lines are drawn at real-world intervals (default 50 cm) and
    labeled with the actual distance from origin.  The viewport is
    recomputed every render() call to fit all recorded points.
    """

    # colours (BGR)
    _COL_BG = (255, 255, 255)       # white background
    _COL_GRID = (200, 200, 200)      # light grey grid lines
    _COL_LABEL = (100, 100, 100)     # grey text labels
    _COL_HOME = (0, 200, 0)          # green origin marker
    _COL_PATH = (50, 50, 200)        # red-ish path line
    _COL_POS = (0, 0, 200)           # red current position marker
    _COL_EST_YAW = (0, 180, 0)       # green estimated-yaw arrow
    _COL_REAL_YAW = (0, 0, 200)      # red real-yaw arrow (onboard sensor)
    _MARGIN = 44                     # px padding for axis labels
    _HOME_LABEL_OFFSET = 8          # px offset of "HOME" text from origin

    def __init__(self, canvas_size: tuple[int, int] = (500, 500),
                 cm_per_grid: int = 50) -> None:
        """Initialise the renderer.

        Args:
            canvas_size: (width, height) of the output image in pixels.
            cm_per_grid: Spacing between grid lines in real-world cm.
        """
        self._w, self._h = canvas_size
        self._grid_step = cm_per_grid
        self._history: deque[tuple[float, float, float, Optional[float]]] = \
            deque(maxlen=1000)

    # ------------------------------------------------------------------
    def add_point(self, x_cm: float, y_cm: float, yaw_deg: float,
                  real_yaw_deg: Optional[float] = None) -> None:
        """Record a position + heading sample.

        Args:
            x_cm, y_cm:  Estimated position from dead reckoning (cm).
            yaw_deg:     Estimated heading from dead reckoning (degrees,
                          0 = +x direction, increasing CCW).
            real_yaw_deg: Optional second heading from the drone's onboard
                          yaw sensor; drawn as a second arrow in a
                          different colour for comparison.
        """
        self._history.append((x_cm, y_cm, yaw_deg, real_yaw_deg))

    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Clear all point history — call on takeoff for a fresh map."""
        self._history.clear()

    # ------------------------------------------------------------------
    def render(self) -> np.ndarray:
        """Return a BGR image showing the current map state."""
        img = np.full((self._h, self._w, 3), self._COL_BG, dtype=np.uint8)
        if not self._history:
            # Show origin only at a default zoom level
            self._draw_grid(img, [(0.0, 0.0)], 0.5,
                            self._w / 2.0, self._h / 2.0)
            return img

        # Collect all points that define the visible bounds
        all_pts = [(p[0], p[1]) for p in self._history] + [(0.0, 0.0)]
        return self._render_points(img, all_pts)

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _render_points(self, img: np.ndarray,
                       points: list[tuple[float, float]]) -> np.ndarray:
        """Compute scale & offset then draw the full map."""
        # Viewport bounds with 30 % margin
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        half_extent = max(max_x - cx, cx - min_x,
                          max_y - cy, cy - min_y,
                          1.0) * 1.3

        avail_w = self._w - 2.0 * self._MARGIN
        avail_h = self._h - 2.0 * self._MARGIN
        scale = min(avail_w / (2.0 * half_extent),
                    avail_h / (2.0 * half_extent))

        # Canvas offsets — note +cy_off because canvas y increases downward
        cx_off = self._w / 2.0 - scale * cx
        cy_off = self._h / 2.0 + scale * cy   # flipped y

        self._draw_grid(img, points, scale, cx_off, cy_off)

        # ----- Origin marker -----
        ox, oy = self._w2c(0.0, 0.0, scale, cx_off, cy_off)
        cv2.circle(img, (int(ox), int(oy)), 6, self._COL_HOME, -1)
        cv2.putText(img, "HOME",
                    (int(ox) + self._HOME_LABEL_OFFSET,
                     int(oy) + self._HOME_LABEL_OFFSET),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self._COL_HOME, 1)

        # ----- Path history -----
        if len(self._history) > 1:
            pts = np.array([
                self._w2c(*self._history[i][:2], scale, cx_off, cy_off)
                for i in range(len(self._history))
            ], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, self._COL_PATH, 2)

        # ----- Current position marker + yaw arrows -----
        last = self._history[-1]
        px, py = self._w2c(last[0], last[1], scale, cx_off, cy_off)
        cv2.circle(img, (int(px), int(py)), 8, self._COL_POS, 2)

        # Estimated-yaw arrow
        self._draw_yaw_arrow(img, px, py, last[2], scale,
                             self._COL_EST_YAW)
        # Real-yaw arrow (from onboard sensor)
        if last[3] is not None:
            self._draw_yaw_arrow(img, px, py, last[3], scale,
                                 self._COL_REAL_YAW, arrow_len_px=28)

        return img

    # ------------------------------------------------------------------
    def _draw_grid(self, img: np.ndarray,
                   points: list[tuple[float, float]],
                   scale: float, cx_off: float,
                   cy_off: float) -> None:
        """Draw labelled grid lines every cm_per_grid world units."""
        # Find which grid lines are visible
        corners = [
            self._c2w(0, 0, scale, cx_off, cy_off),
            self._c2w(self._w, self._h, scale, cx_off, cy_off),
        ]
        wx_min = min(c[0] for c in corners)
        wx_max = max(c[0] for c in corners)
        wy_min = min(c[1] for c in corners)
        wy_max = max(c[1] for c in corners)

        step = self._grid_step
        # First grid line <= wx_min
        gx0 = math.floor(wx_min / step) * step
        gy0 = math.floor(wy_min / step) * step

        for wx in self._frange(gx0, wx_max + 1, step):
            cx, _ = self._w2c(wx, 0.0, scale, cx_off, cy_off)
            if 0 <= cx <= self._w:
                cv2.line(img, (int(cx), 0), (int(cx), self._h),
                         self._COL_GRID, 1)
                if abs(wx) > 0.1:  # skip origin label here; it's "HOME"
                    label = f"{wx:.0f}cm"
                    tw, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                            0.35, 1)[0]
                    cx_t = int(cx) - tw // 2
                    cy_t = self._MARGIN - 12
                    cv2.putText(img, label, (cx_t, cy_t),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                self._COL_LABEL, 1)

        for wy in self._frange(gy0, wy_max + 1, step):
            _, cy = self._w2c(0.0, wy, scale, cx_off, cy_off)
            if 0 <= cy <= self._h:
                cv2.line(img, (0, int(cy)), (self._w, int(cy)),
                         self._COL_GRID, 1)
                if abs(wy) > 0.1:
                    label = f"{wy:.0f}cm"
                    cv2.putText(img, label,
                                (self._MARGIN - 36, int(cy) + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                self._COL_LABEL, 1)

    # ------------------------------------------------------------------
    def _draw_yaw_arrow(self, img: np.ndarray, px: float, py: float,
                        yaw_deg: float, scale: float,
                        colour: tuple[int, int, int],
                        arrow_len_px: int = 30) -> None:
        """Draw a short arrow from (px,py) in the yaw direction."""
        angle = math.radians(yaw_deg)
        # Canvas y is flipped relative to world y
        dx = arrow_len_px * math.cos(angle)
        dy = -arrow_len_px * math.sin(angle)
        tip = (int(px + dx), int(py + dy))
        cv2.arrowedLine(img, (int(px), int(py)), tip, colour, 2,
                        tipLength=0.35)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _w2c(wx: float, wy: float, s: float,
             cx_off: float, cy_off: float) -> tuple[float, float]:
        """World → canvas (flipped y)."""
        return s * wx + cx_off, -s * wy + cy_off

    @staticmethod
    def _c2w(cx: float, cy: float, s: float,
             cx_off: float, cy_off: float) -> tuple[float, float]:
        """Canvas → world (flipped y)."""
        return (cx - cx_off) / s, (cy_off - cy) / s

    @staticmethod
    def _frange(start: float, stop: float, step: float):
        """Float-range generator (inclusive stop guard)."""
        v = start
        while v <= stop:
            yield v
            v += step
