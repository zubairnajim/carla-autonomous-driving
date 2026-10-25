"""Traffic-light state from the RGB camera with OpenCV colour segmentation.

A lit lamp is a small, compact, very bright and saturated blob above the
horizon, with unlit grey/black housing directly above or below it. Candidates are scored and the state
is smoothed with a majority vote over the last few frames."""
from collections import Counter, deque

import cv2
import numpy as np

RED, YELLOW, GREEN, NONE = "RED", "YELLOW", "GREEN", "NONE"

# OpenCV HSV hue is 0..179
HSV_RANGES = {
    RED: [((0, 100, 215), (10, 255, 255)), ((168, 100, 215), (179, 255, 255))],
    YELLOW: [((14, 100, 215), (34, 255, 255))],
    GREEN: [((42, 90, 215), (98, 255, 255))],
}
DRAW_COLORS = {RED: (0, 0, 255), YELLOW: (0, 220, 255), GREEN: (0, 255, 0), NONE: (160, 160, 160)}


class TrafficLightDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.horizon = cfg.cam_h // 2 - 8          # lamps are above the camera -> above horizon
        self.history = deque(maxlen=cfg.tl_vote_frames)
        self.state = NONE
        self.raw = NONE
        self.box = None

    def _candidates(self, bgr):
        roi = bgr[: self.horizon]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        h, w = sat.shape
        out = []
        for state, ranges in HSV_RANGES.items():
            mask = np.zeros((h, w), np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for i in range(1, n):
                x, y, bw, bh, area = stats[i]
                if not (self.cfg.tl_min_area <= area <= self.cfg.tl_max_area):
                    continue
                if not (0.5 <= bw / bh <= 2.0) or area / (bw * bh) < 0.45:
                    continue
                # housing: directly above or below the lamp there must be unlit
                # housing - grey or black (unsaturated) and not the lamp's colour.
                # This is what separates a lamp from red brick or green foliage.
                span = int(bh * 1.6) + 2
                housing = False
                for rows in (slice(max(y - span, 0), y), slice(y + bh, y + bh + span)):
                    strip_sat, strip_same = sat[rows, x:x + bw], mask[rows, x:x + bw]
                    if strip_sat.size and np.median(strip_sat) < 90 and (strip_same > 0).mean() < 0.15:
                        housing = True
                if not housing:
                    continue
                # relevance: big, and close to / right of the image centre
                u = x + bw / 2
                lateral = abs(u - 0.6 * w) / w
                out.append((area * (1.0 - lateral), state, (x, y, bw, bh)))
        return out

    def detect(self, bgr, gate=None):
        """`gate(box, state) -> bool` lets the caller reject lamps in implausible places."""
        cands = self._candidates(bgr)
        if gate is not None:
            cands = [c for c in cands if gate(c[2], c[1])]
        if cands:
            _, raw, self.box = max(cands, key=lambda c: c[0])
        else:
            raw, self.box = NONE, None
        self.raw = raw                  # this frame's colour, before voting
        self.history.append(raw)
        votes = Counter(self.history)
        self.state = votes.most_common(1)[0][0]
        return self.state

    def draw(self, frame):
        if self.box is None or self.state == NONE:
            return
        x, y, w, h = self.box
        c = DRAW_COLORS[self.state]
        cv2.rectangle(frame, (x - 6, y - 6), (x + w + 6, y + h + 6), c, 2)
        cv2.putText(frame, self.state, (x - 6, max(y - 12, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
