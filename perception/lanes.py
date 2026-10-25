"""Classical OpenCV lane detection: inverse-perspective warp to a bird's-eye
view, top-hat filter to isolate painted markings, sliding-window search and a
polynomial fit per lane line. All fits are metric: lateral = f(forward)."""
import cv2
import numpy as np


class LaneResult:
    def __init__(self):
        self.left = None        # polynomial coeffs, lateral(m) = poly(forward from camera, m)
        self.right = None
        self.center = None
        self.confidence = 0.0
        self.width = None
        self.mask = None        # BEV binary mask (debug view)

    def center_at(self, x_cam):
        return float(np.polyval(self.center, x_cam))


class LaneDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ppm = cfg.bev_ppm
        self.bev_w = int(2 * cfg.bev_half_width * self.ppm)
        self.bev_h = int((cfg.bev_x_max - cfg.bev_x_min) * self.ppm)
        self.width = cfg.lane_width
        self.prev = {"left": None, "right": None}
        self.missed = {"left": 99, "right": 99}

        # Homography from the camera's known pose (flat-ground assumption).
        f = cfg.cam_w / (2 * np.tan(np.radians(cfg.cam_fov) / 2))
        cx, cy = cfg.cam_w / 2, cfg.cam_h / 2

        def ground_px(x, y):
            return [cx + f * y / x, cy + f * cfg.cam_z / x]

        hw = cfg.bev_half_width
        src = np.float32([ground_px(cfg.bev_x_max, -hw), ground_px(cfg.bev_x_max, hw),
                          ground_px(cfg.bev_x_min, hw), ground_px(cfg.bev_x_min, -hw)])
        dst = np.float32([[0, 0], [self.bev_w, 0], [self.bev_w, self.bev_h], [0, self.bev_h]])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)
        k = int(0.8 * self.ppm) | 1
        self.tophat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))

    # BEV pixel <-> metres
    def _col(self, lateral):
        return lateral * self.ppm + self.bev_w / 2

    def _lat(self, col):
        return (col - self.bev_w / 2) / self.ppm

    def _fwd(self, row):
        return self.cfg.bev_x_max - row / self.ppm

    def _row(self, fwd):
        return (self.cfg.bev_x_max - fwd) * self.ppm

    def _binary(self, bgr):
        bev = cv2.warpPerspective(bgr, self.M, (self.bev_w, self.bev_h))
        v = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)[:, :, 2]
        tophat = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, self.tophat_kernel)
        mask = (tophat > self.cfg.lane_tophat_thr).astype(np.uint8)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_kernel)

    def _fit(self, rows, cols):
        if len(rows) < self.cfg.lane_min_pixels:
            return None
        fwd, lat = self._fwd(rows), self._lat(cols)
        span = fwd.max() - fwd.min()
        if span < 4.0:
            return None
        deg = 2 if span > 10.0 else 1
        coeffs = np.polyfit(fwd, lat, deg)
        return np.concatenate([np.zeros(3 - len(coeffs)), coeffs])

    def _search_prior(self, ys, xs, prior, margin=0.5):
        expected = self._col(np.polyval(prior, self._fwd(ys)))
        keep = np.abs(xs - expected) < margin * self.ppm
        return ys[keep], xs[keep]

    def _search_windows(self, mask, ys, xs, lo, hi, n_win=10, margin=0.5):
        """Histogram peak between lateral bounds [lo, hi] m, then sliding windows."""
        c0, c1 = max(int(self._col(lo)), 0), min(int(self._col(hi)), self.bev_w)
        if c1 <= c0:
            return np.array([]), np.array([])
        hist = mask[self.bev_h // 2:, c0:c1].sum(axis=0)
        if hist.max() < 15:
            return np.array([]), np.array([])
        cur = c0 + int(np.argmax(hist))
        m = int(margin * self.ppm)
        win_h = self.bev_h // n_win
        sel = []
        for w in range(n_win):
            y_hi = self.bev_h - w * win_h
            idx = np.nonzero((ys >= y_hi - win_h) & (ys < y_hi) & (np.abs(xs - cur) < m))[0]
            sel.append(idx)
            if len(idx) > 12:
                cur = int(xs[idx].mean())
        sel = np.concatenate(sel)
        return ys[sel], xs[sel]

    def detect(self, bgr):
        res = LaneResult()
        mask = self._binary(bgr)
        res.mask = mask
        ys, xs = mask.nonzero()
        near = self.cfg.bev_x_min + 2.0

        fits = {}
        for side, (lo, hi) in (("left", (-self.width - 0.4, -0.5)), ("right", (0.5, self.width + 0.4))):
            fit = None
            if self.prev[side] is not None and self.missed[side] < 6:
                fit = self._fit(*self._search_prior(ys, xs, self.prev[side]))
            if fit is None:
                fit = self._fit(*self._search_windows(mask, ys, xs, lo, hi))
            # a line must sit on its own side of the car
            if fit is not None:
                lat = np.polyval(fit, near)
                if (side == "left") != (lat < 0) or abs(lat) > self.width + 0.6:
                    fit = None
            fits[side] = fit

        left, right = fits["left"], fits["right"]
        if left is not None and right is not None:
            xs_chk = np.array([near, 12.0, 18.0])
            w = np.polyval(right, xs_chk) - np.polyval(left, xs_chk)
            if w.min() < 2.4 or w.max() > 5.0 or np.ptp(w) > 1.0:
                # inconsistent pair: keep the line with more history
                if self.missed["left"] <= self.missed["right"]:
                    right = None
                else:
                    left = None
            else:
                self.width = float(np.clip(0.9 * self.width + 0.1 * w[0], 3.0, 4.2))

        for side, fit in (("left", left), ("right", right)):
            if fit is None:
                self.missed[side] += 1
                if self.missed[side] >= 6:
                    self.prev[side] = None
            else:
                self.missed[side] = 0
                self.prev[side] = fit

        half = np.array([0.0, 0.0, self.width / 2])
        if left is not None and right is not None:
            res.center, res.confidence = (left + right) / 2, 1.0
        elif left is not None:
            res.center, res.confidence = left + half, 0.6
        elif right is not None:
            res.center, res.confidence = right - half, 0.6
        res.left, res.right, res.width = left, right, self.width
        return res

    def draw(self, frame, res):
        """Overlay the detected lane on the camera image (in place)."""
        if res.center is None:
            return
        overlay = np.zeros((self.bev_h, self.bev_w, 3), np.uint8)
        fwd = np.linspace(self.cfg.bev_x_min, self.cfg.bev_x_max - 3, 30)
        rows = self._row(fwd)
        half = self.width / 2
        l = res.left if res.left is not None else res.center - np.array([0, 0, half])
        r = res.right if res.right is not None else res.center + np.array([0, 0, half])
        pl = np.stack([self._col(np.polyval(l, fwd)), rows], axis=1)
        pr = np.stack([self._col(np.polyval(r, fwd)), rows], axis=1)
        cv2.fillPoly(overlay, [np.int32(np.vstack([pl, pr[::-1]]))], (0, 90, 0))
        for pts, found in ((pl, res.left is not None), (pr, res.right is not None)):
            color = (0, 255, 255) if found else (0, 120, 255)
            cv2.polylines(overlay, [np.int32(pts)], False, color, 4)
        warped = cv2.warpPerspective(overlay, self.Minv, (frame.shape[1], frame.shape[0]))
        cv2.addWeighted(frame, 1.0, warped, 0.6, 0, dst=frame)
