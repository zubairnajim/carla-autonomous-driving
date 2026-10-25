"""Behaviour + control: turns perception into throttle / brake / steer."""
import math
from collections import deque

import carla
import numpy as np

from perception.objects import PERSON
from perception.traffic_lights import GREEN, NONE, RED, YELLOW


class Planner:
    def __init__(self, cfg, max_steer_rad, half_length):
        self.cfg = cfg
        self.max_steer = max_steer_rad
        self.half_length = half_length
        self.steer = 0.0
        self.integral = 0.0
        self.light_latched = False
        self.light_stop = None          # bumper-to-stop-point distance, dead-reckoned between sightings
        self.latched_blind_time = 0.0
        self.light_dists = deque(maxlen=5)
        self.ped_memory = None          # (remaining distance, frames left) of the last stop-for-person

    # --------------------------------------------------------------- steering
    def _pure_pursuit(self, x, y):
        tx = x - self.cfg.rear_axle_x
        delta = math.atan2(2.0 * self.cfg.wheelbase * y, tx * tx + y * y)
        return float(np.clip(delta / self.max_steer, -1.0, 1.0))

    def _steering(self, v, lanes, path, in_junction, dist_junction):
        ld = float(np.clip(4.0 + 0.9 * v, 6.5, 14.0))
        ahead = np.linalg.norm(path, axis=1)
        i = int(np.argmin(np.abs(ahead - ld) + 100.0 * (path[:, 0] < 0)))
        wx, wy = path[i]

        mode = "MAP"
        tx, ty = wx, wy
        if lanes.center is not None and lanes.confidence >= 0.6 and not in_junction and dist_junction > 8.0:
            cy = lanes.center_at(ld - self.cfg.cam_x)
            if abs(cy - wy) < 1.0:            # sanity gate against latching onto the wrong marking
                tx, ty, mode = ld, cy, "CAMERA"
        # CARLA's max steer angle is the inner wheel's; the effective (Ackermann)
        # angle is ~20% smaller, hence the gain. The cross-track term removes
        # the steady-state offset pure pursuit leaves in tight curves.
        target = self.cfg.steer_gain * self._pure_pursuit(tx, ty)
        if mode == "MAP":
            target += self.cfg.cross_track_gain * float(np.clip(path[0][1], -2.0, 2.0))
        target = float(np.clip(target, -1.0, 1.0))
        self.steer += 0.5 * (target - self.steer)
        return self.steer, mode, (tx, ty)

    # ------------------------------------------------------------------ speed
    def _stop_profile(self, distance):
        return math.sqrt(2.0 * self.cfg.comfort_decel * max(distance, 0.0)) if distance > 0.4 else 0.0

    def _light_limit(self, v, tl_state, light_dist, in_junction, dist_junction):
        cfg = self.cfg
        if in_junction or dist_junction > cfg.tl_junction_range:
            self.light_latched = False
            self.light_stop = None
            return None
        if light_dist is not None:
            self.light_dists.append(light_dist)
        else:
            self.light_dists.clear()
        if len(self.light_dists) >= 3:      # median rejects one-frame false detections
            self.light_stop = float(np.median(self.light_dists)) - self.half_length - cfg.tl_stop_gap
        elif self.light_stop is not None:
            self.light_stop -= v * cfg.dt
        d_stop = self.light_stop
        if d_stop is None:
            d_stop = dist_junction - self.half_length - cfg.tl_stop_margin
        if tl_state == GREEN:
            self.light_latched = False
        elif tl_state == RED:
            self.light_latched = True
        elif tl_state == YELLOW and not self.light_latched:
            # only stop for amber if we can do it without slamming the brakes
            self.light_latched = d_stop > v * v / (2.0 * 4.0)
        if tl_state == NONE and self.light_latched and v < 0.2:
            # stopped and the lamp left the field of view: don't wait forever
            self.latched_blind_time += cfg.dt
            if self.latched_blind_time > 45.0:
                self.light_latched = False
        else:
            self.latched_blind_time = 0.0
        self.light_dists = deque(maxlen=5)
        self.ped_memory = None          # (remaining distance, frames left) of the last stop-for-person
        return self._stop_profile(d_stop) if self.light_latched else None

    def _pedestrian_limit(self, v, detections, path):
        cfg = self.cfg
        limit, nearest = None, None
        arclen = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]) + path[0, 0]
        for det in detections:
            if det.cls != PERSON or det.position is None:
                continue
            d = np.linalg.norm(path - np.array(det.position), axis=1)
            i = int(d.argmin())
            lateral, along = float(d[i]), float(arclen[i]) - self.half_length
            if along < -1.0 or along > cfg.ped_range:
                continue
            if lateral < cfg.ped_stop_lateral:
                cand = self._stop_profile(along - 3.0)
                if self.ped_memory is None or along < self.ped_memory[0]:
                    self.ped_memory = (along, int(1.5 / cfg.dt))
            elif lateral < cfg.ped_slow_lateral:
                cand = cfg.ped_slow_kmh / 3.6
            else:
                continue
            if limit is None or cand < limit:
                limit, nearest = cand, along
        # Detections flicker and people vanish below the bonnet line when close:
        # keep honouring the last stop for a moment, dead-reckoning its distance.
        if self.ped_memory is not None:
            along, frames = self.ped_memory
            cand = self._stop_profile(along - 3.0)
            if limit is None or cand < limit:
                limit, nearest = cand, along
            self.ped_memory = (along - v * cfg.dt, frames - 1) if frames > 1 else None
        return limit, nearest

    def step(self, v, lanes, path, route, tl_state, light_dist, detections, free_distance):
        cfg = self.cfg
        in_junction = route.in_junction
        dist_junction = route.distance_to_junction()
        steer, steer_mode, target = self._steering(v, lanes, path, in_junction, dist_junction)

        limits = {"CRUISE": cfg.cruise_kmh / 3.6}
        if in_junction or dist_junction < 25.0:
            limits["JUNCTION"] = cfg.junction_kmh / 3.6
        kappa = route.curvature_ahead()
        if kappa > 1e-3:
            limits["CURVE"] = max(math.sqrt(cfg.lat_accel_max / kappa), 2.5)
        light = self._light_limit(v, tl_state, light_dist, in_junction, dist_junction)
        if light is not None:
            limits["RED LIGHT"] = light
        ped, ped_dist = self._pedestrian_limit(v, detections, path)
        if ped is not None:
            limits["PEDESTRIAN"] = ped
        if free_distance < 40.0:
            limits["OBSTACLE"] = self._stop_profile(free_distance - cfg.follow_gap)
        if route.remaining < 60.0:
            limits["DESTINATION"] = self._stop_profile(route.remaining)

        reason = min(limits, key=limits.get)
        v_ref = limits[reason]

        err = v_ref - v
        self.integral = float(np.clip(self.integral + err * cfg.dt, -3.0, 3.0))
        control = carla.VehicleControl(steer=steer)
        if v_ref < 0.1:
            self.integral = 0.0
            control.brake = 1.0 if (v < 1.0 or free_distance < 2.0) else 0.7
        elif err > -0.4:
            control.throttle = float(np.clip(0.35 * err + 0.08 * self.integral + 0.12, 0.0, 0.7))
        else:
            control.brake = float(np.clip(-0.6 * err, 0.0, 1.0))

        info = dict(reason=reason, v_ref=v_ref, steer_mode=steer_mode, target=target,
                    dist_junction=dist_junction, in_junction=in_junction,
                    light_latched=self.light_latched, ped_dist=ped_dist)
        return control, info
