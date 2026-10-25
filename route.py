"""Rolling waypoint route: gives the car a destination through junctions,
where there are no lane markings for the camera to follow."""
import math
import random

import numpy as np


def world_to_vehicle(transform, xy):
    """World XY points (N,2) -> vehicle frame (x forward, y right)."""
    yaw = math.radians(transform.rotation.yaw)
    d = np.asarray(xy, dtype=np.float64) - np.array([transform.location.x, transform.location.y])
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1)


class Route:
    def __init__(self, carla_map, start_location, seed=0, step=1.0, horizon=70, waypoints=None):
        """Without `waypoints` the route wanders (random turn at each junction);
        with them it is a fixed route from the navigation planner."""
        self.step = step
        self.horizon = horizon
        self.rng = random.Random(seed)
        self.fixed = waypoints is not None
        self.wps = list(waypoints) if self.fixed else [carla_map.get_waypoint(start_location)]
        self._extend()

    @property
    def remaining(self):
        """Metres left to the destination (inf when wandering)."""
        return (len(self.wps) - 1) * self.step if self.fixed else float("inf")

    def _extend(self):
        while not self.fixed and len(self.wps) < self.horizon:
            nxt = self.wps[-1].next(self.step)
            if not nxt:
                break
            self.wps.append(self.rng.choice(nxt))

    def update(self, location):
        """Drop waypoints behind the car and refill the horizon."""
        n = min(len(self.wps), 25)
        d = [self.wps[i].transform.location.distance(location) for i in range(n)]
        self.wps = self.wps[int(np.argmin(d)):]
        self._extend()

    def path_vehicle_frame(self, transform):
        xy = [(w.transform.location.x, w.transform.location.y) for w in self.wps]
        return world_to_vehicle(transform, xy)

    @property
    def in_junction(self):
        return self.wps[0].is_junction

    def distance_to_junction(self):
        """Metres along the route to the next junction entry (inf if none in horizon)."""
        if self.in_junction:
            return 0.0
        for i, w in enumerate(self.wps):
            if w.is_junction:
                return i * self.step
        return float("inf")

    def curvature_ahead(self, n=18):
        """Largest curvature (1/m) over the next n waypoints."""
        yaws = np.unwrap([math.radians(w.transform.rotation.yaw) for w in self.wps[:n]])
        if len(yaws) < 6:
            return 0.0
        return float(np.max(np.abs(yaws[5:] - yaws[:-5])) / (5 * self.step))
