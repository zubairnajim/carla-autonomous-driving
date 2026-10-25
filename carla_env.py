"""CARLA world setup: synchronous mode, ego vehicle, sensors, traffic, walkers."""
import queue
import random

import carla
import numpy as np


class CarlaEnv:
    def __init__(self, cfg):
        self.cfg = cfg
        self.actors = []
        self.walker_controllers = []
        self.sensors = {}
        self.queues = {}
        self.collisions = []
        self.rng = random.Random(cfg.seed)

        self.client = carla.Client(cfg.host, cfg.port)
        self.client.set_timeout(120.0)
        self.world = self.client.get_world()
        if not self.world.get_map().name.endswith(cfg.town):
            self.world = self.client.load_world(cfg.town)
        self.map = self.world.get_map()
        self._original_settings = self.world.get_settings()

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = cfg.dt
        self.world.apply_settings(settings)

        self.tm = self.client.get_trafficmanager(cfg.tm_port)
        self.tm.set_synchronous_mode(True)
        self.tm.set_random_device_seed(cfg.seed)
        self.tm.global_percentage_speed_difference(40.0)

        self.ego = None
        try:
            self._spawn_ego()
            self._spawn_sensors()
            self._spawn_traffic()
            self._spawn_walkers()
        except Exception:
            self.close()
            raise
        self.max_steer_rad = self._max_steer()

    # ------------------------------------------------------------------ spawn
    def _spawn_ego(self):
        bp = self.world.get_blueprint_library().find(self.cfg.ego_blueprint)
        bp.set_attribute("role_name", "hero")
        points = self.map.get_spawn_points()
        self.spawn_points = list(points)
        self.rng.shuffle(points)
        if self.cfg.start_index is not None:
            points.insert(0, self.spawn_points[self.cfg.start_index % len(self.spawn_points)])
        for p in points:
            self.ego = self.world.try_spawn_actor(bp, p)
            if self.ego:
                break
        if not self.ego:
            raise RuntimeError("could not spawn ego vehicle")
        self.actors.append(self.ego)
        self.free_spawn_points = [p for p in points if p.location.distance(self.ego.get_location()) > 8]

    def _add_sensor(self, name, bp, transform, queued=True):
        sensor = self.world.spawn_actor(bp, transform, attach_to=self.ego)
        self.actors.append(sensor)
        self.sensors[name] = sensor
        if queued:
            q = queue.Queue()
            self.queues[name] = q
            sensor.listen(q.put)
        return sensor

    def _spawn_sensors(self):
        cfg, lib = self.cfg, self.world.get_blueprint_library()

        cam = lib.find("sensor.camera.rgb")
        cam.set_attribute("image_size_x", str(cfg.cam_w))
        cam.set_attribute("image_size_y", str(cfg.cam_h))
        cam.set_attribute("fov", str(cfg.cam_fov))
        self._add_sensor("camera", cam, carla.Transform(carla.Location(x=cfg.cam_x, z=cfg.cam_z)))

        lidar = lib.find("sensor.lidar.ray_cast")
        lidar.set_attribute("channels", str(cfg.lidar_channels))
        lidar.set_attribute("range", str(cfg.lidar_range))
        lidar.set_attribute("points_per_second", str(cfg.lidar_pps))
        lidar.set_attribute("rotation_frequency", str(1.0 / cfg.dt))
        lidar.set_attribute("upper_fov", str(cfg.lidar_upper_fov))
        lidar.set_attribute("lower_fov", str(cfg.lidar_lower_fov))
        lidar.set_attribute("dropoff_general_rate", "0.0")
        lidar.set_attribute("noise_stddev", "0.0")
        self._add_sensor("lidar", lidar, carla.Transform(carla.Location(z=cfg.lidar_z)))

        col = self._add_sensor("collision", lib.find("sensor.other.collision"), carla.Transform(), queued=False)
        col.listen(lambda e: self.collisions.append((e.frame, e.other_actor.type_id)))

    def _spawn_traffic(self):
        lib = self.world.get_blueprint_library()
        bps = [b for b in lib.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4]
        for p in self.free_spawn_points[: self.cfg.n_vehicles]:
            bp = self.rng.choice(bps)
            if bp.has_attribute("color"):
                bp.set_attribute("color", self.rng.choice(bp.get_attribute("color").recommended_values))
            v = self.world.try_spawn_actor(bp, p)
            if v:
                v.set_autopilot(True, self.cfg.tm_port)
                self.actors.append(v)

    def _spawn_walkers(self):
        lib = self.world.get_blueprint_library()
        self.world.set_pedestrians_seed(self.cfg.seed)
        self.world.set_pedestrians_cross_factor(self.cfg.walker_cross_factor)
        walker_bps = list(lib.filter("walker.pedestrian.*"))
        controller_bp = lib.find("controller.ai.walker")
        walkers = []
        for _ in range(self.cfg.n_walkers):
            loc = self.world.get_random_location_from_navigation()
            if loc is None:
                continue
            bp = self.rng.choice(walker_bps)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            w = self.world.try_spawn_actor(bp, carla.Transform(loc))
            if w:
                walkers.append(w)
        self.world.tick()
        for w in walkers:
            c = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=w)
            self.walker_controllers.append(c)
        self.world.tick()
        for c in self.walker_controllers:
            c.start()
            c.go_to_location(self.world.get_random_location_from_navigation())
            c.set_max_speed(1.0 + self.rng.random() * 0.8)
        for w in walkers:
            # CARLA >= 0.9.15 disables collisions on AI walkers, which also hides
            # them from ray-cast LiDAR; turn them back on.
            w.set_collisions(True)
        self.actors.extend(walkers)
        self.n_walkers_spawned = len(walkers)

    def _max_steer(self):
        wheels = self.ego.get_physics_control().wheels
        return float(np.deg2rad(max(w.max_steer_angle for w in wheels)))

    # ------------------------------------------------------------------- tick
    def tick(self):
        """Advance one step and return the sensor data belonging to that frame."""
        frame = self.world.tick()
        data = {"frame": frame}
        for name, q in self.queues.items():
            while True:
                item = q.get(timeout=10.0)
                if item.frame == frame:
                    data[name] = item
                    break
        return data

    @staticmethod
    def image_to_bgr(image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
        return arr[:, :, :3].copy()

    @staticmethod
    def lidar_to_points(measurement):
        """Nx3 array in the sensor frame: x forward, y right, z up."""
        pts = np.frombuffer(measurement.raw_data, dtype=np.float32).reshape(-1, 4)
        return pts[:, :3].copy()

    def follow_with_spectator(self):
        t = self.ego.get_transform()
        fwd = t.get_forward_vector()
        loc = t.location - carla.Location(x=fwd.x * 8, y=fwd.y * 8) + carla.Location(z=4.5)
        rot = carla.Rotation(pitch=-18, yaw=t.rotation.yaw)
        self.world.get_spectator().set_transform(carla.Transform(loc, rot))

    # ------------------------------------------------------------------ close
    def close(self):
        for s in self.sensors.values():
            try:
                s.stop()
            except RuntimeError:
                pass
        for c in self.walker_controllers:
            try:
                c.stop()
            except RuntimeError:
                pass
        ids = [a.id for a in self.walker_controllers + self.actors]
        self.actors, self.walker_controllers, self.sensors = [], [], {}
        try:
            self.client.set_timeout(10.0)
            self.client.apply_batch_sync([carla.command.DestroyActor(i) for i in ids])
            self.tm.set_synchronous_mode(False)
            self.world.apply_settings(self._original_settings)
        except RuntimeError:
            print("Could not clean up: the simulator is not responding.")
