"""All tunable parameters for the autonomous driving stack."""
from dataclasses import dataclass


# Height of each lamp above the road (Town01/02 pole signals). Converts a lamp's
# image row into range, which then selects the LiDAR returns on its pole.
TL_LAMP_HEIGHTS = {"RED": 2.95, "YELLOW": 2.63, "GREEN": 2.30}


@dataclass
class Config:
    # --- simulator ---
    host: str = "127.0.0.1"
    port: int = 2000
    tm_port: int = 8000
    town: str = "Town02"
    dt: float = 0.05                 # fixed simulation step (20 Hz)
    seed: int = 7
    start_index: int = None          # spawn point for the ego (None = seeded random)
    n_vehicles: int = 15
    n_walkers: int = 40
    walker_cross_factor: float = 0.4  # fraction of walkers allowed to cross roads

    # --- ego / sensors (positions relative to vehicle centre, metres) ---
    ego_blueprint: str = "vehicle.tesla.model3"
    cam_w: int = 800
    cam_h: int = 600
    cam_fov: float = 100.0
    cam_x: float = 1.5
    cam_z: float = 1.6
    lidar_z: float = 1.9
    lidar_channels: int = 32
    lidar_range: float = 50.0
    lidar_pps: int = 300_000
    lidar_upper_fov: float = 3.0
    lidar_lower_fov: float = -22.0

    # --- lane detection (bird's-eye view) ---
    bev_x_min: float = 5.0           # metres ahead of camera
    bev_x_max: float = 25.0
    bev_half_width: float = 5.0
    bev_ppm: int = 20                # pixels per metre
    lane_width: float = 3.5          # prior, refined online
    lane_tophat_thr: int = 30
    lane_min_pixels: int = 60

    # --- traffic lights ---
    tl_min_area: int = 4
    tl_max_area: int = 2500
    tl_vote_frames: int = 5
    tl_junction_range: float = 45.0  # only obey lights when a junction is this close
    tl_stop_margin: float = 9.0      # fallback: stop this far before the junction entry
    tl_stop_gap: float = 5.5         # stop with the bumper this far before the light pole
    tl_lamp_heights = TL_LAMP_HEIGHTS
    tl_lateral_min: float = -1.5     # lamp must be within this lateral band (m, + = right)
    tl_lateral_max: float = 8.0
    tl_past_junction: float = 3.0    # ... and no further than this beyond the junction entry

    # --- pedestrians (YOLOX through cv2.dnn) ---
    yolo_model: str = "models/object_detection_yolox_2022nov.onnx"
    yolo_every: int = 3              # run detector every N frames
    yolo_conf: float = 0.45
    yolo_nms: float = 0.5
    ped_slow_lateral: float = 2.6    # pedestrian this close to our path -> slow down
    ped_stop_lateral: float = 1.8    # ... -> stop
    ped_range: float = 22.0

    # --- LiDAR obstacle corridor ---
    obst_min_h: float = 0.4          # height band above ground treated as obstacle
    obst_max_h: float = 1.8
    corridor_half_width: float = 1.15
    obst_min_points: int = 8

    # --- control ---
    cruise_kmh: float = 30.0
    junction_kmh: float = 18.0
    ped_slow_kmh: float = 10.0
    comfort_decel: float = 2.5
    lat_accel_max: float = 2.0
    follow_gap: float = 4.0
    steer_gain: float = 1.3
    cross_track_gain: float = 0.06   # steer per metre of lateral error (map mode)
    wheelbase: float = 2.9
    rear_axle_x: float = -1.4
