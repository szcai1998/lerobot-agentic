import mujoco
import numpy as np


class InvalidDepthError(Exception):
    """Raised when sampled depth violates operational camera bounds."""


class CameraGeometry:
    """
    Derives intrinsic matrix K and extrinsic transformation T_world_cam dynamically
    from MuJoCo camera parameters, performing metric unprojection with bound checks.
    """
    def __init__(self, model: mujoco.MjModel, camera_name: str, width: int = 640, height: int = 480):
        self.model = model
        self.camera_name = camera_name
        self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if self.camera_id == -1:
            raise ValueError(f"Camera '{camera_name}' not found in MuJoCo model.")
        self.width = width
        self.height = height

        # Derive intrinsics from vertical field of view (fovy)
        fovy_rad = np.deg2rad(self.model.cam_fovy[self.camera_id])
        self.fy = (height / 2.0) / np.tan(fovy_rad / 2.0)
        self.fx = self.fy  # Standard square pixels
        self.cx = width / 2.0
        self.cy = height / 2.0
        self.K = np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        self.inv_K = np.linalg.inv(self.K)

    def unproject_pixel_to_world(
        self,
        u: float,
        v: float,
        depth: float,
        data: mujoco.MjData,
        min_depth: float = 0.05,
        max_depth: float = 2.5
    ) -> np.ndarray:
        """
        Unprojects a 2D pixel coordinate (u, v) and metric depth into 3D world coordinates.
        Raises InvalidDepthError if depth reading is outside valid physical range.
        """
        if not (min_depth <= depth <= max_depth) or np.isnan(depth):
            raise InvalidDepthError(f"INVALID_DEPTH: Sampled depth {depth:.4f}m outside [{min_depth}, {max_depth}]m")

        # Camera frame optical ray
        p_cam = depth * (self.inv_K @ np.array([u, v, 1.0], dtype=np.float64))

        # Extrinsics from MuJoCo data (cam_xpos and cam_xmat)
        cam_pos = data.cam_xpos[self.camera_id]
        cam_rot = data.cam_xmat[self.camera_id].reshape(3, 3)

        # MuJoCo camera coordinate convention: +X right, +Y up, -Z optical axis
        # Standard robotics optical frame: +X right, +Y down, +Z optical axis
        r_mujoco_optical = np.array([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0]
        ], dtype=np.float64)

        p_world = cam_pos + cam_rot @ (r_mujoco_optical @ p_cam)
        return p_world
