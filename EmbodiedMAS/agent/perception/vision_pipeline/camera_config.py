"""Agent 挂载相机：捕获参数、外参模型、默认内参入口、深度有效阈值（单一事实来源）。"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# 1. 捕获参数与深度阈值
# ---------------------------------------------------------------------------

AGENT_CAMERA_CAPTURE_PARAMS: dict[str, Any] = {
    "width": 640,
    "height": 640,
    "fov_degrees": 90.0,
    "enable_color": True,
    "enable_depth": True,
    # CaptureAPI serializes omitted protobuf fields as zero, rather than UE's
    # camera defaults: mode=0 disables depth and color_format=0 selects R8.
    "qps": 30.0,
    "color_source": 2,  # FinalColorLDR
    "color_format": 2,  # RGBA8
    "enable_post_process": True,
    "enable_temporal_aa": True,
    "depth_mode": 1,  # linear camera-forward depth in UE UU
    "depth_near": 10.0,
    "depth_far": 10000.0,
}

# Human 与所有 vision 后端共用，相对 agent 原点的挂载偏移（UE cm）。
CAMERA_FORWARD_OFFSET_UU = 9.0
CAMERA_UP_OFFSET_UU = 80.0

VALID_DEPTH_MIN = 0.0
VALID_DEPTH_MAX = 10000.0

Orientation = Union[float, Tuple[float, float, float, float], Any]

__all__ = [
    "AGENT_CAMERA_CAPTURE_PARAMS",
    "CAMERA_FORWARD_OFFSET_UU",
    "CAMERA_UP_OFFSET_UU",
    "VALID_DEPTH_MIN",
    "VALID_DEPTH_MAX",
    "Orientation",
    "orientation_to_yaw_deg",
    "camera_world_position_from_agent",
    "cam_arrays_from_agent",
    "default_intrinsic_matrix",
    "valid_depth_mask",
]


# ---------------------------------------------------------------------------
# 私有 helper
# ---------------------------------------------------------------------------


def _quaternion_components(
    orientation: Orientation,
) -> Tuple[float, float, float, float]:
    if isinstance(orientation, (tuple, list)) and len(orientation) >= 4:
        return (
            float(orientation[0]),
            float(orientation[1]),
            float(orientation[2]),
            float(orientation[3]),
        )
    return (
        float(getattr(orientation, "x", 0.0)),
        float(getattr(orientation, "y", 0.0)),
        float(getattr(orientation, "z", 0.0)),
        float(getattr(orientation, "w", 1.0)),
    )


def _orientation_to_yaw_rad(orientation: Orientation) -> float:
    if isinstance(orientation, (int, float)):
        return math.radians(float(orientation))
    qx, qy, qz, qw = _quaternion_components(orientation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


# ---------------------------------------------------------------------------
# 2. 朝向
# ---------------------------------------------------------------------------


def orientation_to_yaw_deg(orientation: Orientation) -> float:
    """四元数或 float yaw（度）→ 水平 yaw 度。"""
    return float(math.degrees(_orientation_to_yaw_rad(orientation)))


# ---------------------------------------------------------------------------
# 3. 外参
# ---------------------------------------------------------------------------


def camera_world_position_from_agent(
    agent_x: float,
    agent_y: float,
    agent_z: float,
    orientation: Orientation,
) -> Tuple[float, float, float]:
    """
    由 Agent 世界坐标与朝向推算相机光心（UE cm）。

    orientation:
        - float: 水平 yaw（度）；
        - (qx, qy, qz, qw) 四元数，或带 x/y/z/w 属性的对象。
    """
    yaw_rad = _orientation_to_yaw_rad(orientation)
    cam_x = float(agent_x) + CAMERA_FORWARD_OFFSET_UU * math.cos(yaw_rad)
    cam_y = float(agent_y) + CAMERA_FORWARD_OFFSET_UU * math.sin(yaw_rad)
    cam_z = float(agent_z) + CAMERA_UP_OFFSET_UU
    return (cam_x, cam_y, cam_z)


def cam_arrays_from_agent(
    agent_x: float,
    agent_y: float,
    agent_z: float,
    orientation: Orientation,
) -> Tuple[Any, Any]:
    """agent 位姿 → (cam_world_pos, cam_world_rot) numpy 数组；rot 为 [0, 0, yaw_deg]。"""
    import numpy as np

    cx, cy, cz = camera_world_position_from_agent(agent_x, agent_y, agent_z, orientation)
    yaw_deg = orientation_to_yaw_deg(orientation)
    cam_pos = np.array([cx, cy, cz], dtype=np.float32)
    cam_rot = np.array([0.0, 0.0, yaw_deg], dtype=np.float32)
    return cam_pos, cam_rot


# ---------------------------------------------------------------------------
# 4. 内参
# ---------------------------------------------------------------------------


def default_intrinsic_matrix(
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Any:
    """从 config 读 FOV/分辨率，返回内参 K（numpy 3×3）。"""
    try:
        from .Location.coordinate_projection import intrinsic_matrix_from_wh_fov
    except ImportError:
        from Location.coordinate_projection import intrinsic_matrix_from_wh_fov

    w = int(width if width is not None else AGENT_CAMERA_CAPTURE_PARAMS["width"])
    h = int(height if height is not None else AGENT_CAMERA_CAPTURE_PARAMS["height"])
    fov = float(AGENT_CAMERA_CAPTURE_PARAMS["fov_degrees"])
    return intrinsic_matrix_from_wh_fov(w, h, fov)


# ---------------------------------------------------------------------------
# 5. 深度有效性
# ---------------------------------------------------------------------------


def valid_depth_mask(zs: Any) -> Any:
    """有效深度布尔 mask（阈值 VALID_DEPTH_MIN / VALID_DEPTH_MAX）。"""
    import numpy as np

    arr = np.asarray(zs)
    return (
        (arr > VALID_DEPTH_MIN)
        & (arr < VALID_DEPTH_MAX)
        & np.isfinite(arr)
    )
