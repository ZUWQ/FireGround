from __future__ import annotations

import math
from typing import Optional

import tongsim as ts


# 导航目标世界高度（与 VL/VLM base agent 的 Z_HEIGHT 对齐）
NAV_TARGET_Z = 1000.0

# navigate_to_location 全局默认（conn / actor_id / target_location 由每次调用传入）
NAV_ACCEPT_RADIUS_UU = 50.0
NAV_ALLOW_PARTIAL = True
NAV_SPEED_UU_PER_SEC = 100.0
NAV_TIMEOUT_SEC = 60.0

# prompt navigate.relative 距离单位：米 → 仿真 UU（UE 惯例 1m = 100 UU）
METERS_TO_UU = 100.0

DIRECTION_TO_ANGLE: dict[str, float] = {
    "forward": 0.0,
    "left": 90.0,
    "right": -90.0,
    "backward": 180.0,
    "front_left": 45.0,
    "front_right": -45.0,
}

# 灭火器水平前向长度（UU，与 Human_Agent/action_h / 引擎参考一致）
EXTINGUISHER_FORWARD_LENGTH_UU = 57.0
# 俯仰计算时竖直参考：喷嘴/参考点相对 actor 根的高度偏移
EXTINGUISHER_PITCH_REF_Z_OFFSET_UU = 26.0


def _actor_key(actor: bytes | str | dict) -> bytes | str:
    """已部署 agent 的 RPC 用 id；dict 时取 ``id`` 字段。"""
    if isinstance(actor, dict):
        return actor["id"]
    return actor


def _get_yaw_from_quaternion(q: ts.Quaternion) -> float:
    return _quaternion_to_yaw(q.x, q.y, q.z, q.w)


def world_position_from_yaw_distance_2d(
    x: float,
    y: float,
    z: float,
    yaw_deg: float,
    distance: float,
) -> Optional[tuple[float, float, float]]:
    """
    世界 XY 平面极坐标偏移（与 Try_Agent/try_sos_agent.calculate_sos_location 一致）。
    yaw_deg 为世界系平面绝对方位角（度）；distance <= 0 时返回 None。
    """
    if distance <= 0:
        return None
    psi = math.radians(float(yaw_deg))
    d = float(distance)
    return (
        float(x) + d * math.cos(psi),
        float(y) + d * math.sin(psi),
        float(z),
    )


def _quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """
    从四元数 (x, y, z, w) 提取 yaw 角（弧度）。
    """
    yaw = math.atan2(
        2 * (w * z + x * y),
        1 - 2 * (y * y + z * z),
    )
    return yaw


def _compute_relative_xy(
    agent_xy: tuple[float, float],
    quat_xyzw: tuple[float, float, float, float],
    target_xy: tuple[float, float],
) -> dict:
    """
    计算 target 相对 agent 在 XY 平面上的相对位置与偏移角（度）。
    quat_xyzw = (x, y, z, w)。
    """
    x1, y1 = agent_xy
    x2, y2 = target_xy
    dx = x2 - x1
    dy = y2 - y1
    yaw = _quaternion_to_yaw(*quat_xyzw)
    cos_yaw = math.cos(-yaw)
    sin_yaw = math.sin(-yaw)
    x_rel = dx * cos_yaw - dy * sin_yaw
    y_rel = dx * sin_yaw + dy * cos_yaw
    angle_rad = math.atan2(y_rel, x_rel)
    angle_deg = math.degrees(angle_rad)
    return {
        "relative_position": (x_rel, y_rel),
        "angle_deg": angle_deg,
    }


def _extinguisher_ab_from_target(tf: ts.Transform, target: ts.Vector3) -> tuple[float, float]:
    """
    由 agent 位姿与目标世界坐标求 ``set_extinguisher_rotation(a, b)`` 的角度（度）。
    与 ``Human_Agent/action_h._extinguisher_ab_from_target`` 几何一致。
    """
    quat_xyzw = (tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w)
    yaw_ref = _quaternion_to_yaw(*quat_xyzw)
    ax = float(tf.location.x)
    ay = float(tf.location.y)
    az = float(tf.location.z)

    L = EXTINGUISHER_FORWARD_LENGTH_UU
    z_off = EXTINGUISHER_PITCH_REF_Z_OFFSET_UU

    n0x = ax + L * math.cos(yaw_ref)
    n0y = ay + L * math.sin(yaw_ref)
    n0z = az + z_off

    tx, ty, tz = float(target.x), float(target.y), float(target.z)
    rel = _compute_relative_xy((ax, ay), quat_xyzw, (tx, ty))
    x_rel, y_rel = rel["relative_position"]
    phi_rad = math.atan2(y_rel, x_rel)
    a = math.degrees(phi_rad)

    d0x = n0x - ax
    d0y = n0y - ay
    cos_g = math.cos(phi_rad)
    sin_g = math.sin(phi_rad)
    d1x = d0x * cos_g - d0y * sin_g
    d1y = d0x * sin_g + d0y * cos_g
    n1x = ax + d1x
    n1y = ay + d1y
    n1z = n0z

    wx = tx - n1x
    wy = ty - n1y
    wz = tz - n1z
    r_xy = math.hypot(wx, wy)
    if r_xy > 1e-8:
        pitch_rad = math.atan2(wz, r_xy)
    else:
        pitch_rad = 0.0
    b = math.degrees(pitch_rad)
    b = max(-90.0, min(90.0, b))
    return (a, b)


def direction_to_angle(direction: str) -> float:
    """离散方向 → 相对朝向角（度，左正右负）。"""
    key = str(direction).strip().lower()
    try:
        return DIRECTION_TO_ANGLE[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(DIRECTION_TO_ANGLE))
        raise ValueError(f"Unknown direction {direction!r}. Expected one of: {allowed}.") from exc


def validate_relative_distance(value: object) -> float:
    """Normalize a positive finite displacement whose simulator value is finite."""
    try:
        distance = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("navigate/relative requires a positive finite distance in meters") from exc
    if (isinstance(value, bool) or not math.isfinite(distance) or distance <= 0
            or not math.isfinite(distance * METERS_TO_UU)):
        raise ValueError("navigate/relative requires a positive finite distance in meters")
    return distance


def meters_to_uu(distance_m: float) -> float:
    """米 → UU。"""
    return float(distance_m) * METERS_TO_UU


