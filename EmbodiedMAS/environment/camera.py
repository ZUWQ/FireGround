"""TongSim first-person camera capture and image persistence."""
from __future__ import annotations

import binascii
import json
import time
import struct
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import tongsim as ts
from tongsim.core.world_context import WorldContext

_OBSERVATION_FILE = Path(__file__).resolve()


def _infer_log_dir_from_calling_agent() -> Path:
    """Compatibility default; callers may explicitly inject their output directory."""
    return Path(__file__).resolve().parents[1] / "outputs" / "captures"


class ObservationCamera:
    """获取第一视角相机 RGB 和深度图像的类，参考 capture_demo.py"""
    _GLOBAL_AGENT_CAMERAS: dict[str, bytes | str] = {}

    def __init__(self, context: WorldContext, log_dir: Optional[Path] = None):
        self._context = context
        self._conn = context.conn
        self._log_dir = Path(log_dir if log_dir is not None else _infer_log_dir_from_calling_agent()).expanduser().resolve()
        self._log_dir.mkdir(parents=True, exist_ok=True)
        # 缓存 agent_id -> camera_id 的映射（实例级缓存）
        self._agent_cameras: dict[str, bytes | str] = {}
        # 缓存 agent_id -> 保存目录的映射
        self._agent_capture_dirs: dict[str, Path] = {}

    @staticmethod
    def _normalize_agent_id(agent_id: bytes | str | None) -> str:
        """将 agent_id 标准化为字符串格式"""
        if agent_id is None:
            return ""
        if isinstance(agent_id, (bytes, bytearray)):
            return binascii.hexlify(agent_id).decode("ascii")
        return str(agent_id)

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """清理名称，只保留字母数字和特定字符"""
        if not name:
            return "agent"
        return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name)

    def register_agent_camera(self, agent_id: bytes | str | None, camera_id: bytes | str | None) -> None:
        """注册或更新与 agent 关联的 camera（实例缓存 + 全局缓存）"""
        if camera_id is None:
            return
        agent_key = self._normalize_agent_id(agent_id)
        if not agent_key:
            return
        self._agent_cameras[agent_key] = camera_id
        ObservationCamera._GLOBAL_AGENT_CAMERAS[agent_key] = camera_id

    def unregister_agent_camera(self, agent_id: bytes | str | None) -> None:
        """移除 agent 的 camera 缓存（实例缓存 + 全局缓存）"""
        agent_key = self._normalize_agent_id(agent_id)
        if not agent_key:
            return
        self._agent_cameras.pop(agent_key, None)
        ObservationCamera._GLOBAL_AGENT_CAMERAS.pop(agent_key, None)
        self._agent_capture_dirs.pop(agent_key, None)

    def _ensure_capture_dir(self, agent_cache_key: str, agent_label: str) -> Path:
        """确保保存目录存在，如果不存在则创建"""
        capture_dir = self._agent_capture_dirs.get(agent_cache_key)
        if capture_dir is None:
            timestamp = datetime.now().strftime("%m%d_%H%M%S")
            folder_name = f"{agent_label}_{timestamp}"
            capture_dir = self._log_dir / folder_name
            capture_dir.mkdir(parents=True, exist_ok=True)
            self._agent_capture_dirs[agent_cache_key] = capture_dir
        return capture_dir

    def _camera_tag(self, camera_id: Any) -> str:
        """将 camera_id 转换为字符串标签"""
        if isinstance(camera_id, (bytes, bytearray)):
            return camera_id.hex()
        if isinstance(camera_id, str):
            return camera_id
        return str(camera_id)

    def _save_color_image(self, frame: dict[str, Any], path: Path, fmt: str = "png") -> Path | None:
        """保存颜色图像，参考 capture_demo.py"""
        rgba = frame.get("rgba8")
        if not rgba:
            return None
        width = int(frame.get("width", 0))
        height = int(frame.get("height", 0))
        if width <= 0 or height <= 0:
            return None

        # 优先使用 PIL
        try:
            from PIL import Image  # type: ignore
            img = Image.frombytes("RGBA", (width, height), bytes(rgba), "raw", "BGRA")
            ext = fmt.lower()
            if ext in ("jpg", "jpeg"):
                img = img.convert("RGB")
                out = path.with_suffix(".jpg")
                img.save(out, quality=95)
                return out
            else:
                out = path.with_suffix(".png")
                img.save(out)
                return out
        except Exception:
            pass

        # 回退：手动 PNG 编码
        def _crc(chunk_type: bytes, data: bytes) -> bytes:
            c = binascii.crc32(chunk_type)
            c = binascii.crc32(data, c)
            return struct.pack(">I", (c & 0xFFFFFFFF))

        def _chunk(chunk_type: bytes, data: bytes) -> bytes:
            return struct.pack(">I", len(data)) + chunk_type + data + _crc(chunk_type, data)

        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
        raw = bytearray()
        mv = memoryview(rgba)
        row_bytes = width * 4
        for y in range(height):
            raw.append(0)
            start = y * row_bytes
            row = mv[start : start + row_bytes]
            # BGRA -> RGBA
            for x in range(width):
                b = row[x * 4 + 0]
                g = row[x * 4 + 1]
                r = row[x * 4 + 2]
                a = row[x * 4 + 3] if len(row) >= (x * 4 + 4) else 255
                raw += bytes((r, g, b, a))
        idat = zlib.compress(bytes(raw))
        png = sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")
        out = path.with_suffix(".png")
        out.write_bytes(png)
        return out

    def _save_depth_image(self, frame: dict[str, Any], path: Path) -> Path | None:
        """保存深度图像为 EXR 格式，参考 capture_demo.py"""
        depth_data = frame.get("depth") or frame.get("depth_r32")
        if not depth_data:
            return None
        
        out = path.with_suffix(".exr")
        out.write_bytes(depth_data)
        return out

    async def _find_best_camera(self, agent_id: bytes | str | None, agent_name: str | None, prefer_depth: bool = False) -> Optional[bytes | str]:
        """查找最佳相机，支持缓存和智能匹配"""
        agent_key = self._normalize_agent_id(agent_id)
        
        # 检查缓存
        if agent_key in self._agent_cameras:
            return self._agent_cameras[agent_key]
        
        # 获取所有相机并匹配
        all_cameras = await ts.CaptureAPI.list_cameras(self._conn)
        candidates = []
        
        for cam_desc in all_cameras:
            cam_info = cam_desc.get("camera", {})
            cam_name = cam_info.get("name", "")
            cam_id = cam_info.get("id")
            if not cam_id:
                continue
            
            # 匹配相机名称
            if (str(agent_id) in cam_name 
                or (agent_name and agent_name in cam_name)
                or (agent_name and f"AgentCamera_{agent_name}" in cam_name)
                or f"AgentCamera_{agent_id}" in cam_name):
                candidates.append((cam_name, cam_id))

        if candidates:
            if prefer_depth:
                for name, cid in candidates:
                    if "depth" in name.lower():
                        return cid
            return candidates[0][1]

        # Fallback: 返回最后一个相机（通常是最新生成的）
        if all_cameras:
            last_cam = all_cameras[-1].get("camera", {})
            fallback_id = last_cam.get("id")
            if fallback_id:
                self._agent_cameras[agent_key] = fallback_id
                return fallback_id

        return None

    async def resolve_camera_id(
        self,
        agent_id: bytes | str | dict,
        *,
        prefer_depth: bool = True,
    ) -> Optional[bytes | str]:
        """
        解析与 :meth:`capture_rgb_and_depth` 一致的 agent 关联 Capture 相机 ID，
        复用 ``_find_best_camera`` 的缓存与名称匹配规则。
        """
        if isinstance(agent_id, dict):
            agent_id_str = agent_id.get("id", agent_id)
            agent_name = agent_id.get("name")
            dict_cam_id = agent_id.get("camera_id")
            if dict_cam_id:
                agent_key = self._normalize_agent_id(agent_id_str)
                self._agent_cameras[agent_key] = dict_cam_id
        else:
            agent_id_str = agent_id
            agent_name = None

        return await self._find_best_camera(agent_id_str, agent_name, prefer_depth=prefer_depth)

    async def capture_rgb_image(
        self, 
        agent_id: bytes | str | dict, 
        image_prefix: Optional[str] = None, 
        image_format: str = "png"
    ) -> Optional[str]:
        """获取并保存 RGB 图像"""
        return await self._capture_impl(
            agent_id, 
            image_prefix, 
            image_format, 
            include_color=True, 
            include_depth=False, 
            return_type="color"
        )

    async def capture_depth_image(
        self, 
        agent_id: bytes | str | dict, 
        image_prefix: Optional[str] = None
    ) -> Optional[str]:
        """获取并保存深度图像，返回 .exr 文件路径"""
        return await self._capture_impl(
            agent_id, 
            image_prefix, 
            "exr", 
            include_color=True, 
            include_depth=True, 
            return_type="depth"
        )

    @staticmethod
    def _capture_metadata(snapshot: dict[str, Any], captured_at: float) -> dict[str, Any]:
        """Keep calibration and camera pose from the exact RGB-D snapshot."""
        metadata: dict[str, Any] = {"captured_at_monotonic": captured_at}
        for key in ("frame_id", "game_time", "depth_mode", "width", "height"):
            if key in snapshot:
                metadata[key] = snapshot[key]
        for source, target in (("depth_near", "near"), ("depth_far", "far")):
            if source in snapshot:
                metadata[target] = float(snapshot[source])
        pose = snapshot.get("world_pose")
        if pose is not None:
            metadata["world_position"] = [float(getattr(pose.location, axis)) for axis in ("x", "y", "z")]
            euler = ts.math.quaternion_to_euler(pose.rotation, is_degree=True)
            # SDK conversion round-trips the UE Rotator angles (roll, pitch, yaw).
            metadata["world_rotation"] = [float(getattr(euler, axis)) for axis in ("x", "y", "z")]
        intrinsics = snapshot.get("intrinsics")
        if intrinsics is not None:
            metadata["intrinsics"] = {key: float(intrinsics[key]) for key in ("fx", "fy", "cx", "cy")}
        return metadata

    async def capture_rgb_and_depth(
        self,
        agent_id: bytes | str | dict,
        image_prefix: Optional[str] = None,
        image_format: str = "png",
    ) -> dict[str, Any]:
        """
        单次快照同时保存 RGB 与深度图，避免分别调用 capture_rgb_image / capture_depth_image
        导致的两次 capture_snapshot。
        """
        if isinstance(agent_id, dict):
            agent_id_str = agent_id.get("id", agent_id)
            agent_name = agent_id.get("name")
            dict_cam_id = agent_id.get("camera_id")
            if dict_cam_id:
                agent_key = self._normalize_agent_id(agent_id_str)
                self._agent_cameras[agent_key] = dict_cam_id
        else:
            agent_id_str = agent_id
            agent_name = None

        agent_key = self._normalize_agent_id(agent_id_str)
        agent_label = self._sanitize_name(agent_name or agent_key or str(agent_id_str))

        camera_id = await self._find_best_camera(agent_id_str, agent_name, prefer_depth=True)
        if camera_id is None:
            return {"rgb": None, "depth": None}

        snapshot = await ts.CaptureAPI.capture_snapshot(
            self._conn,
            camera_id,
            include_color=True,
            include_depth=True,
            timeout_seconds=3.0,
        )
        if not snapshot:
            return {"rgb": None, "depth": None}
        metadata = self._capture_metadata(snapshot, time.monotonic())

        capture_dir = self._ensure_capture_dir(agent_key or agent_label, agent_label)
        timestamp = datetime.now().strftime("%m%d_%H%M%S_%f")[:-2]
        prefix = image_prefix or f"{agent_label}"
        base_name = f"{prefix}_{timestamp}"

        rgb_path: Optional[str] = None
        depth_path: Optional[str] = None
        if snapshot.get("rgba8"):
            c_path = capture_dir / base_name
            saved_color = self._save_color_image(snapshot, c_path, fmt=image_format)
            if saved_color:
                rgb_path = str(saved_color)
        d_path = capture_dir / base_name
        saved_depth = self._save_depth_image(snapshot, d_path)
        if saved_depth:
            depth_path = str(saved_depth)

        metadata_path = (capture_dir / base_name).with_suffix(".camera.json")
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
        return {"rgb": rgb_path, "depth": depth_path, "camera_metadata": metadata}

    async def _capture_impl(
        self, 
        agent_id: bytes | str | dict, 
        image_prefix: str | None, 
        image_format: str, 
        include_color: bool, 
        include_depth: bool,
        return_type: str
    ) -> Optional[str]:
        """统一的内部捕获实现"""
        # 解析 agent_id
        if isinstance(agent_id, dict):
            agent_id_str = agent_id.get("id", agent_id)
            agent_name = agent_id.get("name")
            dict_cam_id = agent_id.get("camera_id")
            if dict_cam_id:
                agent_key = self._normalize_agent_id(agent_id_str)
                self._agent_cameras[agent_key] = dict_cam_id
        else:
            agent_id_str = agent_id
            agent_name = None

        agent_key = self._normalize_agent_id(agent_id_str)
        agent_label = self._sanitize_name(agent_name or agent_key or str(agent_id_str))

        # 查找相机
        camera_id = await self._find_best_camera(agent_id_str, agent_name, prefer_depth=(return_type == "depth"))
        if camera_id is None:
            return None

        # 捕获快照
        snapshot = await ts.CaptureAPI.capture_snapshot(
            self._conn,
            camera_id,
            include_color=include_color,
            include_depth=include_depth,
            timeout_seconds=3.0,
        )

        if not snapshot:
            return None

        # 保存文件
        capture_dir = self._ensure_capture_dir(agent_key or agent_label, agent_label)
        timestamp = datetime.now().strftime("%m%d_%H%M%S_%f")[:-2]
        prefix = image_prefix or f"{agent_label}"
        base_name = f"{prefix}_{timestamp}"

        saved_path = None
        if include_color and snapshot.get('rgba8'):
            c_path = capture_dir / base_name
            saved_path_color = self._save_color_image(snapshot, c_path, fmt=image_format)
            if return_type == "color":
                saved_path = saved_path_color

        if include_depth:
            d_path = capture_dir / base_name
            saved_path_depth = self._save_depth_image(snapshot, d_path)
            if return_type == "depth":
                saved_path = saved_path_depth

        return str(saved_path) if saved_path else None

    async def get_latest_image_path(self, agent_id: bytes | str | dict) -> Optional[str]:
        """获取最近保存的图像路径"""
        # 解析 agent_id
        if isinstance(agent_id, dict):
            agent_id_str = agent_id.get("id", agent_id)
            agent_name = agent_id.get("name")
        else:
            agent_id_str = agent_id
            agent_name = None

        agent_key = self._normalize_agent_id(agent_id_str)
        agent_label = self._sanitize_name(agent_name or agent_key or str(agent_id_str))

        # 获取保存目录
        capture_dir = self._agent_capture_dirs.get(agent_key or agent_label)
        if capture_dir is None or not capture_dir.exists():
            return None

        # 查找所有图像文件
        image_files = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.exr"]:
            image_files.extend(capture_dir.glob(ext))

        if not image_files:
            return None

        return str(max(image_files, key=lambda p: p.stat().st_mtime))


