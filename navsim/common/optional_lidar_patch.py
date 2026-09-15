"""Runtime patch for NAVSIM datasets whose metadata has ``lidar_path=None``.

SimScale camera-only caches can contain frames without lidar blobs. The stock
NAVSIM dataclasses call ``Path(lidar_path)`` before checking whether lidar is
actually requested, so camera-only ReCogDrive caching can fail on ``Path(None)``.
This module keeps the original files untouched and patches that behavior only
for entrypoints that explicitly import it.
"""

from pathlib import Path as _Path
from typing import Optional

import navsim.common.dataclasses as _dataclasses

_ORIGINAL_LIDAR_FROM_PATHS = _dataclasses.Lidar.from_paths.__func__
_ORIGINAL_PATH = _dataclasses.Path
_APPLIED = False


def _optional_path(path: Optional[object]):
    return None if path is None else _Path(path)


def _lidar_from_paths(cls, sensor_blobs_path, lidar_path, sensor_names):
    if "lidar_pc" in sensor_names and (sensor_blobs_path is None or lidar_path is None):
        return cls()
    return _ORIGINAL_LIDAR_FROM_PATHS(cls, sensor_blobs_path, lidar_path, sensor_names)


def apply_optional_lidar_patch() -> None:
    """Allow NAVSIM scene loading to tolerate missing lidar paths."""
    global _APPLIED
    if _APPLIED:
        return

    _dataclasses.Path = _optional_path
    _dataclasses.Lidar.from_paths = classmethod(_lidar_from_paths)
    _APPLIED = True


def remove_optional_lidar_patch() -> None:
    """Restore stock NAVSIM dataclass behavior."""
    global _APPLIED
    if not _APPLIED:
        return

    _dataclasses.Path = _ORIGINAL_PATH
    _dataclasses.Lidar.from_paths = classmethod(_ORIGINAL_LIDAR_FROM_PATHS)
    _APPLIED = False
