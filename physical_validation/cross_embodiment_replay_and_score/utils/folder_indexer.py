# -*- coding: utf-8 -*-
"""
Directory indexing utilities for traversing first-, second-, and third-level subdirectories.

Provided functions:
- find_level1_subdirs(base_dir): Return full paths of all first-level subdirectories (stable natural sort).
- level1_count(base_dir): Return the total number of first-level subdirectories.
- level1_path_by_index(base_dir, index): Return the first-level subdirectory path by 0-based natural-sort index.
- find_level3_subdirs(base_dir): Return full paths of all third-level subdirectories (stable natural sort).
- level3_count(base_dir): Return the total number of third-level subdirectories.
- level3_path_by_index(base_dir, index): Return the third-level subdirectory path by 0-based natural-sort index.
- find_level2_subdirs(base_dir): Return full paths of all second-level subdirectories (stable natural sort).
- level2_count(base_dir): Return the total number of second-level subdirectories.
- level2_path_by_index(base_dir, index): Return the second-level subdirectory path by 0-based natural-sort index.

Natural sort note: names containing numbers are ordered numerically first
(for example, session_2 before session_10); names without numbers use normal lexicographic order.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from typing import List, Dict, Literal, Optional, Union, Tuple

_num_re = re.compile(r"(\d+)")
_num_prefix3_re = re.compile(r"^\d{3}")

# Disk cache default switch (manual in-code control).
# Set this to True if you want FolderIndexer to persist index JSON by default.
# This module intentionally does NOT read env vars to toggle caching.
FOLDER_INDEX_DISK_CACHE_DEFAULT_ENABLED: bool = True

def _natural_key(s: str):
    parts = _num_re.split(os.path.basename(s))
    key = []
    for p in parts:
        if p.isdigit():
            try:
                key.append(int(p))
            except Exception:
                key.append(p)
        else:
            key.append(p.lower())
    return key

def _sorted_subdirs(parent: str) -> List[str]:
    try:
        names = [n for n in os.listdir(parent)]
    except Exception:
        return []
    res = []
    for name in names:
        p = os.path.join(parent, name)
        try:
            if os.path.isdir(p):
                res.append(p)
        except Exception:
            continue
    res.sort(key=_natural_key)
    return res

# session_ee is for end-effector trajectories in txt format; session_jnt is for joint trajectories in txt format.
Mode = Literal["session_ee", "session_jnt"]

class FolderIndexer:
    """Configurable folder indexer.

    Parameters:
      base_dir: root directory to search under.
      mode:
                - "session_ee": end-effector trajectory directory in txt format (default).
                    Matches session_ee*; also backward-compatible with session* (excluding session_jnt*).
                - "session_jnt": joint trajectory directory in txt format. Matches session_jnt*.
      level: 1/2/3 = how many nested directory levels to traverse before filtering.

    Notes:
      - Traversal uses `_sorted_subdirs()` which is stable and naturally sorted.
      - Filtering is applied only on the final level directory (d1/d2/d3).
    """

    def __init__(self, base_dir: str, mode: Mode = "session_ee", level: int = 3):
        self.base_dir = os.path.abspath(str(base_dir))
        self.mode: Mode = mode
        self.level = int(level)
        if self.level not in (1, 2, 3):
            raise ValueError(f"level must be 1/2/3, got {level!r}")
        if self.mode not in ("session_ee", "session_jnt"):
            raise ValueError(f"mode must be 'session_ee' or 'session_jnt', got {mode!r}")

        # In-memory cache: avoid repeated traversal on slow filesystems (e.g. s3fs/FUSE).
        # This cache is especially important because callers may call count()/path_by_index()
        # repeatedly inside multiprocessing workers.
        self._paths_cache: Optional[List[str]] = None

        # Optional in-memory cache for per-index trajectory file paths.
        # Format: List[Optional[(l_pose, r_pose, l_grip, r_grip)]].
        self._traj_paths_cache: Optional[List[Optional[Tuple[str, str, str, str]]]] = None

    def _cache_enabled(self) -> bool:
        """Whether to use on-disk cache for directory index.

        Controlled ONLY by in-code constant `FOLDER_INDEX_DISK_CACHE_DEFAULT_ENABLED`.
        """
        return bool(FOLDER_INDEX_DISK_CACHE_DEFAULT_ENABLED)

    def _cache_file_path(self) -> str:
        home = os.path.expanduser("~")
        if (not home) or (home == "~"):
            cache_dir = os.path.join(tempfile.gettempdir(), "rm75_TeleAI", "folder_indexer")
        else:
            cache_dir = os.path.join(home, ".cache", "rm75_TeleAI", "folder_indexer")
        key = f"{self.base_dir}|{self.mode}|{self.level}"
        h = hashlib.sha1(key.encode("utf-8"), usedforsecurity=False).hexdigest()  # type: ignore[call-arg]
        return os.path.join(cache_dir, f"index_{h}.json")

    def _try_load_paths_from_disk_cache(self) -> Optional[List[str]]:
        if not self._cache_enabled():
            return None
        p = self._cache_file_path()
        try:
            if not os.path.isfile(p):
                return None
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if not isinstance(obj, dict):
                return None
            if obj.get("base_dir") != self.base_dir:
                return None
            if obj.get("mode") != self.mode:
                return None
            if int(obj.get("level", -1)) != int(self.level):
                return None
            paths = obj.get("paths")
            if not isinstance(paths, list) or not all(isinstance(x, str) for x in paths):
                return None

            abs_paths = [os.path.abspath(x) for x in paths]

            # Optional per-index trajectory file paths cache.
            traj_obj = obj.get("traj_paths", None)
            if isinstance(traj_obj, list) and len(traj_obj) == len(abs_paths):
                traj_cache: List[Optional[Tuple[str, str, str, str]]] = []
                ok = True
                for item in traj_obj:
                    if item is None:
                        traj_cache.append(None)
                        continue
                    if not (isinstance(item, list) and len(item) == 4 and all(isinstance(s, str) for s in item)):
                        ok = False
                        break
                    traj_cache.append((
                        os.path.abspath(item[0]),
                        os.path.abspath(item[1]),
                        os.path.abspath(item[2]),
                        os.path.abspath(item[3]),
                    ))
                if ok:
                    self._traj_paths_cache = traj_cache

            return abs_paths
        except Exception:
            return None

    def _try_write_paths_to_disk_cache(self, paths: List[str]) -> None:
        if not self._cache_enabled():
            return
        p = self._cache_file_path()
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            tmp = p + ".tmp"

            traj_paths_json = None
            # When disk cache is enabled, also persist per-index trajectory file
            # paths to avoid repeated os.listdir() later.
            if int(self.level) in (1, 2, 3):
                traj_paths_json = self._build_all_traj_paths_for_sessions(paths)
                # Keep in-memory cache in sync.
                self._traj_paths_cache = [
                    (tuple(x) if isinstance(x, list) else None)  # type: ignore[misc]
                    for x in traj_paths_json
                ]
            obj = {
                "version": 2,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "base_dir": self.base_dir,
                "mode": self.mode,
                "level": int(self.level),
                "paths": paths,
                "traj_paths": traj_paths_json,
            }
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False)
            os.replace(tmp, p)
        except Exception:
            # Cache is best-effort; never fail indexing because of cache IO.
            return

    def _build_all_traj_paths_for_sessions(self, session_dirs: List[str]) -> List[Optional[List[str]]]:
        """Precompute all per-index trajectory file paths for session-based modes.

        This avoids calling os.listdir() inside scoring workers for each idx.
        Returns JSON-friendly list aligned with `session_dirs`.
        Each element is either None (if left/right not found) or [l_pose, r_pose, l_grip, r_grip].
        """
        res: List[Optional[List[str]]] = []
        for d in session_dirs:
            try:
                subs = _sorted_subdirs(d)
                left_dir = None
                right_dir = None
                for p in subs:
                    name = os.path.basename(p).lower()
                    if left_dir is None and name.startswith("left"):
                        left_dir = p
                    elif right_dir is None and name.startswith("right"):
                        right_dir = p
                    if left_dir is not None and right_dir is not None:
                        break
                if (not left_dir) or (not right_dir):
                    res.append(None)
                    continue

                if self.mode == "session_jnt":
                    l_pose = os.path.join(left_dir, "Merged_Trajectory", "merged_trajectory_jnt.txt")
                    r_pose = os.path.join(right_dir, "Merged_Trajectory", "merged_trajectory_jnt.txt")
                else:
                    # session_ee
                    l_pose = os.path.join(left_dir, "Merged_Trajectory", "merged_trajectory.txt")
                    r_pose = os.path.join(right_dir, "Merged_Trajectory", "merged_trajectory.txt")

                l_grip = os.path.join(left_dir, "Clamp_Data", "clamp_data_tum.txt")
                r_grip = os.path.join(right_dir, "Clamp_Data", "clamp_data_tum.txt")
                res.append([l_pose, r_pose, l_grip, r_grip])
            except Exception:
                res.append(None)
        return res

    def _match(self, path: str) -> bool:
        name = os.path.basename(path)
        lower = name.lower()
        if self.mode == "session_ee":
            # Prefer explicit prefix if present; otherwise keep backward-compatible
            # behavior by matching session* but excluding session_jnt*.
            return lower.startswith("session_ee") or (lower.startswith("session") and (not lower.startswith("session_jnt")))
        if self.mode == "session_jnt":
            return lower.startswith("session_jnt")
        return _num_prefix3_re.match(name) is not None

    def find_subdirs(self) -> List[str]:
        if self._paths_cache is not None:
            return list(self._paths_cache)

        cached = self._try_load_paths_from_disk_cache()
        if cached is not None:
            self._paths_cache = cached
            # Cache upgrade: older cache files may only contain `paths`.
            # If disk cache is enabled and per-index traj paths are missing,
            # backfill traj_paths and rewrite the cache once.
            if (
                self._cache_enabled()
                and int(self.level) in (1, 2, 3)
                and self._traj_paths_cache is None
            ):
                try:
                    self._traj_paths_cache = [
                        (tuple(x) if isinstance(x, list) else None)  # type: ignore[misc]
                        for x in self._build_all_traj_paths_for_sessions(cached)
                    ]
                    # Rewrite upgraded cache (best-effort).
                    self._try_write_paths_to_disk_cache(cached)
                except Exception:
                    pass
            return list(cached)

        if not os.path.isdir(self.base_dir):
            self._paths_cache = []
            return []

        if self.level == 1:
            subs = _sorted_subdirs(self.base_dir)
            res = [p for p in subs if self._match(p)]
            self._paths_cache = res
            # Persist directory list and per-index trajectory paths.
            self._try_write_paths_to_disk_cache(res)
            return list(res)

        if self.level == 2:
            level2: List[str] = []
            for d1 in _sorted_subdirs(self.base_dir):
                for d2 in _sorted_subdirs(d1):
                    if self._match(d2):
                        level2.append(d2)
            self._paths_cache = level2
            # Persist directory list and per-index trajectory paths.
            self._try_write_paths_to_disk_cache(level2)
            return list(level2)

        # level == 3
        level3: List[str] = []
        for d1 in _sorted_subdirs(self.base_dir):
            for d2 in _sorted_subdirs(d1):
                for d3 in _sorted_subdirs(d2):
                    if self._match(d3):
                        level3.append(d3)
        self._paths_cache = level3
        self._try_write_paths_to_disk_cache(level3)
        return list(level3)

    def count(self) -> int:
        return len(self.find_subdirs())

    def path_by_index(self, index: int) -> str:
        paths = self.find_subdirs()
        if index < 0 or index >= len(paths):
            raise IndexError(f"index {index} out of range (0..{len(paths)-1})")
        return paths[index]

    def left_right_subdirs(
        self,
        index: int,
        left_prefix: str = "left",
        right_prefix: str = "right",
    ) -> Dict[str, List[str]]:
        """Return children of selected directory grouped by left/right prefixes."""
        target_dir = self.path_by_index(index)
        subs = _sorted_subdirs(target_dir)
        lefts: List[str] = []
        rights: List[str] = []
        lp = left_prefix.lower()
        rp = right_prefix.lower()
        for p in subs:
            name = os.path.basename(p).lower()
            if name.startswith(lp):
                lefts.append(p)
            elif name.startswith(rp):
                rights.append(p)
        return {"left": lefts, "right": rights}
    
    def build_traj_paths(self, index: int) -> Union[Tuple[str, str, str, str], str]:
        """Build trajectory file path(s) from index.

        - mode == "session_ee" or "session_jnt":
            returns 4-tuple (left_pose_path, right_pose_path, left_gripper_path, right_gripper_path)
        """
        paths = self.find_subdirs()
        totalcount = len(paths)
        if index < 0 or index >= totalcount:
            raise IndexError(f"index {index} out of range (0..{totalcount-1})")

        # Fast path: if we have cached full traj paths for this index, use it.
        if self._traj_paths_cache is not None and index < len(self._traj_paths_cache):
            cached_entry = self._traj_paths_cache[index]
            if cached_entry is not None:
                return cached_entry

        # Ensure in-memory traj cache list exists, so we only compute once per idx.
        if self._traj_paths_cache is None:
            self._traj_paths_cache = [None] * totalcount

        lr_paths = self.left_right_subdirs(index)
        left_path: Optional[str] = None
        right_path: Optional[str] = None
        if self.mode=="session_jnt":
            left_path = os.path.join(lr_paths["left"][0],f"Merged_Trajectory/merged_trajectory_jnt.txt")
            right_path = os.path.join(lr_paths["right"][0],f"Merged_Trajectory/merged_trajectory_jnt.txt")
        elif self.mode=="session_ee":
            left_path = os.path.join(lr_paths["left"][0],f"Merged_Trajectory/merged_trajectory.txt")
            right_path = os.path.join(lr_paths["right"][0],f"Merged_Trajectory/merged_trajectory.txt")
        else:
            raise ValueError(f"Unsupported mode {self.mode!r}")
        left_gripper = os.path.join(lr_paths["left"][0],f"Clamp_Data/clamp_data_tum.txt")
        right_gripper = os.path.join(lr_paths["right"][0],f"Clamp_Data/clamp_data_tum.txt")
        assert left_path is not None and right_path is not None

        out = (left_path, right_path, left_gripper, right_gripper)
        # Save into in-memory cache for subsequent calls.
        try:
            self._traj_paths_cache[index] = out
        except Exception:
            pass
        return out

    def preload(self) -> List[str]:
        """Force building (and possibly persisting) the index cache."""
        return self.find_subdirs()

