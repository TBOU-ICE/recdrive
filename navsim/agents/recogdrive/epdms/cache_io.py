"""IO helpers for navsim_v2 (EPDMS-format) metric caches.

The v2 caches are pickled inside the *navsim_v2* package, so their class paths
(``navsim.planning.metric_caching.metric_cache.MetricCache`` etc.) would resolve
to this repo's **v1** classes on unpickle. ``EpdmsUnpickler`` redirects every
scoring-stack class to the v2 copies ported into this subpackage, which keeps
data (v2 payload) and methods (v2 code) consistent without touching any
existing v1 module.
"""

from __future__ import annotations

import csv
import lzma
import pickle
from pathlib import Path
from typing import Any, Dict, List

_PP = "navsim.planning.simulation.planner.pdm_planner"
_EPDMS = "navsim.agents.recogdrive.epdms"

# pickle module path -> ported module path
MODULE_REDIRECTS: Dict[str, str] = {
    "navsim.planning.metric_caching.metric_cache": f"{_EPDMS}.metric_cache",
    f"{_PP}.observation.pdm_observation": f"{_EPDMS}.pdm_observation",
    f"{_PP}.observation.pdm_occupancy_map": f"{_EPDMS}.pdm_occupancy_map",
    f"{_PP}.observation.pdm_object_manager": f"{_EPDMS}.pdm_object_manager",
    f"{_PP}.utils.pdm_path": f"{_EPDMS}.pdm_path",
    "navsim.common.enums": f"{_EPDMS}.enums",
}


class EpdmsUnpickler(pickle.Unpickler):
    """Unpickler that binds v2-cache objects to the ported v2 classes."""

    def find_class(self, module: str, name: str):  # noqa: D102
        module = MODULE_REDIRECTS.get(module, module)
        return super().find_class(module, name)


def load_metric_cache_v2(path: str | Path) -> Any:
    """Load one lzma-compressed v2 metric cache pickle."""
    with lzma.open(str(path), "rb") as f:
        return EpdmsUnpickler(f).load()


class MetricCacheIndexV2:
    """token -> pickle path index for a v2 metric cache directory.

    Prefers the metadata CSVs written by navsim's metric caching; falls back to
    a filesystem walk so a partially built cache is usable as well.
    """

    def __init__(self, cache_root: str | Path):
        self.cache_root = Path(cache_root)
        self.paths: Dict[str, Path] = {}

        metadata_dir = self.cache_root / "metadata"
        csv_files = sorted(metadata_dir.glob("*.csv")) if metadata_dir.is_dir() else []
        for csv_path in csv_files:
            try:
                with csv_path.open("r", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        file_name = row.get("file_name") or next(iter(row.values()))
                        if not file_name:
                            continue
                        p = Path(file_name)
                        # Trust metadata CSVs written by the caching job. Per-path
                        # is_file() on CPFS for ~100k entries takes many minutes and
                        # was stalling every DDP rank at startup.
                        # .../<log_name>/<scenario_type>/<token>/metric_cache.pkl
                        self.paths[p.parent.name] = p
            except Exception:
                continue

        if not self.paths:
            for p in self.cache_root.glob("*/*/*/metric_cache.pkl"):
                self.paths[p.parent.name] = p

    def __contains__(self, token: str) -> bool:
        return token in self.paths

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def tokens(self) -> List[str]:
        return list(self.paths.keys())

    def load(self, token: str) -> Any:
        return load_metric_cache_v2(self.paths[token])
