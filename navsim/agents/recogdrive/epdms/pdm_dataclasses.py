# Ported verbatim from navsim_v2 (common/dataclasses.py::PDMResults) so the v2
# scorer in this subpackage does not depend on this repo's v1 dataclasses.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class PDMResults:
    """Helper dataclass to record PDM results."""

    no_at_fault_collisions: float
    drivable_area_compliance: float
    driving_direction_compliance: float
    traffic_light_compliance: float

    ego_progress: float
    time_to_collision_within_bound: float
    lane_keeping: float
    history_comfort: float

    multiplicative_metrics_prod: float
    weighted_metrics: npt.NDArray[np.float64]
    weighted_metrics_array: npt.NDArray[np.float64]

    pdm_score: float

    @classmethod
    def get_empty_results(cls) -> "PDMResults":
        """Returns an instance of the class where all values are NaN."""
        return PDMResults(
            no_at_fault_collisions=np.nan,
            drivable_area_compliance=np.nan,
            driving_direction_compliance=np.nan,
            traffic_light_compliance=np.nan,
            ego_progress=np.nan,
            time_to_collision_within_bound=np.nan,
            lane_keeping=np.nan,
            history_comfort=np.nan,
            multiplicative_metrics_prod=np.nan,
            weighted_metrics=np.nan,
            weighted_metrics_array=np.nan,
            pdm_score=np.nan,
        )
