"""Self-contained port of navsim_v2's EPDMS scoring stack (see per-file headers).

Used by the EPDMS RL expert training; deliberately isolated from the repo's v1
pdm_planner modules so the two metric versions can coexist in one process.
"""

from navsim.agents.recogdrive.epdms.cache_io import (  # noqa: F401
    EpdmsUnpickler,
    MetricCacheIndexV2,
    load_metric_cache_v2,
)
from navsim.agents.recogdrive.epdms.reward import (  # noqa: F401
    EPDMS_WEIGHTS,
    MULTIPLICATIVE_KEYS,
    TokenV2Scores,
    build_v2_simulator_and_scorer,
    epdms_score,
    score_token_proposals_v2,
    transform_trajectory,
    two_frame_extended_comfort,
)
