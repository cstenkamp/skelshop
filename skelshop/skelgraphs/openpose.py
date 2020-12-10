import os

__all__ = [
    "MODE_SKELS",
    "BODY_ALL",
    "FACE_IN_BODY_25_ALL_REDUCER",
    "BODY_IN_BODY_25_ALL_REDUCER",
    "LHAND_IN_BODY_25_ALL_REDUCER",
    "RHAND_IN_BODY_25_ALL_REDUCER",
    "FACE_IN_BODY_25_ALL",
    "BODY_IN_BODY_25_ALL",
    "RHAND_IN_BODY_25_ALL",
    "LHAND_IN_BODY_25_ALL",
    "BODY_25",
    "BODY_25_JOINTS",
    "BODY_25_LINES",
    "UPPER_BODY_25_LINES",
    "FACE_LINES",
    "HAND_LINES",
    "HAND",
]

from .openpose_base import (
    BODY_25,
    BODY_25_JOINTS,
    BODY_25_LINES,
    FACE_LINES,
    HAND,
    HAND_LINES,
    UPPER_BODY_25_LINES,
)

if "LEGACY_SKELS" in os.environ:
    from .openpose_legacy import BODY_135 as BODY_ALL
    from .openpose_legacy import FACE_IN_BODY_25_ALL_REDUCER, MODE_SKELS

    # TODO also reducers etc
else:
    from .openpose_multi import BODY_25_ALL as BODY_ALL
    from .openpose_multi import (
        BODY_IN_BODY_25_ALL,
        BODY_IN_BODY_25_ALL_REDUCER,
        FACE_IN_BODY_25_ALL,
        FACE_IN_BODY_25_ALL_REDUCER,
        LHAND_IN_BODY_25_ALL,
        LHAND_IN_BODY_25_ALL_REDUCER,
        MODE_SKELS,
        RHAND_IN_BODY_25_ALL,
        RHAND_IN_BODY_25_ALL_REDUCER,
    )
