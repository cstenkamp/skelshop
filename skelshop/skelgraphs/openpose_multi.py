from .base import SkeletonType
from .openpose_base import (
    BODY_25,
    BODY_25_JOINTS,
    BODY_25_LINES,
    FACE_LINES,
    HAND_LINES,
)
from .reducer import SkeletonReducer
from .utils import incr


def compose_body(body=None, left_hand=None, right_hand=None, face=None):
    lines = {}
    if body is not None:
        lines["body"] = body
    if left_hand is not None:
        lines["left hand"] = incr(25, left_hand)
    if right_hand is not None:
        lines["right hand"] = incr(46, right_hand)
    if face is not None:
        lines["face"] = incr(67, face)
    return lines


BODY_25_HANDS_LINES = compose_body(BODY_25_LINES, HAND_LINES, HAND_LINES)
BODY_25_ALL_LINES = compose_body(BODY_25_LINES, HAND_LINES, HAND_LINES, FACE_LINES)
FACE_IN_BODY_25_ALL_LINES = compose_body(face=FACE_LINES)
BODY_IN_BODY_25_ALL_LINES = compose_body(body=BODY_25_LINES)
LHAND_IN_BODY_25_ALL_LINES = compose_body(left_hand=HAND_LINES)
RHAND_IN_BODY_25_ALL_LINES = compose_body(right_hand=HAND_LINES)

BODY_25_HANDS = SkeletonType(BODY_25_HANDS_LINES, BODY_25_JOINTS)
BODY_25_ALL = SkeletonType(BODY_25_ALL_LINES, BODY_25_JOINTS, composed=True)
FACE_IN_BODY_25_ALL = SkeletonType(
    FACE_IN_BODY_25_ALL_LINES, BODY_25_JOINTS, composed=True
)
BODY_IN_BODY_25_ALL = SkeletonType(
    BODY_IN_BODY_25_ALL_LINES, BODY_25_JOINTS, composed=True
)
LHAND_IN_BODY_25_ALL = SkeletonType(
    LHAND_IN_BODY_25_ALL_LINES, BODY_25_JOINTS, composed=True
)
RHAND_IN_BODY_25_ALL = SkeletonType(
    RHAND_IN_BODY_25_ALL_LINES, BODY_25_JOINTS, composed=True
)

FACE_IN_BODY_25_ALL_REDUCER = SkeletonReducer(FACE_IN_BODY_25_ALL)
BODY_IN_BODY_25_ALL_REDUCER = SkeletonReducer(BODY_IN_BODY_25_ALL)
LHAND_IN_BODY_25_ALL_REDUCER = SkeletonReducer(LHAND_IN_BODY_25_ALL)
RHAND_IN_BODY_25_ALL_REDUCER = SkeletonReducer(RHAND_IN_BODY_25_ALL)

MODE_SKELS = {
    "BODY_25": BODY_25,
    "BODY_25_HANDS": BODY_25_HANDS,
    "BODY_25_ALL": BODY_25_ALL,
}
