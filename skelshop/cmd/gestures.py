import time
from contextlib import contextmanager
from functools import reduce
from itertools import repeat
from typing import Any, Callable, Deque, Dict, Iterable, List, cast

import click
import h5py
import numpy as np
import opencv_wrapper as cvw
import pygame as pg

from skelshop.config import conf
from skelshop.drawsticks import ScaledVideo, SkelDraw, get_skel
from skelshop.face.pipe import mk_conf_thresh
from skelshop.io import ShotSegmentedReader
from skelshop.pipebase import IterStage
from skelshop.pipeline import pipeline_options
from skelshop.player import imdisplay
from skelshop.skelgraphs.openpose import (
    BODY_IN_BODY_25_ALL_REDUCER,
    FACE_IN_BODY_25_ALL_REDUCER,
    LHAND_IN_BODY_25_ALL_REDUCER,
    MODE_SKELS,
    RHAND_IN_BODY_25_ALL_REDUCER,
)

BUFFER_SIZE = 10

# ######### debug: show skels ##############

FORMAT = (360, 640)


@contextmanager
def debug_screen(size):
    screen = pg.display.set_mode((size[1], size[0]))
    yield screen


def show_frame(frame, screen, skel_h5f, img=None):
    img = img if img is not None else np.ones((*FORMAT, 3))
    posetrack = False  # TODO is posetrack False?!
    skel = get_skel(skel_h5f, False)
    skel_draw = SkelDraw(skel, posetrack, ann_ids=True, scale=1)
    skel_draw.draw_bundle(img, frame)
    # img = (img * 255).astype(np.uint8)
    imdisplay(img, screen)
    pg.display.flip()
    time.sleep(1 / 60)


# ########### end that ################


@contextmanager
def maybe_load(vid_in):
    if vid_in is not None:
        with cvw.load_video(vid_in) as vid:
            yield vid
    else:
        yield None


class Body:
    def __init__(self, skel_h5f, num):
        self.skel_h5f = skel_h5f
        self.num = num
        self.skel = MODE_SKELS[skel_h5f.attrs["mode"]]

    def update_numarr(self, numarr):
        self.numarr = numarr

    def get(self, name):
        return self.numarr[self.skel.names.index(name)]


FrameBuf = Deque[List[Body]]


@click.command()
@click.argument("h5infn", type=click.Path(exists=True))
@click.option("--vid-in", type=click.Path(exists=True))
@click.option("--start-frame", type=int, default=0)
@click.option("--end-frame", type=int, default=None)
# --skel-thresh-pool and --skel-thresh-val can be changed as env-vars and not via click
@pipeline_options(allow_empty=True)  # brauch nur shot_seg/segs_file
def gestures(h5infn, pipeline, vid_in, start_frame, end_frame):
    """
    Detect gestures on sorted etc data.
    """
    buffer = FrameBuf(maxlen=BUFFER_SIZE)

    if pipeline.metadata.get("shot_seg") == "none":
        raise click.BadOptionUsage("--shot-seg", "--shot-seg is required!")

    with h5py.File(h5infn, "r") as skel_h5f, debug_screen(FORMAT) as screen, maybe_load(
        vid_in
    ) as vid_read:
        if vid_read is not None:
            vid_iter = iter(ScaledVideo(vid_read, vid_in, 1))
        else:
            vid_iter = repeat(None, skel_h5f.attrs["num_frames"])
        read = ShotSegmentedReader(skel_h5f, infinite=True)
        # poses = [i for i in peekable(iter(peekable(iter(read)).peek()))]
        # for frame in read(): for skel_id, skel in shot: print(skel)
        stage = IterStage(read.iter_from_frame(0))
        frame_iter = pipeline(stage)
        # Each stage acts as an iterator, typically yielding some kind of pose bundle. A pose bundle is an iterator of skeletons, either with ids or not depending on whether it has been tracked.
        # TODO assert it's tracked at this point!
        conf_thresh = mk_conf_thresh(conf.DEFAULT_THRESH_POOL, conf.DEFAULT_THRESH_VALS)
        # skel_type = MODE_SKELS[skel_h5f.attrs["mode"]]; [i for i in skel_type.iter_limbs(posture)]
        bodies = []
        for i in range(2):  # TODO get that number from the skeletons
            bodies.append(Body(skel_h5f, i))
        for scene in frame_iter:
            for frame in scene:
                img = (next(vid_iter) * 0.3).astype(np.uint8)
                for num, skel in frame:
                    bodies[num].update_numarr(threshold_limbs(skel, conf_thresh))
                    frame.bundle[num] = bodies[num].numarr
                    if num > 0:  # TODO delete this
                        break
                # bewegt sich was, ist die Hand drauf, #Personen begrenzen (Mahnaz-Talk)
                # im video einblenden ob geste erkannt ist oder nicht (video dimmen?)
                # er wird mir code zukommen lassen (trian taphylos) -> "velocity from keypoints"

                if len(frame.bundle) > 2:
                    img = make_red(img)
                if not (
                    bodies[0].get("left wrist")[2] > 0
                    or bodies[0].get("right wrist")[2] > 0
                ):  # frame.bundle[num][bodies[0].get('right wrist')][2]
                    img = make_red(img)

                buffer.append(bodies)

                # breakpoint in skelshop.track.track.PoseTrack.pose_track -> the specs have a spec.prev_frame_buf_size, and that many get added to
                # posetrack.prev_tracked, which is: FrameBuf = (collections.deque(maxlen=spec.prev_frame_buf_size)), where FrameBuf = Deque[List[TrackedPose]]
                # trackedpose? gets filled in a complicated way.

                show_frame(frame, screen, skel_h5f=skel_h5f, img=img)


def make_red(img):
    return np.stack(
        (img[:, :, 0], img[:, :, 1], (img[:, :, 2] * 3).astype(np.uint8)), axis=-1
    )  # np.floor_divide(img[0], 2).astype(np.uint8)


def threshold_limbs(skel, conf_thresh):
    arr = skel.all()
    parts = {}
    for part, reducer in [
        ("body", BODY_IN_BODY_25_ALL_REDUCER),
        ("left hand", LHAND_IN_BODY_25_ALL_REDUCER),
        ("right hand", RHAND_IN_BODY_25_ALL_REDUCER),
        ("face", FACE_IN_BODY_25_ALL_REDUCER),
    ]:
        kps = reducer.reduce_arr(arr)
        # tmp = [i for i in reducer.dense_skel.iter_limbs(kps, gen_id=True)]
        tmp: Any = [
            i
            for i in cast(
                Iterable[Any], reducer.dense_skel.iter_limbs(kps, gen_id=True),
            )
            if conf_thresh([i[1][0][2], i[1][1][2]], part)
        ]
        add_unique: Callable[[List, Any], List] = lambda lst, elem: (
            lst if elem[0] in [i[0] for i in cast(Any, lst)] else lst + [elem]
        )
        tmp2: List[Any] = reduce(
            add_unique, tmp, [],
        )
        tmp3: Dict[Any, Any] = dict(tmp2)
        parts[part] = [
            tmp3.get(i, (np.array((0, 0, 0)), None))[0] for i in range(len(kps))
        ]
    # TODO this only holds if skel.__class__ is Body25All
    return skel.__class__.from_parts(
        parts["body"], parts["left hand"], parts["right hand"], parts["face"],
    ).keypoints
