import time
from contextlib import contextmanager
from functools import reduce
from typing import Any, Callable, Dict, Iterable, List, cast

import click
import h5py
import numpy as np
import pygame as pg

from skelshop.config import conf
from skelshop.drawsticks import SkelDraw, get_skel
from skelshop.face.pipe import mk_conf_thresh
from skelshop.io import ShotSegmentedReader
from skelshop.pipebase import IterStage
from skelshop.pipeline import pipeline_options
from skelshop.player import imdisplay
from skelshop.skelgraphs.openpose import (
    BODY_IN_BODY_25_ALL_REDUCER,
    FACE_IN_BODY_25_ALL_REDUCER,
    LHAND_IN_BODY_25_ALL_REDUCER,
    RHAND_IN_BODY_25_ALL_REDUCER,
)

# ######### debug: show skels ##############

FORMAT = (360, 640)


@contextmanager
def debug_screen(size):
    screen = pg.display.set_mode((size[1], size[0]))
    yield screen


def show_frame(frame, screen, skel_h5f=None):
    img = np.ones((*FORMAT, 3))
    if skel_h5f:
        posetrack = False  # TODO is posetrack False?!
        skel = get_skel(skel_h5f, False)
        skel_draw = SkelDraw(skel, posetrack, ann_ids=True, scale=1)
    else:
        posetrack = False  # TODO is posetrack False?!
        skel_draw = SkelDraw(frame.cls, posetrack, ann_ids=True, scale=1)
    skel_draw.draw_bundle(img, frame)
    img = (img * 255).astype(np.uint8)
    imdisplay(img, screen)
    pg.display.flip()
    time.sleep(1 / 60)


# ########### end that ################


@click.command()
@click.argument("h5infn", type=click.Path(exists=True))
@click.option("--start-frame", type=int, default=0)
@click.option("--end-frame", type=int, default=None)
# --skel-thresh-pool and --skel-thresh-val can be changed as env-vars and not via click
@pipeline_options(allow_empty=True)  # brauch nur shot_seg/segs_file
def gestures(h5infn, pipeline, start_frame, end_frame):
    """
    Detect gestures on sorted etc data.
    """
    if pipeline.metadata.get("shot_seg") == "none":
        raise click.BadOptionUsage("--shot-seg", "--shot-seg is required!")

    with h5py.File(h5infn, "r") as skel_h5f, debug_screen(FORMAT) as screen:
        read = ShotSegmentedReader(skel_h5f, infinite=True)
        # poses = [i for i in peekable(iter(peekable(iter(read)).peek()))]
        # for frame in read(): for skel_id, skel in shot: print(skel)
        stage = IterStage(read.iter_from_frame(0))
        frame_iter = pipeline(stage)
        # Each stage acts as an iterator, typically yielding some kind of pose bundle. A pose bundle is an iterator of skeletons, either with ids or not depending on whether it has been tracked.
        # TODO assert it's tracked at this point!
        # TODO wie ist das mit dem threshold
        conf_thresh = mk_conf_thresh(conf.DEFAULT_THRESH_POOL, conf.DEFAULT_THRESH_VALS)
        # skel_type = MODE_SKELS[skel_h5f.attrs["mode"]]; [i for i in skel_type.iter_limbs(posture)]
        for scene in frame_iter:
            for frame in scene:
                for num, skel in frame:
                    parts = {}
                    for part, reducer in [
                        ("body", BODY_IN_BODY_25_ALL_REDUCER),
                        ("left hand", LHAND_IN_BODY_25_ALL_REDUCER),
                        ("right hand", RHAND_IN_BODY_25_ALL_REDUCER),
                        ("face", FACE_IN_BODY_25_ALL_REDUCER),
                    ]:
                        kps = reducer.reduce_arr(skel.all())
                        # tmp = [i for i in reducer.dense_skel.iter_limbs(kps, gen_id=True)]
                        tmp: Any = [
                            i
                            for i in cast(
                                Iterable[Any],
                                reducer.dense_skel.iter_limbs(kps, gen_id=True),
                            )
                            if conf_thresh([i[1][0][2], i[1][1][2]], part)
                        ]
                        add_unique: Callable[[List, Any], List] = lambda lst, elem: (
                            lst
                            if elem[0] in [i[0] for i in cast(Any, lst)]
                            else lst + [elem]
                        )
                        tmp2: List[Any] = reduce(
                            add_unique, tmp, [],
                        )
                        tmp3: Dict[Any, Any] = dict(tmp2)
                        parts[part] = [
                            tmp3.get(i, (np.array((0, 0, 0)), None))[0]
                            for i in range(len(kps))
                        ]
                        # show_frame(h5) -> get_skel_draw(h5).draw_bundle(..) -> SkelDraw( get_skel() -> MODE_SKELS[mode] ).draw_bundle(..).
                        #   -> to draw I need to make this a SkeletonType.
                        # draw_skel (what REALLY draws it) gets numarr, which are the keypoints. draw_skel gets called by draw_bundle, which needs a bundle which is "frame" here.
                        # ... that means I need to remove the respective keypoints from the DumpReaderPoseBundle.bundle[pers_id] here.

                        # frame.bundle[num] = skel #PoseBody25All
                        # BODY_IN_BODY_25_ALL_REDUCER.sparse_skel is SkeletonType and has lines_flat etc
                    frame.bundle[num] = skel.__class__.from_parts(
                        parts["body"],
                        parts["left hand"],
                        parts["right hand"],
                        parts["face"],
                    ).keypoints
                    # TODO this only holds if skel.__class__ is Body25All

                # show_frame(frame, screen)

                show_frame(frame, screen, skel_h5f=skel_h5f)
