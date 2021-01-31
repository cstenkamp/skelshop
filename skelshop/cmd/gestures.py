import time
from collections import namedtuple
from contextlib import contextmanager
from functools import reduce
from itertools import repeat
from typing import Any, Callable, Deque, Dict, Iterable, Iterator, List, Tuple, cast

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
from skelshop.pose import PoseBase
from skelshop.skelgraphs.openpose import (
    BODY_IN_BODY_25_ALL_REDUCER,
    FACE_IN_BODY_25_ALL_REDUCER,
    LHAND_IN_BODY_25_ALL_REDUCER,
    MODE_SKELS,
    RHAND_IN_BODY_25_ALL_REDUCER,
)

# from skelshop.config import set_conf, DefaultConf
# new_conf = DefaultConf()
# new_conf.DEFAULT_THRESH_VALS['body'] = 0.9
# set_conf(new_conf)

BUFFER_SIZE = 200
MAX_BODIES = 2

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
    def __init__(self, skel_h5f, num, max_buffer=0):
        self.skel_h5f = skel_h5f
        self.num = num
        self.skel = MODE_SKELS[skel_h5f.attrs["mode"]]
        self.max_buffer = max_buffer
        self.visible_at = set()
        if max_buffer:
            self.buffer = Deque[np.array](maxlen=self.max_buffer)

    def update_numarr(self, numarr, framenum):
        self.visible_at.add(framenum)
        self.numarr = numarr
        if self.max_buffer:
            self.buffer.append(numarr)

    def get(self, name, numarr=None):
        numarr = numarr if numarr is not None else self.numarr
        return numarr[self.skel.names.index(name)]

    def check_gestures(self):
        """checks if in the frame at self.max_buffer//2 there was a gesture"""
        if len(self.buffer) < self.max_buffer:
            return False
        for bodypart in ["left wrist", "right wrist"]:
            movement = sum(
                [
                    self.get(bodypart, numarr=i)[:2]
                    - self.get(bodypart, numarr=self.buffer[self.max_buffer // 2])[:2]
                    for i in self.buffer
                ]
            )
            if min(movement) > 3:  # type: ignore
                return True


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

    class FakeFrame(namedtuple("Frame", "bundle cls")):
        def __iter__(self) -> Iterator[Tuple[int, "PoseBase"]]:
            for idx, pose in self.bundle.items():
                yield idx, self.cls.from_keypoints(pose)

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

        nframe_before_scene = 0
        nframe = 0

        for nscene, scene in enumerate(frame_iter):

            print(f"Scene #{nscene+1} at {nframe}")
            # in every new scene, start anew with the bodies
            bodies = [
                Body(skel_h5f, i, max_buffer=BUFFER_SIZE) for i in range(MAX_BODIES)
            ]  # TODO get that number from the skeletons
            if BUFFER_SIZE > 0:
                # if we have a buffer, we also need to delay displaying the video
                img_delay_buffer = Deque[np.array](maxlen=BUFFER_SIZE // 2)
            newscene = True

            for nframe_in_scene, frame in enumerate(scene):
                nframe = nframe_before_scene + nframe_in_scene
                img = (next(vid_iter) * 0.3).astype(
                    np.uint8
                )  # background scene should be dimmed
                for num, skel in frame:
                    bodies[num].update_numarr(
                        threshold_limbs(skel, conf_thresh), nframe
                    )
                    frame.bundle[num] = bodies[num].numarr
                # bewegt sich was, ist die Hand drauf, #Personen begrenzen (Mahnaz-Talk)
                # im video einblenden ob geste erkannt ist oder nicht (video dimmen?)
                # er wird mir code zukommen lassen (trian taphylos) -> "velocity from keypoints"

                # if there are more than 2 persons in this shot or no wrist is visible --> make red
                if len(frame.bundle) > MAX_BODIES:
                    img = make_red(img)
                if not (
                    bodies[0].get("left wrist")[2] > 0
                    or bodies[0].get("right wrist")[2] > 0
                ):
                    img = make_red(img)

                if BUFFER_SIZE > 0:
                    # replace current img with the one from the buffer
                    if len(img_delay_buffer) == img_delay_buffer.maxlen:
                        if newscene:
                            print(f"Buffer full at {nframe}")
                            newscene = False

                        visible_bodies = [
                            b
                            for b in bodies
                            if nframe - BUFFER_SIZE // 2 in b.visible_at
                        ]
                        visible_img = img_delay_buffer[0]

                        # if you spotted a gesture --> make green
                        gstr = False
                        for body in visible_bodies:
                            gstr = gstr or body.check_gestures()
                        if gstr:
                            visible_img = make_green(visible_img)

                        show_frame(
                            FakeFrame(
                                {
                                    b.num: b.buffer[BUFFER_SIZE // 2]
                                    for b in visible_bodies
                                },
                                frame.cls,
                            ),
                            screen,
                            skel_h5f=skel_h5f,
                            img=visible_img,
                        )

                    # we only have to start filling the img_delay_buffer once the body-buffer is half full (we drop the first buffer_size//2 frames of every scene)
                    if max(len(i.buffer) for i in bodies) >= bodies[0].max_buffer // 2:
                        img_delay_buffer.append(img)

                else:
                    show_frame(frame, screen, skel_h5f=skel_h5f, img=img)
                # breakpoint in skelshop.track.track.PoseTrack.pose_track -> the specs have a spec.prev_frame_buf_size, and that many get added to
                # posetrack.prev_tracked, which is: FrameBuf = (collections.deque(maxlen=spec.prev_frame_buf_size)), where FrameBuf = Deque[List[TrackedPose]]
                # trackedpose? gets filled in a complicated way.
            nframe_before_scene += nframe_in_scene


def make_red(img):
    return np.stack(
        (img[:, :, 0], img[:, :, 1], (img[:, :, 2] * 3).astype(np.uint8)), axis=-1
    )  # np.floor_divide(img[0], 2).astype(np.uint8)


def make_green(img):
    return np.stack(
        (img[:, :, 0], (img[:, :, 1] * 3), (img[:, :, 2]).astype(np.uint8)), axis=-1
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
