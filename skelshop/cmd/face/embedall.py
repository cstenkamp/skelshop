from contextlib import ExitStack

import click
from imutils.video.count_frames import count_frames

from skelshop.config import conf
from skelshop.dump import add_basic_metadata
from skelshop.face.cmd import check_extractor, mode_of_extractor_info, open_skels
from skelshop.face.consts import DEFAULT_FRAME_BATCH_SIZE
from skelshop.face.io import FaceWriter, write_faces
from skelshop.face.modes import EXTRACTORS
from skelshop.utils.h5py import h5out
from skelshop.utils.video import decord_video_reader, load_video_rgb

DEFAULT_THRESH_VAL = conf.DEFAULT_THRESH_VALS["body"]
DEFAULT_THRESH_POOL = conf.DEFAULT_THRESH_POOL


@click.command()
@click.argument(
    "face_extractor", type=click.Choice(list(EXTRACTORS.keys())),
)
@click.argument("video", type=click.Path(exists=True))
@click.argument("h5fn", type=click.Path())
@click.option("--from-skels", type=click.Path(exists=True))
@click.option("--start-frame", type=int, default=0)
@click.option(
    "--skel-thresh-pool", type=click.Choice(["min", "max", "mean"]), default=None,
)
@click.option("--skel-thresh-val", type=float, default=None)
@click.option("--batch-size", type=int, default=DEFAULT_FRAME_BATCH_SIZE)
@click.option("--batch-size", type=int)
@click.option("--write-bboxes/--no-write-bboxes")
@click.option("--write-chip/--no-write-chip")
def embedall(
    face_extractor,
    video,
    h5fn,
    from_skels,
    start_frame,
    skel_thresh_pool,
    skel_thresh_val,
    batch_size,
    write_bboxes,
    write_chip,
):
    """
    Create a HDF5 face dump from a video using dlib.
    """
    skel_thresh_pool = skel_thresh_pool or DEFAULT_THRESH_POOL
    skel_thresh_val = skel_thresh_val or DEFAULT_THRESH_VAL
    from skelshop.face.pipe import all_faces_from_skel_batched, iter_faces_from_dlib

    check_extractor(face_extractor, from_skels)
    extractor_info = EXTRACTORS[face_extractor]
    num_frames = count_frames(video) - start_frame
    with ExitStack() as stack:
        h5f = stack.enter_context(h5out(h5fn))
        add_basic_metadata(h5f, video, num_frames)
        has_fod = extractor_info["type"] == "dlib"
        writer = FaceWriter(
            h5f,
            write_fod_bbox=write_bboxes and has_fod,
            write_chip_bbox=write_bboxes,
            write_chip=write_chip,
        )
        kwargs = {
            "include_chip": write_chip,
            "include_bboxes": write_bboxes,
        }
        if batch_size is not None:
            kwargs["batch_size"] = batch_size
        if extractor_info["type"] == "dlib":
            vid_read = stack.enter_context(load_video_rgb(video))
            face_iter = iter_faces_from_dlib(
                vid_read,
                detector=extractor_info["detector"],
                keypoints=extractor_info["keypoints"],
                **kwargs,
            )
        else:
            import h5py

            skels_h5 = h5py.File(from_skels, "r")
            if skels_h5.attrs.get("fmt_type") != "trackshots":
                raise click.BadOptionUsage(
                    "--from-skels",
                    f"--from-skels needs an h5-file of type 'trackshots', got '{skels_h5.attrs.get('fmt_type')}'!",
                )
            vid_read = decord_video_reader(video)
            skel_read = open_skels(from_skels)
            mode = mode_of_extractor_info(extractor_info)
            face_iter = all_faces_from_skel_batched(
                vid_read,
                skel_read,
                thresh_pool=skel_thresh_pool,
                thresh_val=skel_thresh_val,
                mode=mode,
                **kwargs,
            )
        write_faces(face_iter, writer)
