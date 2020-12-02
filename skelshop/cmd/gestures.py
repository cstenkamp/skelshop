import click
import h5py
import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt
import numpy as np

from skelshop.drawsticks import SkelDraw, get_skel
from skelshop.io import ShotSegmentedReader
from skelshop.pipebase import IterStage
from skelshop.pipeline import pipeline_options


def imshow(img, title="", hsv=False, **kwargs):
    if hsv:
        img = mplcolors.hsv_to_rgb(img)
    plt.imshow(img, cmap="Greys_r", **kwargs)
    if title or title == 0:
        plt.title(title)
    plt.show()


@click.command()
@click.argument("h5infn", type=click.Path(exists=True))
@click.option("--start-frame", type=int, default=0)
@click.option("--end-frame", type=int, default=None)
# @click.option("--skel-thresh-val", type=float, default=DEFAULT_THRESH_VAL) #see cmd/face/embedall
@pipeline_options(allow_empty=True)  # brauch nur shot_seg/segs_file
def gestures(h5infn, pipeline, start_frame, end_frame):
    """
    Detect gestures on sorted etc data.
    """
    if pipeline.metadata.get("shot_seg") == "none":
        raise click.BadOptionUsage("--shot-seg", "--shot-seg is required!")

    def get_skel_draw(h5f):
        posetrack = False  # TODO is posetrack False?!
        skel = get_skel(h5f, posetrack)
        return SkelDraw(skel, posetrack, ann_ids=True, scale=1)

    with h5py.File(h5infn, "r") as skel_h5f:
        read = ShotSegmentedReader(skel_h5f, infinite=True)
        # poses = [i for i in peekable(iter(peekable(iter(read)).peek()))]
        # for frame in read(): for skel_id, skel in shot: print(skel)
        stage = IterStage(read.iter_from_frame(0))
        frame_iter = pipeline(stage)
        # Each stage acts as an iterator, typically yielding some kind of pose bundle. A pose bundle is an iterator of skeletons, either with ids or not depending on whether it has been tracked.
        # TODO assert it's tracked at this point!
        for scene in frame_iter:
            for frame in scene:
                img = np.zeros((360, 640, 3))
                get_skel_draw(skel_h5f).draw_bundle(img, frame)
                imshow(img)
