from contextlib import contextmanager

import cv2
import opencv_wrapper as cvw
from decord.video_reader import VideoReader


class RGBVideoCapture(cvw.VideoCapture):
    def __iter__(self):
        for frame in super().__iter__():
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)
            yield frame

    def skip_frame(self):
        self.skip_frames(1)


class NumpyVideoReader(VideoReader):
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        frame = super().next()
        return frame.asnumpy()

    def skip_frame(self):
        self.skip_frames(1)


@contextmanager
def load_video_rgb(filename, lib="decord"):
    if lib == "decord":
        vid_read = NumpyVideoReader(filename, ctx=decord_dev())
        try:
            yield vid_read
        finally:
            del vid_read
    else:
        if lib != "opencv":
            raise ValueError("lib must be decord or opencv")
        video = RGBVideoCapture(filename)
        if not video.isOpened():
            raise ValueError(f"Could not open video with filename {filename}")
        try:
            yield video
        finally:
            video.release()


_decord_dev = None


def decord_dev():
    from decord import cpu, gpu

    global _decord_dev
    if _decord_dev is None:
        gpu_dev = gpu(0)
        if gpu_dev.exist:
            _decord_dev = gpu_dev
        else:
            _decord_dev = cpu(0)
    return _decord_dev


def decord_video_reader(path):
    return VideoReader(path, ctx=decord_dev())
