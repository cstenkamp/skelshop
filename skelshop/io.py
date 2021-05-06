from functools import partial
from itertools import repeat
from typing import Any, Dict, Iterator, List, Tuple

import h5py
import hdf5plugin
from numpy import ndarray

from .openpose import POSE_CLASSES
from .pose import DumpReaderPoseBundle, UntrackedDumpReaderPoseBundle
from .sparsepose import SparsePose, create_csr, create_growable_csr


def get_pose_nz(pose):
    for limb_idx, limb in enumerate(pose):
        if not limb[2]:
            continue
        yield limb_idx, limb


def grow_ds(ds, extra):
    ds.resize(len(ds) + extra, axis=0)


def add_empty_rows_grp(indptr, data, new_rows):
    grow_ds(indptr, new_rows)
    indptr[-new_rows:] = len(data)


class NullWriter:
    def __init__(self, h5f, *args, **kwargs):
        pass

    def add_pose(self, frame_num, pose_id, pose):
        pass

    def start_shot(self, start_frame=None):
        pass

    def register_frame(self, frame_num):
        pass

    def end_shot(self):
        pass


class UnsegmentedWriter:
    """
    Write a skeleton dump without any shot segmentation. This typically implies
    that poses are not tracked.
    """

    def __init__(self, h5f: h5py.File, num_kps=None, **create_kwargs):
        """
        Constructs an unsegmented pose writer
        """
        self.h5f = h5f
        self.num_kps = num_kps
        self.timeline_grp = self.h5f.create_group("/timeline", track_order=True)
        self.pose_grps: Dict[int, List[Any]] = {}
        self.start_frame = 0
        self.create_kwargs = create_kwargs

    def _pose_grp(self, pose_id: int, frame_num: int):
        if pose_id in self.pose_grps:
            return self.pose_grps[pose_id]
        path = f"/timeline/pose{pose_id}"
        pose_grp = create_growable_csr(
            self.h5f, path, self.num_kps, **self.create_kwargs
        )
        pose_grp.attrs["start_frame"] = frame_num
        last_frame_num = frame_num - 1
        self.pose_grps[pose_id] = [
            pose_grp,
            pose_grp["data"],
            pose_grp["indices"],
            pose_grp["indptr"],
            last_frame_num,
        ]
        return self.pose_grps[pose_id]

    def add_pose(self, frame_num: int, pose_id: int, pose: ndarray):
        """
        Add a pose
        """
        pose_grp, data, indices, indptr, last_frame_num = self._pose_grp(
            pose_id, frame_num
        )
        new_rows = frame_num - last_frame_num
        add_empty_rows_grp(indptr, data, new_rows)
        new_data = []
        new_indices = []
        for limb_idx, limb in get_pose_nz(pose):
            new_data.append(limb)
            new_indices.append(limb_idx)
        if new_data:
            grow_ds(data, len(new_data))
            grow_ds(indices, len(new_indices))
            data[-len(new_data) :] = new_data
            indices[-len(new_indices) :] = new_indices
        self.pose_grps[pose_id][-1] = frame_num

    def start_shot(self, start_frame: int = 0):
        """
        Start a shot. This should only be called once at the beginning of writing.
        """
        self.start_frame = start_frame

    def register_frame(self, frame_num):
        pass

    def end_shot(self):
        """
        Ends a shot. This should only be called once at the end of writing.
        """
        self.timeline_grp.attrs["start_frame"] = self.start_frame
        timeline_last_frame_num = 0
        for pose_grp, data, indices, indptr, last_frame_num in self.pose_grps.values():
            add_empty_rows_grp(indptr, data, 1)
            pose_grp.attrs["end_frame"] = last_frame_num + 1
            timeline_last_frame_num = max(timeline_last_frame_num, last_frame_num)
        self.timeline_grp.attrs["end_frame"] = timeline_last_frame_num + 1


class ShotSegmentedWriter:
    """
    Write a skeleton dump with any shot segmentation. This typically implies
    that poses are are tracked.
    """

    def __init__(self, h5f: h5py.File, num_kps=None, **create_kwargs):
        """
        Constructs a shot segmented writer
        """
        self.h5f = h5f
        self.num_kps = num_kps
        self.h5f.create_group("/timeline", track_order=True)

        self.pose_data: Dict[int, Dict[int, ndarray]] = {}
        self.pose_starts: Dict[int, int] = {}
        self.pose_ends: Dict[int, int] = {}
        self.shot_idx = 0
        self.shot_start = 0
        self.last_frame = 0
        self.create_kwargs = create_kwargs

    def start_shot(self, start_frame=None):
        """
        Start a new shot
        """
        if start_frame is not None:
            self.shot_start = start_frame

    def add_pose(self, frame_num: int, pose_id: int, pose: ndarray):
        """
        Add a pose
        """
        if pose_id not in self.pose_data:
            self.pose_starts[pose_id] = frame_num
            self.pose_data[pose_id] = {}
        self.pose_data[pose_id][frame_num] = pose
        self.pose_ends[pose_id] = frame_num
        self.last_frame = frame_num

    def register_frame(self, frame_num: int):
        """
        Register frame_num as existing within the current shot
        """
        self.last_frame = frame_num

    def end_shot(self):
        """
        End the current shot
        """
        shot_grp = self.h5f.create_group(
            f"/timeline/shot{self.shot_idx}", track_order=True
        )
        shot_grp.attrs["start_frame"] = self.shot_start
        shot_grp.attrs["end_frame"] = self.last_frame + 1
        for pose_id, poses in self.pose_data.items():
            data: List[ndarray] = []
            indices: List[int] = []
            indptr: List[int] = []
            pose_first_frame = self.pose_starts[pose_id]
            pose_last_frame = self.pose_ends[pose_id] + 1
            last_frame_num = pose_first_frame - 1

            def add_empty_rows(num_rows):
                for _ in range(num_rows):
                    indptr.append(len(data))

            for frame_num, pose in poses.items():
                add_empty_rows(frame_num - last_frame_num)
                for limb_idx, limb in get_pose_nz(pose):
                    data.append(limb)
                    indices.append(limb_idx)
                last_frame_num = frame_num
            # Extra empty row to insert final nnz entry
            add_empty_rows(1)

            pose_group = create_csr(
                self.h5f,
                f"/timeline/shot{self.shot_idx}/pose{pose_id}",
                self.num_kps,
                data=data,
                indices=indices,
                indptr=indptr,
                **self.create_kwargs,
            )
            pose_group.attrs["start_frame"] = pose_first_frame
            pose_group.attrs["end_frame"] = pose_last_frame
        self.pose_data = {}
        self.shot_idx += 1
        self.shot_start = self.last_frame + 1


def get_endnum(haystack, expect):
    assert haystack.startswith(expect)
    idx_str = haystack[4:]
    assert idx_str.isnumeric()
    return int(idx_str)


def enumerated_poses(grp):
    return ((get_endnum(k, "pose"), v) for k, v in grp.items())


def read_grp(grp) -> Tuple[int, int, Iterator[Any]]:
    return (grp.attrs["start_frame"], grp.attrs["end_frame"], enumerated_poses(grp))


class ShotSegmentedReader:
    """
    Reads a shot segmented skeleton dump.
    """

    def __init__(self, h5f: h5py.File, bundle_cls=DumpReaderPoseBundle, infinite=False):
        """
        Constructs the reader from a HDF5 file and. If `infinite` is True, all
        shot iterators will terminate with a final empty shot which infinitely
        yields empty pose bundles.
        """
        self.h5f = h5f
        self.limbs = self.h5f.attrs.get("limbs")
        assert self.h5f.attrs.get("fmt_type") == "trackshots"
        self.mk_bundle = partial(bundle_cls, cls=POSE_CLASSES[self.h5f.attrs["mode"]])
        self.empty_bundle = self.mk_bundle({})
        self.infinite = infinite

    def _mk_reader(self, start_frame, end_frame, bundles):
        return ShotReader(start_frame, end_frame, bundles, self.limbs, self.mk_bundle)

    def __getitem__(self, idx):
        shot_name = f"shot{idx}"
        shot_grp = self.h5f[f"/timeline/{shot_name}"]
        start_frame, end_frame, bundles = read_grp(shot_grp)
        return self._mk_reader(start_frame, end_frame, bundles)

    def _iter(self, req_start=0):
        end_frame = req_start
        for shot_name, shot_grp in self.h5f["/timeline"].items():
            shot_idx = get_endnum(shot_name, "shot")
            start_frame, end_frame, bundles = read_grp(shot_grp)
            if end_frame <= req_start:
                continue
            if shot_idx == 0 and start_frame > 0 and req_start < start_frame:
                yield (
                    -1,
                    "empty_shot",
                    req_start,
                    start_frame,
                    lambda: EmptyShot(req_start, start_frame, self.empty_bundle),
                )
            yield (
                shot_idx,
                shot_name,
                start_frame,
                end_frame,
                lambda: self._mk_reader(start_frame, end_frame, bundles),
            )
        if self.infinite:
            yield (
                shot_idx + 1,
                "empty_shot",
                end_frame,
                None,
                lambda: EmptyShot(end_frame, None, self.empty_bundle),
            )

    def __iter__(self):
        """
        Returns a shot iterator of all shots
        """
        for shot_idx, shot_name, start_frame, end_frame, mk_shot in self._iter():
            yield mk_shot()

    def iter_from_shot(self, start_shot: int):
        """
        Returns a shot iterator starting from shot 0-index `start_shot`.
        """
        for shot_idx, shot_name, start_frame, end_frame, mk_shot in self._iter():
            if shot_idx >= start_shot:
                yield mk_shot()

    def iter_from_frame(self, start_frame: int):
        """
        Returns a shot iterator starting from frame 0-index `start_frame`.
        """
        started = False
        for (
            shot_idx,
            shot_name,
            shot_start_frame,
            shot_end_frame,
            mk_shot,
        ) in self._iter(start_frame):
            if started:
                yield mk_shot()
            elif shot_start_frame <= start_frame and (
                shot_end_frame is None or start_frame < shot_end_frame
            ):
                yield mk_shot()
                started = True


class UnsegmentedReader:
    """
    Reads a non-shot segmented skeleton dump.
    """

    def __init__(self, h5f, bundle_cls=UntrackedDumpReaderPoseBundle):
        self.h5f = h5f
        assert self.h5f.attrs["fmt_type"] == "unseg"
        mk_bundle = partial(bundle_cls, cls=POSE_CLASSES[self.h5f.attrs["mode"]])
        self.shot_reader = ShotReader(
            *read_grp(self.h5f["/timeline"]), self.h5f.attrs["limbs"], mk_bundle
        )

    def __iter__(self):
        """
        Returns a pose bundle iterator of all frames.
        """
        return iter(self.shot_reader)

    def iter_from(self, start_frame: int):
        """
        Returns a pose bundle iterator starting at frame `start_frame`.
        """
        return self.shot_reader.iter_from(start_frame)


class EmptyShot:
    def __init__(self, start_frame, end_frame, empty_bundle):
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.empty_bundle = empty_bundle

    def __iter__(self):
        return self.iter_from(self.start_frame)

    def iter_from(self, start_frame):
        if self.end_frame:
            return repeat(self.empty_bundle, self.end_frame - start_frame)
        else:
            return repeat(self.empty_bundle)


class ShotReader:
    """
    A reader for a single shot. Typically this is returned from ShotSegmentedReader.
    """

    def __init__(self, start_frame, end_frame, bundles, num_limbs, mk_bundle):
        self.num_limbs = num_limbs
        self.mk_bundle = mk_bundle
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.poses = []
        for pose_num, pose_grp in bundles:
            start_frame = pose_grp.attrs["start_frame"]
            end_frame = pose_grp.attrs["end_frame"]
            sparse_pose = SparsePose(pose_grp, num_limbs)
            self.poses.append((pose_num, start_frame, end_frame, sparse_pose))

    def __getitem__(self, tpl):
        if isinstance(tpl, int):
            frame_num = tpl
            pose_nums = None
        else:
            frame_num, pose_nums = tpl
        abs_frame_num = frame_num + self.start_frame
        bundle = {}
        for pose_num, start_frame, end_frame, sparse_pose in self.poses:
            if pose_nums is not None and pose_num not in pose_nums:
                continue
            if abs_frame_num < start_frame or abs_frame_num >= end_frame:
                continue
            row_num = abs_frame_num - start_frame
            bundle[pose_num] = sparse_pose.get_row(row_num)
        return self.mk_bundle(bundle)

    def __iter__(self):
        """
        Returns a pose bundle iterator of all frames.
        """
        return self.iter_from(self.start_frame)

    def iter_from(self, start_frame):
        """
        Returns a pose bundle iterator starting at frame `start_frame`.
        """
        for frame in range(start_frame, self.end_frame):
            bundle = {}
            for pose_num, start_frame, end_frame, sparse_pose in self.poses:
                if start_frame <= frame < end_frame:
                    row_num = frame - start_frame
                    bundle[pose_num] = sparse_pose.get_row(row_num)
            yield self.mk_bundle(bundle)


class EnumerateIterable:
    """
    Like enumerate(...) but produces an iterable (can consume mulitple times)
    rather than an iterator (can only consume once).
    """

    def __init__(self, wrapped):
        self.wrapped = wrapped

    def __iter__(self):
        return enumerate(self.wrapped)


class AsIfTracked:
    """
    Adapter wrapper for readers producing UntrackedDumpReaderPoseBundle such as
    UnsegmentedReader by default that makes them appear to produce bundles like
    TrackedDumpReaderPoseBundle.
    """

    def __init__(self, wrapped):
        self.wrapped = wrapped

    def __iter__(self):
        return (EnumerateIterable(frame) for frame in iter(self.wrapped))

    def iter_from(self, start_frame):
        return (
            EnumerateIterable(frame) for frame in self.wrapped.iter_from(start_frame)
        )

    @property
    def total_frames(self):
        return self.wrapped.total_frames  # TODO not everything has this..


class AsIfSingleShot:
    """
    Adapter wrapper for ShotSegmentedReader to make it act more like an
    UnsegmentedReader.
    """

    def __init__(self, wrapped):
        self.wrapped = wrapped

    def __iter__(self):
        return self.iter_from(0)

    def iter_from(self, start_frame):
        for shot in self.wrapped.iter_from_frame(start_frame):
            for payload in shot:
                yield payload


NO_COMPRESSION: Tuple[Dict[str, Any], Dict[str, Any]] = ({}, {})

BLOSC_ZSTD_5 = hdf5plugin.Blosc(
    cname="zstd", clevel=5, shuffle=hdf5plugin.Blosc.BITSHUFFLE
)
BLOSC_ZSTD_9 = hdf5plugin.Blosc(
    cname="zstd", clevel=9, shuffle=hdf5plugin.Blosc.BITSHUFFLE
)


COMPRESSIONS = {
    "none": NO_COMPRESSION,
    "lossless": ({"compression": BLOSC_ZSTD_5}, {}),
    "lossless9": ({"compression": BLOSC_ZSTD_9}, {}),
    "lossy": ({"compression": BLOSC_ZSTD_5}, {"scaleoffset": 2}),
    "lossy9": ({"compression": BLOSC_ZSTD_9}, {"scaleoffset": 2}),
}
