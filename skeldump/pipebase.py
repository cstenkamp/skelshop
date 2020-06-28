import collections
from abc import ABC, abstractmethod
from typing import Any, Deque, Optional


class PipelineStageBase(ABC):
    prev: Optional["PipelineStageBase"] = None

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        ...

    def send_back(self, name, *args, **kwargs):
        meth = getattr(self, name, None)
        if meth is not None:
            meth(*args, **kwargs)
            return
        if self.prev is not None:
            self.prev.send_back(name, *args, **kwargs)


class FilterStageBase(PipelineStageBase, ABC):
    prev: PipelineStageBase


class RewindStage(FilterStageBase):
    def __init__(self, size, prev: PipelineStageBase):
        self.prev = prev
        self.buf: Deque[Any] = collections.deque(maxlen=size)
        self.rewinded = 0

    def __next__(self):
        if self.rewinded > 0:
            res = self.buf[-self.rewinded]
            self.rewinded -= 1
            return res
        else:
            item = next(self.prev)
            self.buf.append(item)
            return item

    def rewind(self, iters):
        self.rewinded += iters
        if self.rewinded > len(self.buf):
            raise Exception("Can't rewind that far")


class IterStage(PipelineStageBase):
    def __init__(self, wrapped):
        self.wrapped = iter(wrapped)

    def __next__(self):
        return next(self.wrapped)
