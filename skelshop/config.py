import os


class DefaultConf:
    # see https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/demo_not_quick_start.md
    DEFAULT_THRESH_VALS = {
        "body": 0.05,
        "left hand": 0.2,
        "right hand": 0.2,
        "face": 0.4,
    }
    # DEFAULT_THRESH_VALS = {"body": 0, "left hand": 0, "right hand": 0, "face": 0}
    DEFAULT_THRESH_POOL = "min"

    def __init__(self):
        keys = [key for key in dir(self) if not key.startswith("_")]
        self.__dict__ = {key: os.environ.get(key) or getattr(self, key) for key in keys}
        for key, val in self.__dict__.items():
            if isinstance(val, dict):
                for k2, v2 in val.items():
                    varname = f'{key}_{k2.upper().replace(" ", "_")}'
                    if os.environ.get(varname):
                        self.__dict__[key][k2] = os.environ[varname]


conf = DefaultConf()


def set_conf(new_conf):
    global conf
    conf = new_conf
