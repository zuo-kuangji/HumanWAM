import dataclasses
import enum
import pathlib


def conv(o, depth=0):
    if depth > 14:
        return repr(o)
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return {"__type__": type(o).__name__, **{f.name: conv(getattr(o, f.name), depth + 1) for f in dataclasses.fields(o)}}
    if isinstance(o, (list, tuple)):
        return [conv(x, depth + 1) for x in o]
    if isinstance(o, dict):
        return {str(k): conv(v, depth + 1) for k, v in o.items()}
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    if isinstance(o, enum.Enum):
        return str(o)
    if isinstance(o, pathlib.Path):
        return str(o)
    return repr(o)
