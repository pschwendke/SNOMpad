from enum import Enum, auto
from inspect import Signature, Parameter, signature
from typing import Iterable

import attr

from ..buffer import CircularArrayBuffer, ExtendingArrayBuffer, H5Buffer


class BackendType(Enum):
    numpy = auto()
    hdf5 = auto()


@attr.s(order=False)
class BufferConfig:
    """Holds the configuration of the buffer."""
    fname: str = attr.ib(default="")
    backend: BackendType = attr.ib(default=BackendType.numpy)
    size: int = attr.ib(default=200_000)
    continuous: bool = attr.ib(default=True)

    def __attrs_post_init__(self):
        super().__init__()  # tsk tsk tsk...

    def config(self):
        return attr.asdict(self)


def buffer_class(
        backend: BackendType, continuous=True, **kw
):
    """
    Determines the correct buffer class from the configuration arguments.
    """
    if backend is BackendType.numpy:
        if continuous:
            return CircularArrayBuffer
        else:
            return ExtendingArrayBuffer
    elif backend is BackendType.hdf5:
        return H5Buffer


def parameter_merge(*args: Iterable[Signature]):
    "first takes precedence"
    collected = {}
    for sigs in reversed(args):
        for name, p in sigs.parameters.items():
            collected[name] = p
    return collected


def keep_param(p: Parameter):
    return p.kind not in [Parameter.VAR_KEYWORD, Parameter.VAR_POSITIONAL]


def possible_kws(cls):
    sig = parameter_merge(*map(signature, cls.mro()))
    sig = {n: p for n, p in sig.items() if keep_param(p)}
    return list(sig.keys())


def prepare_buffer(exp_config, buffer_config):
    cfg = buffer_config.config()
    cls = buffer_class(**cfg)
    kws = dict()
    kws["vars"] = exp_config.signals()
    kws |= {k: v for k, v in cfg.items() if k in possible_kws(cls)}
    return cls(**kws)

