from enum import Enum, auto
from inspect import Signature, Parameter, signature
from typing import Iterable

from ..buffer import CircularArrayBuffer, ExtendingArrayBuffer, H5Buffer


class BackendType(Enum):
    numpy = auto()
    hdf5 = auto()


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


def prepare_buffer(**cfg):
    cls = buffer_class(**cfg)
    kws = {k: v for k, v in cfg.items() if k in possible_kws(cls)}
    return cls(**kws)


