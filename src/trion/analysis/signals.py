from itertools import chain
import os.path as pth
from enum import Enum, auto
from bidict import bidict
from functools import total_ordering, singledispatch
import numpy as np
import toml


"""
This module defines the possible experimental configurations. 

The module defines types to describe the currently supported
experimental configurations and signal types. The main class defined here is 
the `Experiment` class, which handles the type of the experiments. The module
also defines the following enumeration types:
- `Scan`, the possible SNOM scan types
- `Acquisition`: near field acquisition modes: self-homodyne, pshet
- `Detector`: detector configurations
- `Signals`: the possible experimental signal types.
"""
# How to handle pump probe? We will add a boolean value to the experiment..


class NamedEnum(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name

    def __repr__(self) -> str:
        return '<%s.%s>' % (self.__class__.__name__, self.name)


@total_ordering
class SortableEnum(Enum):
    def index(self) -> int:
        return list(type(self)).index(self)

    def __lt__(self, other) -> bool:
        if self.__class__ is other.__class__:
            return self.index() < other.index()
        return NotImplemented


class Signals(SortableEnum, NamedEnum):
    sig_a = auto()
    sig_b = auto()
    sig_d = auto()
    sig_s = auto()
    tap_x = auto()
    tap_y = auto()
    tap_p = auto()
    ref_x = auto()
    ref_y = auto()
    ref_p = auto()
    chop = auto()


def signal_colormap(scale=255, filename=None) -> dict:
    """
    Obtain the standard colormap.

    By default, the colormap is in a format compatible with pyqtgraph and Qt,
    where RGB values range from 0-255. In order to use the colors in matplotlib,
    they must be rescaled to 0-1. This is done by passing "scale=1".

    Parameters
    ----------
    scale : int
        Scale of the output colors (ie: 0-255 or 0-1). Defaults to 255
    filename : path-like
        Name of the 'toml' file containing the color map.


    Returns
    -------
    cmap : dict
        signal -> color mapping.
    """
    filename = filename or pth.join(pth.dirname(pth.abspath(__file__)), "signal_colors.toml")
    scale = 255/scale  # 256 <-> 1
    with open(filename, "r") as f:
        cmap = toml.load(f)
    cmap = {k: np.array(v)/scale for k, v in cmap.items()}
    cmap.update({Signals[k]: v for k, v in cmap.items()})
    cmap.update({k.lower(): v for k, v in cmap.items() if isinstance(k, str)})
    return cmap


class Scan(NamedEnum):
    point = auto()
    stepped_retraction = auto()
    stepped_image = auto()
    stepped_line = auto()
    continuous_retraction = auto()
    continuous_image = auto()
    continuous_line = ()
    noise_sampling = auto()
    delay_scan = auto()
    delay_lines = auto()


class Demodulation(NamedEnum):
    shd = auto()
    pshet = auto()
    # nanospectroscopy


class Detector(NamedEnum):
    single = auto()
    dual = auto()
    balanced = auto()


detection_signals = bidict({
    Detector.single: frozenset([Signals.sig_a]),
    Detector.dual: frozenset([Signals.sig_a, Signals.sig_b]),
    Detector.balanced: frozenset([Signals.sig_d, Signals.sig_s]),
})
all_detector_signals = frozenset(chain(*detection_signals.values()))

modulation_signals = bidict({
    Demodulation.shd: frozenset([Signals.tap_x, Signals.tap_y]),
    Demodulation.pshet: frozenset([Signals.tap_x, Signals.tap_y, Signals.ref_x, Signals.ref_y])
})

all_modulation_signals = frozenset(chain(*modulation_signals.values()))


def is_optical_signal(role: Signals) -> bool:
    return role in all_detector_signals


@singledispatch
def is_tap_modulation(role) -> bool:
    raise TypeError(f"Didn't find implementation for {type(role)}")


@is_tap_modulation.register(Signals)
def is_tap_modulation_sig(role: Signals) -> bool:
    return role in [Signals.tap_x, Signals.tap_y, Signals.tap_p]


@is_tap_modulation.register(str)
def is_tap_modulation_str(role: str) -> bool:
    try:
        s = Signals[role]
    except KeyError:
        return False
    return is_tap_modulation(s)
