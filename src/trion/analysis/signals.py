from itertools import chain
import os.path as pth
from enum import Enum, auto
from bidict import bidict
from functools import total_ordering, singledispatch
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
    sig_A = auto()
    sig_B = auto()
    sig_d = auto()
    sig_s = auto()
    tap_x = auto()
    tap_y = auto()
    ref_x = auto()
    ref_y = auto()
    #chop = auto()
    tap_p = auto()


def signal_colormap(filename=None):
    filename = filename or pth.join(pth.dirname(pth.abspath(__file__)), "signal_colors.toml")
    with open(filename, "r") as f:
        cmap = toml.load(f)
    return {Signals[k]: v for k, v in cmap.items()}

class Scan(NamedEnum):
    point = auto()
    # approach curve?
    # image scan, or AFM, or SNOM?


class Demodulation(NamedEnum):
    shd = auto()  # self-homodyne
    pshet = auto()
    # nanospectroscopy


class Detector(NamedEnum):
    single = auto()
    dual = auto()
    balanced = auto()


detection_signals = bidict({
    Detector.single: frozenset([Signals.sig_A]),
    Detector.dual: frozenset([Signals.sig_A, Signals.sig_B]),
    Detector.balanced: frozenset([Signals.sig_d, Signals.sig_s]),
})
all_detector_signals = frozenset(chain(*detection_signals.values()))

acquisition_signals = bidict({
    Demodulation.shd: frozenset([Signals.tap_x, Signals.tap_y]),
    Demodulation.pshet: frozenset([Signals.tap_x, Signals.tap_y, Signals.ref_x, Signals.ref_y])
})

all_acquisition_signals = frozenset(chain(*acquisition_signals.values()))


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
