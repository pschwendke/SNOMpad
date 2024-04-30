from itertools import chain
from enum import Enum, auto
from bidict import bidict
from functools import total_ordering

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


class Scan(NamedEnum):
    stepped_retraction = auto()
    stepped_image = auto()
    stepped_line = auto()
    continuous_retraction = auto()
    continuous_image = auto()
    continuous_line = auto()
    noise_sampling = auto()
    delay_scan = auto()


class Demodulation(NamedEnum):
    shd = auto()
    pshet = auto()
    none = auto()


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
