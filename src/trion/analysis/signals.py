from itertools import chain


from enum import Enum, auto
from typing import Iterable
import attr
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
# How to handle pump probe? We will add a boolean value to the experiment..

# TODO: use an implementation where auto() copies the name instead of a number


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
    #ref_x = auto()
    #ref_y = auto()
    #chop = auto()


class Scan(NamedEnum):
    point = auto()
    # approach curve?
    # image scan, or AFM, or SNOM?


class Acquisition(NamedEnum):
    shd = auto()  # self-homodyne
    # pshet = auto()
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
    Acquisition.shd: frozenset([Signals.tap_x, Signals.tap_y]),
})

all_acquisition_signals = frozenset(chain(*acquisition_signals.values()))


@attr.s(order=False)
class Experiment:
    """
    Describes a TRION experimental configuration.

    Parameters
    ----------
    scan: Scan
        AFM scan protocol, such as single-point, approach curve or AFM
    acquisition: Acquisition
        Interferometric Near field acquisition modes: self-homodyne, pshet, etc.
    detector: Detector
        Optical detector configuration
    """
    scan: Scan = attr.ib()
    acquisition: Acquisition = attr.ib()
    detector: Detector = attr.ib()

    def signals(self) -> Iterable[Signals]:
        """
        Lists the signals for the current Experimental configuration.
        """
        return sorted(chain(
            detection_signals[self.detector],
            acquisition_signals[self.acquisition]
        ))

    def is_valid(self) -> bool:
        """
        Check if the experimental configuration is valid.
        """
        # Currently this is a bit basic, but this will get more complicated
        return (
            type(self.scan) is Scan and
            type(self.acquisition) is Acquisition and
            type(self.detector) is Detector
        )

    @classmethod
    def from_signals(cls, signals: Iterable[Signals]):
        acq_sigs = {s for s in signals
                    if s in all_acquisition_signals}
        det_sigs = {s for s in signals
                    if s in all_detector_signals}
        scan = Scan.single
        acq = acquisition_signals.inverse[acq_sigs]
        det = detection_signals.inverse[det_sigs]
        return cls(scan, acq, det)


def is_optical_signal(role: Signals) -> bool:
    return role in all_detector_signals


def is_tap_modulation(role: Signals) -> bool:
    return role in [Signals.tap_x, Signals.tap_y]


# def is_pshet_modulation(role: str) -> bool:
#     return role.startswith("ref")