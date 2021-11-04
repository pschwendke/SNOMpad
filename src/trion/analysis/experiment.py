from itertools import chain
from typing import Iterable

import attr

from trion.analysis.signals import Scan, Demodulation, Detector, Signals, \
    detection_signals, modulation_signals, all_modulation_signals, \
    all_detector_signals

# TODO: Should probably factor this out into an object to specify the acquisition (roles + channels.. the channel map),
#   and one to specify the analysis (ie: so we can acquire pshet channels, but analyze only the sHD part...
#   The channel map is enough only for single pixels. For scans, we need to add more info. The detector bit is also part
#   of the channel map...
@attr.s(order=False)
class Experiment:
    """
    Describes a TRION experimental configuration.

    Parameters
    ----------
    scan: Scan
        AFM scan protocol, such as single-point, approach curve or AFM
    acquisition: Demodulation
        Interferometric Near field acquisition modes: self-homodyne, pshet, etc.
    detector: Detector
        Optical detector configuration
    nreps: int
        Number of repetitions. Default `1`
    npts: int
        Number of points per frame. Ignored if `continuous`. Default 200_000.
    continuous:
        Continous acquisition. Default True.
    """
    scan: Scan = attr.ib(default=Scan.point)
    acquisition: Demodulation = attr.ib(default=Demodulation.shd)
    detector: Detector = attr.ib(default=Detector.single)
    frame_reps: int = attr.ib(default=1)
    npts: int = attr.ib(default=200_000)
    continuous: bool = attr.ib(default=True)

    def __attrs_post_init__(self):
        super().__init__()

    def signals(self) -> Iterable[Signals]:
        """
        Lists the signals for the current Experimental configuration.
        """
        return sorted(chain(
            detection_signals[self.detector],
            modulation_signals[self.acquisition]
        ))

    def is_valid(self) -> bool:
        """
        Check if the experimental configuration is valid.
        """
        # Currently this is a bit basic, but this will get more complicated
        return (
                type(self.scan) is Scan and
                type(self.acquisition) is Demodulation and
                type(self.detector) is Detector
        )

    def axes(self) -> list:
        if self.frame_reps > 1:
            ax = ["frame_rep"]
        else:
            ax = []
        return ax

    def to_dict(self):
        return attr.asdict(self)

    @classmethod
    def from_dict(cls, **cfg):
        cfg = cfg.copy()
        for key, enum in [("scan", Scan),
                          ("demodulation", Demodulation),
                          ("detector", Detector),
                          ]:
            v = cfg["key"]
            if isinstance(v, str):
                cfg[key] = enum[cfg[key]]
        return cls(**cfg)




