from itertools import chain
from typing import Iterable

import attr

from trion.analysis.signals import Scan, Acquisition, Detector, Signals, \
    detection_signals, acquisition_signals, all_acquisition_signals, \
    all_detector_signals


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
    nreps: int
        Number of repetitions. Default `1`
    npts: int
        Number of points per frame. Ignored if `continuous`. Default 200_000.
    continuous:
        Continous acquisition. Default True.
    """
    scan: Scan = attr.ib(default=Scan.point)
    acquisition: Acquisition = attr.ib(default=Acquisition.shd)
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
                          ("acquisition", Acquisition),
                          ("detector", Detector),
                          ]:
            v = cfg["key"]
            if isinstance(v, str):
                cfg[key] = enum[cfg[key]]
        return cls(**cfg)




