# Mostly classes to analyse scans data, e.g. line scans, retraction curves etc
from snompad.analysis.base import BaseScanDemod, BaseDemodReader
from snompad.analysis.scans import Retraction, Image, Line, Noise, Delay
from snompad.analysis.demod import RetractionDemod, LineDemod, DelayDemod


def load(filename: str) -> BaseScanDemod or BaseDemodReader:
    """ Loads scan from file and returns BaseScanDemod object. The function is agnostic to scan type.
    """
    if '_demod' in filename:
        return load_demod(filename)
    if 'retraction' in filename:
        return Retraction(filename)
    elif 'image' in filename:
        return Image(filename)
    elif 'line' in filename:
        return Line(filename)
    elif 'noise' in filename:
        return Noise(filename)
    elif 'delay' in filename:
        return Delay(filename)
    else:
        raise NotImplementedError


def load_demod(filename: str) -> BaseDemodReader:
    """ Loads respective demod classes to
    """
    if 'retraction' in filename:
        return RetractionDemod(filename)
    elif 'line' in filename:
        return LineDemod(filename)
    elif 'delay' in filename:
        return DelayDemod(filename)
    else:
        raise NotImplementedError
