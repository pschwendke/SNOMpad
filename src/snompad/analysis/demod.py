# Classes to load demod files. Very similar to scan demodulation classes, just without reshaping and demod methods.
# This just avoids loading the original scan file.
from .base import BaseDemodReader


class RetractionDemod(BaseDemodReader):
    def plot(self):
        raise NotImplementedError


class LineDemod(BaseDemodReader):
    def plot(self):
        raise NotImplementedError


class DelayDemod(BaseDemodReader):
    def plot(self):
        raise NotImplementedError
