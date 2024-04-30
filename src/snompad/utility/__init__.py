from .signals import Signals, Demodulation, Scan
from .acquisition_logger import notebook_logger

# for conversion of carriage position to delay time
c = 299_792_458   # m/s
n_air = 1.00028520   # 400 - 1_000 nm wavelength [https://doi.org/10.1364/AO.47.004856]
c_air = c/n_air
