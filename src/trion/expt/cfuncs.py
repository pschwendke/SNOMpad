
from nidaqmx._lib import lib_importer, ctypes_byte_str
from nidaqmx.errors import check_for_error


def self_cal(device: str):
    cfunc = lib_importer.windll.DAQmxSelfCal
    if cfunc.argtypes is None:
        with cfunc.arglock:
            cfunc.argtypes = [ctypes_byte_str]
    retcode = cfunc(device)
    check_for_error(retcode)
    return True # success!