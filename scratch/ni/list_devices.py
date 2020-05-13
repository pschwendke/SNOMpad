# list available devices

from nidaqmx.system import System

sys = System()

print(sys.devices.device_names)