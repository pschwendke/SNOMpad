# ToDo: standardize units

import sys
import clr
import logging
from time import sleep

# from snompad.demodulation.signals import Demodulation, NamedEnum
from snompad.utility.signals import Demodulation

# import the neaSpec SDK
neaspec_folder = '//nea-server/updates/SDK'  # Import all DLLs in the folder or 'N:/updates/SDK' to mapped drive
sys.path.append(neaspec_folder)  # Import all DLLs in the folder
import Nea.Client.Hardware.SDK as neaSDK  # Import the DLL as element neaSDK
clr.AddReference('Nea.Client.Hardware')  # Load the main DLL

logger = logging.getLogger(__name__)


class NeaSNOM:
    def __init__(self):
        # connects to the microscope of neaSNOM
        # returns an instance of a microscope neaMIC and the connection to the SDK nea_client
        self.scan = None
        self.tracked_channels = None
        self.tip_velocity = 5.                # tip velocity when moving to target position (um/s)
        self.nea_client = neaSDK.Connection('nea-server')               # Open up connection to microscope
        self.nea_mic = self.nea_client.Connect()              # Define the Microscope
        sleep(0.1)                              # Short delay makes things work fine (?)
        self.connected = True
        logger.info('Microscope connected')
        self.stop()

        logger.debug("Client Version: " + self.nea_mic.ClientVersion)    # get Client Version
        logger.debug("Server Version: " + self.nea_mic.ServerVersion)    # get Server Version

    def __del__(self):
        logger.debug('NeaSNOM.__del__()')
        if self.connected:
            self.disconnect()

    def engage(self, setpoint: float = 0.8, settle_s: float = 3):
        logger.info('Engaging sample')
        if not self.nea_mic.IsInContact:
            self.nea_mic.AutoApproach(setpoint)
        logger.info(f'Waiting {settle_s} s for the AFM to settle')
        sleep(settle_s)

    def goto_xy(self, x, y):
        # ToDo: units
        if not (0 < x < 100 and 0 < y < 100):
            logger.error('SNOM error: coordinates out of bounds: 0 < (x, y) < 100')
        reached = self.nea_mic.GotoTipPosition(x, y, self.tip_velocity)
        if not reached:
            raise RuntimeError('Did not reach target position')

    def stop(self):
        logger.info('Stopping any Scan')
        self.nea_mic.CancelCurrentProcedure()
        self.nea_mic.RegulatorOff()

    def disconnect(self):
        if self.nea_mic is not None:
            self.stop()
        logger.info('Disconnecting microscope')
        self.nea_mic = self.nea_client.Disconnect()
        self.connected = False

    def set_pshet(self, mod: Demodulation):
        logger.info(f'Settig NeaSNOM modulation to {mod.value}')
        if mod == Demodulation.shd:
            mod_setter = self.nea_mic.PrepareApproachCurveAfm()
        elif mod == Demodulation.pshet:
            mod_setter = self.nea_mic.PrepareApproachCurvePsHet()
        mod_setter.Start()
        sleep(0.1)
        self.nea_mic.CancelCurrentProcedure()

    def prepare_retraction(self, mod: Demodulation, z_size: float, z_res: int, afm_sampling_time: float):
        # ToDo: units
        #  write doc string
        self.tracked_channels = ['Z', 'M1A', 'M1P']
        if mod not in [Demodulation.pshet, Demodulation.shd]:
            logger.error(f'SNOM error: {mod.value} is not implemented')

        if mod == Demodulation.shd:
            self.scan = self.nea_mic.PrepareApproachCurveAfm()
        elif mod == Demodulation.pshet:
            self.scan = self.nea_mic.PrepareApproachCurvePsHet()

        self.scan.set_ApproachCurveHeight(z_size)
        self.scan.set_ApproachCurveResolution(z_res)
        self.scan.set_SamplingTime(afm_sampling_time)

    def prepare_image(self, mod: Demodulation, x_center, y_center, x_size, y_size, x_res, y_res,
                      angle, sampling_time_ms, serpent: bool = False):
        # ToDo: units
        #  write doc string
        if serpent:
            self.tracked_channels = ['Z', 'M1A', 'M1P']
        else:
            self.tracked_channels = ['Z', 'R-Z', 'M1A', 'R-M1A', 'M1P', 'R-M1P']
        if mod not in [Demodulation.pshet, Demodulation.shd]:
            logger.error(f'SNOM error: {mod.value} is not implemented')

        step = x_size / x_res
        vel = step / sampling_time_ms * 1000
        logger.info(f'Tip velocity is set to {vel:.2} um s^-1')
        if vel > 4:
            logger.warning('WARNING: tip velocity > 4 um s^-1')
        if mod == Demodulation.pshet:
            self.scan = self.nea_mic.PreparePsHetScan()
        elif mod == Demodulation.shd:
            self.scan = self.nea_mic.PrepareAfmScan()

        self.scan.set_CenterX(x_center)
        self.scan.set_CenterY(y_center)
        self.scan.set_ScanAreaWidth(x_size)
        self.scan.set_ScanAreaHeight(y_size)
        self.scan.set_ResolutionColumns(x_res)
        self.scan.set_ResolutionRows(y_res)
        self.scan.set_ScanAngle(angle)
        self.scan.set_SamplingTime(sampling_time_ms)
        if serpent:
            self.scan.set_ResolutionRows(y_res * 5)
            self.scan.TakeRows = 1
            self.scan.SkipRows = 4

    def start(self) -> dict:
        image = self.scan.Start()
        data = {ch: image.GetChannel(ch) for ch in self.tracked_channels}
        return data

    def get_current(self) -> (float, float, float, float, float):
        """
        returns tuple (x, y, z, amp, phase)
        """
        x = self.nea_mic.TipPositionX  # um
        y = self.nea_mic.TipPositionY  # um
        z = self.nea_mic.GetChannel('Z').CurrentValue   # um
        amp = self.nea_mic.GetChannel('M1A').CurrentValue  # nm
        phase = self.nea_mic.GetChannel('M1P').CurrentValue  # rad
        return x, y, z, amp, phase
