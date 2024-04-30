import logging
import serial
from time import sleep, monotonic

from ..utility import c_air

logger = logging.getLogger(__name__)

# dicts for decoding status and error codes
status_code = {
    1: 'End of run -',
    2: 'End of run +',
    4: 'ZM (not used)'
}
ctrl_error_code = {
    1: 'End of run negative. -> Check carriage position and cables.',
    2: 'End of run positive. -> Check carriage position and cables.',
    4: 'Current limit. -> Check that carriage is free to move.',
    8: 'RMS current limit. -> Check that carriage is free to move.',
    16: 'Fuse broken. -> Cycle power.',
    32: 'Following error. -> Check carriage freedom, payload parameter. Restore factory settings. ',
    64: 'Time out homing. -> Check acceleration. Restore factory settings.',
    128: 'Bad SmartGate. -> Verify stage ID. Disable smart stage verification.',
    256: 'Vin sense error (DC voltage too low). -> Restore factory settings.',
    512: 'Motor Driver Over Temperature Warnin. -> Restore factory settings.g',
    1024: 'Motor Driver Overcurrent shut-down or GVDD undervoltage protection occurred.'
         '-> Restore factory settings.',
    2048: 'Motor thermistance error',
    4096: 'Parameters EEPROM error. -> Restore factory settings.',
    8192: 'Parameters Range error. -> Restore factory settings.',
    16384: 'Sin/Cos radius error. -> Check encoder cable.',
    32768: 'Encoder quadrature error. -> Check encoder cable',
    65536: 'AquadB output error. -> Check manual.',
    131072: 'ISR ratio error. -> Contact Newport',
    262144: 'Motion done timeout error. -> Check manual.',
    524288: 'Power error. -> Check carriage freedom. Restore factory settings. Contact Newport.'
}
ctrl_state_code = {
    10: 'NOT INITIALIZED: after reset',
    11: 'NOT INITIALIZED: after CONFIG state',
    12: 'NOT INITIALIZED: after INITIALIZING state',
    13: 'NOT INITIALIZED: after NOT_REFERENCED state',
    14: 'NOT INITIALIZED: after HOMING state',
    15: 'NOT INITIALIZED: after MOVING state',
    16: 'NOT INITIALIZED: after READY state',
    17: 'NOT INITIALIZED: after DISABLE state',
    18: 'NOT INITIALIZED: after JOGGING state',
    19: 'NOT INITIALIZED: error, Stage type not valid',
    20: 'CONFIGURATION',
    30: 'INITIALIZING: launch by USB',
    31: 'INITIALIZING: launch by Remote Control',
    40: 'NOT_REFERENCED',
    50: 'HOMING: launch by USB',
    51: 'HOMING: launch by Remote Control',
    60: 'MOVING',
    70: 'READY: after HOMING state',
    71: 'READY: after MOVING state',
    72: 'READY: after DISABLE state',
    73: 'READY: after JOGGING state',
    80: 'DISABLE: after READY state',
    81: 'DISABLE: after MOVING state',
    82: 'DISABLE: after JOGGING state',
    90: 'JOGGING: after READY state',
    91: 'JOGGING: after DISABLE state'
}
last_error_code = {
    'A': 'Unknown Message Code',
    'B': 'Parameter out of Limits',
    'C': 'Scaling parameters dependence error',
    'D': 'Function Execution not Allowed',
    'E': 'Home sequence already started',
    'F': 'Function Execution not Allowed in NOT INITIALIZED mode',
    'G': 'Function Execution not Allowed in INITIALIZING mode',
    'H': 'Function Execution not Allowed in NOT REFERENCED mode',
    'I': 'Function Execution not Allowed in CONFIG mode',
    'J': 'Function Execution not Allowed in DISABLE mode',
    'K': 'Function Execution not Allowed in READY mode',
    'L': 'Function Execution not Allowed in HOMING mode',
    'M': 'Function Execution not Allowed in MOVING mode',
    'N': 'Function Execution not Allowed in JOGGING mode',
    'O': 'Target Position out of limit',
    'P': 'Current position out of software limit',
    'Q': 'Motion Timeout',
    'R': 'Motion Error',
    'S': 'USB Communication ERROR',
    'T': 'Gathering not completed',
    'U': 'Error during EEPROM access',
    'V': 'Estimated motion time >timeout'
}


class DLStage:
    def __init__(self, port='COM4'):
        """ Class DLstage to control a Newport DL325 delay stage. The carriage position is read and set in mm.
        The velocity is set in mm/s. The delay is calculated relative to reference position.
        ---> When moving carriage with move_abs() or move_rel(), a unit must be specified (m, mm, s, fs, ps) <---
        One beam pass through the delay line is considered -> path difference = (reference pos. - carriage pos.) * 2

        port: COM port where controller is located. For testing, pass a MockStage object.
        """
        config = {
            # 'baudrate': '9600',
            # 'bytesize': 8,
            # 'parity': 'PARITY_NONE',
            # 'stopbits': 1,
            'timeout': 1  # s
        }
        try:
            if port[:3] != 'COM':  # ToDo: Rewrite this for testing
                self.ser = port
            else:
                self.ser = serial.Serial(port, **config)
                self.ser.reset_input_buffer()
            self.ser.write(b'TE\r\n')  # TE calls last error and return starts with 'TE'. Used to check communication.
            returns = self.ser.readline().decode()
            if not returns[:2] == 'TE':
                raise RuntimeError(f'Expected reply to TE, got {returns[:2]} instead.')
        except serial.SerialException:
            logger.error('Serial Exception: delay stage can not be found or can not be configured')
        logger.info(f'Connected to delay stage on port {port}')

    def __del__(self):
        logger.debug('DLStage.__del__()')
        self.disconnect()

    def disconnect(self):
        logger.info('Disconnected from delay stage')
        self.ser.close()

    # COMMUNICATION WITH DELAY STAGE ##################################################################################
    def command(self, cmd: str, val: str = '') -> None or str:
        """ Sends commands to the delay stage controller via serial port and returns the returned string.
        Implemented with return: TS, TP, RF?, MM?
        Implemented without return: RS, PWnn, IE, OR, RFnn, MMnn, PAnn, PRnn, ST
        """
        eol = '\r\n'
        msg = cmd + val + eol
        logger.debug(f'Command to delay stage: {msg}')
        self.ser.write(msg.encode())

        if cmd in ['TS', 'TP'] or (cmd in ['RF', 'MM', 'VA'] and val == '?'):  # return value expected
            returns = self.ser.readline()
            logger.debug(f'Return from delay stage after command ({msg.encode()}): {returns}')
            returns = returns.decode().strip()
            ret_cmd, ret_msg = returns[:len(cmd)], returns[len(cmd):]
            if not ret_cmd == cmd:
                raise RuntimeError(f'Expected reply to {cmd}, got {ret_cmd} instead.')
            ret = ret_msg
        elif cmd in ['RS', 'IE', 'OR', 'ST'] or (cmd in ['MM', 'PW', 'PA', 'PR', 'RF', 'VA'] and val != '?'):  # no returns
            ret = None
        else:
            raise NotImplementedError(f'Passed command {cmd} is not implemented.')

        # check whether error was memorized
        self.ser.write(b'TE\r\n')
        error_code = self.ser.readline().decode()
        if not error_code[:2] == 'TE':
            raise RuntimeError(f'Expected reply to TE, got {error_code[:2]} instead.')
        if error_code[2] != '@':
            error_msg = last_error_code[error_code[2]]
            logger.error(f'Error communicating with delay stage: {error_msg}')

        return ret

    def get_status(self) -> (int, int, int):
        """ Asks for controller status. Returns are 8 hexadecimal characters: 1: status, 2-6: error, 7-8: controller
        state. The codes are returned as integers and can be decoded with the dictionaries status_code, ctrl_error_code,
        ctrl_state_code. The error codes are additive, i.e. multiple errors can be encoded in the same returned integer.
        """
        returns = self.command('TS')
        status = int(returns[0], base=16)
        ctrl_error = int(returns[1:6], base=16)
        ctrl_state = int(returns[-2:], base=16)
        logger.debug('Delay stage controller state: ' + ctrl_state_code[ctrl_state])
        return status, ctrl_error, ctrl_state

    def log_status(self):
        carr, err, state = self.get_status()
        if carr != 0:
            logger.info('Delay stage carriage status: ' + status_code[carr])
        if state != 0:
            logger.info('Delay stage controller status: ' + ctrl_state_code[state])
        if err != 0:
            for mask, msg in ctrl_error_code.items():
                if err & mask:
                    logger.info('Delay stage error message: ' + msg)

    # PREPARE DELAY STAGE #############################################################################################
    def reset(self):
        self.command('RS')  # reset controller, go into 'not initialized' state
        self.ser.write(b'PW1\r\n')  # go into 'config' state, command() is not used, getting errors returns to not init
        self.ser.write(b'FSR\r\n')  # restore factory settings
        self.command(cmd='PW', val='0')  # go back to 'not initialized' state

    def initialize(self):
        status, _, _ = self.get_status()
        if status == 0:
            self.command('IE')
        else:
            logger.error('Could not initialize delay stage. Check if carriage is at end of stage (it should not be).')

    def search_home(self):
        self.command('OR')

    @ property
    def is_ready(self) -> bool:
        """ returns true if controller is in ready state
        """
        _, _, state = self.get_status()
        return state // 10 == 7  # see ctrl_state_code: state in 70's means 'ready'

    def prepare(self) -> bool:
        _, _, state = self.get_status()
        if state < 20:
            self.initialize()
            self.wait_for_stage()
        if state < 60:
            self.search_home()
            self.wait_for_stage()
        if state >= 80:
            self.motor_enabled = True
            self.wait_for_stage()    
        return self.is_ready

    # MOVING DELAY STAGE ##############################################################################################
    @ property
    def velocity(self) -> float:
        """ returns carriage velocity in mm/s.
        """
        vel = self.command('VA', val='?')
        vel = float(vel)  # mm/s
        return vel

    @ velocity.setter
    def velocity(self, vel: float):
        """ sets carriage velocity in mm/s.
        """
        vel = str(vel)  # mm/s
        self.command(cmd='VA', val=vel)

    @ property
    def position(self) -> float:
        """ returns position of carriage on axis in mm.
        """
        pos = self.command('TP')  # str(mm)
        pos = float(pos)  # mm
        return pos

    # @ property
    # def delay(self) -> float:
    #     """ returns delay time added by delay line in seconds. One pass (to retro-reflector and back) is considered.
    #     """
    #     path_diff = (self.position - self.reference) * 2  # m
    #     delay = path_diff / c_air  # s
    #     return delay

    @ property
    def reference(self) -> float:
        """ returns reference carriage position in mm.
        """
        ref = self.command(cmd='RF', val='?')  # str(mm)
        ref = float(ref)  # mm
        return ref

    @ reference.setter
    def reference(self, pos: float):
        """ sets reference carriage position in mm.
        """
        pos = str(pos)  # str(mm)
        self.command(cmd='RF', val=pos)

    @ property
    def motor_enabled(self) -> bool:
        """ Asks whether motor on axis is enabled. True: motor enabled, do not move by hand. False: motor disabled,
        okay to move by hand.
        """
        status = self.command(cmd='MM', val='?')
        return bool(int(status))

    @ motor_enabled.setter
    def motor_enabled(self, enable: bool):
        """ Sets whether motor on axis in enabled. True: motor enabled, do not move by hand. False: motor disabled,
        okay to move by hand.
        """
        self.command(cmd='MM', val=str(int(enable)))

    def prepare_move(self, velocity: float):
        """ prepares axis for move at velocity (mm/s), i.e. performs initialization and homing to put
        controller in ready state.
        """
        _, _, state = self.get_status()
        state = state // 10
        if state == 7:  # ready state
            self.velocity = velocity
            return
        elif state == 6:  # moving
            self.stop()
        elif state == 8:  # disable state, i.e. axis motor disabled
            self.motor_enabled = True
        elif state < 4:  # not initialized or referenced
            self.prepare()
        self.wait_for_stage()
        if self.is_ready:
            self.velocity = velocity
            return
        else:
            logger.error('Delay stage error: could not prepare to move')

    def move_abs(self, val: float, unit: str, velocity: float = 100):
        """ Moves the carriage to a certain position. The position can be given as position on axis or delay time.
        Delay times are calculated taking reference position as t_0.

        Parameters
        ----------
        val: float
            position to where the carriage should move.
        unit: str
            unit of either be position ('m', 'mm'), or delay ('s', 'fs', 'ps')
        velocity: float
            velocity of move in mm/s
        """
        if unit == 'm':
            val *= 1e3  # mm
        elif unit == 'mm':
            pass
        elif unit == 's':
            val = self.reference - 0.5 * val * c_air * 1e3  # mm
        elif unit == 'fs':
            val = self.reference - 0.5 * val * 1e-15 * c_air * 1e3  # mm
        elif unit == 'ps':
            val = self.reference - 0.5 * val * 1e-12 * c_air * 1e3  # mm
        else:
            raise TypeError(f"unit must be in ['m', 'mm', 'fs', 'ps', 's'], got {unit} instead.")
        logger.info(f'Moving to delay position {val:.3f} mm with {velocity} mm/s')
        self.prepare_move(velocity=velocity)
        self.command(cmd='PA', val=str(val))

    def move_rel(self, step: float, unit: str, velocity: float = 100):
        """ Moves the carriage a certain distance. Differences in delay time and position on axis can be given.

        Parameters
        ----------
        step: float
            difference in position that the carriage should move
        unit: str
            unit of either be position ('m', 'mm'), or delay ('s', 'fs', 'ps')
        velocity: float
            velocity of move in mm/s
        """
        if unit == 'm':
            step *= 1e3  # mm
        elif unit == 'mm':
            pass
        elif unit == 's':
            step = -0.5 * step * c_air * 1e3  # mm
        elif unit == 'fs':
            step = -0.5 * step * 1e-15 * c_air * 1e3  # mm
        elif unit == 'ps':
            step = -0.5 * step * 1e-12 * c_air * 1e3  # mm
        else:
            raise TypeError(f"unit must be in ['m', 'mm', 'fs', 'ps', 's'], got {unit} instead.")
        step = str(step)
        logger.info(f'Moving delay position by {step:.2} mm with {velocity} mm/s')
        self.prepare_move(velocity=velocity)
        self.command(cmd='PR', val=step)

    def wait_for_stage(self, timeout: float = 10.0):
        """ Waits for stage during moving or homing, until controller status is 'ready' or timeout (s) is reached.
        """
        logger.info('Waiting for delay stage ...')
        start_time = monotonic()
        done = False
        while not done:
            _, _, status = self.get_status()
            if 50 <= status < 70 or status == 30:   # moving, homing or initializing
                if monotonic() - start_time > timeout:
                    self.stop()
                    logger.error('Delay stage: timeout while moving')
                    return False
                sleep(.05)
            else:
                done = True

    def stop(self):
        self.command('ST')
