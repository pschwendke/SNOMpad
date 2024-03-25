""" Delay stage simulator and tests to develop and maintain delay stage implementation in snompad.expt.dl_ctrl
"""

import numpy as np
import threading
import pytest
from serial import SerialException
from time import sleep
from itertools import product

from trion.expt.dl_ctrl import DLStage, c_air, ctrl_state_code, ctrl_error_code, status_code


def int2hex(num: int, digits: int) -> str:
    """ converts and integer to a string in hex basis with given digits
    """
    string = hex(num)
    string = string[2:]
    assert len(string) <= digits
    for i in range(digits-len(string)):
        string = '0' + string
    return string


# simulation of Newport DL325 delay stage controller
class MockStage:
    def __init__(self):
        self.output_buffer = []
        self.last_error = b'@'
        self.carr_stat = 0
        self.ctrl_err = 0
        self.ctrl_state = 10
        self.position = 150
        self.reference = 0
        self.home = -1.5
        self.lolim = -1.6
        self.hilim = 325
        self.stopped = False
        self.connected = True

    # IO AND ERROR HANDLING ###########################################################################################
    def readline(self):
        """ simulates interaction with COM port
        """
        if not self.connected:
            raise SerialException
        elif len(self.output_buffer) == 0:
            return b''
        else:
            return self.output_buffer.pop(0)

    def readlines(self):
        """ simulates interaction with COM port
        """
        if not self.connected:
            raise SerialException
        elif len(self.output_buffer) == 0:
            return b''
        else:
            ret = self.output_buffer
            self.output_buffer = []
            return ret

    def write(self, msg: bytes):
        """ simulates writing to COM port
        """
        commands = {
            b'IE': self.ie,
            b'FS': self.fs,
            b'MM': self.mm,
            b'OR': self.search_home,
            b'PA': self.pa,
            b'PR': self.pr,
            b'PW': self.pw,
            b'RF': self.rf,
            b'RS': self.rs,
            b'ST': self.st,
            b'TE': self.te,
            b'TP': self.tp,
            b'TS': self.ts,
        }
        if not self.connected:
            raise SerialException
        elif msg[-2:] == b'\r\n':
            msg = msg[:-2]
            try:
                ctrl_daemon = threading.Thread(target=commands[msg[:2]], args=(msg,), daemon=True)
                ctrl_daemon.start()
            except KeyError:
                raise
                # pass

    def not_allowed(self):
        """ memorizes error if command is not allowed in current state
        """
        state = self.ctrl_state // 10
        state_errs = {
            1: b'F',
            2: b'I',
            3: b'G',
            4: b'H',
            5: b'L',
            6: b'M',
            7: b'K',
            8: b'J',
            9: b'N',
        }
        self.last_error = state_errs[state]

    def close(self):
        self.connected = False

    # SIMULATION OF PROCESSES #########################################################################################
    def initializer(self):
        """ simulates initialization sequence
        """
        self.ctrl_state = 30
        sleep(4)
        self.ctrl_state = 40

    def mover(self, pos: float):
        """ simulates movement of carriage
        """
        v = 100  # mm/s
        dt = 0.005
        direction = np.sign(pos - self.position)
        steps = np.arange(self.position, pos, v * dt * direction)  # 0.5 mm per step
        self.stopped = False
        self.ctrl_state = 60
        for p in steps:
            sleep(dt)
            self.position = p
            if self.stopped:
                break
        if not self.stopped:
            self.position = pos
        self.ctrl_state = 71

    def homer(self):
        """ simulates homing sequence
        """
        self.mover(pos=self.home)
        self.ctrl_state = 70

    # SERIAL COMMANDS #################################################################################################
    def ie(self, msg: bytes):
        sleep(.01)
        if not 10 <= self.ctrl_state < 19:
            self.not_allowed()
        elif self.ctrl_err != 0 or self.carr_stat != 0:
            self.last_error = b'D'
        else:
            init = threading.Thread(target=self.initializer(), daemon=True)
            init.start()

    def fs(self, msg: bytes):
        sleep(.01)
        if msg != b'FSR':
            raise NotImplementedError
        if self.ctrl_state != 20:
            self.not_allowed()

    def mm(self, msg: bytes):
        sleep(.01)
        if not 70 <= self.ctrl_state < 90:
            self.not_allowed()
        elif msg == b'MM?':
            if self.ctrl_state // 10 == 7:
                self.output_buffer.append(b'MM1')
            else:
                self.output_buffer.append(b'MM0')
        elif msg == b'MM0':
            self.ctrl_state = 80
        elif msg == b'MM1':
            self.ctrl_state = 72
    
    def search_home(self, msg: str):  # or
        sleep(.01)
        if self.ctrl_state != 40:
            self.not_allowed()
        elif msg == b'OR':
            movement = threading.Thread(target=self.homer, daemon=True)
            movement.start()
            sleep(.01)   # workaround, since mover would define state as moving. I know. Not very elegant...
            self.ctrl_state = 50

    def pa(self, msg: bytes):
        sleep(.01)
        if self.ctrl_state // 10 != 7:
            self.not_allowed()
        else:
            try:
                pos = float(msg[2:])
            except ValueError:
                self.last_error = b'A'
                return
            if pos < self.lolim or pos > self.hilim:
                self.last_error = b'O'
            else:
                movement = threading.Thread(target=self.mover, args=(pos,), daemon=True)
                movement.start()

    def pr(self, msg: bytes):
        sleep(.01)
        if self.ctrl_state // 10 != 7:
            self.not_allowed()
        else:
            try:
                pos = float(msg[2:].decode())
            except ValueError:
                self.last_error = b'A'
                return
            pos += self.position
            if pos < self.lolim or pos > self.hilim:
                self.last_error = b'O'
            else:
                movement = threading.Thread(target=self.mover, args=(pos,), daemon=True)
                movement.start()
    
    def pw(self, msg: bytes):
        sleep(.01)
        if msg == b'PW1':
            if 10 <= self.ctrl_state < 19:
                self.ctrl_state = 20
            else:
                self.not_allowed()
        elif msg == b'PW0' and self.ctrl_state == 20:
            self.ctrl_state = 11
        else:
            self.last_error = b'A'

    def rf(self, msg: bytes):
        sleep(.01)
        if msg == b'RF?':
            ret = f'RF{self.reference}\r\n'
            self.output_buffer.append(ret.encode())
        else:
            try:
                ref = float(msg[2:])
                self.reference = ref
            except ValueError:
                pass
    
    def rs(self, msg: bytes):
        sleep(.01)
        self.__init__()
    
    def st(self, msg: bytes):
        sleep(.01)
        if 60 <= self.ctrl_state < 90:
            self.stopped = True
        else:
            self.not_allowed()

    def te(self, msg: bytes):
        sleep(.01)
        ret = f'TE{self.last_error.decode()}\r\n'
        self.output_buffer.append(ret.encode())
        self.last_error = b'@'

    def tp(self, msg: bytes):
        sleep(.01)
        if msg == b'TP':
            ret = f'TP{self.position:.6f}\r\n'
            self.output_buffer.append(ret.encode())
    
    def ts(self, msg: bytes):
        sleep(.01)
        ret = str(self.carr_stat) + int2hex(self.ctrl_err, 5) + int2hex(self.ctrl_state, 2)
        ret = 'TS' + ret + '\r\n'
        self.output_buffer.append(ret.encode())


# TESTING SUITE #######################################################################################################
def test_get_pos():
    mock = MockStage()
    set_pos = np.random.uniform(0, .325)
    set_ref = np.random.uniform(0, .325)
    mock.position = set_pos * 1e3  # mm
    mock.reference = set_ref * 1e3  # mm
    dls = DLStage(port='', mock=mock)

    get_pos = dls.position
    get_delay = dls.delay
    set_delay = 2 * (set_pos - set_ref) / c_air

    assert np.isclose(set_pos, get_pos, atol=1e-9)
    assert np.isclose(get_delay, set_delay, atol=1e-9)


def test_move_absolute_m():
    mock = MockStage()
    mock.ctrl_state = 70  # put stage in is_ready mode
    dls = DLStage(port='', mock=mock)

    new_pos = np.random.uniform(0, .32)
    dls.move_abs(new_pos, 'm')
    dls.wait_for_stage()

    assert np.isclose(dls.position, new_pos, atol=1e-9)


def test_move_absolute_s():
    mock = MockStage()
    mock.ctrl_state = 70  # put stage in is_ready mode
    dls = DLStage(port='', mock=mock)

    new_delay = np.random.uniform(-10e-12, 100e-12)
    dls.move_abs(new_delay, 's')
    dls.wait_for_stage()

    assert np.isclose(dls.delay, new_delay, atol=1e-12)


def test_move_relative_m():
    mock = MockStage()
    mock.ctrl_state = 70  # put stage in is_ready mode
    dls = DLStage(port='', mock=mock)

    old_pos = mock.position * 1e-3  # m
    d_pos = np.random.uniform(-0.1, 0.1)
    dls.move_rel(d_pos, 'm')
    dls.wait_for_stage()

    assert np.isclose(dls.position, old_pos + d_pos, atol=1e-9)


def test_move_relative_s():
    mock = MockStage()
    mock.ctrl_state = 70  # put stage in is_ready mode
    dls = DLStage(port='', mock=mock)

    old_delay = 2 * (mock.position - mock.reference) * 1e-3 / c_air
    d_delay = np.random.uniform(-10e-12, 10e-12)
    dls.move_rel(d_delay, 's')
    dls.wait_for_stage()

    assert np.isclose(dls.delay, old_delay + d_delay, atol=1e-12)


def test_set_reference():
    mock = MockStage()
    dls = DLStage(port='', mock=mock)

    new_ref = np.random.uniform(0, .325)
    dls.reference = new_ref
    assert np.isclose(mock.reference * 1e-3, new_ref, atol=1e-9)


def test_get_reference():
    mock = MockStage()
    dls = DLStage(port='', mock=mock)

    ref = mock.reference
    assert np.isclose(dls.reference * 1e-3, ref, atol=1e-9)


def test_enable_motor():
    mock = MockStage()
    mock.ctrl_state = 70  # status: is_ready
    dls = DLStage(port='', mock=mock)

    dls.motor_enabled = False
    sleep(.1)
    assert mock.ctrl_state == 80
    assert dls.motor_enabled is False

    dls.motor_enabled = True
    sleep(.1)
    assert mock.ctrl_state == 72
    assert dls.motor_enabled is True


def test_stop():
    mock = MockStage()
    mock.ctrl_state = 70  # status: is_ready
    mock.position = 0
    dls = DLStage(port='', mock=mock)

    dls.move_abs(.320, 'm')
    sleep(1)
    dls.stop()
    dls.wait_for_stage()

    assert 0 < mock.position < 320
    assert mock.ctrl_state == 71  # is_ready after moving


def test_reset():
    mock = MockStage()
    mock.ctrl_state = 70  # status: is_ready
    mock.hilim = 200
    mock.lolim = 100
    dls = DLStage(port='', mock=mock)

    dls.reset()
    sleep(.5)
    assert mock.ctrl_state == 11
    assert mock.lolim == -1.6
    assert mock.hilim == 325


def test_initialize():
    mock = MockStage()
    dls = DLStage(port='', mock= mock)

    assert mock.ctrl_state == 10
    dls.initialize()
    sleep(.1)
    assert mock.ctrl_state == 30
    sleep(5)
    assert mock.ctrl_state == 40


def test_homing():
    mock = MockStage()
    mock.ctrl_state = 40
    dls = DLStage(port='', mock=mock)

    dls.search_home()
    sleep(.2)
    assert mock.ctrl_state == 50
    dls.wait_for_stage()
    assert mock.ctrl_state == 70
    assert mock.position == mock.home


def test_check_ready():
    mock = MockStage()
    dls = DLStage(port='', mock=mock)
    assert not dls.is_ready
    mock.ctrl_state = 70
    assert dls.is_ready
    mock.ctrl_state = 71
    assert dls.is_ready


def test_get_status():
    mock = MockStage()
    dls = DLStage(port='', mock=mock)
    for c in product([0, 1], [0, 256, 2048], [10, 17, 71]):
        mock.carr_stat, mock.ctrl_err, mock.ctrl_state = c
        carr_stat, ctrl_err, ctrl_state = dls.get_status()
        assert (carr_stat, ctrl_err, ctrl_state) == c


def test_print_status(capsys):
    mock = MockStage()
    dls = DLStage(port='', mock=mock)
    for c in product([1, 2], [32, 256, 2048], [10, 17, 71]):
        mock.carr_stat, mock.ctrl_err, mock.ctrl_state = c
        dls.log_status()
        sleep(.1)
        captured = capsys.readouterr()
        assert status_code[c[0]] in captured.out
        assert ctrl_error_code[c[1]] in captured.out
        assert ctrl_state_code[c[2]] in captured.out


def test_disconnect():
    mock = MockStage()
    dls = DLStage(port='', mock=mock)

    dls.disconnect()
    with pytest.raises(SerialException):
        dls.ser.readline()


def test_error_hander():
    pass


def test_error_limit():
    # ToDo: check how the real controller reacts
    pass


def test_error_position():
    pass
