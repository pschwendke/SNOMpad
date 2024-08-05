import sys
import logging


class LogFilter(logging.Filter):
    def filter(self, record):
        ignored = [
            'Moving to delay position',
            'Waiting for delay stage',
            'Delay stage controller status',
        ]
        if any(i in str(record.getMessage()) for i in ignored):
            return False
        return True


def notebook_logger():
    logger = logging.getLogger()
    logger.setLevel('DEBUG')
    logging.getLogger('snompad.drivers.daq_ctrl').setLevel('INFO')
    logging.getLogger('snompad.drivers.delay_ctrl').setLevel('INFO')

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel('INFO')
    handler.setFormatter(logging.Formatter(fmt='%(levelname)s - %(name)s - %(message)s'))
    handler.addFilter(LogFilter())

    logger.addHandler(handler)

    return logger


def gui_logger():
    pass
