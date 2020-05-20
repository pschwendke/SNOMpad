#!python3
# single_point.py: acquire data to file.
#
import argparse
import logging
from time import sleep

import numpy as np
import tqdm

from trion.analysis.signals import Signals
from trion.expt.buffer import ExtendingArrayBuffer
from trion.expt.daq import DaqController

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

logging.getLogger("matplotlib").setLevel(logging.WARNING)

parser = argparse.ArgumentParser(
    description="Acquire raw TRIONs data to file.",
)
parser.add_argument("-d", "--dev", default="DevT", help="NI device.")
parser.add_argument(
    "--signals", nargs="+", action="extend",
    type=Signals,
    choices=[s for s in Signals],
    metavar="sig",
    help="Signals to acquire. Default to 'sig_A tap_x tap_y'"
)
parser.add_argument(
    "-c", "--clock", default="", help="Sample clock channel. Defaults to software triggering ('')"
)
parser.add_argument(
    "-n", "--n_samples", 
    type=int,
    default=1_000_000, 
    help="Maximum number of shots. Defaults to 1M. If negative, acquire continuously."
)
parser.add_argument("--noprogress", action="store_false", dest="pbar", help="Use a progress bar.")
parser.add_argument("-v", "--verbose", action="store_true", help="Logging uses debug mode.")
parser.add_argument("--filename", help="output file name. Defaults to `trions.npy`", default="trions.npy")
args = parser.parse_args()

log_fmt = "%(levelname)-5s - %(name)s - %(message)s" if args.verbose else "%(levelname)-5s - %(message)s"
log_cfg = {
    "level": logging.DEBUG if args.verbose else logging.INFO,
    "format": log_fmt,
}
logging.basicConfig(handlers=[TqdmLoggingHandler()], **log_cfg)

if not args.signals:
    args.signals=[Signals.sig_A, Signals.tap_x, Signals.tap_y]
if args.n_samples < 0:
    # infinite acquisition
    acq_lim = np.inf
    pbar_lim = np.inf#
else:
    acq_lim = args.n_samples
    pbar_lim = args.n_samples

print("Initializing single-point TRIONs acquisition.")
pbar = tqdm.tqdm(
    desc="Acquisition in progress...",
    total=pbar_lim,
    unit=" samples",
    disable=not args.pbar,
)
ctrl = DaqController(args.dev, clock_channel="")
buffer = ExtendingArrayBuffer(vars=args.signals, max_size=acq_lim)
n_read = 0
ctrl.setup(buffer=buffer).start()
try:
    while True:
        try:
            if ctrl.is_done() or n_read >= acq_lim:
                break
            sleep(0.01)
            n = ctrl.reader.read()
            pbar.update(n)
            n_read += n
        except KeyboardInterrupt:
            logging.warning("Acquisition interrupted by user.")
            break
finally:
    ctrl.stop()#.close()
    ctrl.close()
    pbar.close()
    logging.info("Acquisition finished.")

logging.info(f"Saving data to: {args.filename}")
np.save(args.filename, buffer.buf, allow_pickle=False)
