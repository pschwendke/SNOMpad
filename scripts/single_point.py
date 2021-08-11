#!python3
# single_point.py: acquire data to file.
#
import argparse
import logging
from time import sleep

import numpy as np
import tqdm

from trion.analysis.signals import Signals
from trion.analysis.io import export_data
from trion.expt.acquisition import single_point
from trion.expt.buffer import ExtendingArrayBuffer
from trion.expt.daq import DaqController

# TODO: add support for csv export.

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
    help="Signals to acquire. Default to 'sig_a tap_x tap_y'"
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
parser.add_argument(
    "--truncate", action="store_true", help="Truncate output data to the exact number of requested points. Defaults to False."
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
    args.signals=[Signals.sig_a, Signals.tap_x, Signals.tap_y]
if args.n_samples < 0:
    # infinite acquisition
    pbar_lim = np.inf#
else:
    pbar_lim = args.n_samples

print("Initializing single-point TRIONs acquisition.")
pbar = tqdm.tqdm(
    desc="Acquisition in progress...",
    total=pbar_lim,
    unit=" samples",
    disable=not args.pbar,
)
with pbar:
    data = single_point(args.dev, args.signals, args.n_samples, args.clock,
                        args.truncate, pbar)

logging.info(f"Saving data to: {args.filename}")
logging.debug(f"Data shape: {data.shape}")
export_data(args.filename, data, header=[s for s in args.signals])
