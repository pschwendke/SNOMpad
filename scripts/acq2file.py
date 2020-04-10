#!python3
# acq2file.py: acquire data to file.
# 

import argparse
import logging
from time import sleep

import numpy as np
from tqdm import tqdm

from trion.expt.buffer import ExtendingArrayBuffer
from trion.expt.daq import DaqController

logging.getLogger("matplotlib").setLevel(logging.WARNING)


parser = argparse.ArgumentParser(
    description="Acquire raw TRIONs data to file.",
)
parser.add_argument("-d", "--dev", default="DevT", help="NI device.")

parser.add_argument(
    "--signals", nargs="+", action="extend",
    help="Signals to acquire. Default to 'sig_A tap_x tap_y'"
)
parser.add_argument(
    "-n", "--n_samples", 
    type=int,
    default=1_000_000, 
    help="Maximum number of shots. Defaults to 1M. If negative, acquire continuously."
)
parser.add_argument("-p", "--pbar", action="store_true", help="Use a progress bar. Doesn't mix well with 'v'.")
parser.add_argument("-v", "--verbose", action="store_true", help="Logging uses debug mode.")
parser.add_argument("--filename", nargs=1, help="output file name")
args = parser.parse_args()

log_fmt = "%(levelname)-5s - %(name)s - %(message)s" if args.verbose else "%(levelname)-5s - %(message)s"
log_cfg = {
    "level": logging.DEBUG if args.verbose else logging.INFO,
    "format": log_fmt,
}
logging.basicConfig(**log_cfg)
logger = logging.getLogger()

if not args.signals:
    args.signals=["sig_A", "tap_x", "tap_y"]

ctrl = DaqController("DevT", clock_channel="")

buffer = ExtendingArrayBuffer(vars=args.signals)#, max_size=args.n_samples)

#t0 = time()
n_read = 0
#setup the progressbar

ctrl.setup(buffer=buffer)
try:
    pbar = tqdm(
        desc="Acquisition in progress",
        total=args.n_samples,
        unit=" samples",
        disable=not args.pbar,
    )
    ctrl.start()
    while True:
        try:
            if  ctrl.is_done() or n_read >= args.n_samples:
                logging.info("Acquisition complete")
                break
            sleep(0.01)
            n = ctrl.reader.read()
            pbar.update(n)
            n_read += n
        except KeyboardInterrupt:
            logging.warning("User interrupting acquisition.")
            break
    #logging.info(f"{n_read}")
finally:
    pbar.close()
    logging.info("Acquisition finished")
    ctrl.stop()
    
    ctrl.close()



#assert not np.any(np.isnan(buffer.buf))

#print(args)
