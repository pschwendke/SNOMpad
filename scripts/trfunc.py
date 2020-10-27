#!python3
# trfunc.py: acquire transfer function measurement

import argparse
import logging
from collections import ChainMap
from copy import copy
import sys

import numpy as np
import tqdm
import toml

from trion.expt.acquisition import transfer_func_acq


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


parser = argparse.ArgumentParser(
    description="Perform transfer function measurement using slow passage."
)
# parser.add_argument("-d", "--dev", default="Dev0", help="NI device")
# parser.add_argument("-i", "--input", nargs="+", action="extend")
# parser.add_argument("-o", "--output", nargs="+", action="extend")
# parser.add_argument("--f")
parser.add_argument("-v", "--verbose", action="store_true", help="Logging at DEBUG level.")
parser.add_argument("--noprogress", action="store_false", dest="pbar", help="Don't use a progress bar.")
parser.add_argument("--skip", action="store_true", help="Skip measurement")
parser.add_argument("--print_toml", action="store_true", help="print config in TOML")
parser.add_argument("config", help="Configuration file")
parser.add_argument("filename", help="Output file name")

args = parser.parse_args()

log_fmt="%(levelname)-7s | %(relativeCreated)6d - %(message)s"
log_cfg = {
    "level": logging.DEBUG if args.verbose else logging.INFO,
    "format": log_fmt,
}
logging.basicConfig(handlers=[TqdmLoggingHandler()], **log_cfg)
logger = logging.getLogger()

defaults = {
    "device": "Dev0", # perhaps try to find one...
    "sig_channels": ["ai0"],
    "ref_channels": ["ai1"],
    "write_channels": ["ao0", "ao1"],
    "sample_rate": 1E6,  # 1 MHz
    "f_max": 500E3,  # 500 kHz
    "f_step": 1E3,  # 1 kHz
    "amp": 0.5,
    "offset": 0,
}

with open(args.config) as f:
    cfg_file = toml.load(f)

cfg = {**defaults, **cfg_file}

if args.print_toml:
    print(toml.dumps(cfg))

if args.skip:
    logging.info("Skipping measurement")
    sys.exit()

n_samples = int(cfg["sample_rate"]/cfg["f_step"])
freqs = np.arange(0, cfg["f_max"]+1E-3, cfg["f_step"])
t = np.arange(0, n_samples)/cfg["sample_rate"]
assert freqs[-1] == cfg["f_max"]

pbar = tqdm.tqdm(
    desc="Acquisition in progress...",
    total=len(freqs),
    disable=not args.pbar,
)

read_channels = copy(cfg["sig_channels"])
read_channels.extend(cfg["ref_channels"])
read_channels = list(sorted(read_channels))

measured = transfer_func_acq(
    cfg["device"],
    read_channels=read_channels,
    write_channels=cfg["write_channels"],
    freqs=freqs,
    amp=cfg["amp"],
    offset=cfg["offset"],
    n_samples=n_samples,
    sample_rate=cfg["sample_rate"],
    pbar=pbar,
    logger=logger,
)

logger.info(f"Saving to: {args.filename}")
np.savez_compressed(
    args.filename,
    freqs=freqs, t=t, measured=measured
)
