from os import truncate
import sys
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import logging
from trion.expt.acquisition import single_point, Signals
from trion.analysis.io import export_data
from pprint import pprint
import tqdm

import clr
neaspec_folder = '//nea-server/updates/SDK'
sys.path.append(neaspec_folder)
clr.AddReference("Nea.Client.Hardware")
import Nea.Client.Hardware.SDK as neaSDK

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


logging.basicConfig(format="[%(asctime)s] %(levelname)-5s %(name)s: %(message)s", level=logging.DEBUG,
handlers=[TqdmLoggingHandler()])
logging.getLogger("snompad.expt").setLevel(logging.WARNING)

# # # # # # # #   PARAMETERS
setpoint = 0.8
tracked_names = ["Z", "M1A"]
height = 0.2 # microns
npix = 101
targets = np.linspace(0, height, npix, endpoint=True)[:-1]
afm_sampling_time = 50
npts_daq = 100_000
daq_name = "Dev1"
signals = [Signals.sig_a, Signals.tap_x, Signals.tap_y]
clock_chan = "pfi0"
truncate=True


# # # # # # # #   Script
logging.info("Connecting...")
neaClient = neaSDK.Connection("nea-server")
neaMic = neaClient.Connect()
time.sleep(0.1)  # ... argh, smells bad already
tracked_channels = {n: neaMic.GetChannel(n) for n in tracked_names}


logging.info("Going to contact")

neaMic.CancelCurrentProcedure() 
neaMic.RegulatorOff() 
z_channel = tracked_channels["Z"]
off_z = z_channel.CurrentValue
logging.info("Z value in off state: %0.6f", off_z)
try:
    if not neaMic.IsInContact: 
        neaMic.AutoApproach(setpoint)

    logging.info("Waiting a bit")
    time.sleep(5)  # let system settle a bit, for piezo creep I guess?!
    logging.info("Preparing scan")
    scan = neaMic.PrepareApproachCurveAfm()
    scan.set_ApproachCurveHeight(height)
    scan.set_ApproachCurveResolution(npix)
    scan.set_SamplingTime(afm_sampling_time)

    logging.info("Starting scan")
    image = scan.Start()  

    
    afm_tracking = []
    init_z = z_channel.CurrentValue
    pbar = tqdm.tqdm(targets)
    for i, t in enumerate(pbar):
        
        
        #logging.info("Going to position: %f", t)
        
        z = z_channel.CurrentValue
        dist = abs(z-init_z)    
        while dist < t:
            z = z_channel.CurrentValue
            dist = abs(z-init_z)
            pbar.set_postfix(target=t,dist=dist, z=z)
            time.sleep(0.01)
            if scan.IsCompleted:
                logging.warning("Scan is completed while there are targets. (%d of %d)", i, len(targets))
                break
        if scan.IsCompleted:
            break
        scan.Suspend()
        pre_value = [c.CurrentValue for c in tracked_channels.values()]
        data = single_point(daq_name, signals, npts_daq, clock_chan)
        post_values = [c.CurrentValue for c in tracked_channels.values()]
        export_data(f"step_approach_{i:03d}.npz", data, signals)
        afm_tracking.append([t] + pre_value + post_values)
        scan.Resume()

    while not scan.IsCompleted:
        time.sleep(1)
finally:
    # carefull! enclose this in a try..finally block!
    neaMic.CancelCurrentProcedure()
    neaMic.RegulatorOff()
    neaClient.Disconnect()

afm_tracking = np.array(afm_tracking)
np.savetxt("afm_tracking.txt", afm_tracking, fmt="%.06f", header="Z_pre M1A_pre Z_post M1A_post")
    

    
logging.info("Done")