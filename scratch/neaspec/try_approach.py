# try an approach curve

# I should probably try to list the components somewhere...


import sys
import clr
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import logging
from pprint import pprint
logging.basicConfig(format="[%(asctime)s] %(levelname)-5s : %(message)s", level=logging.DEBUG)

setpoint = 0.8

channels_list = ['Z', 'M1A', 'M1P', 'O0A', 'O2A', 'O2P']

sys.path.append(neaspec_folder)
clr.AddReference("Nea.Client.Hardware")
import Nea.Client.Hardware.SDK as neaSDK  # check what's in there...

logging.info("Connecting...")
neaClient = neaSDK.Connection("nea-server")
neaMic = neaClient.Connect()
time.sleep(0.1)  # ... argh, smells bad already

# cancel if there is something
logging.info("Going to contact")
neaMic.CancelCurrentProcedure()  # can we check what's going on? already
neaMic.RegulatorOff()  # retract. Is this a blocking call? we should time it.
try:
    if not neaMic.IsInContact:  # time this one too, while we're at it...
        neaMic.AutoApproach(setpoint)  # and time this one too!

    logging.info("Waiting a bit")
    time.sleep(5)  # let system settle a bit, for piezo creep I guess?!

    # we need to prepare an approach curve scan, such as:
    # parametrize scan
    logging.info("Preparing scan")
    scan = neaMic.PrepareApproachCurveAfm()
    scan.set_ApproachCurveHeight(0.2)
    scan.set_ApproachCurveResolution(200)
    scan.set_SamplingTime(50)
    
    tracked = {n: neaMic.GetChannel(n) for n in ["Z", "M1A"]}
    
    hb_curve = defaultdict(list) # homebrew
    logging.info("Starting scan")
    image = scan.Start()  # can I pause?!

    t0 = time.time()
    step_period = 0.5
    duty_cycle = 0.25
    while not scan.IsCompleted:
        #logging.info(f"scan.Progress: {scan.Progress}")
        dt = time.time()-t0
        if scan.IsSuspended and (dt % step_period < step_period*duty_cycle):
            scan.Resume()
        elif not scan.IsSuspended and (dt % step_period > step_period*duty_cycle):
            scan.Suspend()
        for n, c in tracked.items():
            v = c.CurrentValue
            hb_curve[n].append(v)
            #logging.info(f"current {n}: {v:.06f}")
        time.sleep(0.01)
    logging.info("Done")

finally:
# carefull! enclose this in a try..finally block!
    neaMic.CancelCurrentProcedure()
    neaMic.RegulatorOff()
    neaClient.Disconnect()

z = np.array(hb_curve["Z"])
a = np.array(hb_curve["M1A"])
np.savez("stepped_approach.npz", a=a, z=z)
# somehow full of nans...

# for channel in channels_list:
#     nea_channel = image.GetChannel(channel)
#     nea_ap_data = nea_channel.GetData()
#     #assert nea_ap_data.Rank == 1
#     shape = [nea_ap_data.GetUpperBound(i) for i in range(nea_ap_data.Rank)]
#     tmp = np.zeros(shape)
#     for i in range(len(shape)):
#         tmp[i] = nea_ap_data[i]
#     data_np[channel] = tmp
#     break
    
#np.savez("approach.npy", **data_np)


# maybe show the plot