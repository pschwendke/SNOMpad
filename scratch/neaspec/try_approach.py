# try an approach curve

# I should probably try to list the components somewhere...

neaspec_folder = '//nea-server/updates/SDK'

import sys
import clr
import time

setpoint = 0.8

sys.path.append(neaspec_folder)
clr.addReference("Nea.Client.Hardware")
import Nea.Client.Hardware.SDK as neaSDK  # check what's in there...

neaClient = neaSDK.Connection("nea-server")
neaMic = neaClient.Connect()
time.sleep(0.1)  # ... argh, smells bad already

# cancel if there is something
neaMic.CancelCurrentProcedure()  # can we check what's going on? already
neaMic.RegulatorOff()  # retract. Is this a blocking call? we should time it.
if not neaMic.IsInContact:  # time this one too, while we're at it...
    neaMic.AutoApproach(setpoint)  # and time this one too!

time.sleep(5)  # let system settle a bit, for piezo creep I guess?!

# we need to prepare an approach curve scan, such as:
#  scan =  neaMic.prepareApproachSomething...
# parametrize scan

image = scan.Start()  # can I pause?!
while not scan.isCompleted:
    time.sleep(0.1)

# do something with the data.

neaMic.RegulatorOff()
neaClient.Disconnect()

# maybe show the plot