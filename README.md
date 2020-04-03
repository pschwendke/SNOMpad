# TRION experiment
This repository contains two packages: `trion-data` for data analysis and 
`trion-expt` for the trion experiment. WIP (readme not up to date :))

The data analysis package `trion-data` contains data analysis tools and format
descriptions. It should not depend on `trion-expt` module.

The experimental code, including controllers and gui, is located in `trion-expt`
module. This can depend on the `trion-data` module for specific io.

# Project roadmap

Task list:
1. Finish setup. (literally: setup.py)

2. Single channel sHD for live update  
   Upon completion, we can test and calibrate the cantilever modulation.

3. Single channel sHD to file

4. Dual channel sHD

5. Setup balanced detection
   1. test impedence
   2. Add signal types

6. psHet mode. Does NOT actually require balanced detection

7. Pump-probe

When elements become available:
- Test the impendence of diff-sum
- Check connectivity of DAQ wiring
- Test filtering by RLC circuit
- Setup and test timing.
- Design testing and calibration procedure for cantilever modulation.