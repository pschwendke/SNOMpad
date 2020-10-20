# TRION experiment

This repository contains the code necessary for running the TRIONS experiment. 

two packages: `trion.analysis` for data analysis and 
`trion.expt` for the trion experiment. WIP (readme not up to date :))

# Status

## GUI
Main TRIONS user interface is available as `trion/launcher/launch_trion.py`

Command-line scripts are available under `scripts`

Refer to command line help and source code for details.

# Roadmap

Required changes:
- ~~Add ability to save buffer to file.~~
- Make better test suite for buffers.
- Make go, stop, save etc as actions. Good point for performing improvements.
- Design and implement HDF5 buffer.
- Integrate control of SNOM.
