# TRION experiment
This repository contains two packages: `trion.data` for data analysis and 
`trion.expt` for the trion experiment. WIP (readme not up to date :))

The data analysis package `trion.data` contains data analysis tools and format
descriptions. It should not depend on `trion.expt` module.

The experimental code, including controllers and gui, is located in `trion.expt`
module. This can depend on the `trion.data` module for specific IO.

# Status

Single-point acquisition possible via command line interface using 
`scripts/single_point.py`. Refer to command line help and source code for
details.
