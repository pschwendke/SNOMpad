# trion-data
This package contains tools designed to work with the TRION data.

The experimental variables can be divided in two categories: signals and 
parameters. Signals vary at every laser shot. The signals must be acquired 
simultaneously at every laser shot, yielding a dense table. The signals are the
optical signals, the modulation signals and the chopper
status. The parameters are varied imperatively by the acquisition controller. 
They do not vary shot-to-shot. The slow-scan variables are
the AFM positions, pump-probe delay and interferometer position in spectroscopy
mode.

## Variables
The data follows the following conventions. The experimental setup determines 
the set of variables. This is a bijective mapping: the experimental setup can 
be determined from the set of acquisition variables present in a dataset.
The possible variables are:

### Optical signals (units: Volts)
* `sig_a` : Optical signal A. Used for single channel detection,
* `sig_b` : Optical signal B. Used for dual channel detection,
* `sig_d` : Optical difference signal, for balanced detection,
* `sig_s` : Optical sum signal, for balanced detection,

### Modulations signals (units: Volts unless noted)
* `tap_x` : Cantilever deflection signal "X", always used,
* `tap_y` : Cantilever deflection quadrature "Y", always used,
* `ref_x` : Reference mirror modulation signal "X", in psHet mode,
* `ref_y` : Reference mirror modulation signal "Y", in psHet mode,
* `chop` : Chopper status, bool, for pump-probe.

### Positions (parameters)
* `afm_x` : AFM position x (nm),
* `afm_y` : AFM position y (nm),
* `afm_z` : AFM position z (nm),
* `pos_ref` : Position of reference arm (units?), in spectroscopy mode, 
* `pos_pp` : Position of pump-probe stage (units?), for pump-probe,

## Processed variables
The variables will be processed in order to yield the signal. The 
processed variables are to be determined.

* `tap_p` : Tapping phase (radians),
* `ref_p` : psHet reference mirror phase (radians),

## Miscellaneous data

The experiment can also be characterized by miscellaneous data, such as number
of shots, calibration matrices for modulations, scaling factors for the signals,
etc. We'll see.
