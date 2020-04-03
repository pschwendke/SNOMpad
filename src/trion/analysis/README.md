# trion-data
This package contains tools designed to work with the TRION data.

The acquired variables can be divided in two categories: fast-scan and 
slow-scan. Fast scan data varies at every laser shot. The fast-scan variables 
must be acquired simultaneously at every laser shot, yielding a dense table. 
The fast-scan are the optical signals, the modulation signals and the chopper
status. The slow-scan variables are varied imperatively by the acquisition
controller. They do not vary shot-to-shot. The slow-scan variables are
the AFM positions, pump-probe delay and interferometer position in spectroscopy
mode.

## Acquisition variables
In order to structure the data, the acquisition variables must strictly follow
a convention. The experimental setup determines the set of variables. The
possible variables are:

### Optical signals (all V)
* `sig_A` : Optical signal A. Used for single channel detection,
* `sig_B` : Optical signal B. Used for dual channel detection,
* `sig_d` : Optical difference signal, for balanced detection,
* `sig_s` : Optical sum signal, for balanced detection,

### Modulations (V unless noted)
* `tap_x` : Cantilever deflection signal "X", always used,
* `tap_y` : Cantilever deflection quadrature "Y", always used,
* `ref_x` : Reference mirror modulation signal "X", in psHet mode,
* `ref_y` : Reference mirror modulation signal "Y", in psHet mode,
* `chop` : Chopper status, bool, for pump-probe.

### Positions (slow scan variables)
* `afm_x` : AFM position x (nm),
* `afm_y` : AFM position y (nm),
* `afm_z` : AFM position z (nm),
* `pos_ref` : Position of reference arm (units?), in spectroscopy mode, 
* `pos_pp` : Position of pump-probe stage (units?), for pump-probe,

## Processed variables
The acquired variables will be processed in order to yield the signal. The 
processed variables are to be determined.

* `tap_phi` : Tapping phase (radians),
* `ref_phi` : psHet reference mirror phase (radians),

## Physical mapping.