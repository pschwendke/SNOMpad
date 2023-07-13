### Implemented acquisition routines
- SinglePoint
- SteppedRetraction
- SteppedImage
- ContinuousRetraction
- ContinuousImage
- NoiseScan
- (DelayScan)
- (LineScan)


### TRION scan files

The acquisition routines are aimed at producing a standardised data format.
Raw data, tracked data, data returned by the NeaSNOM, and metadata are stored in one hdf5 file.
No units are converted all values are recorded as returned by the instrument.
The structure is as follows:

>#### scan file: 'YYYYMMDD-hhmmss_acquisition_mode.h5'
- attributes: dict containing all collected metadata
    > #### group: 'afm_data'
    - **datasets**: 'x', 'z', 'amp' etc.
      - tracked data, either on grid (stepped), or indexed (continuous)
      - an 'idx' array links each row of data to a dataset it 'daq_data'
    > #### group: 'nea_data'
    - **datasets**: 'M1A', 'O1A' etc. 
      - images or curves as returned by the NeaScan API
    > #### group: 'daq_data'
    - **datasets**: '1', '2', '3' etc.
      - one dataset per chunk of DAQ data
      - data points on axis=0, signals on axis=1
      - numbered as they are acquired 
    
    
