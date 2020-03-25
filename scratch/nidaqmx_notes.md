Notes from reading the NI-DAQmx
# Tasks
Tasks are the fundamental building blocs. They collect mutiple virtual channels
as well as their common auxiliaries (timing, triggers...). They are a state 
machine.
- Create task, its state is  "unverified"
- While task is "unverified", you can change is timing, triggering and 
attributes
- The validity of the configuration is checked when transitioning to "verified".
 This usually occurs automatically when starting the task.
- The "reserved" state indicates the task has successfully secured access to 
the required components. 
- The "commited" state indicates some preliminary configuration steps has been 
successfully completed. The "Commit" operation should be performed explicitely 
if there are many start-stop cycles.
- The "running" state indicates the task is currently running. When stopping, it
goes back to "reserved". The state becomes "running" when starting the task. 
Start should be called explicitely when there are multiple calls to "read" or 
"write".
- When Aborting, it goes back to its preceeding state, ie: last human
intervention.

# General IO considerations

## Default ports
Default ports are listeed under `NI-DAQmx Device Considerations/Physical 
Channels/X Series Physical Channels`

## Data formats and organization
Data has both formats and organization. Format has to do with the data type
(ex: int vs floats). Organization has to do with data structure (ex: scalar vs
array).

### Analog Data formats:
- Waveform: channel name, timing, unit information and 64bit scaled floating
point. The exact information is configurable. For AI, the timing data includes
a start time t0 and dt for every point. Waveform timing has some limitations.
Accurate determination of t0 requires careful of the calls. dt similarly can
only be used for certain timing types (ie: probably not external...). The dt is
a constant value, not a per-sample timestamp.
- 64-bit floating point numbers: read or write scaled data. Better performance
than waveform.
- Unsigned and signed integer: native hardware format. Maximum performance. Does
not use software calibration.
- Raw data: device dependant. 1D array. Ordering of samples (channels before 
samples or vice versa) is sample dependent (ie: interleaved or not). Does not 
use software calibration.

### Digital data formats:
- Waveform. Similar to Analog waveform. Can be organized asa logical unit for 
complex digital communcation schemes, it seems.
- Line format: Boolean, for individual lines. Limited to single sample r/w. Low
performance.
- Port format: integer. Matches the logical organization of the hardware. The
ports in the same order they are added to the task, LSB first. Line 0 is the LSB
of the port. If only part of the port is used, the rest is padded with 0s.

### Counter data formats:
- 64 bit floating point: for scaled data.
- Unsigned integer: native hardware format.

### Data Organization:
- waveform (single???)
- 1D waveform array: list of waveforms, ie: list of lists...
- Scalar: for single point values, for software controlled single point IO...
- Array: for multiple channels and/or multiple samples. Required for
simultaneous reads. Reading and writing multiple samples is more efficient than
going one at a time.
- Raw.

## Buffering
Space in computer memory for samples. For input, device transfers to buffer,
where it is available for application. Buffer is created when `Timing` function 
is used and 'sample mode' is set to 'finite' or 'continuous'. If buffer size
is set to 0 somehow, or data transfer mechanism is "programmed IO", there is no
buffer.

For finite acquisition, buffer size is number of channels x samples per channel.
For continuous acquisition, it is identical, but there is a minimum value of
10 kS for us.
The buffer size can be set manually. We probably should increase it to >20 kS?

### Continuous acquisition
Continuous acquisition uses a circular buffer. In case of buffer underflow, the
next read call blocks. In case of buffer overflow, an error is raised. The error
behavior can be overidden by using the "Overwrite Mode" attribute.

Buffer parameters:
- Current read position: index of next read.
- Total samples per channel acquired: amount of data transfered to buffer
(left to read, or total?)
- Available samples per channel: available to read.


# Digital input
## Setup and Hold time: 
for DI, the signal must stay stable both before and after the clock edge. The 
time before is called "setup time", the time after is called "hold time".

# Counters
Enable multiple logical tasks related to timing measurement and generation, in 
both time and frequency domains.

## Counter 'advanced'(?!) terminals
- `CtrNSource`: input terminal for measurement or counter timebase
- `CtrNGate`: controls counter behavior, generally w/r to timing: start trigger,
  pause trigger
- `CtrNAux`: Auxiliary input, depends on application
- `CtrNInternalOutput`: raw output of the counter. Pulses when count reaches the
  specified value. If necessary, it is further processed depending on 
  application. For pulse generation, it is directly routed to `Ctr*n*Out`.
- `CtrNSampleClock`: import or export a sample clock.

There are (maybe?) the basic terminals:`GATE`, `SOURCE`/`CLK` and `OUT`. They
are not in the routing table in NI-MAX though...

# Timing and sychronization
## Clocks
- AI sample clock: causes sampling
- AI sample clock timebase: source of AI Sample clock. It is divided down to 
produce `AI sample clock`.
- DI sample clock and sample clock timebase: same logic.
- Counter timebase: clock connected to source terminal of counter (???)
- Master timebase: onboard clock used by others.

"When you use the Timing function/VI to select the source of the sample clock 
signal for your analog input task on an M Series device, you choose a signal at
some other terminal to act as the source for the ai/SampleClock terminal. In 
other words, NI-DAQmx connects your chosen terminal (a PFI terminal pin, for 
instance) to the ai/SampleClock terminal. Selecting the ai/SampleClock terminal
as the sample clock source returns an error because a terminal cannot be
connected to itself. "

## Sample timing types
Can be set using a `Timing function`, or by attribute
Types are:
- Sample clock: use a clock (see above)
- On demand: when read or write functions execute, a form of soft timing
- Change detection: digital read when a change is detected on a digital line or 
port(ie: rising or falling edge).
- Handshake: for digital communcations
- Burst handshake: something about digital
- Implicit: for period or frequency sampling using counters.

## Trigger
We won't use hardware trigger I think. The "Start Trigger" starts an entire 
acquisition sequence (ie: a batch of points.). If needed, a trigger can be 
generated in software by `Send Software trigger`.

## Synchronization
- "For synchronizing multiple tasks on a single device, even at different rates 
and on different subsystems, you do not need to synchronize any  clocks. Because
the device derives those clocks from the same internal oscillator, you need to
share only the Start Trigger among the tasks so the clocks start at the same 
time.
To perform Start Trigger synchronization, configure start triggering on all 
slave tasks, setting the trigger source to the internal Start Trigger 
terminal from the master task, such as ai/StartTrigger. You do not have to 
configure start triggering on the master task. All tasks include an 
implicit Start Trigger, which occurs immediately when the task starts."
- Synchronization modes "Sample clock", "Reference clock", "Master timebase", 
"sample clock timebase" and "Mixed-clock" are for multiple devices

## Events: 
Timing outputs that do not directly correspond to a trigger or clock
- Change detection event: DIO device when logical level change has been 
detected.
- Counter output: Signal produced by counter when it reaches its count
- Ready for start: signal produced when a device is ready for a start trigger
- Sample complete event: produced when device acquires a sample from every 
channel in a task. (The Sample Complete Event is not exportable.)

## Software event:
For use in software.
- Every N samples acquired into buffer: produced when n samples are acquired 
from device to PC buffer. Only for buffered tasks.
- Every N samples transferred from buffer: idem, but works the other way around:
from PC to device.
- Done: produced when task completes execution, and some other conditions are 
met...
- Signal: occurs when a specific harware signal occurs.  Supported signals 
include the counter output event, change detection event, sample complete 
event, and the sample clock. 

## Exported signals
Clocks, triggers and events can be exported. Exported signals have multiple 
possible behaviors: pulse, level, toggle
- pulse: the exported signal yields a positive edge.
- level: the exported signal yields a positive edge of some duration. Exact 
behavior depends on source
- toggle: every signal event, the signal changes polarity. (mostly for counters 
it seems.)

## Signal routes
A route connects two terminals. Can be created using a Task, then called 
Task-based routing. It is made implicitely when creating a task. Can be made 
explicitely sugin 'Export Signal'. In both cases, the route is managed by the 
Task. When a task is cleared, the route is unreserved, but may not be cleared.
Otherwise, an "Immediate Route" can be created. Must be handled manually. 
Created with `Connect Terminals`, destroyed with `Disconnect terminals`.

Make sure to disconnect all DIO connectors to prevent 'double driving'. They can
be disconnected from the internal driver using `Tri-State Output Terminal`.
Check 'Lazy Line Transitions'. Not sure if I should care or not (!!!).

# Calibration
Calibration occurs in two steps: hardware calibration, and software calibration.

Hardware calibration must be performed using high precision voltage sources etc.
It was performed initially at the factory. The values are stored in memory 
(EEPROM). Performing a new hardware calibration overwrites these values.

Self calibration uses the values from hardware calibration, but adjusts them
using an internal standard. This is intended to compensate for minor
environmental drifts. Self-calibration should be performed after at least 15
minutes warm-up time. Self-calibration does not harm the device in any way, so
this can be done as often as desired. Takes a few seconds. For best results,
stop any ongoing tasks and disconnect any unnecessary external connections 
before running calibration. 

"When you self-calibrate your X Series device, no signal connections are 
necessary. However, values generated on the analog output channels change during
the calibration process. If you have external circuitry connected to the analog
output channels and you do not want changes on these channels, you should
disconnect the circuitry before beginning the self-calibration. "