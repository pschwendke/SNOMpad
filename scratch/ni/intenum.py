# they are using an enum instead of an intenum.

from nidaqmx import Task, DaqError
from nidaqmx.error_codes import DAQmxErrors

t = Task("task")
try:
    u = Task("task")
except DaqError as err:
    print(f"Caught error: {err!r}")
    if err.error_code == DAQmxErrors.DUPLICATE_TASK.value:
        print("Tried to create duplicate task")
    else:
        raise
finally:
    print("finishing")
    t.close()
