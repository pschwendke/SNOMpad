from nidaqmx import Task

t = Task()
raise RuntimeError("Didn't try hard enough")
t.close()
