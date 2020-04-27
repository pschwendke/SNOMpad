#try monkey patching the task object

# from nidaqmx import Task
# t = Task()
# type(t).__del__ = None
# raise RuntimeError("Didn't try hard enough")
# t.close()

from nidaqmx import Task
t = Task()
Task.name = property(lambda self: self._saved_name)
raise RuntimeError("Didn't try hard enough")
t.close()