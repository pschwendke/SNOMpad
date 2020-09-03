# try our lib with qt and traitlets.

# first start with traitlets, attrs, and observe!

import attr
from traitlets import HasTraits, Integer


def func(change):
    print(f"{change.old} -> {change.new}")


def func2(change):
    print(f"{change.old} modified {change.new}")


# works
# class MyData(HasTraits):
#     v = Integer()
#     def __init__(self, v):
#         self.v = v
# m = MyData(3)


# Doesn't work
# @attr.s
# class MyData(HasTraits):
#     v = attr.ib(type=Integer, d)
#
# m = MyData(3)


# Also works!
class MyData(HasTraits):
    v = Integer(default_value=3)
    w = Integer(default_value=4)

m = MyData()

m.observe(func, names=["v"])
m.observe(func2, names=["w"])
m.v = 1
m.v = 2
m.w = 5
m.w = 5 # this one is silent. good.
