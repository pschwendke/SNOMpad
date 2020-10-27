import inspect


class A:
    def __init__(self, a=None, **kw):
        super().__init__(**kw)
        self.a = a


class B(A):
    def __init__(self, b=None, **kw):
        super().__init__(**kw)
        self.b = b


def parameter_merge(*args):
    "first takes precedence"
    collected = {}
    for sigs in reversed(args):
        for name, p in sigs.parameters.items():
            collected[name] = p
    return collected



sig = inspect.signature(B.__init__)

rsig = [inspect.signature(c.__init__) for c in B.mro()]

