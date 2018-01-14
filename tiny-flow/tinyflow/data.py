'''
Cool classes for data processing
'''
class Chainable(object):
    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def __rshift__(self, g):
        return Chainable(lambda *a, **kw: g(self(*a, **kw)))


def chainable(func):
    ''' Decorates function '''
    return Chainable(func)
