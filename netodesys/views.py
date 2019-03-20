import sympy as sym
import numpy as np
from itertools import product

__all__ = []

__all__.extend([
    'VarView'
])

# delegate certain magic methods to numpy
_delegated_mms = ['add', 'contains', 'copy', 'deepcopy', 'eq',
                  'ge', 'gt', 'index', 'le',  'lt', 'matmul',
                  'mul', 'ne', 'neg',  'pos', 'pow', 'radd', 'reduce',
                  'reduce_ex', 'repr', 'rmatmul', 'rmul', 'rpow', 'rsub',
                  'rtruediv', 'sizeof', 'str', 'sub', 'truediv']


def delegated_to_numpy(method_name):
    def wrapped(self, *args, **kwargs):
        arr = self.array
        return getattr(arr, method_name)(*args, **kwargs)

    return wrapped


def reshape(items, net):
    items = np.array(items)
    d = len(items.shape)
    n = len(net)
    m = len(net.vars)
    if d == 1:
        return items.reshape((m, n))
    elif d == 2:
        return items.T.reshape((m, n, -1))


class ViewMeta(type):

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        for mm in _delegated_mms:
            mm = f"__{mm}__"
            setattr(cls, mm, delegated_to_numpy(mm))


class View(object, metaclass=ViewMeta):

    def __init__(self, net, nodes, vars):
        self._net = net
        self._nodes = nodes
        self._vars = vars


class VarView(View):

    def __init__(self, net, var_name):
        self._net = net
        self._var_name = var_name

    def __getitem__(self, node):
        i = self._net.index(node)
        name = self._var_name
        return sym.Symbol(f"{name}_{i}")

    def __len__(self):
        return len(self._net)

    def apply(self, f):
        return np.vectorize(f)(self.array)

    def __iter__(self):
        yield from self.array

    @property
    def shape(self):
        return (len(self._net), 1)

    @property
    def array(self):
        return sym.symarray(self._var_name, len(self._net))

