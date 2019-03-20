import abc
import itertools as it

import sympy as sym
from paramnet import Parametrized
from pyodesys.native import native_sys
from pyodesys.symbolic import SymbolicSys
from sympy import flatten
from sympy.core.numbers import Zero

from netodesys.dict import NodeDict, AdjlistOuterDict
from netodesys.views import VarView

__all__ = []
__all__.extend([
    'Dynamical',
])


def uses_dynamics(method):
    # decorator to lazily update dynamics for methods that require them
    def wrapped(self, *args, **kwargs):
        if self._stale_dynamics:
            self.update_dynamics()
        return method(self, *args, **kwargs)

    return wrapped


def sym_getter(var_name):
    def fget(self):
        return VarView(self, var_name)

    return fget


class Dynamical(Parametrized, is_abstract=True, metaclass=abc.ABCMeta):
    _node = NodeDict()
    _adj = AdjlistOuterDict()
    _pred = AdjlistOuterDict()

    def __init__(self, integrator=None, use_native=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_native = use_native
        self.integrator = integrator

        self._sys = None
        self._stale_dynamics = True
        self._native_sys = None

    def __init_subclass__(cls, vars=None, is_abstract=False, **kwargs):
        super().__init_subclass__(is_abstract=is_abstract, **kwargs)

        if is_abstract:
            return

        if vars is None:
            vars = ['x']

        vars = tuple(vars)
        cls._vars = vars

        for var in vars:
            if hasattr(cls, var):
                raise AttributeError(f"Attribute for symbolic variable {var} "
                                     f"conflicts with existing attr for class"
                                     f"{cls.__name__}.")

            setattr(cls, var, property(sym_getter(var)))

    def expire_dynamics(self):
        self._stale_dynamics = True

    @property
    def vars(self):
        return self._vars

    @property
    @uses_dynamics
    def sys(self):
        return self._sys

    @property
    @uses_dynamics
    def native_sys(self):
        return self._native_sys

    @property
    def stale_dynamics(self):
        return self._stale_dynamics

    @abc.abstractmethod
    def rhs(self):
        pass

    def update_dynamics(self):
        eqs = dict(self.rhs())
        vars = [getattr(self, v) for v in self.vars]
        if all(k in self for k in eqs):
            # by node
            dep = flatten(zip(*vars))
            expr = flatten(sym.Matrix([eqs[node] for node in self]))
        elif all(k in self.vars for k in eqs):
            # by variable
            dep = it.chain.from_iterable(vars)
            expr = flatten(sym.Matrix([eqs[v] for v in self.vars]))
        else:
            raise ValueError(
                "rhs must map either nodes to rhs or variables to rhs")

        dep_expr = [(d, e + Zero()) for d, e in zip(dep, expr)]
        self._sys = SymbolicSys(dep_expr)
        if self.use_native:
            self._native_sys = native_sys[self.integrator].from_other(
                self._sys)

        self._stale_dynamics = False

    @uses_dynamics
    def integrate(self, *args, **kwargs):
        if self.use_native:
            return self._native_sys.integrate(*args, **kwargs)
        else:
            return self._sys.integrate(integrator=self.integrator, *args,
                                       **kwargs)
