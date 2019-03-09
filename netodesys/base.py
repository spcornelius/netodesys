import abc
from collections import OrderedDict, defaultdict

import networkx as nx
import sympy as sym
from pyodesys.native import native_sys
from pyodesys.symbolic import SymbolicSys

from netodesys.dict import make_dict_factories

__all__ = []
__all__.extend([
    'Dynamical',
    'TermwiseDynamical'
])


def uses_dynamics(method):
    def wrapped(self, *args, **kwargs):
        if self._stale_dynamics:
            self.update_dynamics()
        return method(self, *args, **kwargs)

    return wrapped


class Dynamical(object, metaclass=abc.ABCMeta):

    def __init__(self, integrator=None, use_native=False, *args, **kwargs):
        ndf, nadf, aodf, aidf, eadf = make_dict_factories(self)
        self.node_dict_factory = ndf
        self.node_attr_dict_factory = nadf
        self.adjlist_outer_dict_factory = aodf
        self.adjlist_inner_dict_factory = aidf
        self.edge_attr_dict_factory = eadf

        self.integrator = integrator
        self.use_native = use_native
        self._sys = None
        self._stale_dynamics = True
        self._native_sys = None
        super().__init__(*args, **kwargs)

    def index(self, node):
        try:
            return self._node.keys().index(node)
        except ValueError:
            raise nx.NetworkXError(f"The node {node} is not in the graph")

    def expire_dynamics(self):
        self._stale_dynamics = True

    @abc.abstractmethod
    def dep_exprs(self):
        pass

    @property
    def sys(self):
        return self._sys

    @property
    def native_sys(self):
        return self._native_sys

    @property
    def stale_dynamics(self):
        return self._stale_dynamics

    def update_dynamics(self):
        self._sys = SymbolicSys(self.dep_exprs())
        if self.use_native:
            self._native_sys = native_sys[self.integrator].from_other(self._sys)

            # kluge to make pyodesys/pycompilation work with LLVM
            self._native_sys._native.compile_kwargs['flags'] = ["-undefined dynamic_lookup"]

        self._stale_dynamics = False

    @uses_dynamics
    def integrate(self, *args, **kwargs):
        if self.use_native:
            return self._native_sys.integrate(*args, **kwargs)
        else:
            return self._sys.integrate(integrator=self.integrator, *args, **kwargs)


class TermwiseDynamical(Dynamical):

    @abc.abstractmethod
    def R(self, i, x_i):
        pass

    @abc.abstractmethod
    def Q(self, i, x_i, j, x_j):
        pass

    def dep_exprs(self):
        n = len(self)
        x = OrderedDict(zip(self.nodes(), sym.symarray('x', n)))
        dxdt = defaultdict(float)
        for i in self:
            dxdt[x[i]] += self.R(i, x[i])
            for j in self.neighbors(i):
                w = self.edges[i, j].get('weight', 1.0)
                if self.is_directed():
                    Q1, Q2 = self.Q(i, x[i], j, x[j])
                    dxdt[x[i]] += w * Q1
                    dxdt[x[j]] += w * Q2
                else:
                    dxdt[x[i]] += w * self.Q(i, x[i], j, x[j])
        return dxdt
