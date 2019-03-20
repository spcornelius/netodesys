import networkx as nx
import numpy as np

from netodesys import Dynamical, TermwiseDynamical

__all__ = []
__all__.extend([
    'NodewiseLVNet',
    'VarwiseLVNet',
    'TermwiseLVNet'
])


class LVMixin(object):
    e = 0.1

    def is_basal(self, node):
        return set((self.successors(node))) <= set((node,))


class NodewiseLVNet(LVMixin, Dynamical, nx.DiGraph, node_params=['r', 'K']):

    def rhs(self):
        x = self.x
        for u in self:
            eq = self.r[u] * x[u] * (1 - x[u] / self.K[u])
            eq += sum(self.e * self.A[u, v] * x[u] * x[v] for v in
                      self.successors(u))
            eq -= sum(self.A[v, u] * x[u] * x[v] for v in self.predecessors(u))
            yield u, eq


class VarwiseLVNet(LVMixin, Dynamical, nx.DiGraph, node_params=['r', 'K']):

    def rhs(self):
        x = self.x
        r = self.r
        A = self.A
        A = self.e * A - A.T
        K = self.K
        yield 'x', x * (r * (1 - x / K) + np.dot(A, x))


class TermwiseLVNet(LVMixin, TermwiseDynamical, nx.DiGraph,
                    node_params=['r', 'K']):

    def node_term(self, u):
        x = self.x[u]
        return self.r[u] * x * (1 - x / self.K[u])

    def source_term(self, u, v):
        x = self.x
        return self.e * self.A[u, v] * x[u] * x[v]

    def target_term(self, u, v):
        x = self.x
        return -self.A[u, v] * x[u] * x[v]
