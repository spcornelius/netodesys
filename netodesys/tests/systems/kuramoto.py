import networkx as nx
import sympy as sym
import numpy as np

from netodesys import Dynamical, TermwiseDynamical

__all__ = []
__all__.extend([
    'NodewiseKuramotoNet',
    'VarwiseKuramotoNet',
    'TermwiseKuramotoNet'
])


class NodewiseKuramotoNet(Dynamical, nx.Graph, vars=['y']):

    def rhs(self):
        y = self.y
        for u in self:
            yield u, sum(self.A[u, v] * sym.sin(y[v] - y[u])
                         for v in self.neighbors(u))


class VarwiseKuramotoNet(Dynamical, nx.Graph, vars=['y']):

    def rhs(self):
        y = np.array(self.y).flatten()
        sin = np.vectorize(sym.sin)
        yield 'y', np.sum(self.A * sin(np.subtract.outer(y, y)), axis=0)


class TermwiseKuramotoNet(TermwiseDynamical, nx.Graph, vars=['y']):

    def node_term(self, u):
        return 0

    def source_term(self, u, v):
        y = self.y
        return self.A[u, v] * sym.sin(y[v] - y[u])
