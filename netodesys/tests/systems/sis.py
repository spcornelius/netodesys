import networkx as nx
import sympy as sym
import numpy as np

from netodesys import Dynamical, TermwiseDynamical

__all__ = []
__all__.extend([
    'NodewiseSISNet',
    'VarwiseSISNet',
    'TermwiseSISNet'
])


class NodewiseSISNet(Dynamical, nx.Graph, vars=['S', 'I'],
                     node_params=['a', 'b']):

    def rhs(self):
        S = self.S
        I = self.I
        a = self.a
        b = self.b
        A = self.A

        for u in self:
            N = S[u] + I[u]
            dSdt = -a[u] / N * S[u] * I[u] + b[u] * I[u]
            dSdt += sum(A[u, v] * S[v] for v in self.neighbors(u))
            dSdt -= S[u] * sum(A[u, v] for v in self.neighbors(u))

            dIdt = a[u] / N * S[u] * I[u] - b[u] * I[u]
            dIdt += sum(A[u, v] * I[v] for v in self.neighbors(u))
            dIdt -= I[u] * sum(A[u, v] for v in self.neighbors(u))
            yield u, [dSdt, dIdt]


class VarwiseSISNet(Dynamical, nx.Graph, vars=['S', 'I'],
                    node_params=['a', 'b']):

    def rhs(self):
        S = self.S
        I = self.I
        a = self.a
        b = self.b
        L = nx.laplacian_matrix(self).todense().A

        N = S + I
        return {'S': -a * S * I / N + b * I - np.dot(L, S),
                'I': a * S * I / N - b * I - np.dot(L, I)}


class TermwiseSISNet(TermwiseDynamical, nx.Graph, vars=['S', 'I'],
                     node_params=['a', 'b']):

    def node_term(self, u):
        S = self.S[u]
        I = self.I[u]
        N = S + I
        w = sum(self.A[u, v] for v in self.neighbors(u))
        return [-self.a[u] * S * I / N + self.b[u] * I - w * S,
                self.a[u] * S * I / N - self.b[u] * I - w * I]

    def source_term(self, u, v):
        return [self.A[u, v] * self.S[v],
                self.A[u, v] * self.I[v]]
