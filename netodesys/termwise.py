import abc

import networkx as nx
import numpy as np

from netodesys.dynamical import Dynamical

__all__ = []
__all__.extend([
    'TermwiseDynamical',
])


class TermwiseDynamicalMeta(abc.ABCMeta):

    def __new__(mcs, name, bases, attrs, **kwargs):

        if not any(issubclass(base, (TermwiseDirected, TermwiseUndirected)) for
                   base in bases):
            graph_bases = [base for base in bases if
                           issubclass(base, nx.Graph)]
            if graph_bases:
                directed = issubclass(graph_bases[0], nx.DiGraph)
                new_base = TermwiseDirected if directed else TermwiseUndirected
                bases = (new_base,) + bases
        return super().__new__(mcs, name, bases, attrs, **kwargs)


class TermwiseUndirected(object):

    def rhs(self):
        for u in self:
            eq = np.array(self.node_term(u), dtype=object)
            for v in self.neighbors(u):
                eq += np.array(self.source_term(u, v), dtype=object)
            yield u, eq


class TermwiseDirected(object):

    @abc.abstractmethod
    def target_term(self, u, v):
        """ symbolic expression for the term corresponding to the edge
            u (<)---> v in the dynamics of v"""

    def rhs(self):
        for u in self:
            eq = np.array(self.node_term(u), dtype=object)
            for v in self.successors(u):
                eq += np.array(self.source_term(u, v), dtype=object)
            for v in self.predecessors(u):
                eq += self.A[v, u] * np.array(self.target_term(v, u),
                                              dtype=object)
            yield u, eq


class TermwiseDynamical(Dynamical, is_abstract=True,
                        metaclass=TermwiseDynamicalMeta):

    @abc.abstractmethod
    def node_term(self, u):
        """ symbolic expression for the self-dynamics of node u """

    @abc.abstractmethod
    def source_term(self, u, v):
        """ symbolic expression for the term corresponding to the edge
            u (<)---> v in the dynamics of u"""

