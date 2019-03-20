import pytest
import networkx as nx
from netodesys import Dynamical, TermwiseDynamical


@pytest.mark.parametrize("graph_cls", [nx.Graph, nx.DiGraph])
def test_dynamical_abc(graph_cls):

    class A1(Dynamical, graph_cls):
        pass

    class A2(Dynamical, graph_cls):
        def rhs(self):
            return []

    with pytest.raises(TypeError):
        A1()

    A2()


def test_termwise_undirected_abc():

    class A1(TermwiseDynamical, nx.Graph):
        pass

    class A2(TermwiseDynamical, nx.Graph):
        def node_term(self, u):
            return 0

    class A3(TermwiseDynamical, nx.Graph):
        def source_term(self, u, v):
            return 0

    class A4(TermwiseDynamical, nx.Graph):
        def node_term(self, u):
            return 0

        def source_term(self, u, v):
            return 0

    for cls in [A1, A2, A3]:
        with pytest.raises(TypeError):
            cls()

    A4()


def test_termwise_directed_abc():

    class A1(TermwiseDynamical, nx.DiGraph):
        pass

    class A2(TermwiseDynamical, nx.DiGraph):
        def node_term(self, u):
            return 0

    class A3(TermwiseDynamical, nx.DiGraph):
        def source_term(self, u, v):
            return 0

    class A4(TermwiseDynamical, nx.DiGraph):
        def target_term(self, u, v):
            return 0

    class A5(TermwiseDynamical, nx.DiGraph):
        def node_term(self, u):
            return 0

        def source_term(self, u, v):
            return 0

        def target_term(self, u, v):
            return 0

    for cls in [A1, A2, A3, A4]:
        with pytest.raises(TypeError):
            cls()

    A5()
