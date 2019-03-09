from collections import defaultdict
from itertools import product

import networkx as nx
import numpy as np
import pytest
import sympy as sym

from netodesys import Dynamical, TermwiseDynamical

twopi = 2 * np.pi
integrators = ['cvode', 'gsl', 'scipy', 'odeint']


class KuramotoNet(Dynamical, nx.Graph):

    def __init__(self, default_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_weight = default_weight

    def dep_exprs(self):
        n = len(self)
        x = sym.symarray('x', n)
        dxdt = defaultdict(float)

        def w(u, v):
            return self.edges[u, v].get('weight', self.default_weight)

        for node in self:
            i = self.index(node)
            for nbr in self.neighbors(node):
                j = self.index(nbr)
                dxdt[x[i]] += w(node, nbr) * sym.sin(x[j] - x[i])

        return dxdt


class TermwiseKuramotoNet(TermwiseDynamical, nx.Graph):

    def R(self, i, x_i):
        return 0

    def Q(self, i, x_i, j, x_j):
        return sym.sin(x_j - x_i)


class LVMixin(object):
    r = 1.0
    e = 0.1
    K = 10.0

    def is_basal(self, node):
        return set((self.successors(node))) <= set((node,))


class LVNet(LVMixin, Dynamical, nx.DiGraph):

    def dep_exprs(self):
        n = len(self)
        x = sym.symarray('x', n)
        dxdt = defaultdict(float)
        for node in self:
            i = self.index(node)
            if self.is_basal(node):
                dxdt[x[i]] += self.r * x[i] * (1 - x[i] / self.K)
            else:
                dxdt[x[i]] -= self.e * self.r * x[i]
            for nbr in self.successors(node):
                j = self.index(nbr)
                dxdt[x[i]] += self.e * x[i] * x[j]
                dxdt[x[j]] -= x[i] * x[j]
        return dxdt


class TermwiseLVNet(LVMixin, TermwiseDynamical, nx.DiGraph):

    def R(self, i, x_i):
        if self.is_basal(i):
            return self.r * x_i * (1.0 - x_i/self.K)
        else:
            return -self.e * self.r * x_i

    def Q(self, i, x_i, j, x_j):
        return self.e * x_i * x_j, -x_i * x_j


def test_updates():
    net = KuramotoNet()
    net.add_cycle(range(3))

    net.add_node(4)
    assert net.stale_dynamics
    net.update_dynamics()
    assert not net.stale_dynamics

    net.edges[0, 1]['weight'] = 0.99
    assert net.stale_dynamics
    net.update_dynamics()

    net.add_node(10)
    assert net.stale_dynamics
    net.update_dynamics()

    net.add_edge(2, 10)
    assert net.stale_dynamics
    net.update_dynamics()

    net.remove_edge(1, 2)
    assert net.stale_dynamics
    net.update_dynamics()


@pytest.mark.parametrize("integrator,use_native,adaptive", product(['cvode', 'gsl', 'odeint', 'scipy'], [True, False], [True, False]))
def test_integration(integrator, use_native, adaptive):
    if integrator == 'scipy' and use_native is True:
        pytest.skip("No native code support for scipy integration")
    if (integrator == 'scipy'  or integrator == 'odeint') and adaptive is True:
        pytest.skip("Adaptive integration unreliable for scipy/odeint")

    net = KuramotoNet(integrator=integrator, use_native=use_native)
    net.add_cycle(range(3))
    x0 = np.random.uniform(0, 2 * np.pi, size=3)
    t_max = 1000.0

    if adaptive:
        t_out = t_max
    else:
        t_out = np.linspace(0, t_max, 1000)
    res = net.integrate(t_out, x0, rtol=1.0e-8, atol=1.0e-8)
    xf = res.yout[-1] % twopi
    assert np.allclose(xf, xf[0])

    net = LVNet(integrator=integrator, use_native=use_native)
    net.add_edge(0, 1)
    net.add_edge(0, 2)
    x0 = np.ones(3)
    t_max = 1000.0

    if adaptive:
        t_out = t_max
    else:
        t_out = np.linspace(0, t_max, 1000)
    res = net.integrate(t_out, x0, rtol=1.0e-8, atol=1.0e-8, nsteps=10**8)
    xf = res.yout[-1]
    assert np.allclose(xf, [0.95, 0.5, 0.5])


def test_termwise():
    # need to test equality of sympy exprs this way since ==
    # is only True if two formulae have the EXACT same structure
    def eq(expr1, expr2):
        return sym.simplify(expr1 - expr2) == 0

    knet1 = KuramotoNet()
    knet1.add_cycle(range(5))

    knet2 = TermwiseKuramotoNet()
    knet2.add_cycle(range(5))

    for (s1, e1), (s2, e2) in zip(knet1.dep_exprs().items(), knet2.dep_exprs().items()):
        assert eq(s1, s2)
        assert eq(e1, e2)

    lvnet1 = LVNet()
    lvnet1.add_edge(0, 1)
    lvnet1.add_edge(0, 2)

    lvnet2 = TermwiseLVNet()
    lvnet2.add_edge(0, 1)
    lvnet2.add_edge(0, 2)

    for (s1, e1), (s2, e2) in zip(knet1.dep_exprs().items(), knet2.dep_exprs().items()):
        assert eq(s1, s2)
        assert eq(e1, e2)
