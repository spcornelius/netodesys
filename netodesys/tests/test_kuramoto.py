from itertools import product
from wurlitzer import pipes

import numpy as np
import pytest

from .systems import NodewiseKuramotoNet, VarwiseKuramotoNet, \
    TermwiseKuramotoNet
from .util import exprs_equal, check_combo, integrators, ChangesDynamics

classes = [NodewiseKuramotoNet, VarwiseKuramotoNet, TermwiseKuramotoNet]
twopi = 2 * np.pi


@pytest.mark.parametrize("cls", classes)
def test_definition(cls):
    net = cls()

    assert hasattr(net, 'x')
    assert hasattr(net, 'A')


@pytest.mark.parametrize("cls", classes)
def test_updates(cls):
    net = cls()
    net.add_cycle(range(3))
    net.update_dynamics()

    with ChangesDynamics(net):
        net.add_node(4)

    with ChangesDynamics(net):
        net.edges[0, 1]['weight'] = 0.99

    with ChangesDynamics(net):
        net.add_node(10)

    with ChangesDynamics(net):
        net.add_edge(2, 10)

    with ChangesDynamics(net):
        net.remove_edge(1, 2)


def test_equivalence():
    net1 = NodewiseKuramotoNet()
    net2 = VarwiseKuramotoNet()
    net3 = TermwiseKuramotoNet()

    nets = [net1, net2, net3]

    for net in nets:
        net.add_cycle(range(5))

    for eq1, eq2, eq3 in zip(*[net.sys.exprs for net in nets]):
        assert exprs_equal(eq1, eq2)
        assert exprs_equal(eq2, eq3)


@pytest.mark.slow
@pytest.mark.parametrize("cls,integrator,use_native,adaptive",
                         product(classes, integrators,
                                 [True, False], [True, False]))
def test_integration(cls, integrator, use_native, adaptive):
    check_combo(integrator, use_native, adaptive)

    net = cls(integrator=integrator, use_native=use_native)
    net.add_cycle(range(3))
    x0 = np.random.uniform(0, 2 * np.pi, size=3)
    t_max = 1000.0

    if adaptive:
        t_out = t_max
    else:
        t_out = np.linspace(0, t_max, 1000)

    # capture annoying compilation output
    with pipes() as (out, err):
        res = net.integrate(t_out, x0, rtol=1.0e-8, atol=1.0e-8)
    xf = res.yout[-1] % twopi
    assert np.allclose(xf, xf[0])
