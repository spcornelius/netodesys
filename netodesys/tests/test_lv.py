from itertools import product

import numpy as np
import pytest
from wurlitzer import pipes

from .systems import NodewiseLVNet, VarwiseLVNet, \
    TermwiseLVNet
from .util import exprs_equal, check_combo, integrators, ChangesDynamics

classes = [NodewiseLVNet, VarwiseLVNet, TermwiseLVNet]


@pytest.mark.parametrize("cls", classes)
def test_definition(cls):
    net = cls()

    assert hasattr(net, 'x')
    assert hasattr(net, 'K')
    assert hasattr(net, 'r')
    assert hasattr(net, 'A')


@pytest.mark.parametrize("cls", classes)
def test_updates(cls):
    net = cls()

    with ChangesDynamics(net):
        net.add_node(0, r=1.0, K=10.0)

    with ChangesDynamics(net):
        net.add_node(1, r=1.0, K=10.0)

    with ChangesDynamics(net):
        net.add_node(2, r=-0.1, K=np.inf)

    with ChangesDynamics(net):
        net.add_edge(0, 1)

    with ChangesDynamics(net):
        net.add_edge(0, 2)

    with ChangesDynamics(net):
        net.r[0] = 0.99

    with ChangesDynamics(net):
        net.K[1] = 2.0

    with ChangesDynamics(net):
        net.e = 0.15

    with ChangesDynamics(net):
        net.remove_node(1)


def test_equivalence():
    net1 = NodewiseLVNet()
    net2 = VarwiseLVNet()
    net3 = TermwiseLVNet()

    nets = [net1, net2, net3]

    for net in nets:
        net.add_node(0, r=1.0, K=10.0)
        net.add_node(1, r=1.0, K=10.0)
        net.add_node(2, r=-0.1, K=np.inf)

        net.add_edge(0, 1)
        net.add_edge(0, 2)

    for eq1, eq2, eq3 in zip(*[net.sys.exprs for net in nets]):
        assert exprs_equal(eq1, eq2)
        assert exprs_equal(eq2, eq3)


@pytest.mark.slow
@pytest.mark.parametrize("cls,integrator,use_native,adaptive",
                         product(classes,
                                 integrators,
                                 [True, False],
                                 [True, False]))
def test_integration_lv(cls, integrator, use_native, adaptive):
    check_combo(integrator, use_native, adaptive)
    net = cls(integrator=integrator, use_native=use_native)
    net.add_node(0, r=-0.1, K=np.inf)
    net.add_node(1, r=1.0, K=10.0)
    net.add_node(2, r=1.0, K=10.0)
    net.add_edge(0, 1)
    net.add_edge(0, 2)

    x0 = np.ones(3)
    t_max = 1000.0

    if adaptive:
        t_out = t_max
    else:
        t_out = np.linspace(0, t_max, 1000)
    with pipes() as (out, err):
        res = net.integrate(t_out, x0, rtol=1.0e-8, atol=1.0e-8,
                            nsteps=10 ** 8)
    xf = res.yout[-1]
    assert np.allclose(xf, [0.95, 0.5, 0.5])
