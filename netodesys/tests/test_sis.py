from itertools import product
from wurlitzer import pipes

import numpy as np
import pytest

from .systems.sis import NodewiseSISNet, VarwiseSISNet, TermwiseSISNet
from .util import exprs_equal, check_combo, integrators, ChangesDynamics

classes = [NodewiseSISNet, VarwiseSISNet, TermwiseSISNet]
#classes = [NodewiseSISNet]


@pytest.mark.parametrize("cls", classes)
def test_definition(cls):
    net = cls()

    assert hasattr(net, 'S')
    assert hasattr(net, 'I')
    assert hasattr(net, 'a')
    assert hasattr(net, 'b')
    assert hasattr(net, 'A')


@pytest.mark.parametrize("cls", classes)
def test_updates(cls):
    net = cls()

    with ChangesDynamics(net):
        net.add_node(0, a=0.1, b=0.05)

    with ChangesDynamics(net):
        net.add_nodes_from([1,2,3], a=0.1, b=0.05)

    with ChangesDynamics(net):
        net.add_edges_from([(0, 1), (3, 2)])

    with ChangesDynamics(net):
        net.add_edge(1, 3)

    with ChangesDynamics(net):
        net.a[0] = 0.3333

    with ChangesDynamics(net):
        net.b[2] = 0.1

    with ChangesDynamics(net):
        net.remove_node(1)

    with ChangesDynamics(net):
        net.remove_edge(3, 2)


def test_equivalence():
    net1 = NodewiseSISNet()
    net2 = TermwiseSISNet()
    net3 = VarwiseSISNet()

    nets = [net1, net2, net3]

    for net in nets:
        net.add_node(0, a=0.1, b=0.05)
        net.add_node(1, a=0.1, b=0.05)

        net.add_edge(0, 1, weight=0.1)

    for eq1, eq2 in zip(*[net.sys.exprs for net in [net1, net2]]):
        assert exprs_equal(eq1, eq2)

    exprs3 = np.array(net3.sys.exprs)[[0, 2, 1, 3]]
    for eq1, eq3 in zip(net1.sys.exprs, exprs3):
        assert exprs_equal(eq1, eq3)


@pytest.mark.slow
@pytest.mark.parametrize("cls,integrator,use_native,adaptive",
                         product(classes,
                                 integrators,
                                 [True, False],
                                 [True, False]))
def test_integration_lv(cls, integrator, use_native, adaptive):
    check_combo(integrator, use_native, adaptive)
    net = cls(integrator=integrator, use_native=use_native)
    net.add_node(0, a=0.2, b=0.05)
    net.add_node(1, a=0.2, b=0.05)

    net.add_edge(0, 1, weight=0.01)

    x0 = 1000.0*np.ones(2*len(net))
    t_max = 1000.0

    if adaptive:
        t_out = t_max
    else:
        t_out = np.linspace(0, t_max, 1000)
    with pipes() as (out, err):
        res = net.integrate(t_out, x0, rtol=1.0e-8, atol=1.0e-8,
                            nsteps=10 ** 8)
    assert np.allclose(res.yout[-1, 0], 500.0)

