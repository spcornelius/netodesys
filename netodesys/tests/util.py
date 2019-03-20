import pytest
import sympy as sym

__all__ = ['exprs_equal', 'check_combo', 'integrators', 'ChangesDynamics']

integrators = ['cvode', 'gsl', 'scipy', 'odeint']


def exprs_equal(expr1, expr2):
    return sym.simplify(expr1 - expr2) == 0


def check_combo(integrator, use_native, adaptive):
    if integrator == 'scipy' and use_native:
        pytest.skip("No native code support for scipy integration")
    elif integrator in ['scipy', 'odeint'] and adaptive:
        pytest.skip("Adaptive integration unreliable for scipy/odeint")


class ChangesDynamics(object):

    def __init__(self, net):
        self.net = net

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            raise exc_type(exc_value)
        assert self.net.stale_dynamics
        self.net.update_dynamics()
        assert not self.net.stale_dynamics
