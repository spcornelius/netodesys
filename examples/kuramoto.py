import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sympy import sin
from netodesys import Dynamical


class KuramotoNet(Dynamical, nx.Graph, vars=['x'], node_params=['w']):

    def rhs(self):
        x = self.x
        w = self.w
        t = self.t
        for u in self:
            dudt = sin(w[u]*t) + sum(self.A[u, v] * sin(x[v] - x[u])
                                     for v in self.neighbors(u))
            yield u, dudt


def main():
    net = KuramotoNet()
    net.add_node(0, w=2*np.pi+0.1)
    net.add_node(1, w=2*np.pi-0.1)
    net.add_edge(0, 1, weight=10.0)

    x0 = np.random.uniform(0, 2*np.pi, size=2)
    t_max = 3.0
    res = net.integrate(t_max, x0, integrator='cvode', nsteps=10**6)
    plt.figure(figsize=(4,3))
    plt.plot(res.xout, res.yout)
    plt.xlabel("time, $t$")
    plt.ylabel("phase, $x(t)$")
    plt.savefig("kuramoto_example.png", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
