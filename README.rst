netodesys
=========

``netodesys`` provides tools easily define and integrate systems of ordinary differential
equations (ODEs) on networks.

Examples
--------
A system of ODEs on an undirected (directed) network can be defined by subclassing from
NetworkX's `Graph` (`DiGraph`) along with the `Dynamical` mixin. All one needs to define
is a method, `rhs`, that calculates the dynamical equations from the current state of the graph.
The output should be either a dict or list-of-tuples, matching `sympy` variables with the corresponding
differential equations. For example, a network of :math:`N` Kuramoto oscillators, in which the
phase :math:`x_i` of node :math:`i` obeys:

.. math::

\\dot{x}_i = f_i + \\sum_{j=1}^N A_{ij} \\sin (x_j - x_i)

..

...can be represented by

.. code:: python

    >>> from networkx import Graph, DiGraph
    >>> from sympy import sin
    >>> class KuramotoNet(Dynamical, nx.Graph, vars=['x'], node_params=['f']):
    ...     def rhs(self):
    ...         x = self.x
    ...
    ...         for u in self:
    ...             yield u, self.f[u] + sum(self.A[u, v] * sin(x[v] - x[u])
    ...                                      for v in self.neighbors(u))

    >>> net = KuramotoNet()
    >>> net.add_nodes_from([0, 1], f = 1.0)
    >>> net.add_edge(0, 1, weight=0.1)

..

Integrating and plotting the resulting system is as simple as

.. code:: python

    >>> t_max = 100.0
    >>> x0 = [0, 3.1415]
    >>> res = net.integrate(t_max, x0)

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(res.tout, res.xout)
    >>> plt.xlabel("time")
    >>> plt.ylabel("x")
..

...which can in turn be plotted:
Note: As with all mixins, `Parametrized` should be inherited from *first* as above.

Instances of this class will enforce the existence of the parameters for all new nodes/edges.
Trying to add nodes/edges without them raises a `NodeParameterError` or `EdgeParameterError`:

.. code:: python

    >>> G.add_node('a')
    NodeParameterError: Tried to add node 'a' without all required parameters: ['x']

    >>> G.add_nodes_from(['a', 'b'], x=100)
    >>> G.add_edge('a', 'b')
    EdgeParameterError: Tried to add edge ('a', 'b') without all required parameters: ['y']
..

`paramnet` automatically adds named fields for each declared parameter, supporting clean random
access by node or edge

.. code:: python

    >>> G.add_edge('a', 'b', y=3)
    >>> G.x['a']
    100
    >>> G.y['a', 'b']
    3
..

Contrast this with the more cumbersome random access in base NetworkX (which can be used interchangeably if you wish):

.. code:: python

    >>> G.nodes['a']['x']
    100
    >>> G.edges['a', 'b']['y']
    3

..

What's more, `paramnet` maintains the order in which nodes were added, allowing index lookup:

.. code:: python

    >>> G.index('a')
    0
    >>> G.index('b')
    1
    >>> G.add_node('c')
    >>> G.index('c')
    2

..

The fact that nodes are ordered also allows a well-defined representation of *all* values of
a given node (edge) parameter as a vector (matrix). This can be obtained by accessing
the associated attribute without square brackets.

.. code:: python

    >>> G.x
    array([100, 100])
    >>> G.y
    array([[0., 3.],
           [3., 0.]])

    >>> G.A
    array([[0., 1.],
           [1., 0.]])
..

Note the special case for the network adjacency matrix, which is automatically defined
for every graph through the field `A` regardless of whether the associated edge attribute
(`weight`) is listed among the required parameters.

Under the hood, the parameter fields return View objects that wrap most `numpy` functionality,
allowing easy array operations on parameters including vector arithmetic and matrix
multiplication:

.. code:: python

    >>> G = MyGraph()
    >>> G.add_nodes_from([(node, {'x': 5*node+1}) for node in range(5)])
    >>> G.add_cycle(range(5), y=1)

    # number of paths of length two between node pairs
    >>> np.dot(G.A, G.A)
    array([[0., 3., 1., 1., 3.],
           [3., 0., 3., 1., 1.],
           [1., 3., 0., 3., 1.],
           [1., 1., 3., 0., 3.],
           [3., 1., 1., 3., 0.]])

    >>> G.x + 1
    array([ 2,  7, 12, 17, 22])

..

Dependencies
------------
* NetworkX (>= 2.0)
* paramnet
* pyodesys

License
-------

``netodesys`` is released under the MIT license. See LICENSE for details.