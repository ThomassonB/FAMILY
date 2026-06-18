"""
reduce_network.py
-----------------
Utility functions to filter and reduce the multi-scale source network built by
the FAMILY pipeline (Fragmentation Analysis and Multi-scale hIerarchy in Young
star-forming regions).

In FAMILY, compact sources (clumps/cores) detected independently at several
angular resolutions are connected into a directed graph.  Edges between nodes
(sources) carry an overlap weight ``_weight`` in [0, 1] that quantifies the
fractional spatial overlap between two polygonal footprints (see
``polygons_utility``).  The functions below allow one to prune edges that do
not satisfy physically motivated overlap criteria, which is the first step
toward identifying hierarchical fragmentation structures.

References
----------
* networkx documentation : https://networkx.org/
* Python operator module   : https://docs.python.org/3/library/operator.html
"""

import networkx as nx
import numpy as np

from . import polygons_utility as putility
import operator
from scipy import optimize

import copy
import time
from enum import Enum

############################################################
############################################################

# ------------- functions to reduce the network ------------

############################################################
############################################################

def cutEdges(graph, efeature={}):
    """
    Remove edges from a graph in-place based on attribute threshold conditions.

    For each edge attribute specified in ``efeature``, all edges whose
    attribute value satisfies the given comparison operator with respect to the
    reference value are collected and then removed from the graph.  This
    generalised interface allows simultaneous filtering on several edge
    attributes.

    Parameters
    ----------
    graph : networkx.DiGraph or networkx.Graph
        The multi-scale source network to be filtered.  The graph is modified
        **in place** — no copy is returned.
    efeature : dict of { str : (scalar, callable) }, optional
        Dictionary that maps each edge-attribute name to a ``(value, op)``
        pair, where:

        * ``value`` – reference scalar used in the comparison.
        * ``op``    – a two-argument callable from the :mod:`operator` module
          (e.g. ``operator.gt``, ``operator.lt``) or any function with the
          signature ``op(value, edge_attribute) -> bool``.  An edge is
          **removed** when ``op(value, edge_attribute)`` evaluates to
          ``True``.

        See https://docs.python.org/3/library/operator.html for the full list
        of available operator objects.

        Example::

            # Remove all edges whose '_weight' attribute is less than 0.5
            cutEdges(graph, {"_weight": (0.5, operator.gt)})

    Notes
    -----
    Edges matching *any* of the supplied conditions are removed (logical OR).
    If ``efeature`` is empty (default), the graph is left unchanged.
    """
    ebunch = []
    for feat, container in efeature.items():
        value, op = container
        # Collect edges (u, v) for which op(value, edge_attr) is True
        ebunch += [(u, v) for u, v, w in graph.edges.data(feat) if op(value, w)]
    graph.remove_edges_from(ebunch)

def overlapThreshold(graph, min_overlap):
    """
    Remove edges whose spatial overlap weight is below a minimum threshold.

    This is the primary edge-pruning step used in FAMILY to build the
    hierarchical network: two sources at adjacent resolution levels are kept
    connected only if their polygon footprints overlap by at least
    ``min_overlap`` (relative to the smaller source area).

    Internally delegates to :func:`cutEdges` using the ``_weight`` edge
    attribute and ``operator.gt`` so that edges with
    ``_weight < min_overlap`` are discarded.

    Parameters
    ----------
    graph : networkx.DiGraph or networkx.Graph
        The multi-scale source network to be filtered.  Modified **in place**.
    min_overlap : float
        Minimum required fractional overlap in the interval ``[0, 1]``.
        Edges with ``_weight < min_overlap`` are removed.

        * ``0``   – keep all edges (no filtering).
        * ``1``   – keep only edges with perfect spatial coincidence.
        * ``0.75`` – typical value used in the NGC 2264 benchmark (see
          ``testNotebook.ipynb``).

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.DiGraph()
    >>> G.add_edge(0, 1, _weight=0.9)
    >>> G.add_edge(1, 2, _weight=0.4)
    >>> overlapThreshold(G, min_overlap=0.75)
    >>> list(G.edges())   # edge (1, 2) has been removed
    [(0, 1)]
    """
    efeature = {"_weight": [min_overlap, operator.gt]}
    cutEdges(graph, efeature)