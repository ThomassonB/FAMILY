"""
label_nodes.py
==============
Utility functions to classify and annotate nodes within a hierarchical
multi-scale network built from astrophysical source catalogs.

In this framework, each node of the network corresponds to a detected
compact source (clump/core/structure) at a given angular resolution
(observation scale / beam size). Directed edges connect sources across
consecutive resolution levels when a significant spatial overlap is
detected, encoding the parent-child hierarchical relationship between
structures observed at different scales.

Node classification is based on the in-degree and out-degree of each
node in the directed graph:

  - SOURCE      : no parent (indegree = 0), has children  -> top-level structure
  - SINK        : has parent(s), no children (outdegree = 0) -> leaf structure
  - INTERMEDIATE: has both parents and children           -> nested structure
  - ISOLATED    : no connections at all                   -> unrelated structure
  - VIRTUAL     : artificially inserted node used to fill
                  missing detection gaps between levels
                  (see ``network_utility.virtualNodes``)

These labels, together with the hole count and fractality measure,
are used to characterise the hierarchical fragmentation of
star-forming regions (e.g. NGC 2264) observed at multiple wavelengths
and angular resolutions.
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
# ------------- functions to label the nodes ---------------
############################################################
############################################################

class NodeKind(Enum):
    """
    Enumeration of topological roles for nodes in the multi-scale network.

    Each source detected at a given resolution is assigned one of the
    following kinds based on its connectivity in the directed graph,
    where edges go from coarser (parent) to finer (child) resolution levels.

    Attributes
    ----------
    VIRTUAL : int
        Placeholder node artificially inserted to represent a missing
        detection at an intermediate scale. Used to correct for
        observational incompleteness when estimating fractality.
    SOURCE : int
        Node with no incoming edges (indegree = 0) but at least one
        outgoing edge. Represents a top-level structure with no
        detected parent at a coarser scale.
    SINK : int
        Node with at least one incoming edge but no outgoing edges
        (outdegree = 0). Represents a leaf structure with no detected
        children at finer scales.
    INTERMEDIATE : int
        Node with both incoming and outgoing edges. Represents a
        structure nested within a coarser-scale parent and itself
        containing finer-scale children.
    ISOLATED : int
        Node with neither incoming nor outgoing edges. Represents a
        structure with no detected hierarchical relationship to any
        other source in the network.
    """
    VIRTUAL = 0
    SOURCE = 1
    SINK = 2
    INTERMEDIATE = 3
    ISOLATED = 4

def labelKind(graph):
    """
    Add label inplace for nodes of graph. Labels are defined in NodeKind class.

    Parameters
    ----------
    graph : networkx.network
        network to labelise
    """
    for (node, indeg), (_, outdeg) in zip(graph.in_degree(), graph.out_degree()):
        if not indeg and not outdeg:
            graph.nodes[node]['_Kind'] = NodeKind.ISOLATED
        elif not indeg and outdeg:
            graph.nodes[node]['_Kind'] = NodeKind.SOURCE
        elif indeg and outdeg:
            graph.nodes[node]['_Kind'] = NodeKind.INTERMEDIATE
        else:
            graph.nodes[node]['_Kind'] = NodeKind.SINK

def setLevel(graph, levels):
    """
    Add integer level label inplace for nodes of graph.

    Parameters
    ----------
    graph : networkx.network
        network to labelise
    levels : list of float
        contains the ordered physical levels. The label corresponds to the index of the associated physical level.
    """
    for node, beam in graph.nodes("_beam"):
        graph.nodes[node]["_level"] = levels.index(beam)

def getHoles(graph, levels):
    """
    Add holes attribute inplace for nodes of graph. A hole is defined as an absence of node in an intermediate level between two nodes.
    The attribute '_Holes' is an array_like object which value is 0 where no node is missing (no hole) and 1 where a node is missing.
    The index of this iterable correspond to the index of levels.

    For example :
        A 5 levels objects is [0, 0, 0, 0, 0]. If the outcoming node v have a hole in the level 2, it will receive [0, 0, 1, 0, 0].

    Parameters
    ----------
    graph : networkx.network
        network to labelise
    levels : list of float
        contains the ordered physical levels. The label corresponds to the index of the associated physical level.
    """
    holes = {node:0 for node in graph.nodes}
    for u, v, dl in graph.edges.data('_deltal'):
        if (dl != 1) and graph.edges[u, v]["dir"]:
            holes[v] += dl - 1
    nx.set_node_attributes(graph, holes, "_Holes")

    # ------------------ Old function not counting all the holes
    #for u, v, dl in graph.edges.data('_deltal'):
    #    Nvec = np.zeros_like(levels)
    #    if dl != 1:
    #        ro = levels.index(graph.nodes[u]['_beam'])
    #        rl = levels.index(graph.nodes[v]['_beam'])
    #        Nvec[rl + 1:ro] = 1
    #    graph.nodes[v]['_Holes'] = np.sum(Nvec)

def prepareNetwork(graph, eta=2, verbose=False):
    """ Prepare an empty network by adding labels, measuring holes and fractality """
    from . import network_utility as utility
    if verbose:
        tini = time.time()
        print(f"Starting preparation for {len(graph)} nodes and {len(graph.edges)} edges")
        to = time.time()

    labelKind(graph)

    if verbose:
        print(f"labeling nodes ended in {time.time() - to} s")
        to = time.time()

    levels = utility.getLevels(graph)
    getHoles(graph, levels)

    if verbose:
        print(f"holes measurement ended in {time.time() - to} s")
        to = time.time()

    utility.fractality(graph, eta)

    if verbose:
        print(f"fractality measurement ended in {time.time() - to} s")
        to = time.time()

    setLevel(graph, levels)

    if verbose:
        print(f"levels set in {time.time() - to} s \nEnd of preparation, total time : {time.time() - tini}")