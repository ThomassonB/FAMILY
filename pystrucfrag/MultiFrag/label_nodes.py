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
    for node, phlevel in graph.nodes("_phlevel"):
        graph.nodes[node]["_level"] = levels.index(phlevel)

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
    #        ro = levels.index(graph.nodes[u]['_phlevel'])
    #        rl = levels.index(graph.nodes[v]['_phlevel'])
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