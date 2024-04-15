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
    Cut inplace edges of graph accordingly to efeature operation.

    Parameters
    ----------
    graph : networkx.network
        network to filter
    efeature : dict { attribute : ( value, operator ) }
        key is the attribute name and the associated value is a list or a tuple.
        This list/tuple contains the mathematical operation used to filter and the value of reference for this operation.

        The operator variable is an operator object, see https://docs.python.org/3/library/operator.html for a complete
        list of available operations.
    """
    ebunch = []
    for feat, container in efeature.items():
        value, op = container
        ebunch += [(u, v) for u, v, w in graph.edges.data(feat) if op(value, w)]
    graph.remove_edges_from(ebunch)

def overlapThreshold(graph, min_overlap):
    """
    Cut inplace edges of graph whose _weight attribute is lower than min_overlap.

    Parameters
    ----------
    graph : networkx.network
        network to filter
    min_overlap : float between 0 and 1
        lower limit value used to filter
    """
    efeature = {"_weight": [min_overlap, operator.gt]}
    cutEdges(graph, efeature)