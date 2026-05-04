import networkx as nx
import numpy as np
import shapely as shp

from . import polygons_utility as putility
from . import network_utility as utility
from . import label_nodes
NodeKind = label_nodes.NodeKind
 
import operator
from scipy import optimize

import copy
import time
from enum import Enum


############################################################
############################################################

# ------------- functions to derive the structures ---------

############################################################
############################################################

class StructureMode(Enum):
    ISOLATED = 1
    LINEAR = 2
    HIERARCHICAL = 3

def getStructures(network):
    from . import analyse
    from shapely.ops import unary_union

    levels = sorted(list(utility.getSetAttribute(network, "_beam")))
    for idx, component in enumerate(utility.getComponents(network)):
        #####
        # Statistic
        #####
        Nfrag = nodeKindRepartition(component, levels)
        # Nfragl = list(map(sum, zip(*Nfrag)))
        Nfragl = np.sum(Nfrag, axis=0) #nb de fragments à chaque niveau
        #Nfragt = np.sum(Nfrag, axis=1) #nb de fragments de chaque type

        Missed = np.sum([nhole for n, nhole in component.nodes('_Holes')
                         if component.nodes[n]["_Kind"].value in (NodeKind.SINK.value, NodeKind.INTERMEDIATE.value)])

        productivity_parents = productivityPerSource(component, levels, Nfrag=Nfrag)

        classes = [att for n, att in component.nodes('_Class')
                   if component.nodes[n]['_Kind'].value in (3, 4)]

        #####
        # Polygon
        #####
        polygons = [poly for node, poly in component.nodes('_Polygon')]
        Poly = unary_union(polygons)
        centroid = shp.centroid(Poly).xy

        #####
        # Metrics
        #####
        MeanFractality = np.mean([att for n, att in component.nodes('_Fractality')
                                  if component.nodes[n]['_Kind'].value in (1, 4)])

        mode = setMode(Nfrag)

        try:
            pMissed = Missed / (Missed + len(component.nodes))
        except:
            pMissed = 1

        kwargs = dict(
                        sinks=sum(Nfrag[3]),
                        sources=sum(Nfrag[1]),
                        productivity=productivity_parents,
                        nl=Nfragl,
                        sourcel=Nfrag[1],
                        sinkl=Nfrag[3],
                        mode=mode,
                        fractality=MeanFractality,
                        missed=Missed,
                        percmissed=pMissed,
                        triplets=utility.transitiveTriplets(component),
                        maxR=max([at for n, at in component.nodes('_beam')]),
                        YSO=len([x for x in classes if x is not None]),
                        gas=len([x for x in classes if x is None]),
                        nbunch=list(component.nodes),
                        polygon=Poly,
                        size=np.sqrt(Poly.area),  # typical size in arcsec -> kAU (*1000 for AU)
                        position=centroid)

        yield analyse.Structure(label=idx, component=component, **kwargs)

def nodeKindNumber(graph):
    """
    Count and return the number of node of each kind in graph
    order is [virtual, source, intermediate, sink, isolated]

    Parameters
    ----------
    graph : networkx.network
        network to count the nodes on

    Returns
    -------
    list
        list containing the number of each kind of node in the network
        order is setup automatically with the associated value in NodeKind()
    """
    tpls = [0, 0, 0, 0, 0]  # virtual, source, intermediate, sink, isolated
    label, counts = np.unique([kind.value for n, kind in graph.nodes('_Kind')], return_counts=True)
    for idx, t in zip(label, counts):
        tpls[idx] = t
    return tpls

def nodeKindRepartition(graph, levels):
    """
    Count and return the number of node of each kind in graph for each level of the network

    Parameters
    ----------
    graph : networkx.network
        network to count the nodes on
    levels : list of float
        contains the ordered physical levels. The label corresponds to the index of the associated physical level.

    Returns
    -------
    list of list
        [[virtual], [source], [intermediate], [sink], [isolated]]
        each sublist contains at idx = l the number of each kind of node in the network at level l
    """
    virtu = []
    iso = []
    so = []
    inter = []
    si = []
    for l in levels:
        feature = {'node': {'_beam': [l, operator.eq]}}
        comp = utility.selector(graph, feature)

        tpls = nodeKindNumber(comp)

        virtu.append(tpls[NodeKind.VIRTUAL.value])
        iso.append(tpls[NodeKind.ISOLATED.value])
        so.append(tpls[NodeKind.SOURCE.value])
        inter.append(tpls[NodeKind.INTERMEDIATE.value])
        si.append(tpls[NodeKind.SINK.value])

    return [virtu, so, inter, si, iso]

def structureCentroid(component):
    """
    measure geometrical centroid of a structure

    Parameters
    ----------
    component : networkx.network
        network containing the nodes to compute centroid on

    Returns
    -------
    x array_like, y array_like
        x and y coordinates of the centroid
    """
    x = np.mean([x_coord
                 for node, x_coord in component.nodes("_X")
                 if component.nodes[node]['_Kind'].value in (1, 4)])

    y = np.mean([y_coord
                 for node, y_coord in component.nodes("_Y")
                 if component.nodes[node]['_Kind'].value in (1, 4)])
    return x, y


def cumulativeSource(graph, levels):
    """
    Count and return the number of source node that are in scales higher than the scale associated to the index of the list

    Parameters
    ----------
    graph : networkx.network
        network to use
    levels : list of float
        contains the ordered physical levels. The label corresponds to the index of the associated physical level.

    Returns
    -------
    list
        idx correspond to level[idx], for example if level[idx] is 10kAU, list[idx] contains the number of source nodes
        that are localised in scales >= 10kAU
    """
    return [len([node
                 for node, r in graph.nodes('_beam')
                 if graph.nodes[node]['_Kind'].value == 1 and r >= l])
            for l in levels]


def productivityPerSource(graph, levels, Nfrag=None):
    """
    Compute the productivity at each level by comparing the total number of nodes in this level with the number of
    sources in higher levels

    Parameters
    ----------
    graph : networkx.network
        network to use
    levels : list of float
        contains the ordered physical levels. The label corresponds to the index of the associated physical level.
    Nfrag : optional, default is None
        output of nodeKindRepartition()
        if None, compute nodeKindRepartition()

    Returns
    -------
    list
        contains the productivity at each level
    """
    if Nfrag is None:
        Nfrag = nodeKindRepartition(graph, levels)

    norm = cumulativeSource(graph, levels)
    productivity_parents = []
    for i, l in enumerate(levels):
        if norm[i]:
            productivity_parents.append(np.sum(Nfrag, axis=0)[i] / norm[i])
        else:
            productivity_parents.append(0)

        if productivity_parents[i] == 0:
            productivity_parents[i] = np.nan

    return productivity_parents


def productivityScaleByScale(graph, levels, Nfrag=None):
    """
    Compute the productivity at each level l considering the number at level l+1 (higher level):
        (N_l - Nsources_l) / N_{l+1}

    Parameters
    ----------
    graph : networkx.network
        network to use
    levels : list of float
        contains the ordered physical levels. The label corresponds to the index of the associated physical level.
    Nfrag : optional, default is None
        output of nodeKindRepartition()
        if None, compute nodeKindRepartition()

    Returns
    -------
    list
        contains the productivity at each level
    """
    if Nfrag is None:
        Nfrag = nodeKindRepartition(graph, levels)

    Nfragl = np.sum(Nfrag, axis=0)
    productivitySbS = []
    for i, l in enumerate(levels[:-1]):

        if Nfragl[i + 1]:
            productivitySbS.append((Nfragl[i] - Nfrag[1][i]) / Nfragl[i + 1])
        else:
            productivitySbS.append(np.nan)

    if np.sum(Nfrag, axis=0)[-1]:
        productivitySbS.append(1)
    else:
        productivitySbS.append(np.nan)

    return productivitySbS


def holesPerSource(graph, levels, Nholes=None):
    """
    #### TO BE CHECKED ####

    Parameters
    ----------
    graph : networkx.network
        network to use
    levels : list of float
        contains the ordered physical levels. The label corresponds to the index of the associated physical level.
    Nholes : optional, default is None
        if None, compute the total number of holes for intermediates and sinks nodes in graph

    Returns
    -------
    list
        contains the productivity at each level
    """
    if Nholes is None:
        Nholes = np.sum([nhole for n, nhole in graph.nodes('_Holes') if graph.nodes[n]["_Kind"].value in (3, 4)])

    norm = cumulativeSource(graph, levels)
    Nholes_parents = []
    for i, l in enumerate(levels):

        if norm[i]:
            Nholes_parents.append(Nholes[1][i] / norm[i])
        else:
            Nholes_parents.append(0)

    return Nholes_parents

def setMode(Nfrag):
    """
    Determine the mode of fragmentation of a structure considering the number of node type at each level

    Parameters
    ----------
    Nfrag : list, output of nodeKindRepartition()

    Returns
    -------
    int
        the mode of fragmentation associated to the organisation of nodes in the levels
    """
    Nfragl = np.sum(Nfrag, axis=0)
    if sum(Nfrag[NodeKind.ISOLATED.value]):
        return StructureMode.ISOLATED
    elif all([x <= 1 for x in Nfragl]) and sum(Nfrag[3]) == 1:
        return StructureMode.LINEAR
    else:
        return StructureMode.HIERARCHICAL