"""
multiscale_structures.py
========================
Analysis of multiscale fragmentation structures in star-forming regions.

This module provides tools to characterise the hierarchical, linear, or isolated
fragmentation modes of interstellar gas structures identified across multiple
angular resolution levels (e.g., Herschel multi-wavelength observations).

Each compact source (clump, core, condensation) detected at a given angular
resolution is represented as a node in a directed network (graph). Edges connect
sources that overlap spatially between consecutive resolution levels, encoding
the parent-child fragmentation relationships. This module operates on such
networks to derive physical descriptors of the resulting multiscale structures.

The fragmentation mode classification follows three categories:
    - ISOLATED  : a single source with no multi-level connectivity.
    - LINEAR    : a chain of sources with at most one descendant at each level
                  (no branching, single sink).
    - HIERARCHICAL : a branching tree of sources exhibiting genuine hierarchical
                     fragmentation (multiple sinks or branching nodes).

The fractality metric F is defined as the mean number of child fragments per
source node (productivity), and is related to the 2D fragmentation exponent
phi_2D = log(F) / log(2). The 3D exponent phi_3D is corrected for projection
effects following standard assumptions (see Thomasson et al.).

References
----------
Thomasson B. et al. (in prep.) — FAMILY: Fragmentation Analysis of
Multi-Level Interstellar Yields.

Dependencies
------------
networkx, numpy, shapely, scipy
"""

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
    """
    Enumeration of the three fragmentation modes that can be assigned to a
    multiscale structure.

    Attributes
    ----------
    ISOLATED : int
        The structure contains only a single, non-connected source at one level.
    LINEAR : int
        Sources form a linear chain across levels with no branching
        (at most one descendant per node, exactly one sink).
    HIERARCHICAL : int
        Sources form a branching tree, indicating genuine hierarchical
        fragmentation with multiple sinks or multiple children per node.
    """
    ISOLATED = 1
    LINEAR = 2
    HIERARCHICAL = 3


def getStructures(network):
    """
    Generator that derives and yields :class:`~analyse.Structure` objects from
    each connected component of the multiscale source network.

    For every connected component (i.e. independent multiscale structure) found
    in the network graph, this function computes a comprehensive set of
    descriptors — fragmentation statistics, spatial geometry, fractality, and
    source classification — and packages them into a ``Structure`` instance.

    Parameters
    ----------
    network : networkx.DiGraph
        The full multiscale network where nodes are compact sources (clumps,
        cores, condensations) observed at different angular resolutions
        (``_beam`` attribute), and edges encode parent-child overlaps between
        consecutive resolution levels.

    Yields
    ------
    analyse.Structure
        One ``Structure`` object per connected component of ``network``,
        carrying the following attributes:

        - ``label``       : integer index of the component.
        - ``component``   : the subgraph of ``network`` for this structure.
        - ``sinks``       : total number of sink nodes (finest-scale, leaf sources).
        - ``sources``     : total number of source nodes (coarsest-scale, root sources).
        - ``productivity``: list of mean number of children per source node at each level
                            (see :func:`productivityPerSource`).
        - ``nl``          : total number of nodes per resolution level.
        - ``sourcel``     : number of source nodes per level.
        - ``sinkl``       : number of sink nodes per level.
        - ``mode``        : fragmentation mode (:class:`StructureMode`).
        - ``fractality``  : mean fractality index F = mean productivity over source nodes.
        - ``missed``      : estimated number of undetected (missing) fragments inferred
                            from holes in sink/intermediate nodes.
        - ``percmissed``  : fraction of missing fragments relative to total node count.
        - ``triplets``    : number of transitive triplets in the component graph,
                            a measure of tree-like topology.
        - ``maxR``        : largest angular resolution (beam size) present in this structure.
        - ``YSO``         : number of sink/intermediate nodes associated with a known
                            Young Stellar Object (YSO) classification.
        - ``gas``         : number of sink/intermediate nodes with no YSO classification
                            (pure gas condensations).
        - ``nbunch``      : list of node identifiers in the component.
        - ``polygon``     : :class:`shapely.geometry.Polygon` enclosing the full structure
                            (union of all source polygons).
        - ``size``        : characteristic spatial size of the structure in arcsec
                            (square root of polygon area; multiply by distance in kAU/arcsec
                            to convert to kAU).
        - ``position``    : (x, y) sky coordinates of the polygon centroid.

    Notes
    -----
    The ``_Holes`` node attribute stores the estimated number of undetected
    fragments below the detection threshold for each sink or intermediate node.
    These are summed to provide the ``missed`` and ``percmissed`` statistics,
    which quantify the incompleteness of the fragmentation tree.
    """
    from . import analyse
    from shapely.ops import unary_union

    levels = sorted(list(utility.getSetAttribute(network, "_beam")))
    for idx, component in enumerate(utility.getComponents(network)):

        # ------------------------------------------------------------------
        # Fragmentation statistics
        # ------------------------------------------------------------------
        # Nfrag[kind][level_idx] = number of nodes of that kind at that level
        Nfrag = nodeKindRepartition(component, levels)

        # Total number of nodes per resolution level (summed over all node kinds)
        Nfragl = np.sum(Nfrag, axis=0)

        # Estimate of missed (undetected) fragments: sum over sink and
        # intermediate nodes of their '_Holes' attribute, which encodes
        # how many sub-fragments were expected but not observed.
        Missed = np.sum([nhole for n, nhole in component.nodes('_Holes')
                         if component.nodes[n]["_Kind"].value in (NodeKind.SINK.value, NodeKind.INTERMEDIATE.value)])

        # Mean number of total nodes per source node at each level
        productivity_parents = productivityPerSource(component, levels, Nfrag=Nfrag)

        # Collect source classifications (YSO or None) for sink/intermediate nodes
        classes = [att for n, att in component.nodes('_Class')
                   if component.nodes[n]['_Kind'].value in (3, 4)]

        # ------------------------------------------------------------------
        # Spatial geometry: build the convex hull / union polygon
        # ------------------------------------------------------------------
        polygons = [poly for node, poly in component.nodes('_Polygon')]
        Poly = unary_union(polygons)          # union of all source footprints
        centroid = shp.centroid(Poly).xy      # sky position of the structure centroid

        # ------------------------------------------------------------------
        # Fractality metric
        # ------------------------------------------------------------------
        # Mean fractality F over source and sink nodes only (kinds 1 and 4).
        # F ~ N^(1/n_levels) and is related to the fragmentation exponent:
        #   phi_2D = log(F) / log(2)
        MeanFractality = np.mean([att for n, att in component.nodes('_Fractality')
                                  if component.nodes[n]['_Kind'].value in (1, 4)])

        # Assign the fragmentation mode (ISOLATED / LINEAR / HIERARCHICAL)
        mode = setMode(Nfrag)

        # Fraction of undetected fragments (incompleteness estimator)
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
                        size=np.sqrt(Poly.area),  # typical size in arcsec; multiply by d [kAU/arcsec] to get kAU
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