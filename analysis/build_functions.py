

"""
build_functions.py
==================

Functions to construct the multiscale network (graph) from a set of source
catalogs observed at different angular resolutions.

The network is a directed graph (NetworkX DiGraph) in which:

- **Nodes** represent individual sources (compact objects) extracted from
  astronomical catalogs at a given angular resolution.  Each node carries the
  full set of catalog attributes (position, flux, size, beam, …) together
  with the associated shapely Polygon footprint.

- **Edges** connect sources from different resolution levels whose sky
  footprints spatially overlap.  The overlap fraction is stored as the edge
  weight ``_weight``, and the level separation ``_deltal`` records how many
  resolution steps separate the two connected sources.

The typical call order is::

    graph = networkx.DiGraph()
    addNodes(graph, catalogs, polygons)
    addEdges(graph, polygons, resolutions)

Dependencies
------------
networkx, numpy, shapely (via polygons_utility)
"""

import networkx as nx
import numpy as np
from . import polygons_utility as putility

############################################################
############################################################

# ------------- functions to construct the network ---------

############################################################
############################################################


def generateNodes(polygons, start, prop):
    """
    Yield labelled nodes with their attributes for one resolution level.

    Each source in the catalog is assigned a unique integer label in the global
    network by offsetting its local catalog index by ``start``.  The shapely
    Polygon footprint of each source is attached to its attribute dictionary
    under the key ``_Polygon``.

    Parameters
    ----------
    polygons : list of shapely.geometry.Polygon
        Footprint polygons for all sources at one resolution level, ordered
        consistently with the corresponding catalog rows.
    start : int
        Global index offset for this resolution level.  The k-th source in
        the catalog receives the node label ``k + start``.
    prop : dict of {int : dict}
        Attribute dictionaries for every source in the catalog, keyed by the
        local (zero-based) catalog index.  Each inner dict holds all catalog
        column values for that source (e.g. position, flux, beam size).
        **Modified in place**: the key ``_Polygon`` is added to each inner
        dict.

    Yields
    ------
    tuple : (int, dict)
        ``(node_label, attribute_dict)`` ready to be consumed by
        ``networkx.Graph.add_nodes_from()``.
    """
    for idx, polygon in enumerate(polygons):
        node_number = idx + start
        prop[idx]["_Polygon"] = polygon
        yield (node_number, prop[idx])


def buildNodes(graph, polygons, starts, tables):
    """
    Populate a graph with nodes from all resolution levels.

    Iterates over every resolution level simultaneously and calls
    :func:`generateNodes` to yield the labelled nodes, which are then
    inserted into the graph.  The graph is modified **in place**.

    Parameters
    ----------
    graph : networkx.DiGraph
        Empty (or pre-existing) directed graph to which the nodes are added.
    polygons : list of list of shapely.geometry.Polygon
        Outer list has one entry per resolution level; each inner list
        contains the footprint polygons of all sources at that level.
    starts : array-like of int
        Global index offsets for each resolution level, as returned by
        ``numpy.cumsum([0, n_0, n_1, ...])``.
    tables : list of dict of {int : dict}
        Catalog attribute dictionaries for each resolution level, as returned
        by ``pandas.DataFrame.to_dict(orient='index')``.

    Notes
    -----
    The list comprehension drives the side-effect of adding nodes; its return
    value (a list of ``None``) is discarded.
    """
    [graph.add_nodes_from([tpl for tpl in generateNodes(poly, start, table)])
    for poly, start, table in zip(polygons, starts, tables)]


def addNodes(graph, catalogs, polygons):
    """
    Compute global node labels and add all nodes to the graph.

    This is the high-level entry point for node construction.  It converts
    the list of pandas DataFrames into the format expected by
    :func:`buildNodes` and computes the cumulative index offsets that give
    each source a unique integer label across all resolution levels.

    Parameters
    ----------
    graph : networkx.DiGraph
        Empty directed graph to be populated with nodes.
    catalogs : list of pandas.DataFrame
        One DataFrame per resolution level.  Each row corresponds to one
        source; columns become node attributes.
    polygons : list of list of shapely.geometry.Polygon
        One inner list per resolution level, containing the footprint polygon
        of each source in the same row order as the corresponding DataFrame.

    Notes
    -----
    The global label of source *k* in catalog *i* is
    ``k + starts[i]``, where ``starts = cumsum([0, len(cat_0), len(cat_1), …])``.
    """
    starts = np.cumsum([0] + [len(catalog) for catalog in catalogs])
    tables = [catalog.to_dict(orient='index') for catalog in catalogs]
    buildNodes(graph, polygons, starts, tables)


def generateEdges(polygons, cascade, starts, res):
    """
    Yield directed edges between spatially overlapping sources at different levels.

    For each pair of resolution levels ``(starting, ending)`` listed in
    ``cascade``, the overlap matrix between all source footprints is computed.
    Every non-zero overlap defines a directed edge from the higher-resolution
    (finer) source (``ending`` level) to the lower-resolution (coarser) source
    (``starting`` level).

    Parameters
    ----------
    polygons : list of list of shapely.geometry.Polygon
        One inner list per resolution level (same ordering as catalogs).
    cascade : list of tuple of (int, int)
        Pairs ``(starting, ending)`` that define which level combinations
        should be connected by edges.  ``starting < ending`` ensures edges
        go from coarser to finer resolution levels.
    starts : array-like of int
        Global index offsets for each resolution level (see :func:`addNodes`).
    res : list of float
        Angular resolution values (e.g. in arcsec) for each catalog level,
        in the same order as ``polygons``.  Reserved for potential use in
        linkage or ratio attributes.

    Yields
    ------
    tuple : (int, int, dict)
        ``(from_node, to_node, attribute_dict)`` where

        - ``from_node`` : global label of the finer-resolution source.
        - ``to_node``   : global label of the coarser-resolution source.
        - ``_weight``   : fractional overlap area between the two footprints,
          as returned by :func:`polygons_utility.overlapMatrix`.
        - ``_deltal``   : difference in resolution level indices
          ``ending - starting``.

    Calls
    -----
    :func:`polygons_utility.overlapMatrix`
    """
    for starting, ending in cascade:
        #print(starting, ending)
        area = putility.overlapMatrix(polygons[starting], polygons[ending])
        intersects = np.where(area > 0)
        for row, col in zip(*intersects):
            from_node = col + starts[ending]
            to_node = row + starts[starting]
            prop = dict(_weight=area[row, col],
                        #_linkage=max(res) / res[ending],
                        #_r=res[ending] / res[starting],
                        _deltal=ending - starting)
            yield (from_node, to_node, prop)


def buildEdges(graph, *args):
    """
    Wrap :func:`generateEdges` and insert the resulting edges into a graph.

    All positional arguments beyond ``graph`` are forwarded directly to
    :func:`generateEdges`.  The graph is modified **in place**.

    Parameters
    ----------
    graph : networkx.DiGraph
        Directed graph that already contains all nodes (built with
        :func:`buildNodes`).
    *args :
        Arguments passed transparently to :func:`generateEdges`:
        ``polygons``, ``cascade``, ``starts``, ``res``.
    """
    graph.add_edges_from([tpl for tpl in generateEdges(*args)])


def addEdges(graph, polygons, res):
    """
    Compute the cascade of level pairs and add all overlap edges to the graph.

    This is the high-level entry point for edge construction.  It builds the
    exhaustive list of ordered level pairs ``(i, j)`` with ``i < j`` (the
    *cascade*), computes cumulative offsets, and delegates to
    :func:`buildEdges`.  The graph is modified **in place**.

    Parameters
    ----------
    graph : networkx.DiGraph
        Directed graph populated with nodes (see :func:`addNodes`).
    polygons : list of list of shapely.geometry.Polygon
        One inner list per resolution level, containing the footprint polygon
        of each source.
    res : list of float
        Angular resolution values for each catalog level, in the same order
        as ``polygons``.  Controls how many levels are considered and is
        passed to :func:`generateEdges` for optional use in edge attributes.

    Notes
    -----
    The cascade connects *all* pairs of levels, not only adjacent ones, so
    that sources separated by more than one resolution step are also linked
    if their footprints overlap (``_deltal > 1``).
    """
    starts = np.cumsum([0] + [len(poly) for poly in polygons])
    cascade = [(i, j + 1)
               for i in range(len(res) - 1)
               for j in range(len(res) - 1)
               if i < j + 1]
    buildEdges(graph, polygons, cascade, starts, res)

