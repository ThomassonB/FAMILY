"""
network_utility.py
==================

Utility functions for the manipulation, analysis, and characterisation of
hierarchical source networks built by the FAMILY pipeline.

In FAMILY, a *network* (directed acyclic graph, DAG) encodes the spatial
relationships between compact sources (clumps) detected at multiple angular
resolutions / wavelengths.  Each node represents one source at one
observational scale (characterised by its beam FWHM in arcseconds, stored as
the ``_beam`` node attribute), and each directed edge connects a source at a
coarser scale to a source at a finer scale when they spatially overlap above a
user-defined threshold.

The functions provided here cover:

- **Graph filtering** -- :func:`selector`
- **Edge labelling** -- :func:`labelDirectedges`
- **Fractal dimension** -- :func:`fractality`
- **Virtual node insertion** -- :func:`virtualNodes`
- **Graph traversal helpers** -- :func:`getComponents`, :func:`getSetAttribute`,
  :func:`getLevels`, :func:`getNodeAttributes`, :func:`getEdgeAttributes`,
  :func:`getNodeAttributeName`, :func:`getEdgeAttributeName`
- **Graph manipulation** -- :func:`addComponents`, :func:`replaceComponents`,
  :func:`deleteNode`
- **Pixel statistics** -- :func:`meanPix`, :func:`maxPix`, :func:`ListPix`
- **I/O** -- :func:`saveNetwork`, :func:`loadNetwork`
- **Statistical summaries** -- :func:`statistics`, :func:`transitiveTriplets`
- **DataFrame export** -- :func:`toDataFrame`
- **Fractal scaling helpers** -- :func:`ScaledtoReal`, :func:`RealtoScaled`,
  :func:`ScaletoHave`

Dependencies
------------
networkx, numpy, scipy (optional, for :func:`fractality`),
pandas (optional, for :func:`toDataFrame`), pickle, tkinter (for I/O dialogs)
"""

import networkx as nx
import numpy as np

import operator
import copy
import time

############################################################
############################################################

# ------------- misc. utilitaries functions ----------------

############################################################
############################################################

def selector(graph, feature):
    """
    Reduce the network by selecting specific features. See feature in Parameters section below for more information.

    Parameters
    ----------
    graph : networkx.network
        the graph to filter
    feature : dict of dict { node / edges : { attribute : ( value, operator ) } }
        contains as a first key 'node' and/or 'edges' string to determine which part of the network to filter.
        The associated value is a dict in which the key is the attribute name and the value is a list or a tuple.
        This list/tuple contains the mathematical operation used to filter and the value of reference for this operation.

        The operator variable is an operator object, see https://docs.python.org/3/library/operator.html for a complete
        list of available operations.

        For example:
            { 'node' : { 'X' : (100, operator.gt) } selects all the nodes which possess a 'X' coordinate greater than 100

    Returns
    -------
    networkx.network
        a copy of graph filtered by the desire selection
    """
    if feature is None:
        return graph

    n = set()
    e = set()
    graphc = copy.deepcopy(graph)
    for obj, dic in feature.items():
        for feat, container in dic.items():
            value, op = container
            if obj == 'edge':
                [e.add((u, v)) for u, v, val in graphc.edges.data(feat) if op(value, val)]
            elif obj == 'node':
                [n.add(node) for node, val in graphc.nodes(feat) if op(value, val)]
            else:
                print(obj + " is not a valid object.")
    if e:
        graphc = graphc.edge_subgraph(e)
    if n or (not e and not n):
        graphc = graphc.subgraph(n)
    return graphc

def labelDirectedges(graph):
    """
    Label edges of the network as *direct* or *non-direct* in-place.

    A *direct* edge connects two nodes at adjacent resolution levels
    (i.e. level ``l`` → level ``l-1``) and is therefore the only path
    between the two nodes in the DAG.  Edges that *skip* one or more
    intermediate levels (level ``l`` → level ``l-2``, ``l-3``, …) are
    labelled as non-direct because the existence of a shorter path through
    intermediate nodes implies that the long-range edge is redundant for the
    purpose of connectivity.

    The boolean edge attribute ``'dir'`` is set to ``True`` for direct edges
    and ``False`` otherwise.  This labelling is used downstream by
    :func:`virtualNodes` to decide where virtual (placeholder) nodes should
    be inserted to fill resolution gaps.

    Parameters
    ----------
    graph : networkx.DiGraph
        The hierarchical source network.  Modified **in-place**.
    """
    
    nx.set_edge_attributes(graph, False, 'dir')
    for u, v in graph.edges:
        i = 0
        for _ in nx.all_simple_paths(graph, u, v):
            i += 1
            if i > 1:
                # More than one path exists → the edge is non-direct (redundant)
                break

        if i == 1:
            # Unique path → the edge is a direct, level-adjacent connection
            graph[u][v]['dir'] = True

def fractality(graph, eta=2):
    """
    Compute and store the *fractality coefficient* for every source node.

    The fractality coefficient ``α`` (stored as the ``'_Fractality'`` node
    attribute on source/isolated nodes) quantifies how the number of
    connected sources scales with the observation resolution.  It is derived
    from the mass–size fractal relationship:

    .. math::

        N = \\sum_{i} \\alpha^{\\gamma_i}, \\quad
        \\gamma_i = \\frac{\\ln(\\lambda_{\\max} / \\lambda_i)}{\\ln \\eta}

    where ``N`` is the total number of nodes in the connected sub-network
    rooted at a given source node, ``λ_i`` is the beam size at level ``i``,
    ``λ_max`` is the coarsest (largest) beam size, and ``η`` is a reference
    scaling ratio (default ``η = 2``).

    The coefficient ``α`` is found by numerically solving the above equation
    using a Levenberg–Marquardt root-finding algorithm
    (``scipy.optimize.root`` with ``method='lm'``).

    The result ``α`` is related to the 2-D fragmentation rate ``φ`` by:

    .. math::

        \\phi_{\\rm 2D} = \\frac{\\ln \\alpha}{\\ln 2}

    Parameters
    ----------
    graph : networkx.DiGraph
        The hierarchical source network.  The ``'_Fractality'`` attribute is
        set **in-place** on source and isolated nodes.
    eta : float, optional
        Reference spatial scaling ratio between consecutive resolution levels.
        Default is ``2`` (i.e. each coarser level covers twice the spatial
        scale of the previous one).

    Notes
    -----
    Only nodes whose ``'_Kind'`` attribute has an integer value of ``1``
    (source — coarsest node of a hierarchical structure) or ``4`` (isolated
    — single-level structure) are processed.  The Levenberg–Marquardt solver
    is preferred over other methods because the objective function is
    non-standard and poorly conditioned for generic solvers.

    See Also
    --------
    virtualNodes : insert virtual nodes before computing fractality to account
        for missing resolution levels.
    """
    from scipy import optimize

    def f(alpha, res, N):
        """
        Residual function for the fractality root-finding problem.

        Parameters
        ----------
        alpha : float
            Current estimate of the fractality coefficient.
        res : array_like
            Sequence of ``γ_i`` exponents, one per resolution level.
        N : int
            Total number of nodes in the sub-network.

        Returns
        -------
        float
            Residual ``N - Σ α^{γ_i}``, should equal zero at the solution.
        """
        S = 0
        for r in res:
            S += alpha ** r
        return N - S

    for g in getComponents(graph):
        for node, kind in g.nodes('_Kind'):
            if kind.value in (1, 4):  # source node or isolated (single-level) node
                # Collect all nodes reachable from this source node
                conn = [node] + [v for v in g.nodes if nx.has_path(g, node, v)]
                subg = g.subgraph(conn)

                # Resolution levels (beam sizes) present in the sub-network, sorted ascending
                sl = sorted(list(getSetAttribute(subg, "_beam")))

                # γ exponents: logarithmic scale ratio relative to the coarsest level
                gamma = np.log(max(sl) / np.array(sl)) / np.log(eta)

                N = len(subg.nodes)

                # Solve N = Σ α^γ_i numerically; Levenberg-Marquardt is robust for this equation
                sol = optimize.root(f, 0.5, args=(gamma, N), method='lm')
                graph.nodes[node]['_Fractality'] = sol.x

def virtualNodes(graph, levels):
    """
    Fill resolution gaps in the network by inserting *virtual* (placeholder) nodes.

    In multi-resolution source catalogues it is common for a compact source
    detected at a coarse resolution to have no counterpart at one or more
    intermediate resolutions.  Such *missing levels* create edges in the DAG
    that skip one or more resolution steps (``_deltal > 1``) and introduce a
    bias in the fractality computation because the implicit assumption of the
    mass–size fractal model is that each hierarchy level is sampled.

    This function inserts synthetic *virtual* nodes (``_Kind = NodeKind.VIRTUAL``)
    at the missing intermediate levels and rewires the edges so that every
    direct connection spans exactly one resolution step.  The returned graph
    can then be passed to :func:`fractality` to obtain an upper-bound estimate
    of the fractality coefficient (the actual value bracketed between the
    raw-network estimate and the virtual-node estimate).

    Only *direct* edges (``graph.edges[u, v]['dir'] == True``, as set by
    :func:`labelDirectedges`) are affected; redundant edges are left unchanged.

    Parameters
    ----------
    graph : networkx.DiGraph
        The hierarchical source network, with edge attributes ``'_deltal'``
        (level difference) and ``'dir'`` (directness flag) already set.
    levels : list of float
        Ordered list of beam sizes (in arcseconds), from the finest to the
        coarsest resolution.  The position of a beam size in this list
        determines the integer *level index* used internally.

    Returns
    -------
    networkx.DiGraph
        A deep copy of ``graph`` with virtual nodes and edges inserted.
        The original graph is **not** modified.

    Notes
    -----
    Virtual nodes are named ``'virtual-<idx>'`` where ``<idx>`` is an
    auto-incremented integer counter.  Each virtual node carries the beam
    attribute of the level it fills and ``_Kind = NodeKind.VIRTUAL``.
    Virtual edges receive ``_virtual=True`` and ``_weight=1``.
    """
    from . import label_nodes as ln

    g = copy.deepcopy(graph)
    nbunch = []  # list of (node_id, attr_dict) tuples to add
    ebunch = []  # list of (u, v, attr_dict) tuples to add
    virtual_idx = 1
    for u, v, dl in graph.edges.data('_deltal'):
        if (dl != 1) and graph.edges[u, v]["dir"]:
            # This direct edge skips one or more resolution levels → fill the gap

            level_max = levels.index(graph.nodes[u]['_beam'])  # coarser-end level index
            level_min = levels.index(graph.nodes[v]['_beam'])  # finer-end level index

            virtual_idx_list = []  # indices of virtual nodes created for this gap
            for level in range(level_min + 1, level_max):
                # Create one virtual node per missing intermediate level
                tpl = (f"virtual-{virtual_idx}",
                       dict(_beam=levels[level], _Kind=ln.NodeKind.VIRTUAL, _level=level))
                nbunch.append(tpl)

                virtual_idx_list.append(virtual_idx)
                virtual_idx += 1

            # Wire the virtual nodes into a chain connecting u (coarse) → … → v (fine)
            zipper = zip(range(level_min + 1, level_max), virtual_idx_list)
            for level, idx in zipper:

                if len(virtual_idx_list) == 1:
                    # Single missing level: u → virtual → v
                    tpls = [
                        (f"virtual-{idx}", v, dict(_virtual=True, _weight=1)),
                        (u, f"virtual-{idx}", dict(_virtual=True, _weight=1))]

                elif level == level_min + 1:
                    # Finest virtual node in the chain: virtual → v
                    tpls = [(f"virtual-{idx}", v, dict(_virtual=True, _weight=1))]

                elif level == level_max - 1:
                    # Coarsest virtual node in the chain: u → virtual, virtual → previous virtual
                    tpls = [
                        (u, f"virtual-{idx}", dict(_virtual=True, _weight=1)),
                        (f"virtual-{idx}", f"virtual-{idx-1}", dict(_virtual=True, _weight=1))]
                else:
                    # Intermediate virtual node: virtual → previous virtual
                    tpls = [(f"virtual-{idx}", f"virtual-{idx-1}", dict(_virtual=True, _weight=1))]

                ebunch += tpls

    g.add_nodes_from(nbunch)
    g.add_edges_from(ebunch)
    return g


def getSetAttribute(graph, attr):
    """
    Return the set of unique values for a given node attribute across the whole graph.

    Parameters
    ----------
    graph : networkx.DiGraph
        The network to query.
    attr : str
        Name of the node attribute.

    Returns
    -------
    set
        Unique attribute values present in the graph.
    """
    return set(att for _, att in graph.nodes(attr))


def getLevels(graph):
    """
    Return the sorted list of unique beam sizes (resolution levels) present in the graph.

    The beam size in arcseconds is stored in the ``'_beam'`` node attribute.
    Levels are returned in ascending order (finest resolution first).

    Parameters
    ----------
    graph : networkx.DiGraph
        The hierarchical source network.

    Returns
    -------
    list of float
        Sorted beam sizes from finest (smallest) to coarsest (largest).
    """
    return sorted(list(getSetAttribute(graph, "_beam")))  # low scale to high scale


def getNodeAttributes(graph, attribute):
    """
    Return a list of values for a given node attribute, in node-iteration order.

    Parameters
    ----------
    graph : networkx.DiGraph
        The network to query.
    attribute : str
        Name of the node attribute.

    Returns
    -------
    list
        Attribute values for all nodes, in the order returned by
        ``graph.nodes()``.
    """
    return [att for _, att in graph.nodes(attribute)]


def getEdgeAttributes(graph, attribute):
    """
    Return a list of values for a given edge attribute, in edge-iteration order.

    Parameters
    ----------
    graph : networkx.DiGraph
        The network to query.
    attribute : str
        Name of the edge attribute.

    Returns
    -------
    list
        Attribute values for all edges.
    """
    return [att for _, att in graph.edges.data(attribute)]


def getNodeAttributeName(graph, show=True):
    """
    Retrieve all node attribute names present in the graph.

    Parameters
    ----------
    graph : networkx.DiGraph
        The network to inspect.
    show : bool, optional
        If ``True`` (default), print the attribute names to stdout.
        If ``False``, return them as a set.

    Returns
    -------
    set or None
        Set of attribute name strings when ``show=False``; ``None`` otherwise.
    """
    st = set(k for n in graph.nodes for k in graph.nodes[n].keys())
    if show:
        print('\n Nodes attributes : \n', st, '\n')
    else:
        return st


def getEdgeAttributeName(graph, show=True):
    """
    Retrieve all edge attribute names present in the graph.

    Parameters
    ----------
    graph : networkx.DiGraph
        The network to inspect.
    show : bool, optional
        If ``True`` (default), print the attribute names to stdout.
        If ``False``, return them as a set.

    Returns
    -------
    set
        Set of attribute name strings.
    """
    st = set(k for u, v in graph.edges for k in graph.edges[u, v].keys())
    if show:
        print('\n Edges attributes : \n', st, '\n')
    return st


def addComponents(graph, components):
    """
    Merge a list of sub-graphs into ``graph`` using :func:`networkx.compose`.

    Parameters
    ----------
    graph : networkx.DiGraph
        Base graph to merge into.
    components : iterable of networkx.DiGraph
        Sub-graphs to add.

    Returns
    -------
    networkx.DiGraph
        The merged graph containing all nodes and edges from ``graph`` and
        every element of ``components``.
    """
    for c in components:
        graph = nx.compose(graph, c)
    return graph


def replaceComponents(graph, components):
    """
    Replace specific connected components in a graph with updated versions.

    All nodes belonging to each sub-graph in ``components`` are first removed
    from ``graph``, then the updated sub-graphs are merged back in.  This is
    useful when a component has been modified externally and needs to be
    reintegrated into the full network.

    Parameters
    ----------
    graph : networkx.DiGraph
        The full network to update.
    components : iterable of networkx.DiGraph
        Updated sub-graphs whose nodes should replace their counterparts in
        ``graph``.

    Returns
    -------
    networkx.DiGraph
        Updated graph with the components replaced.
    """
    [graph.remove_nodes_from(list(c.nodes)) for c in components]
    graph = addComponents(graph, components)
    return graph


def meanPix(graph, image, name, verbose=False):
    """
    Compute the mean pixel value inside each source polygon and store it as a node attribute.

    For each node in ``graph``, the mean of all image pixel values that fall
    within the source's footprint polygon (``'Polygon'`` node attribute) is
    computed and stored under the attribute ``name``.

    Parameters
    ----------
    graph : networkx.DiGraph
        The network whose nodes have a ``'Polygon'`` attribute containing
        Shapely polygon objects representing the source footprints.
    image : array_like
        2-D array of pixel values (e.g. a flux map).
    name : str
        Name of the new node attribute where the mean pixel value is stored.
    verbose : bool, optional
        If ``True``, print progress information.  Default is ``False``.
    """
    from imageutility import MeanPixelInPolygon

    polygons = [poly for _, poly in graph.nodes('Polygon')]
    pixels = MeanPixelInPolygon(polygons, image, verbose=verbose)
    attrs = {node: {name: pix} for node, pix in zip(graph.nodes, pixels)}
    nx.set_node_attributes(graph, attrs)


def maxPix(graph, image, name, verbose=False):
    """
    Compute the maximum pixel value inside each source polygon and store it as a node attribute.

    Parameters
    ----------
    graph : networkx.DiGraph
        The network whose nodes have a ``'Polygon'`` attribute.
    image : array_like
        2-D array of pixel values.
    name : str
        Name of the new node attribute where the maximum pixel value is stored.
    verbose : bool, optional
        If ``True``, print progress information.  Default is ``False``.
    """
    from imageutility import MaxPixelInPolygon

    polygons = [poly for n, poly in graph.nodes('Polygon')]
    pixels = MaxPixelInPolygon(polygons, image, verbose=verbose)
    attrs = {node: {name: pix} for node, pix in zip(graph.nodes, pixels)}
    nx.set_node_attributes(graph, attrs)


def ListPix(graph, image, name, verbose=False):
    """
    Collect the list of all pixel values inside each source polygon and store them as a node attribute.

    Unlike :func:`meanPix` and :func:`maxPix` which store scalar summaries,
    this function stores the full distribution of pixel values within each
    source footprint, which can be used for more detailed statistical analyses
    (e.g. computing the PDF of column densities per source).

    Parameters
    ----------
    graph : networkx.DiGraph
        The network whose nodes have a ``'Polygon'`` attribute.
    image : array_like
        2-D array of pixel values.
    name : str
        Name of the new node attribute where the pixel list is stored.
    verbose : bool, optional
        If ``True``, print progress information.  Default is ``False``.
    """
    from imageutility import PixelsInPolygon

    polygons = [poly for n, poly in graph.nodes('Polygon')]
    pixels = PixelsInPolygon(polygons, image, verbose=verbose)
    attrs = {node: {name: pix} for node, pix in zip(graph.nodes, pixels)}
    nx.set_node_attributes(graph, attrs)


def getComponents(graph, n="all"):
    """
    Return the weakly connected components of the network.

    In the context of FAMILY, each weakly connected component corresponds to
    one *hierarchical structure* (or one isolated source), i.e. a set of
    sources at different resolution levels that are spatially related through
    the overlap criterion.

    Parameters
    ----------
    graph : networkx.DiGraph
        The full hierarchical source network.
    n : "all" or int or list of int, optional
        Which components to return:

        - ``"all"`` (default) — return all components.
        - ``int`` — return the single component at that index (as a list of
          one element for consistency).
        - ``list of int`` — return the components at those indices.

    Returns
    -------
    list of networkx.DiGraph
        Copies of the requested weakly connected sub-graphs.
    """
    clst = [graph.subgraph(c).copy() for c in nx.weakly_connected_components(graph)]

    if n == "all":
        return clst

    elif type(n) is list:
        return [clst[idx] for idx in n]

    elif type(n) is int:
        return [clst[n]]

    else:
        print("n has to be list or integer")
        return


def deleteNode(graph, feature, **kwargs):
    """
    Remove nodes matching a feature selector from a copy of the graph and re-index.

    Nodes are identified using :func:`selector` with the provided ``feature``
    dictionary.  After removal the remaining nodes are relabelled with
    consecutive integers starting from ``0`` to maintain a compact index.

    Parameters
    ----------
    graph : networkx.DiGraph
        The source network.
    feature : dict
        Feature selector dictionary as accepted by :func:`selector`.
        If empty, no nodes are removed.
    **kwargs
        Additional keyword arguments (currently unused; reserved for future
        extensions).

    Returns
    -------
    networkx.DiGraph or None
        A new graph with the selected nodes removed and nodes re-indexed, or
        ``None`` if ``feature`` is empty.
    """
    if len(feature) != 0:
        print("... Deleting indicated nodes")
        g = graph.copy()
        nbunch = list(selector(g, feature).nodes)
        g.remove_nodes_from(nbunch)
        # Re-index nodes with consecutive integers after deletion
        mapping = {node: new_node for new_node, (node, attrs) in enumerate(g.nodes.items())}
        return nx.relabel_nodes(g, mapping)
    else:
        print("... No nodes to be deleted")


def saveNetwork(graph):
    """
    Serialise a network to disk using Python's ``pickle`` format.

    Opens GUI dialogs (via ``tkinter``) to let the user specify the output
    directory and file name.  The network is split into two files:

    - ``<name>/Nodes.pkl`` — dictionary of node attributes.
    - ``<name>/Edges.pkl`` — dictionary-of-dictionaries representation of
      edges and their attributes.

    The directory ``<path>/<name>/`` is created if it does not already exist.

    Parameters
    ----------
    graph : networkx.DiGraph
        The hierarchical source network to save.

    Notes
    -----
    This function requires an active display (e.g. it cannot be run in a
    headless environment).  The companion function :func:`loadNetwork` can be
    used to restore the saved network.
    """
    import pickle
    import os
    from tkinter import filedialog, simpledialog

    time.sleep(0.5)
    name = simpledialog.askstring(title="Enter the file name", prompt=" ")
    time.sleep(0.5)
    path = filedialog.askdirectory()
    time.sleep(0.5)

    file = path + "/" + name

    if not os.path.isdir(file):
        os.makedirs(file)

    nodes = dict(graph.nodes.data())
    edges = nx.to_dict_of_dicts(graph)

    with open(f'{file}/Nodes.pkl', 'wb') as outp:
        pickle.dump(nodes, outp, pickle.HIGHEST_PROTOCOL)
    with open(f'{file}/Edges.pkl', 'wb') as outp:
        pickle.dump(edges, outp, pickle.HIGHEST_PROTOCOL)


def loadNetwork(isdirected=True):
    """
    Restore a network previously saved with :func:`saveNetwork`.

    Opens a GUI directory chooser to locate the folder containing the
    ``Nodes.pkl`` and ``Edges.pkl`` files.  The graph is reconstructed from
    these two files and the sorted array of unique beam sizes (resolution
    levels) is also returned for convenience.

    Parameters
    ----------
    isdirected : bool, optional
        If ``True`` (default), reconstruct the graph as a
        :class:`networkx.DiGraph` (directed).  If ``False``, use an undirected
        :class:`networkx.Graph`.

    Returns
    -------
    graph : networkx.DiGraph or networkx.Graph
        The reconstructed hierarchical source network with all node and edge
        attributes restored.
    scales : numpy.ndarray
        Sorted array of unique beam sizes (in arcseconds) present in the
        network, from finest to coarsest resolution.
    """
    import pickle
    from tkinter import filedialog

    time.sleep(0.5)
    path = filedialog.askdirectory()
    time.sleep(0.5)

    with open(path + '/Nodes.pkl', 'rb') as inp:
        nodes = pickle.load(inp)
    with open(path + '/Edges.pkl', 'rb') as inp:
        edges = pickle.load(inp)

    if isdirected:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()

    graph.add_nodes_from(nodes)
    nx.set_node_attributes(graph, nodes)

    # Reconstruct edge list with attributes from the dict-of-dicts format
    ebunch = [(u, v, attr) for u, d in edges.items() for v, attr in d.items()]
    graph.add_edges_from(ebunch)

    scales = sorted(list(set(dict(nx.get_node_attributes(graph, '_beam')).values())))
    return graph, np.array(scales)

def statistics(graph, attribute):
    """
    Compute descriptive statistics for a numerical node attribute.

    Parameters
    ----------
    graph : networkx.DiGraph
        The network to analyse.
    attribute : str
        Name of the node attribute for which statistics are computed.
        Nodes with a falsy attribute value (``None``, ``0``, etc.) are
        excluded.

    Returns
    -------
    dict
        Dictionary with keys ``'count'``, ``'mean'``, ``'std'``,
        ``'25%'``, ``'50%'`` (median), ``'75%'``, ``'min'``, and ``'max'``.
    """
    d = {}
    lst = [att for _, att in graph.nodes(attribute) if att]

    d["count"] = len(lst)
    d["mean"] = np.mean(lst)
    d["std"] = np.std(lst)
    d["25%"], d["50%"], d["75%"] = np.percentile(lst, [25, 50, 75])
    d["min"] = min(lst)
    d["max"] = max(lst)

    return d


def distanceMatrix(graph, p=0.05, verbose=True):
    """
    Compute the pairwise distance matrix between source polygons in the network.

    Parameters
    ----------
    graph : networkx.DiGraph
        The network whose nodes carry a ``'Polygon'`` attribute (Shapely
        polygon objects representing the source footprints on the sky).
    p : float, optional
        Fractional precision parameter passed to the distance computation
        routine.  Default is ``0.05`` (5 %).
    verbose : bool, optional
        If ``True`` (default), print progress information.

    Returns
    -------
    numpy.ndarray
        Square matrix of shape ``(N, N)`` where ``N`` is the number of nodes,
        containing the pairwise distances between source polygons.
    """
    from polygons import distancePolyst
    polygons = [poly for n, poly in graph.nodes("Polygon")]
    return distancePolyst(polygons, p, verbose)


def transitiveTriplets(digraph):
    """
    Count transitive and non-transitive triplets in a directed graph.

    A *transitive triplet* is a set of three nodes ``(u, v, w)`` such that
    edges ``u → v``, ``v → w``, and ``u → w`` all exist (the path through
    ``v`` is "closed" by a direct edge).  In the context of the hierarchical
    source network, transitive triplets indicate sources that are connected at
    multiple resolution scales simultaneously.

    Parameters
    ----------
    digraph : networkx.DiGraph
        The directed network to analyse.

    Returns
    -------
    triangles : int
        Number of transitive triplets found.
    possibles : int
        Number of open paths ``u → v → w`` for which the closing edge
        ``u → w`` does **not** exist (non-transitive).
    nontrans : list of tuple
        List of ``(u, v, w)`` tuples for each non-transitive triplet.
    """
    triangles = 0
    possibles = 0
    nontrans = []
    for u, v in digraph.edges:
        for w in digraph.out_edges(v):
            if digraph.has_edge(u, w[1]):
                triangles += 1
            else:
                possibles += 1
                nontrans.append((u, v, w[1]))
    return triangles, possibles, nontrans


def toDataFrame(graph, names=None):
    """
    Export node attributes of a network to a :class:`pandas.DataFrame`.

    Each row corresponds to one node; each column corresponds to one node
    attribute.  This is the primary way to inspect source properties
    (coordinates, beam size, polygon, fractality, etc.) in a tabular form.

    Parameters
    ----------
    graph : networkx.DiGraph
        The network to export.
    names : list of str or None, optional
        Subset of attribute names to export.  If ``None`` (default), all
        node attributes found in the graph are exported.

    Returns
    -------
    pandas.DataFrame
        Table of node attributes with one row per node.
    """
    import pandas as pd

    if not names:
        names = getNodeAttributeName(graph, show=False)

    data = {
        attribute: getNodeAttributes(graph, attribute)
        for attribute in names
    }

    return pd.DataFrame(data)



def angularPosition(network):
    """
    Compute the angular separation between connected sources, normalised by the parent source radius.

    For each directed edge ``(in_node, out_node)`` in the network, the angular
    distance on the sky between the centroid of the parent source (``in_node``)
    and the centroid of the child source (``out_node``) is computed and
    normalised by the effective radius ``_R`` of the parent.  This gives a
    dimensionless measure of the relative offset between hierarchically nested
    sources.

    Parameters
    ----------
    network : networkx.DiGraph
        The hierarchical source network.  Nodes must carry ``'_X'``, ``'_Y'``
        (sky-plane coordinates) and ``'_R'`` (effective radius) attributes.

    Returns
    -------
    numpy.ndarray
        Array of normalised angular separations, one per edge.
    """
    from polygons import sepAngular

    sizes = []
    lst = [[], [], [], []]

    for in_node, out_node in network.edges:
        sizes.append(network.nodes("_R")[in_node])
        lst[0].append(network.nodes("_X")[in_node])
        lst[1].append(network.nodes("_Y")[in_node])
        lst[2].append(network.nodes("_X")[out_node])
        lst[3].append(network.nodes("_Y")[out_node])

    return sepAngular(lst[0], lst[1], lst[2], lst[3]) / sizes

def ScaledtoReal(x, r, ro):
    """
    Compute the real value according to the scaled value in the case of a fractal behavior.
    For example if x is defined with respect to a scale reduction of ro, compute the actual value of x at the specific
    scaling ratio r

    Parameters
    ----------
    x : float
        value of reference when the scale is reduced by a factor ro
    r : array_like or float
        actual scaling ratio
    ro : float
        scaling ratio of reference

    Returns
    -------
    The actual value after a scale reduction of r
    """
    return x ** (np.log(r) / np.log(ro))

def RealtoScaled(x, r, ro):
    """
    Inverse operation as ScaledtoReal

    Compute the scaled value according to the real value in the case of a fractal behavior.

    Parameters
    ----------
    x : float
        actual value when the scale is reduced by a factor r
    r : array_like or float
        actual scaling ratio
    ro : float
        scaling ratio of reference

    Returns
    -------
    The actual value after a scale reduction of r
    """
    return x ** (np.log(ro) / np.log(r))

def ScaletoHave(xs, xr, ro):
    """
    Compute the scales necessary to get the values xs for a fractal behavior.

    Parameters
    ----------
    xs : array_like or float
        actual value when the scale is reduced by the factor we are looking for
    xr : float
        value of reference after a reduction of ro
    ro : float
        scaling ratio of reference

    Returns
    -------
    The serie of scales we need to get xs values
    """
    return ro ** (np.log(xr) / np.log(xs))
