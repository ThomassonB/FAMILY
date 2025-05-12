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
    Add label inplace for edges of graph that does not 'jump' levels i.e that goes from level l to level l-1 and not l-2 or l-3.

    Parameters
    ----------
    graph : networkx.network
        network to labelise
    """
    
    nx.set_edge_attributes(graph, False, 'dir')
    for u, v in graph.edges:
        i = 0
        for _ in nx.all_simple_paths(graph, u, v):
            i += 1
            if i > 1:
                break

        if i == 1:
            graph[u][v]['dir'] = True

def fractality(graph, eta=2):
    from scipy import optimize
    """
    Add _Fractality attribute to source nodes. The coefficient of a node n is computed by considering the subgraph
    composed of all the nodes connected with n through a path. The computation make used of a zero search function.

    Parameters
    ----------
    graph : networkx.network
        network to measure the fractality on
    eta : float
        scaling ratio of reference for fractal consistency

    Calls
    ----------
    scipy.optimize.root()
    """
    def f(alpha, res, N):
        """ Formula used to derive fractality coefficient with a zero search algorithm """
        S = 0
        for r in res:
            S += alpha ** r
        return N - S

    for g in getComponents(graph):
        for node, kind in g.nodes('_Kind'):
            if kind.value in (1, 4): #source or isolated
                conn = [node] + [v for v in g.nodes if nx.has_path(g, node, v)]
                subg = g.subgraph(conn)
                sl = sorted(list(getSetAttribute(subg, "_phlevel")))
                gamma = np.log(max(sl) / np.array(sl)) / np.log(eta)
                N = len(subg.nodes)
                sol = optimize.root(f, 0.5, args=(gamma, N), method='lm') #or df-sane, in either case the other solvers are trash with this function
                graph.nodes[node]['_Fractality'] = sol.x

                #print('Res = ', sl)
                #print("gamma = ", gamma)
                #print("N = ", N)
                #print("x = ", sol.x)

def virtualNodes(graph, levels):
    """
    Create a copy of graph and fill the holes with virtual nodes. Virtual edges are added to connect the components.

    Parameters
    ----------
    graph : networkx.network
        network to labelise
    levels : list of float
        contains the ordered physical levels. The label corresponds to the index of the associated physical level.

    Returns
    -------
    networkx.network
        modified network with virtual nodes added
    """
    from . import label_nodes as ln

    g = copy.deepcopy(graph)
    nbunch = []
    ebunch = []
    virtual_idx = 1
    for u, v, dl in graph.edges.data('_deltal'):
        if (dl != 1) and graph.edges[u, v]["dir"]:

            level_max = levels.index(graph.nodes[u]['_phlevel'])
            level_min = levels.index(graph.nodes[v]['_phlevel'])

            virtual_idx_list = []
            for level in range(level_min + 1, level_max):
                tpl = (f"virtual-{virtual_idx}", 
                       dict(_phlevel=levels[level], _Kind=ln.NodeKind.VIRTUAL, _level=level)) 
                nbunch.append(tpl)

                virtual_idx_list.append(virtual_idx)
                virtual_idx += 1

            zipper = zip(range(level_min + 1, level_max), virtual_idx_list)
            for level, idx in zipper:

                if len(virtual_idx_list) == 1:
                    tpls = [
                        (f"virtual-{idx}", v, dict(_virtual=True, _weight=1)),
                        (u, f"virtual-{idx}", dict(_virtual=True, _weight=1))]
                
                elif level == level_min + 1:
                    tpls = [(f"virtual-{idx}", v, dict(_virtual=True, _weight=1))]
                    
                elif level == level_max - 1:
                    tpls = [
                        (u, f"virtual-{idx}", dict(_virtual=True, _weight=1)),
                        (f"virtual-{idx}", f"virtual-{idx-1}", dict(_virtual=True, _weight=1))]
                else:
                    tpls = [(f"virtual-{idx}", f"virtual-{idx-1}", dict(_virtual=True, _weight=1))]
                
                ebunch += tpls
            
    g.add_nodes_from(nbunch)
    g.add_edges_from(ebunch)
    return g


def getSetAttribute(graph, attr):
    return set(att for _, att in graph.nodes(attr))

def getLevels(graph):
    return sorted(list(getSetAttribute(graph, "_phlevel")))  # low scale to high scale

def getNodeAttributes(graph, attribute):
    return [att for _, att in graph.nodes(attribute)]

def getEdgeAttributes(graph, attribute):
    return [att for _, att in graph.edges.data(attribute)]

def getNodeAttributeName(graph, show=True):
    st = set(k for n in graph.nodes for k in graph.nodes[n].keys())
    if show:
        print('\n Nodes attributes : \n', st, '\n')
    else:
        return st

def getEdgeAttributeName(graph, show=True):
    st = set(k for u, v in graph.edges for k in graph.edges[u, v].keys())
    if show:
        print('\n Edges attributes : \n', st, '\n')
    return st

def addComponents(graph, components):
    for c in components:
        graph = nx.compose(graph, c)
    return graph


def replaceComponents(graph, components):
    [graph.remove_nodes_from(list(c.nodes)) for c in components]
    graph = addComponents(graph, components)
    return graph


def meanPix(graph, image, name, verbose=False):
    from imageutility import MeanPixelInPolygon

    polygons = [poly for _, poly in graph.nodes('Polygon')]
    pixels = MeanPixelInPolygon(polygons, image, verbose=verbose)
    attrs = {node: {name: pix} for node, pix in zip(graph.nodes, pixels)}
    nx.set_node_attributes(graph, attrs)


def maxPix(graph, image, name, verbose=False):
    from imageutility import MaxPixelInPolygon

    polygons = [poly for n, poly in graph.nodes('Polygon')]
    pixels = MaxPixelInPolygon(polygons, image, verbose=verbose)
    attrs = {node: {name: pix} for node, pix in zip(graph.nodes, pixels)}
    nx.set_node_attributes(graph, attrs)


def ListPix(graph, image, name, verbose=False):
    from imageutility import PixelsInPolygon

    polygons = [poly for n, poly in graph.nodes('Polygon')]
    pixels = PixelsInPolygon(polygons, image, verbose=verbose)
    attrs = {node: {name: pix} for node, pix in zip(graph.nodes, pixels)}
    nx.set_node_attributes(graph, attrs)


def getComponents(graph, n="all"):
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
    if len(feature) != 0:
        print("... Deleting indicated nodes")
        g = graph.copy()
        nbunch = list(selector(g, feature).nodes)
        g.remove_nodes_from(nbunch)
        mapping = {node: new_node for new_node, (node, attrs) in enumerate(g.nodes.items())}
        return nx.relabel_nodes(g, mapping)
    else:
        print("... No nodes to be deleted")


def saveNetwork(graph):
    import pickle
    import os
    from tkinter import filedialog, simpledialog

    time.sleep(0.5)
    name = simpledialog.askstring(title="Enter the file name", prompt=" ")
    time.sleep(0.5)
    path = filedialog.askdirectory()
    time.sleep(0.5)

    file = path+"/"+name

    if not os.path.isdir(file):
        os.makedirs(file)

    nodes = dict(graph.nodes.data())
    edges = nx.to_dict_of_dicts(graph)

    with open(f'{file}/Nodes.pkl', 'wb') as outp:
        pickle.dump(nodes, outp, pickle.HIGHEST_PROTOCOL)
    with open(f'{file}/Edges.pkl', 'wb') as outp:
        pickle.dump(edges, outp, pickle.HIGHEST_PROTOCOL)


def loadNetwork(isdirected=True):
    import pickle
    from tkinter import filedialog

    time.sleep(0.5)
    path = filedialog.askdirectory()
    time.sleep(0.5)

    with open(path+'/Nodes.pkl', 'rb') as inp:
        nodes = pickle.load(inp)
    with open(path+'/Edges.pkl', 'rb') as inp:
        edges = pickle.load(inp)

    if isdirected:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()

    graph.add_nodes_from(nodes)
    nx.set_node_attributes(graph, nodes)

    ebunch = [(u, v, attr) for u, d in edges.items() for v, attr in d.items()]

    graph.add_edges_from(ebunch)

    scales = sorted(list(set(dict(nx.get_node_attributes(graph, '_phlevel')).values())))
    return graph, np.array(scales)

def statistics(graph, attribute):
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
    from polygons import distancePolyst
    polygons = [poly for n, poly in graph.nodes("Polygon")]
    return distancePolyst(polygons, p, verbose)


def transitiveTriplets(digraph):
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
    import pandas as pd
    """
    nodes to panda dataframe, names list of attributes
    """
    if not names:
        names = getNodeAttributeName(graph, show=False)

    data = {
        attribute: getNodeAttributes(graph, attribute)
        for attribute in names
    }

    #for i, polygon in enumerate(data["_Polygon"]):
    #    data["_Polygon"][i] = polygon.exterior.xy

    return pd.DataFrame(data)



def angularPosition(network):
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
