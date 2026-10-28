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
                sl = sorted(list(getSetAttribute(subg, "_beam")))
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

            level_max = levels.index(graph.nodes[u]['_beam'])
            level_min = levels.index(graph.nodes[v]['_beam'])

            virtual_idx_list = []
            for level in range(level_min + 1, level_max):
                tpl = (f"virtual-{virtual_idx}", 
                       dict(_beam=levels[level], _Kind=ln.NodeKind.VIRTUAL, _level=level)) 
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
    return sorted(list(getSetAttribute(graph, "_beam")))  # low scale to high scale

def getNodeAttributes(graph, attribute):
    return [att for _, att in graph.nodes(attribute)]

def getNodeAttributeName(graph, show=True):
    st = set(k for n in graph.nodes for k in graph.nodes[n].keys())
    if show:
        print('\n Nodes attributes : \n', st, '\n')
    else:
        return st

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