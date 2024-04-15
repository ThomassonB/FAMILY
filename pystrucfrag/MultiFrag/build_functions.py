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
    Prepare the nodes labelling and the associated attributes to be put in the final network

    Parameters
    ----------
    polygons : list
        contains shapely.Polygon objects
    start : int
        reference number to start the labelling of nodes
    prop : dict {key:value}
        contains all the entry of the initial catalog
        key is the column name to be passed to an attribute, value is the value of the entry

    Returns
    -------
    generator of tuple
        ( int node label in the final network , dict attributes )
    """
    for idx, polygon in enumerate(polygons):
        node_number = idx + start
        prop[idx]["_Polygon"] = polygon
        yield (node_number, prop[idx])

def buildNodes(graph, polygons, starts, tables):
    """
    Add in place the graph nodes with the attributes of polygons and tables, labels of starts

    Parameters
    ----------
    graph : networkx.network
        Network to place the nodes in, supposed to be empty
    polygons : list of list
        contains lists of shapely.Polygon objects. Each list is associated to one catalog
    starts : list of int
        list of reference number to start the labelling of nodes. These references are associated to the length of the catalogs
    tables : list of dict {key:value}
        contains all the entry of the catalogs
        key is the column name to be passed to an attribute, value is the value of the entry
    """
    [graph.add_nodes_from([tpl for tpl in generateNodes(poly, start, table)])
    for poly, start, table in zip(polygons, starts, tables)]

def addNodes(graph, catalogs, polygons):
    """
    Define the labelling of nodes and call buildNodes to add the nodes with the catalogs attributes

    Parameters
    ----------
    graph : networkx.network
        Network to place the nodes in, supposed to be empty
    catalogs : list of pandas.DataFrame
        list that contains all the initial catalogs
    polygons : list of list
        contains lists of shapely.Polygon objects. Each list is associated to one catalog
    """
    starts = np.cumsum([0] + [len(catalog) for catalog in catalogs])
    tables = [catalog.to_dict(orient='index') for catalog in catalogs]
    buildNodes(graph, polygons, starts, tables)

def generateEdges(polygons, cascade, starts, res):
    """
    Define the labelling of nodes and call buildNodes to add the nodes with the catalogs attributes

    Parameters
    ----------
    polygons : list of shapely.Polygons
        Contains all the polygons of all the catalogs
    cascade : list of tuple
        Contains the incoming node and the outcoming node to define an adge
    starts : list of int
        list of reference number to start the labelling of nodes. These references are associated to the length of the catalogs
    res : list
        Contains the values of resolution in the same order as given in buildNetwork()

    Calls
    ----------
    polygons_utility.overlapMatrix()

    Returns
    ----------
    generator of tuple (incoming node, outcoming node, attributes) to build the edges
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
    """Wrapp generateEdges() in order to add the edges in a networkx.network graph object"""
    graph.add_edges_from([tpl for tpl in generateEdges(*args)])

def addEdges(graph, polygons, res):
    """
    Add edges inplace between the polygons with respect to their spatial covering.

    Parameters
    ----------
    graph : networkx.network
        Network to add the edges on
    polygons : list of list of shapely.Polygons
        Contains the lists of polygons of each catalog. One list correspond to one catalog.
    res : list
        Contains the values of resolution in the same order as given in buildNetwork()
    """
    starts = np.cumsum([0] + [len(poly) for poly in polygons])
    cascade = [(i, j + 1)
               for i in range(len(res) - 1)
               for j in range(len(res) - 1)
               if i < j + 1]
    buildEdges(graph, polygons, cascade, starts, res)

