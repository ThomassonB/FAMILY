import shapely.geometry as shp
import pandas as pd
import networkx as nx
import numpy as np

from . import polygons_utility as putility
from . import network_utility as utility
from . import load, plotter
from .standard_variables import strings_ref, ellipse_params_labels

import operator
import time

import matplotlib.pyplot as plt

class DataSet:
    def __init__(self, items=None):
        self._data = {}
        if items:
            for item in items:
                self.add(item)

    def __getitem__(self, name):
        return self._data[name]

    def __contains__(self, name):
        return name in self._data

    def __iter__(self):
        return iter(self._data.values())

    def add(self, data):
        if data.name in self:
            raise ValueError(f"Nom déjà présent : {data.name}")
        self._data[data.name] = data

    def get(self, name):
        return self._data[name]

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

class Data:
    def __init__(self, init_file, reader_to_df):
        self.path = init_file
        metadata = load.load_data(file=init_file, reader_to_df=reader_to_df)
        [setattr(self, key, value) for key, value in metadata.items()]

        self.color = 'r'

    def __str__(self):
        ok_values = {'comment', 'fits_img', 'catalog', 'beam ["]', 'wavelength [µm]', 'distance [pc]'}
        lines = [f"name \t\t- \t{self.name}"]
        lines.extend(f"{key} \t- \t{value}" for key, value in vars(self).items() if key in ok_values)
        lines.append("")
        lines.append(self.df.to_string())
        return "\n".join(lines)

    def __iter__(self):
        for obj in self.df:
            yield obj

    def setColor(self, color):
        self.color = color

    def setWindow(self, window):
        self.window = shp.Polygon(window)

    def setDistance(self, distance):
        self.distance = distance

    def addSerie(self, name, array):
        self.catalog[name] = array

    def plot(self, figsize=(20, 15)):
        import aplpy

        fig = plt.figure(num=f"beam:{self.beam}", figsize=figsize)

        f = aplpy.FITSFigure(self.image_path, figure=fig)
        f.show_grayscale(stretch='sqrt')

        polygons = putility.buildPolygons(self.catalog, self.strings, ptype = self.object_type)

        PTS = []
        for poly in polygons:
            x, y = poly.exterior.xy
            PTS.append(putility.reshape_coord_for_poly(x, y))

        f.show_polygons(PTS,
                        facecolor="None", edgecolor=self.color,
                        lw=4, alpha=1)
        return fig

class Network:
    def __init__(self, dataset, min_overlap = 0, graph = None, n_poly=128):
        self.min_overlap = min_overlap
        self.n_poly = n_poly
        self.structures = None
        self.levels = None
        
        if graph is not None:
            self.network = graph
        else:
            if dataset is None:
                raise ValueError("dataset must be provided when graph is None")

            data = self._sorted_data(dataset)
            self.levels = tuple(sorted(d.beam for d in data))
            self._build_complete(data)

        # sort the dataset from the lowest level to the highest
        # avoid edges direction problems
        #levels = [data.beam for data in dataset]
        #self.levels, self.data = zip(*sorted(zip(levels, dataset.values())))
        #if graph is None:
        #    self._buildComplete(n_poly=n_poly)
        #else:
        #    self.network = graph

    def __str__(self):
        lines = [
            "Network",
            f"min_overlap - {self.min_overlap}",
            f"n_nodes - {self.network.number_of_nodes()}",
            f"n_edges - {self.network.number_of_edges()}",
            f"n_components - {len(self.components)}",
        ]

        if self.levels is not None:
            lines.append(f"levels - {self.levels}")

        return "\n".join(lines)

    def __contains__(self, component):
        return component in self._components

    def __iter__(self):
        yield from self.components

    def __len__(self):
        return len(self.components)

    @staticmethod
    def _sorted_data(dataset):
        values = dataset.values()
        return tuple(sorted(values, key=lambda d: d.beam))
        
    @property
    def components(self):
        return utility.getComponents(self.network)

    def _build_network(self, data):
        from . import build_functions as bf
        
        G = nx.DiGraph()

        polygons = [putility.buildPolygons(d.df, N=self.n_poly) for d in data]

        catalogs = [d.df for d in data]
        bf.addNodes(G, catalogs, polygons)

        ang_res = [d.beam for d in data]
        bf.addEdges(G, polygons, ang_res)

        self.network = G

    def _selectOverlap(self):
        from . import reduce_network as reduc_net
        reduc_net.overlapThreshold(self.network, self.min_overlap)

    def _cutUndirectedEdges(self):
        from . import reduce_network as reduc_net
        utility.labelDirectedges(self.network)
        reduc_net.cutEdges(self.network, efeature={"dir":[0, operator.eq]})

    def _prepare(self, base=2):
        from . import label_nodes
        label_nodes.prepareNetwork(self.network, base)

    def _build_complete(self, data):
        self._build_network(data)
        self._selectOverlap()
        self._cutUndirectedEdges()
        self._prepare()

    def getAttributes(self):
        lst = set(k for n in self.network.nodes for k in self.network.nodes[n].keys())
        prt = "\n"
        for name in lst:
            prt += f"{name}\n"
        print('Nodes attributes : \n', prt, '\n')

    def extractStructures(self):
        from . import multiscale_structures as ms
        self.structures = list(ms.getStructures(self.network))
        return self.structures  

    def getStructuresTable(self):
        if hasattr(self, "structures"):
            table = {}
            for structure in self.structures:
                table[structure.label] = dict(structure._convertVars())
            return pd.DataFrame.from_dict(table, orient="index")
        else:
            print('No structures to put in table\n Try to call extractStructures() method first')

class Structure:
    TABLECONTENT = ( "sinks",
                     "sources",
                     "mode",
                     "fractality",
                     "missed",
                     "percmissed",
                     #"triplets",
                     "maxR",
                     "YSO",
                     "gas",
                     "size",
                     "position",
                     "polygon",
                     "component"
                   )
    
    def __init__(self, label, component, **kwargs):
        self.label = label
        self.component = component
        self.levels = len(set(att for node, att in component.nodes('_level')))
        self.scales = set(att for node, att in component.nodes('_beam'))

        [setattr(self, key, value) for key, value in kwargs.items() if key not in ('label', 'component', 'levels', 'scales')]

    def __iter__(self):
        for node in self.component.nodes:
            yield node

    def __str__(self):
        table = {}
        table[self.label] = dict(self._convertVars())
        return pd.DataFrame.from_dict(table, orient="index").__str__()

    def __repr__(self):
        table = {}
        table[self.label] = dict(self._convertVars())
        return pd.DataFrame.from_dict(table, orient="index").__repr__()

    def __contains__(self, item):
        return item in self.component.nodes

    def __len__(self):
        return len(self.component.nodes)

    def _convertVars(self):
        for column, value in vars(self).items():
            if column in self.TABLECONTENT:
                if column == "mode":
                    yield column, value.name
                elif column == "position":
                    yield "xposition", value[0][0]
                    yield "yposition", value[1][0]
                else:
                    yield column, value

    def plot(self, image, figsize=(20, 15),
             subset_color={0:"b", 1:"r", 2:"g", 3:"m", 4:"orange", 5:"c", 6:"k", 7:"y", 8:"b", 9:"r"},
             network_prop={"node_size": 700, "alpha": 0.75, "with_labels": True, "width": 3},
             **kwargs):

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 2)
        
        # draw tree
        ax_network = fig.add_subplot(gs[0, 1])
        plotter.plotSubGraph(self.component, subset_color, ax_network, **network_prop)

        # draw map
        import aplpy

        ax_ellipses = fig.add_subplot(gs[0, 0])
        x0 = ax_ellipses.get_position().x0
        y0 = ax_ellipses.get_position().y0
        x1 = ax_ellipses.get_position().x1
        y1 = ax_ellipses.get_position().y1
        
        dx = x1 - x0
        dy = y1 - y0
        #fig.delaxes(ax_ellipses)
        
        ax_ellipses.remove()

        f = aplpy.FITSFigure(image, figure=fig, subplot=[x0, y0, dx, dy])
        f.show_grayscale(stretch='sqrt')

        x, y = self.position
        radius = 1.5 * self.size
        f.recenter(x, y, radius=radius)

        P = []
        N = []

        polygons = nx.get_node_attributes(self.component, "_Polygon").items()
        scales = utility.getLevels(self.component)
        for b in scales[::-1]:
            [P.append(p) for n, p in polygons if self.component.nodes[n]["_beam"] == b]
            [N.append(n) for n, p in polygons if self.component.nodes[n]["_beam"] == b]

        for node, poly in zip(N, P):
            x, y = poly.exterior.xy
            pts = putility.reshape_coord_for_poly(x, y)

            f.show_polygons([pts],
                            facecolor="None",
                            edgecolor=subset_color[self.component.nodes[node]["_level"]], lw=4,
                            alpha=1)
        return fig
