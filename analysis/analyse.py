

"""
analyse.py — Core analysis module for FAMILY.

This module implements the main data structures and analysis pipeline for studying
hierarchical fragmentation of star-forming regions from multi-resolution observations
(e.g., Herschel PACS/SPIRE continuum maps at different angular resolutions).

The general workflow is:
    1. Load multi-resolution source catalogs into ``Data`` objects and group them
       in a ``DataSet``.
    2. Build a ``Network`` that connects sources across resolution levels via
       spatial overlap (a directed acyclic graph / DAG).
    3. Identify connected components of the DAG and classify them into
       ``Structure`` objects (isolated, linear, or hierarchical), from which
       physical quantities such as the fragmentation rate (fractality) can be
       derived.

References
----------
See the FAMILY repository and associated publication for details on the
fragmentation metric and the multiscale graph-building algorithm.
"""

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
    """
    An ordered collection of :class:`Data` objects, each identified by a unique name.

    This container holds one :class:`Data` instance per angular-resolution level
    (e.g., one per Herschel band).  It supports iteration, membership testing,
    and dictionary-like access by name.

    Parameters
    ----------
    items : iterable of :class:`Data`, optional
        Initial :class:`Data` objects to populate the collection.

    Examples
    --------
    >>> dataset = DataSet(Data(cfg, reader) for cfg in path.glob("*.toml"))
    """

    def __init__(self, items=None):
        self._data = {}
        if items:
            for item in items:
                self.add(item)

    def __getitem__(self, name):
        """Return the :class:`Data` object associated with *name*."""
        return self._data[name]

    def __contains__(self, name):
        """Return ``True`` if a :class:`Data` object with *name* is present."""
        return name in self._data

    def __iter__(self):
        """Iterate over all :class:`Data` objects in insertion order."""
        return iter(self._data.values())

    def add(self, data):
        """
        Add a :class:`Data` object to the collection.

        Parameters
        ----------
        data : :class:`Data`
            The dataset to add. Its ``name`` attribute must be unique within
            this collection.

        Raises
        ------
        ValueError
            If a :class:`Data` object with the same name already exists.
        """
        if data.name in self:
            raise ValueError(f"Nom déjà présent : {data.name}")
        self._data[data.name] = data

    def get(self, name):
        """
        Return the :class:`Data` object associated with *name*.

        Parameters
        ----------
        name : str
            Identifier of the desired dataset.

        Returns
        -------
        :class:`Data`
        """
        return self._data[name]

    def items(self):
        """Return ``(name, Data)`` pairs, analogous to :meth:`dict.items`."""
        return self._data.items()

    def keys(self):
        """Return the names of all stored :class:`Data` objects."""
        return self._data.keys()

    def values(self):
        """Return all :class:`Data` objects in insertion order."""
        return self._data.values()


class Data:
    """
    Container for a single-resolution source catalog and its associated metadata.

    On construction, the TOML configuration file *init_file* is parsed by
    :func:`load.load_data`, which populates instance attributes such as:

    Attributes
    ----------
    path : str or Path
        Path to the configuration file used to initialise this object.
    name : str
        Short label identifying the resolution level (e.g., ``'L01'``).
    df : pandas.DataFrame
        Normalised source catalog (positions, sizes, fluxes, …).
    fits_img : str
        Path to the continuum FITS image from which sources were extracted.
    catalog : str
        Path/name of the original source catalog file.
    beam : float
        Angular resolution (FWHM of the beam) in arcseconds.
    wavelength : float
        Observation wavelength in micrometres.
    distance : float
        Distance to the observed region in parsecs.
    fov_window : shapely.Polygon or int
        Polygon defining the field-of-view extraction window,
        or ``-1`` if the full image is used.
    color : str
        Matplotlib color string used when plotting sources from this dataset.

    Parameters
    ----------
    init_file : str or Path
        Path to the TOML configuration file describing this dataset.
    reader_to_df : callable
        Function ``f(path) -> pandas.DataFrame`` used to parse the source
        catalog into a standardised DataFrame.
    """

    def __init__(self, init_file, reader_to_df):
        self.path = init_file
        # Parse the TOML config and set all metadata as instance attributes
        metadata = load.load_data(file=init_file, reader_to_df=reader_to_df)
        [setattr(self, key, value) for key, value in metadata.items()]

        # Default plot colour; can be overridden with setColor()
        self.color = 'r'

    def __str__(self):
        """Return a human-readable summary of the dataset and its catalog."""
        ok_values = {'comment', 'fits_img', 'catalog', 'beam ["]', 'wavelength [µm]', 'distance [pc]'}
        lines = [f"name \t\t- \t{self.name}"]
        lines.extend(f"{key} \t- \t{value}" for key, value in vars(self).items() if key in ok_values)
        lines.append("")
        lines.append(self.df.to_string())
        return "\n".join(lines)

    def __iter__(self):
        """Iterate over column names of the source catalog DataFrame."""
        for obj in self.df:
            yield obj

    def setColor(self, color):
        """
        Set the display colour for this dataset.

        Parameters
        ----------
        color : str
            Any Matplotlib-compatible colour string (e.g., ``'b'``, ``'#FF0000'``).
        """
        self.color = color

    def setWindow(self, window):
        """
        Define a polygonal field-of-view window for source extraction.

        Parameters
        ----------
        window : array-like of shape (N, 2)
            Sequence of ``(RA, Dec)`` vertices (in degrees) that delimit the
            extraction region.
        """
        self.window = shp.Polygon(window)

    def setDistance(self, distance):
        """
        Override the source distance stored in the configuration file.

        Parameters
        ----------
        distance : float
            Distance to the region in parsecs.
        """
        self.distance = distance

    def addSerie(self, name, array):
        """
        Append an extra column to the source catalog.

        Parameters
        ----------
        name : str
            Column name.
        array : array-like
            Values to assign; must have the same length as the catalog.
        """
        self.catalog[name] = array

    def plot(self, figsize=(20, 15)):
        """
        Display the source ellipses overlaid on the continuum FITS image.

        Uses APLpy for WCS-aware rendering.  Each source footprint is drawn
        as a polygon with the dataset's assigned colour.

        Parameters
        ----------
        figsize : tuple of float, optional
            Width and height of the figure in inches.  Default is ``(20, 15)``.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import aplpy

        fig = plt.figure(num=f"beam:{self.beam}", figsize=figsize)

        f = aplpy.FITSFigure(self.image_path, figure=fig)
        f.show_grayscale(stretch='sqrt')

        # Build Shapely polygons from the source ellipses in the catalog
        polygons = putility.buildPolygons(self.catalog, self.strings, ptype=self.object_type)

        PTS = []
        for poly in polygons:
            x, y = poly.exterior.xy
            PTS.append(putility.reshape_coord_for_poly(x, y))

        f.show_polygons(PTS,
                        facecolor="None", edgecolor=self.color,
                        lw=4, alpha=1)
        return fig


class Network:
    """
    Multiscale hierarchical network connecting sources across resolution levels.

    The network is a directed acyclic graph (DAG) in which nodes represent
    individual sources and directed edges connect a source at a coarser
    resolution (parent) to one or more sources at a finer resolution (children)
    whose footprints spatially overlap with the parent above a given threshold.

    The construction pipeline is:
        1. ``_build_network`` — create nodes and raw overlap edges.
        2. ``_selectOverlap`` — prune edges below *min_overlap*.
        3. ``_cutUndirectedEdges`` — keep only downward (finer → coarser) edges.
        4. ``_prepare`` — label nodes with their hierarchical level.

    After building the network, :meth:`extractStructures` identifies connected
    components and classifies them as isolated, linear, or hierarchical
    (see :class:`Structure`).

    Parameters
    ----------
    dataset : :class:`DataSet`
        Collection of single-resolution catalogs to be cross-matched.
    min_overlap : float, optional
        Minimum fractional area overlap required to draw an edge between two
        sources.  Range ``[0, 1]``.  Default is ``0`` (all overlaps kept).
    graph : networkx.DiGraph, optional
        Pre-built graph to use instead of constructing one from *dataset*.
        Useful when injecting virtual nodes for uncertainty estimation.
    n_poly : int, optional
        Number of vertices used to discretise each source ellipse into a
        polygon.  Higher values give more accurate overlaps at the cost of
        computation time.  Default is ``128``.

    Attributes
    ----------
    network : networkx.DiGraph
        The underlying directed graph.
    levels : tuple of float
        Angular resolutions (arcsec) of all resolution levels, sorted from
        finest to coarsest.
    data : list of :class:`Data`
        Ordered list of the input datasets.
    structures : list of :class:`Structure` or None
        Extracted structures; ``None`` until :meth:`extractStructures` is called.
    """

    def __init__(self, dataset, min_overlap=0, graph=None, n_poly=128):
        self.min_overlap = min_overlap
        self.n_poly = n_poly
        self.structures = None
        self.levels = None
        self.data = list(dataset._data.values())

        if graph is not None:
            # Use a supplied graph (e.g., with virtual nodes for uncertainty analysis)
            self.network = graph
        else:
            if dataset is None:
                raise ValueError("dataset must be provided when graph is None")

            # Sort datasets by angular resolution (finest first)
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
        """Return a concise summary of the network topology."""
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
        """Return ``True`` if *component* is among the network's connected components."""
        return component in self._components

    def __iter__(self):
        """Iterate over all connected components of the network."""
        yield from self.components

    def __len__(self):
        """Return the number of connected components."""
        return len(self.components)

    @staticmethod
    def _sorted_data(dataset):
        """
        Return datasets sorted by angular resolution (finest beam first).

        Parameters
        ----------
        dataset : :class:`DataSet`

        Returns
        -------
        tuple of :class:`Data`
        """
        values = dataset.values()
        return tuple(sorted(values, key=lambda d: d.beam))

    @property
    def components(self):
        """
        List of weakly-connected subgraphs of the network.

        Returns
        -------
        list of networkx.DiGraph
            Each element is the induced subgraph of one connected component.
        """
        return utility.getComponents(self.network)

    def _build_network(self, data):
        """
        Construct the raw DAG from source polygons and overlap edges.

        For each pair of adjacent resolution levels, an edge is drawn from a
        coarser source to every finer source whose polygon intersects with it.
        Node attributes include the source's polygon, beam size, position, and
        photometric properties as read from the catalog.

        Parameters
        ----------
        data : tuple of :class:`Data`
            Datasets sorted by angular resolution (finest first).
        """
        from . import build_functions as bf

        G = nx.DiGraph()

        # Discretise source ellipses into N-vertex polygons
        polygons = [putility.buildPolygons(d.df, N=self.n_poly) for d in data]

        catalogs = [d.df for d in data]
        bf.addNodes(G, catalogs, polygons)

        ang_res = [d.beam for d in data]
        bf.addEdges(G, polygons, ang_res)

        self.network = G

    def _selectOverlap(self):
        """
        Remove edges whose fractional overlap is below ``self.min_overlap``.

        Delegates to :func:`reduce_network.overlapThreshold`.
        """
        from . import reduce_network as reduc_net
        reduc_net.overlapThreshold(self.network, self.min_overlap)

    def _cutUndirectedEdges(self):
        """
        Remove bidirectional (undirected) edges, keeping only parent→child links.

        An edge is considered undirected when both sources overlap each other by
        a similar amount, making the parent–child relationship ambiguous.
        Such edges are labelled with ``dir=0`` and subsequently pruned.
        """
        from . import reduce_network as reduc_net
        utility.labelDirectedges(self.network)
        reduc_net.cutEdges(self.network, efeature={"dir": [0, operator.eq]})

    def _prepare(self, base=2):
        """
        Assign hierarchical level labels to all nodes.

        Parameters
        ----------
        base : int, optional
            Base of the logarithm used for level indexing.  Default is ``2``.
        """
        from . import label_nodes
        label_nodes.prepareNetwork(self.network, base)

    def _build_complete(self, data):
        """
        Run the full network-construction pipeline.

        Sequentially calls :meth:`_build_network`, :meth:`_selectOverlap`,
        :meth:`_cutUndirectedEdges`, and :meth:`_prepare`.

        Parameters
        ----------
        data : tuple of :class:`Data`
            Datasets sorted by angular resolution (finest first).
        """
        self._build_network(data)
        self._selectOverlap()
        self._cutUndirectedEdges()
        self._prepare()

    def getAttributes(self):
        """
        Print all node attribute names present in the network.

        Useful for exploring the graph after construction.
        """
        lst = set(k for n in self.network.nodes for k in self.network.nodes[n].keys())
        prt = "\n"
        for name in lst:
            prt += f"{name}\n"
        print('Nodes attributes : \n', prt, '\n')

    def extractStructures(self):
        """
        Identify and classify all multiscale structures in the network.

        Each weakly-connected component of the DAG is wrapped in a
        :class:`Structure` object and categorised as ISOLATED, LINEAR, or
        HIERARCHICAL depending on its topology.  Results are stored in
        ``self.structures``.

        Returns
        -------
        list of :class:`Structure`
        """
        from . import multiscale_structures as ms
        self.structures = list(ms.getStructures(self.network))
        return self.structures

    def getStructuresTable(self):
        """
        Compile the properties of all extracted structures into a DataFrame.

        The table contains one row per structure and columns corresponding to
        the entries of :attr:`Structure.TABLECONTENT` (mode, fractality,
        position, polygon, …).

        Returns
        -------
        pandas.DataFrame
            Table of structure properties, indexed by structure label.

        Notes
        -----
        :meth:`extractStructures` must be called first; if not,
        a reminder message is printed and ``None`` is returned.
        """
        if hasattr(self, "structures"):
            table = {}
            for structure in self.structures:
                table[structure.label] = dict(structure._convertVars())
            return pd.DataFrame.from_dict(table, orient="index")
        else:
            print('No structures to put in table\n Try to call extractStructures() method first')


class Structure:
    """
    A single multiscale star-forming structure extracted from the network.

    A structure corresponds to one weakly-connected component of the network
    DAG.  Its *mode* reflects the topological complexity of the component:

    - **ISOLATED** — a single source with no parent or child at any level.
    - **LINEAR** — a chain with exactly one source per level (no branching).
    - **HIERARCHICAL** — a branching tree, indicating genuine hierarchical
      fragmentation.

    The *fractality* attribute quantifies the mean number of children per
    parent node, which is related to the 2-D fragmentation rate
    ``phi_2D = log(fractality) / log(2)``.

    Parameters
    ----------
    label : int
        Unique integer identifier for this structure.
    component : networkx.DiGraph
        The induced subgraph representing this structure.
    **kwargs
        Additional scalar attributes (e.g., ``mode``, ``fractality``,
        ``polygon``, ``position``, ``size``) set by
        :func:`multiscale_structures.getStructures`.

    Attributes
    ----------
    label : int
        Unique identifier.
    component : networkx.DiGraph
        Subgraph of the full network corresponding to this structure.
    levels : int
        Number of distinct resolution levels spanned by this structure.
    scales : set of float
        Set of beam sizes (arcsec) present in this structure.
    mode : enum
        Classification of the structure topology (ISOLATED / LINEAR /
        HIERARCHICAL).
    fractality : float
        Mean branching ratio across all non-leaf nodes.
    missed : int
        Number of resolution levels at which no source was detected
        (inferred missing nodes).
    percmissed : float
        Fraction of expected nodes that are missing.
    maxR : float
        Maximum projected radius of the structure in degrees.
    size : float
        Characteristic angular size of the structure (degrees).
    position : tuple of array
        ``(RA, Dec)`` centroid of the structure in degrees.
    polygon : shapely.Polygon
        Convex-hull polygon encompassing all source footprints.
    """

    # Columns exported to the structures table via getStructuresTable()
    TABLECONTENT = (
        "sinks",
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
        "component",
    )

    def __init__(self, label, component, **kwargs):
        self.label = label
        self.component = component
        # Count how many distinct resolution levels the structure spans
        self.levels = len(set(att for node, att in component.nodes('_level')))
        # Collect the set of angular resolutions (beam sizes) present
        self.scales = set(att for node, att in component.nodes('_beam'))

        # Set all additional properties (mode, fractality, polygon, …) as attributes
        [setattr(self, key, value) for key, value in kwargs.items()
         if key not in ('label', 'component', 'levels', 'scales')]

    def __iter__(self):
        """Iterate over node identifiers of the structure's subgraph."""
        for node in self.component.nodes:
            yield node

    def __str__(self):
        """Return a DataFrame-like string representation of the structure."""
        table = {}
        table[self.label] = dict(self._convertVars())
        return pd.DataFrame.from_dict(table, orient="index").__str__()

    def __repr__(self):
        """Return a DataFrame-like repr of the structure."""
        table = {}
        table[self.label] = dict(self._convertVars())
        return pd.DataFrame.from_dict(table, orient="index").__repr__()

    def __contains__(self, item):
        """Return ``True`` if *item* is a node of the structure's subgraph."""
        return item in self.component.nodes

    def __len__(self):
        """Return the total number of source nodes in this structure."""
        return len(self.component.nodes)

    def _convertVars(self):
        """
        Yield ``(column_name, value)`` pairs for tabular export.

        Handles special cases:
        - ``mode`` is exported as its string name.
        - ``position`` is split into separate ``xposition`` and ``yposition``
          columns (RA and Dec in degrees).

        Yields
        ------
        tuple
            ``(column_name, value)`` pairs for the columns listed in
            :attr:`TABLECONTENT`.
        """
        for column, value in vars(self).items():
            if column in self.TABLECONTENT:
                if column == "mode":
                    yield column, value.name
                elif column == "position":
                    # Unpack (RA_array, Dec_array) into scalar columns
                    yield "xposition", value[0][0]
                    yield "yposition", value[1][0]
                else:
                    yield column, value

    def plot(self, image, figsize=(20, 15),
             subset_color={0: "b", 1: "r", 2: "g", 3: "m", 4: "orange",
                           5: "c", 6: "k", 7: "y", 8: "b", 9: "r"},
             network_prop={"node_size": 700, "alpha": 0.75, "with_labels": True, "width": 3},
             **kwargs):
        """
        Produce a two-panel diagnostic figure for this structure.

        The left panel shows the source footprints (coloured by resolution
        level) overlaid on the continuum FITS image, zoomed to the structure's
        extent.  The right panel displays the hierarchical tree (DAG subgraph)
        laid out in a multipartite layout with levels on the horizontal axis.

        Parameters
        ----------
        image : str or astropy.io.fits.HDUList
            Path to a FITS file, or an already-opened FITS object, to use as
            the background image.
        figsize : tuple of float, optional
            Figure size in inches ``(width, height)``.  Default ``(20, 15)``.
        subset_color : dict, optional
            Mapping from resolution level index (int) to a Matplotlib colour
            string.  Used to colour both the polygons and the network nodes.
        network_prop : dict, optional
            Keyword arguments forwarded to :func:`plotter.plotSubGraph` for
            controlling node/edge rendering in the tree panel.
        **kwargs
            Additional keyword arguments (currently unused).

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 2)

        # --- Right panel: hierarchical tree ---
        ax_network = fig.add_subplot(gs[0, 1])
        plotter.plotSubGraph(self.component, subset_color, ax_network, **network_prop)

        # --- Left panel: spatial map with source footprints ---
        import aplpy

        ax_ellipses = fig.add_subplot(gs[0, 0])
        x0 = ax_ellipses.get_position().x0
        y0 = ax_ellipses.get_position().y0
        x1 = ax_ellipses.get_position().x1
        y1 = ax_ellipses.get_position().y1

        dx = x1 - x0
        dy = y1 - y0
        # Remove the Matplotlib axes so APLpy can place its own WCS-aware axes
        ax_ellipses.remove()

        f = aplpy.FITSFigure(image, figure=fig, subplot=[x0, y0, dx, dy])
        f.show_grayscale(stretch='sqrt')

        # Centre the view on the structure's centroid with a margin of 1.5×size
        x, y = self.position
        radius = 1.5 * self.size
        f.recenter(x, y, radius=radius)

        # Collect and sort polygons from coarsest to finest resolution
        P = []
        N = []

        polygons = nx.get_node_attributes(self.component, "_Polygon").items()
        scales = utility.getLevels(self.component)
        for b in scales[::-1]:
            [P.append(p) for n, p in polygons if self.component.nodes[n]["_beam"] == b]
            [N.append(n) for n, p in polygons if self.component.nodes[n]["_beam"] == b]

        # Overlay each source polygon, coloured by its resolution level
        for node, poly in zip(N, P):
            x, y = poly.exterior.xy
            pts = putility.reshape_coord_for_poly(x, y)

            f.show_polygons([pts],
                            facecolor="None",
                            edgecolor=subset_color[self.component.nodes[node]["_level"]],
                            lw=4, alpha=1)
        return fig
