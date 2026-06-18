"""
plotter.py — Visualisation tools for the FAMILY pipeline.

This module provides plotting utilities for the multi-scale hierarchical
structure analysis performed by FAMILY (Fragmentation Analysis of Molecular
clouds In multi-Level hierarchY). It covers:

  - Standalone plotting functions for fragmentation curves, hierarchical
    networks overlaid on astronomical images (via APLpy/FITS), and
    size-binned fragmentation statistics.
  - An interactive Tkinter GUI (``InspectNetwork``) that allows the user
    to browse every detected multi-scale structure: clicking through
    structures updates both the FITS sky map (centred on the structure)
    and the corresponding hierarchical network graph.
  - A companion Tkinter table viewer (``Tables``) with drop-down filters
    for rapid inspection of the structure catalogue.

Dependencies
------------
matplotlib, numpy, networkx, tkinter, astropy, aplpy, and the internal
modules ``network_utility`` and ``image_utility``.
"""

import operator
import tkinter as tk
import matplotlib.pyplot as plt

import matplotlib
import numpy as np
# Force the TkAgg backend so that matplotlib figures can be embedded
# inside Tkinter windows (required for the InspectNetwork GUI).
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter.colorchooser import askcolor

import networkx as nx
from . import network_utility as utility
from . import image_utility as iu
from astropy.io import fits
import aplpy


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def random_color():
    """Return a random RGB colour as a tuple of three integers in [0, 255].

    Used to assign a distinct colour to each observation scale when the
    ``InspectNetwork`` GUI is first initialised.

    Returns
    -------
    tuple of int
        (R, G, B) values drawn uniformly from [0, 255].
    """
    color = np.random.randint(0, 255, size=3)
    return tuple(color)


# ---------------------------------------------------------------------------
# Network / graph plotting
# ---------------------------------------------------------------------------

def plotSubGraph(network, subset_color, ax, **kwargs):
    """Draw a hierarchical network graph with nodes coloured by their scale level.

    Each node in ``network`` corresponds to a source detected at a given
    observation scale (``_level`` attribute). Nodes are laid out horizontally
    by level using a multipartite layout, making the hierarchical structure
    immediately readable.

    Parameters
    ----------
    network : networkx.Graph
        The hierarchical network whose nodes carry a ``_level`` attribute
        encoding the observation scale (integer index).
    subset_color : dict
        Mapping from level index to a matplotlib-compatible colour specifier.
    ax : matplotlib.axes.Axes
        Axes on which to draw the network.
    **kwargs
        Additional keyword arguments forwarded to ``networkx.draw``.
    """
    color = [subset_color[data["_level"]] for v, data in network.nodes(data=True)]
    pos = nx.multipartite_layout(network, subset_key="_level", align="horizontal")
    nx.draw(network, pos, ax=ax, node_color=color, **kwargs)
    plt.axis("equal")


# ---------------------------------------------------------------------------
# Fragmentation curve computation and plotting
# ---------------------------------------------------------------------------

def getFragCurve(network, distance, mode="all"):
    """Compute the mean fragmentation (productivity) curve across all structures.

    For each observation scale level, the *productivity* of a structure is
    defined as the ratio of the number of child sources at that scale to the
    number at the top (parent) level.  This function aggregates productivity
    values across all structures of the requested mode and returns the
    mean ± standard error as a function of physical scale.

    Parameters
    ----------
    network : analyse.Network
        The multi-scale network object containing extracted structures and
        their scale levels.
    distance : float
        Distance to the star-forming region in parsecs (pc). Used to convert
        angular beam sizes (arcsec) to physical scales (AU).
    mode : str, optional
        Structural mode to include. One of ``"all"`` (default),
        ``"HIERARCHICAL"``, ``"LINEAR"``, or ``"ISOLATED"``.

    Returns
    -------
    x : numpy.ndarray
        Physical scales [AU] sorted in ascending order.
    meany : list of float
        Mean productivity at each scale level.
    stdy : list of float
        Standard error (std / sqrt(N)) of the productivity at each scale level.
    """
    def ReshapeData(y):
        """Transpose a 2-D productivity array so that rows correspond to scale levels.

        ``y`` is collected as a list of per-structure arrays; this helper
        reshapes it so that each element of the output is the list of
        productivity values from all structures at a single scale level.
        """
        yl = np.reshape(np.ravel(y), np.shape(y)[::-1], order="F")
        return yl

    # Collect productivity arrays; optionally filter by structural mode.
    if mode == "all":
        y = [structure.productivity for structure in network.structures]
    else:
        y = [structure.productivity for structure in network.structures
            if structure.mode.name == mode]
    
    # Convert scale levels (beam sizes in arcsec) to physical scales in AU.
    x = np.sort(network.levels) * distance

    # Reshape so that yl[i] contains all productivity values at scale i.
    yl = ReshapeData(y)

    # Compute mean and standard error, ignoring NaN (missing levels).
    meany = [np.nanmean(ylst) for ylst in yl]
    stdy = [np.nanstd(ylst) / np.sqrt(len(ylst)) for ylst in yl]

    return x, meany, stdy


def plotFragmentationCurve_Sizes(structures, distance, figs=None, **kwargs):
    """Plot the mean number of sub-structures per size bin as a function of scale.

    For each structure, the sizes (``_R``) of its constituent nodes are
    histogrammed into predefined beam-size bins.  Isolated bins (separated
    from neighbours by empty bins) are masked as NaN to avoid artefacts.
    The mean and standard error across all structures are then plotted as
    error bars on a log-scaled x-axis.

    Parameters
    ----------
    structures : list
        List of structure objects whose ``component`` attribute is a
        ``networkx.Graph`` with node attribute ``_R`` (source radius in AU)
        and whose ``size`` attribute gives the top-level size.
    distance : float
        Distance to the region in parsecs, used to scale the bin edges from
        arcsec-equivalent values to AU.
    figs : tuple of (Figure, Axes) or None, optional
        Existing ``(fig, ax)`` pair to draw on. If ``None`` (default), a new
        figure is created.
    **kwargs
        Additional keyword arguments forwarded to ``ax.errorbar``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    import pandas as pd

    # Bin edges corresponding to the five Herschel PACS/SPIRE beam sizes
    # (arcsec) plus boundary guards, scaled to AU via the source distance.
    bins = np.array([1.9, 2.1, 8.4, 13.5, 18.2, 24.9, 36.3, 100])
    bins = bins * distance

    R_lists = []   # per-structure list of node radii
    R0s = []       # top-level size of each structure (for reference line)
    Rmax = 0       # track the global maximum radius to set the last bin edge
    for s in structures:
        R = []
        R0s.append( s.size )

        for node, size in s.component.nodes('_R'):
            Rmax = max(Rmax, size)
            R.append(size)

        R_lists.append(R)

    # Extend the last bin to cover the largest observed radius.
    bins[-1] = Rmax
    x = (bins[1:] + bins[:-1])/2    # bin centres
    dx = bins[1:] - bins[:-1]        # bin widths (used as x error bars)

    # Matrix: rows = structures, columns = scale bins.
    ys = np.zeros(shape=(len(structures), len(x)))
    here = np.zeros_like(ys)  # mask: 1 where a bin is populated

    for idx, r in enumerate(R_lists):
        count, bins = np.histogram(r, bins=bins)
        df = pd.DataFrame((count != 0))

        # Mask isolated populated bins (surrounded by empty ones) to avoid
        # artefacts from sparse sampling at the extremes of the size range.
        tresh = 1
        df1 = df.cumsum().mask(df)
        m1 = df1.apply(lambda x: x.map(x.value_counts())).le(tresh)
        m2 = df1.ne(df1.iloc[0]) & df1.ne(df1.iloc[-1])

        df[m1 & m2] = np.nan
        arr = np.ravel(df.to_numpy())

        ys[idx, :] = count * arr
        here[idx, :] = arr

    print(ys)

    # Average over structures; standard error accounts for the number of
    # structures that actually contribute to each bin.
    meany = np.nanmean(ys, axis=1)
    stdy = np.nanstd(ys, axis=1) / np.sqrt(np.sum(here, axis=1))

    if figs is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = figs
    ax.errorbar(np.array(x), meany, stdy, np.array(dx), "o", **kwargs)
    print("R0s", R0s)
    # Horizontal dashed reference line at N=1 (no fragmentation).
    ax.plot([min(R0s), max(R0s)], [1, 1], color="k", ls="--")

    ax.set_xscale("log")
    #ax.set_yscale("log")

    #plt.tick_params(
    #    axis='x',  # changes apply to the x-axis
    #    which='both',  # both major and minor ticks are affected
    #)
    #ax.minorticks_off()

    #ax.set_xticks(np.array(x))
    #ax.set_xticklabels(["1.4", "6", "10", "13", "18", "26"])

    #ax.set_ylabel(r"$N(R_l)/N_{sources}$", fontsize=20)
    #ax.set_xlabel("Scale [kAU]", fontsize=20)

    return fig, ax


# ---------------------------------------------------------------------------
# FITS sky-map preparation
# ---------------------------------------------------------------------------

def prepareAplpy(image, path=True, **kwargs):
    """Create a publication-quality APLpy figure from a FITS image.

    Loads a FITS file, masks zero-valued pixels, and sets up an APLpy
    ``FITSFigure`` with a colour scale, colour bar, WCS tick labels,
    coordinate grid, and an optional north-arrow overlay.

    Parameters
    ----------
    image : str or array-like
        Path to the FITS file (if ``path=True``) or an image array.
    path : bool, optional
        If ``True`` (default), ``image`` is treated as a file path and read
        with ``image_utility.OpenImage``.
    **kwargs
        Optional configuration keys:

        * ``fig`` — existing ``matplotlib.figure.Figure`` to draw into.
        * ``figsize`` — tuple passed to ``plt.figure`` when creating a new figure.
        * ``north`` — bool; if ``True``, rotate the map so that North is up.
        * ``subplot`` — APLpy subplot specification (list of 4 floats).
        * ``window`` — ``(x, y, height, width)`` in world coordinates to
          recenter the view.
        * ``cmap`` — colour map name (default ``'Greys'``).
        * ``stretch`` — colour stretch (e.g. ``'log'``, ``'sqrt'``).
        * ``cb label`` — label string for the colour bar axis.
        * ``arrow_north`` — ``(x, y, dx, dy)`` in world coordinates for a
          north-direction arrow annotation.

    Returns
    -------
    fig : aplpy.FITSFigure
        The configured APLpy figure object.
    """
    img, hdr = iu.OpenImage(image, path=path)
    # Replace zero flux values with NaN so they are transparent in the plot.
    img[img == 0] = np.nan
    new_fits = fits.PrimaryHDU(data=img, header=hdr)

    # Create or reuse a matplotlib figure.
    if "fig" not in kwargs:
        fig_all = plt.figure("Map", figsize=kwargs.get("figsize"))
    else:
        fig_all = kwargs.get("fig")

    # Build the APLpy figure, optionally rotating to align North with the
    # vertical axis and/or placing it in a subplot position.
    if kwargs.get("north") != None:
        if "subplot" not in kwargs:
            fig = aplpy.FITSFigure(new_fits, figure=fig_all, north=kwargs.get("north"))
        else:
            fig = aplpy.FITSFigure(new_fits, figure=fig_all, north=kwargs.get("north"), subplot=kwargs.get("subplot"))
    else:
        if "subplot" not in kwargs:
            fig = aplpy.FITSFigure(new_fits, figure=fig_all)
        else:
            fig = aplpy.FITSFigure(new_fits, figure=fig_all, subplot=kwargs.get("subplot"))

    # Recenter on a specific sky region if a window is provided.
    if kwargs.get("window") != None:
        x, y, height, width = kwargs.get("window")
        fig.recenter(x, y, height=height, width=width)

    if "cmap" not in kwargs:
        kwargs["cmap"] = 'Greys'

    fig.show_colorscale(cmap=kwargs.get("cmap"), stretch=kwargs.get("stretch"))
    fig.add_colorbar()

    if kwargs.get("cb label") != None:
        fig.colorbar.set_axis_label_text(kwargs.get("cb label"))

    fig.colorbar.set_font(size=16)
    fig.colorbar.set_axis_label_font(size=18)

    # WCS coordinate labels in decimal-degree format.
    fig.tick_labels.set_xformat('dd.ddd')
    fig.tick_labels.set_yformat('dd.ddd')
    fig.tick_labels.set_font(size=16)

    fig.axis_labels.set_xtext('RAJ2000')
    fig.axis_labels.set_ytext('DEJ2000')
    fig.axis_labels.set_font(size=18)

    # Semi-transparent white grid for readability on grey-scale maps.
    fig.add_grid()
    fig.grid.set_color('white')
    fig.grid.set_alpha(0.5)
    fig.grid.set_linestyle('solid')
    #fig.grid.set_xspacing(0.3)
    #fig.grid.set_yspacing(0.3)

    # Optional north-direction arrow (useful when North is not up).
    if kwargs.get("arrow_north") != None:
        x, y, dx, dy = kwargs.get("arrow_north")
        fig.show_arrows(x, y, dy, dx, facecolor="white", ec="k", width=10)

    return fig


# ---------------------------------------------------------------------------
# Interactive GUI — network and sky-map inspector
# ---------------------------------------------------------------------------

class InspectNetwork:
    """Interactive Tkinter GUI for inspecting multi-scale hierarchical structures.

    Launches a window composed of:

    * **Left panel** — APLpy sky map (FITS) with the source footprints
      (polygons) of each observation scale overlaid in distinct colours.
      The view re-centres on the selected structure.
    * **Right panel** — Multipartite network graph of the selected structure,
      with each node coloured by its observation scale.
    * **Control bar** — Drop-down menu to switch the background FITS image,
      entry boxes to jump to a structure or node by index, and per-scale
      colour-picker buttons.

    Parameters
    ----------
    network : analyse.Network
        The multi-scale network object returned by ``analyse.Network``.
        Must have already called ``extractStructures()``.

    Examples
    --------
    >>> from analysis import plotter
    >>> plotter.InspectNetwork(network)
    """
    
    def __init__(self, network):
        # Store the raw networkx graph and derive convenience tables.
        self.network = network.network

        # DataFrame of individual clumps (nodes) with their attributes.
        self.clumps = utility.toDataFrame(self.network)
        # DataFrame of extracted multi-scale structures with summary statistics.
        self.structures = network.getStructuresTable()
        
        # List of FITS file paths, one per observation scale.
        self.images = [data.fits_img for data in network.data]
        # Sorted list of beam sizes (arcsec) used as scale identifiers.
        self.scales = network.levels
        
        self.root = tk.Tk()

        self._initialise_tkinterWindow()
        self._initialise_tkinterButtons()
        self._initialise_tkinterColors()
        self._initialise_tkinterPlots()

        self.root.geometry("900x600")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.mainloop()
        

    def _initialise_tkinterWindow(self):
        """Set up the top-level Tkinter frame layout.

        Creates two main frames (plot area and control bar) and two
        labelled sub-frames within the control bar (image options and
        colour selectors).
        """
        # ---------------- setup 2 main Frames
        self.frame_plot = tk.Frame(self.root)#, text='Structure and network')
        self.frame_plot.grid(row=0, column=0)

        self.frame_buttons = tk.Frame(self.root)#, text='Node options')
        self.frame_buttons.grid(row=1, column=0)
        
        # ----- subdivide into 2
        self.frame_image_opt = tk.LabelFrame(self.frame_buttons, text='Image view')
        self.frame_image_opt.grid(row=0, column=0, columnspan=4)

        self.frame_colors = tk.LabelFrame(self.frame_buttons, text='Colors')
        self.frame_colors.grid(row=0, column=4)

    def _initialise_tkinterButtons(self):
        """Populate the control bar with interactive widgets.

        Widgets created:

        * Drop-down menu to select the background FITS image.
        * Button to open the structure/clump table viewer.
        * Entry + button to jump to a structure by its index.
        * Entry + button to jump to a node by its global node ID.
        """
        # -------------- get the path for emission maps
        self.input = tk.StringVar()
        self.input.set("Select an image")

        self.optmenu = tk.OptionMenu(self.frame_image_opt, self.input, *self.images, command=self.change_image)
        self.optmenu.grid(row=0, column=0, columnspan=2)

        # ------------------ open the table view
        tk.Button(self.frame_image_opt, text="View structures table", command=self.openTables).grid(row=1, column=0, columnspan=2)

        # ------------------ selection button for structure
        self.sidx = tk.StringVar()
        self.sidx.set("0")
        tk.Button(self.frame_image_opt, text="Structure request", command=self.updatesidx).grid(row=2, column=0)
        tk.Entry(self.frame_image_opt, textvariable=self.sidx).grid(row=3, column=0)

        # ------------------ selection button for node
        self.nidx = tk.StringVar()
        self.nidx.set("0")
        tk.Button(self.frame_image_opt, text="Node request", command=self.updatenidx).grid(row=2, column=1)
        tk.Entry(self.frame_image_opt, textvariable=self.nidx).grid(row=3, column=1)

    def _initialise_tkinterColors(self):
        """Create per-scale colour-picker buttons in the colour sub-frame.

        Each observation scale (beam size) gets its own button. Clicking a
        button opens a colour-chooser dialog; the chosen colour is stored in
        a ``tk.StringVar`` and applied to both the network graph nodes and the
        polygon overlays on the sky map.  Colours are initialised randomly
        with a fixed seed for reproducibility.
        """
        # ------------------ nodes properties
        ## ----- one StringVar per scale to hold the current hex colour
        self.subset_colors = {f"{b}":tk.StringVar() for b in self.scales}
        np.random.seed(42)
        [color.set('#%02x%02x%02x' % random_color()) 
         for color in self.subset_colors.values()]

        for i, b in enumerate(sorted(self.scales, reverse=True)):
            attribute = f"color_{b}"
            setattr(self, attribute,
                    tk.Button(self.frame_colors, text=f'{b}', command=lambda att=attribute: self.setcolor(att)))
            getattr(self, attribute).pack()

    def _initialise_tkinterPlots(self):
        """Set up the matplotlib figure embedded in the Tkinter window.

        The figure contains two panels:

        * Left — APLpy FITS figure showing the highest-resolution image,
          which will be updated (recentred and re-overlaid) on each structure
          selection.
        * Right — Axes reserved for the multipartite network graph.

        The figure is embedded via ``FigureCanvasTkAgg`` with a standard
        ``NavigationToolbar2Tk`` for zooming and panning.
        """
        self.fig = plt.figure()
        gs = self.fig.add_gridspec(1, 2)

        # ------------------ prepare network plot
        self.net_ax = self.fig.add_subplot(gs[0, 1])

        # ------------------ aplpy display properties (initial defaults)
        self.aplstretch = "sqrt"
        self.vmin = 1e20   # minimum colour scale value [H2/cm²]
        self.vmax = 1e23   # maximum colour scale value [H2/cm²]

        # ------------------ setup aplpy figure
        import aplpy
        
        # Temporarily create a matplotlib axes to extract its bounding box,
        # which is then passed to APLpy as the subplot specification so that
        # the FITS figure occupies exactly the same area.
        ax_ellipses = self.fig.add_subplot(gs[0, 0])
        x0 = ax_ellipses.get_position().x0
        y0 = ax_ellipses.get_position().y0
        x1 = ax_ellipses.get_position().x1
        y1 = ax_ellipses.get_position().y1
        
        dx = x1 - x0
        dy = y1 - y0
        ellipses_subplot = [x0, y0, dx, dy]
        
        ax_ellipses.remove()
        
        # Use the highest-resolution image (last in the list) as the default
        # background; the user can switch via the drop-down menu.
        self.aplpyfig = aplpy.FITSFigure(self.images[-1], figure=self.fig, subplot=ellipses_subplot)
        self.aplpyfig.show_grayscale(stretch=self.aplstretch)
        plt.close(fig=self.fig)
        
        self.aplpyfig.set_auto_refresh(True)
    
        # Semi-transparent coordinate grid overlay.
        self.aplpyfig.add_grid()
        self.aplpyfig.grid.set_color('white')
        self.aplpyfig.grid.set_alpha(0.5)
        self.aplpyfig.grid.set_linestyle('solid')

        # Embed the matplotlib figure in the Tkinter frame.
        self.canvas = FigureCanvasTkAgg(self.fig, self.frame_plot)
        self.canvas._tkcanvas.pack()
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame_plot)
        self.toolbar.update()
    

    def setcolor(self, attribute):
        """Open a colour-chooser dialog and update the colour for a given scale.

        Parameters
        ----------
        attribute : str
            Name of the instance attribute of the form ``"color_<beam_size>"``.
            The beam size is extracted from the suffix and used as a key in
            ``self.subset_colors``.
        """
        colors = askcolor(title="Tkinter Color Chooser")
        # Update the button background to reflect the chosen colour.
        getattr(self, attribute).configure(bg=colors[1])

        # Propagate the new colour to the StringVar used by the drawing routines.
        idx = attribute.split("_")[-1]
        self.subset_colors[idx].set(colors[1])
        #self.aplpyfig.get_layer(f"current{idx}").set_ec(colors[1])

    def openTables(self):
        """Open the ``Tables`` viewer for the clump and structure catalogues."""
        Tables("Table view", self.clumps, self.structures)

    #def updatecontrast(self, val):
    #    self.vmin = val[0]
    #    self.vmax = val[1]
    #    self.f.show_grayscale(vmin=self.vmin, vmax=self.vmax, stretch=self.stretch)
    #    self.fig.canvas.draw_idle()

    def updatesidx(self):
        """Callback for the *Structure request* button.

        Reads the structure index from the entry widget and triggers a full
        plot update (sky map + network graph).
        """
        print(self.sidx.get())
        self.plot("structures", int(self.sidx.get()))

    def updatenidx(self):
        """Callback for the *Node request* button.

        Reads the global node ID from the entry widget and resolves it to the
        structure index that contains it, then triggers a full plot update.
        """
        self.plot("nodes", int(self.nidx.get()))

    def plot(self, type, idx):
        """Update both the sky map and the network graph for the given index.

        Parameters
        ----------
        type : str
            Either ``"structures"`` (idx is a row index in the structures
            DataFrame) or ``"nodes"`` (idx is a global node ID; the enclosing
            structure is found automatically).
        idx : int
            Index of the structure or node to display.
        """
        # idx is the index of the component, not the node
        if type == "nodes":
            # Find the structure that contains the requested node.
            idx = [node for node, net in enumerate(self.structures["component"]) if idx in net.nodes][0]

        self.drawNetwork(idx)
        self.drawMap(idx)
        
    def drawMap(self, idx):
        """Re-centre the sky map on structure ``idx`` and overlay its polygons.

        For each observation scale, the source footprint polygons belonging to
        the selected structure are drawn on the APLpy figure using the
        per-scale colour stored in ``self.subset_colors``.  The view is
        re-centred on the structure's centroid with a radius 1.5× the
        structure size.

        Parameters
        ----------
        idx : int
            Row index in ``self.structures`` of the structure to display.
        """
        import networkx as nx

        print(f"seen component number {idx}")
        x, y = self.structures["xposition"][idx], self.structures["yposition"][idx]
        component = self.structures["component"][idx]

        # Use 1.5× the structure extent as the field-of-view radius.
        radius = 1.5 * self.structures["size"][idx]

        #try:
        #    [self.aplpyfig.remove_layer(f"current{i}") for i in range(len(self.scales))]
        #except:
        #    pass

        print(f"recentering in {x,y} at radius {radius}")
        self.aplpyfig.recenter(x, y, radius=radius)

        polygons = nx.get_node_attributes(component, '_Polygon').items()

        # Draw polygons scale by scale, largest beam first (back to front).
        for i, b in enumerate(sorted(self.scales, reverse=True)):
            P = [p for node, p in polygons if component.nodes[node]["_beam"] == b]
            PTS = []
            for poly in P:
                x, y = poly.exterior.xy
                # APLpy expects polygon vertices as an (N, 2) array in
                # [RA, Dec] order (world coordinates).
                PTS.append(np.reshape(np.ravel([x, y]), (len(x), 2), order='F'))

            alpha = 0.5
            self.aplpyfig.show_polygons(PTS,
                                        facecolor="None",
                                        edgecolor=self.subset_colors[f"{b}"].get(),
                                        lw=2,
                                        alpha=alpha, layer=f"current{i}")
        
    def drawNetwork(self, idx):
        """Draw the hierarchical network graph for structure ``idx``.

        Nodes are positioned with a horizontal multipartite layout (x-axis =
        scale level) and coloured according to the per-scale colour map.

        Parameters
        ----------
        idx : int
            Row index in ``self.structures`` of the structure to display.
        """
        self.net_ax.clear()

        component = self.structures["component"][idx]

        # Assign the current per-scale colour to every node.
        color = [self.subset_colors[f"{data['_beam']}"].get() for v, data in component.nodes(data=True)]
        pos = nx.multipartite_layout(component, subset_key="_level", align="horizontal")
        nx.draw(component, pos, ax=self.net_ax, node_color=color, node_size=200, with_labels=True)
        plt.axis("equal")

        #self.fig.canvas.draw_idle()
        plt.close(fig=1)

    def change_image(self, event):
        """Swap the background FITS image to the one selected in the drop-down.

        Reads the new FITS file directly into the existing APLpy figure object
        by updating its internal WCS and data attributes, then refreshes the
        grey-scale display.

        Parameters
        ----------
        event : str
            The selected image path (passed automatically by the OptionMenu
            ``command`` callback).
        """
        data = self.input.get()
        hdu = 0
        #_, hdu = iu.OpenImage(data)

        # Update APLpy internals without creating a new figure object.
        self.aplpyfig._data, self.aplpyfig._header, self.aplpyfig._wcs, self.aplpyfig._wcsaxes_slices = \
            self.aplpyfig._get_hdu(data, hdu, north=False)

        # Synchronise the WCS pixel dimensions with the new header.
        dimensions = [0, 1]
        self.aplpyfig._wcs.nx = self.aplpyfig._header['NAXIS%i' % (dimensions[0] + 1)]
        self.aplpyfig._wcs.ny = self.aplpyfig._header['NAXIS%i' % (dimensions[1] + 1)]

        self.aplpyfig.show_grayscale(stretch=self.aplstretch)


# ---------------------------------------------------------------------------
# Structure catalogue table viewer
# ---------------------------------------------------------------------------

class Tables:
    """Tkinter table viewer for the clump and structure catalogues.

    Opens a separate window displaying the structure summary table with
    drop-down combo-box filters for the most relevant physical attributes
    (mode, fractality, missed fraction, size, YSO content, etc.).
    Selecting a filter value instantly reduces the table rows to those
    matching all active filters simultaneously.

    Parameters
    ----------
    title : str
        Window title string.
    df_clumps : pandas.DataFrame
        DataFrame of individual clumps (nodes) produced by
        ``network_utility.toDataFrame``.
    df_structures : pandas.DataFrame
        DataFrame of multi-scale structures produced by
        ``analyse.Network.getStructuresTable``.
    """

    # Columns exposed as filter drop-downs in the GUI.
    FILTER_LIST = ( "sinks",
                    "sources",
                    "mode",
                    "fractality",
                    "missed",
                    "percmissed",
                     "maxR",
                     "YSO",
                     "gas",
                     "size")

    def __init__(self, title, df_clumps, df_structures):
        from tkinter import ttk as ttk

        self.root = tk.Tk()
        self.root.title(title)
        self.df_clumps = df_clumps
        self.df_structures = df_structures

        combofr = tk.Frame(self.root)
        combofr.pack()
        self.tree = ttk.Treeview(self.root, show='headings')
        self.filters = []

        # ----------- setup horizontal bar
        self.Hbar = tk.Scrollbar(self.tree, orient="horizontal", command=self.tree.xview)

        # ----------- setup columns to filter
        for col in self.FILTER_LIST:
            name = 'combo_' + col
            self.filters.append(name)
            setattr(self.root, "label_" + col, ttk.Label(combofr, text=col))

            # Populate the combo-box with all unique values found in the column,
            # plus an empty entry that means "no filter" for this column.
            setattr(self.root, name, ttk.Combobox(combofr, values=[''] + sorted(set(self.df_structures[col])),
                                                  state="readonly"))

            getattr(self.root, 'label_' + col).pack(side=tk.LEFT)
            getattr(self.root, name).pack(side=tk.LEFT)
            getattr(self.root, name).bind('<<ComboboxSelected>>', self.select_from_filters)

        # Populate the tree view with all columns.
        self.tree["columns"] = list(self.df_structures)
        self.tree.pack(expand=tk.TRUE, fill=tk.BOTH)
        self.Hbar.pack(side=tk.BOTTOM, fill='x')

        for i in sorted(self.df_structures):
            self.tree.column(i, width=40, anchor="w")
            self.tree.heading(i, text=i, anchor="w")

        # Insert one row per structure; the 'component' column is replaced by
        # the list of node IDs for compact display.
        for i, row in self.df_structures.iterrows():
            row['component'] = list(row['component'].nodes)
            self.tree.insert("", "end", text=i, values=list(row))

    def select_from_filters(self, event):
        """Re-populate the table with only the rows that match all active filters.

        Called automatically whenever the user changes a combo-box selection.
        Empty combo-box values are treated as "no constraint" for that column.

        Parameters
        ----------
        event : tkinter.Event
            The ``<<ComboboxSelected>>`` event (not used directly).
        """
        # --------------- reduce the table with the filter requested
        self.tree.delete(*self.tree.get_children())

        # Build a single lambda that is True only when every active filter
        # matches the corresponding column value (string comparison).
        # -------------- str x because all is str
        all_filter = lambda x: all(
            str(x[f.split('_')[-1]]) == getattr(self.root, f).get() or getattr(self.root, f).get() == ''
            for f in self.filters)

        for i, row in self.df_structures.iterrows():
            if all_filter(row):
                row['component'] = list(row['component'].nodes)
                self.tree.insert("", "end", values=list(row))
