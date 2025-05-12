import operator
import tkinter as tk
import matplotlib.pyplot as plt

import matplotlib
import numpy as np
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter.colorchooser import askcolor

import networkx as nx
from . import network_utility as utility
from . import image_utility as iu
from astropy.io import fits
import aplpy

def random_color():
    color = np.random.randint(0, 255, size=3)
    return tuple(color)

def plotSubGraph(network, subset_color, ax, **kwargs):
    color = [subset_color[data["_level"]] for v, data in network.nodes(data=True)]
    pos = nx.multipartite_layout(network, subset_key="_level", align="horizontal")
    nx.draw(network, pos, ax=ax, node_color=color, **kwargs)
    plt.axis("equal")

def getFragCurve(network, distance, mode="all"):
    def ReshapeData(y):
        yl = np.reshape(np.ravel(y), np.shape(y)[::-1], order="F")
        return yl

    if mode == "all":
        y = [structure.productivity for structure in network.structures]
    else:
        y = [structure.productivity for structure in network.structures
            if structure.mode.name == mode]
    
    x = np.sort(network.levels) * distance

    yl = ReshapeData(y)

    meany = [np.nanmean(ylst) for ylst in yl]
    stdy = [np.nanstd(ylst) / np.sqrt(len(ylst)) for ylst in yl]

    return x, meany, stdy

def plotFragmentationCurve_Sizes(structures, distance, figs=None, **kwargs):
    import pandas as pd

    bins = np.array([1.9, 2.1, 8.4, 13.5, 18.2, 24.9, 36.3, 100])
    bins = bins * distance

    R_lists = []
    R0s = []
    Rmax = 0
    for s in structures:
        R = []
        R0s.append( s.size )

        for node, size in s.component.nodes('_R'):
            Rmax = max(Rmax, size)
            R.append(size)

        R_lists.append(R)

    bins[-1] = Rmax
    x = (bins[1:] + bins[:-1])/2
    dx = bins[1:] - bins[:-1]

    ys = np.zeros(shape=(len(structures), len(x)))
    here = np.zeros_like(ys)

    for idx, r in enumerate(R_lists):
        count, bins = np.histogram(r, bins=bins)
        df = pd.DataFrame((count != 0))

        tresh = 1
        df1 = df.cumsum().mask(df)
        m1 = df1.apply(lambda x: x.map(x.value_counts())).le(tresh)
        m2 = df1.ne(df1.iloc[0]) & df1.ne(df1.iloc[-1])

        df[m1 & m2] = np.nan
        arr = np.ravel(df.to_numpy())

        ys[idx, :] = count * arr
        here[idx, :] = arr

    print(ys)

    meany = np.nanmean(ys, axis=1)
    stdy = np.nanstd(ys, axis=1) / np.sqrt(np.sum(here, axis=1))

    if figs is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = figs
    ax.errorbar(np.array(x), meany, stdy, np.array(dx), "o", **kwargs)
    print("R0s", R0s)
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

def prepareAplpy(image, path=True, **kwargs):

    img, hdr = iu.OpenImage(image, path=path)
    img[img == 0] = np.nan
    new_fits = fits.PrimaryHDU(data=img, header=hdr)

    if "fig" not in kwargs:
        fig_all = plt.figure("Map", figsize=kwargs.get("figsize"))
    else:
        fig_all = kwargs.get("fig")

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

    ######### recenter to window
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

    fig.tick_labels.set_xformat('dd.ddd')
    fig.tick_labels.set_yformat('dd.ddd')
    fig.tick_labels.set_font(size=16)

    fig.axis_labels.set_xtext('RAJ2000')
    fig.axis_labels.set_ytext('DEJ2000')
    fig.axis_labels.set_font(size=18)

    ###grid
    fig.add_grid()
    fig.grid.set_color('white')
    fig.grid.set_alpha(0.5)
    fig.grid.set_linestyle('solid')
    #fig.grid.set_xspacing(0.3)
    #fig.grid.set_yspacing(0.3)

    if kwargs.get("arrow_north") != None:
        x, y, dx, dy = kwargs.get("arrow_north")
        fig.show_arrows(x, y, dy, dx, facecolor="white", ec="k", width=10)

    return fig
    
class InspectNetwork:
    
    def __init__(self, network):
        self.network = network.network

        self.clumps = utility.toDataFrame(self.network)
        self.structures = network.getStructuresTable()
        
        self.images = [data.image_path for data in network.data]
        self.scales = network.levels
        
        self.root = tk.Tk()

        self._initialise_tkinterWindow()
        self._initialise_tkinterButtons()
        self._initialise_tkinterColors()
        self._initialise_tkinterPlots()

        self.root.geometry("1200x1200")
        self.root.mainloop()
        

    def _initialise_tkinterWindow(self):
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
        # ------------------ nodes properties
        ## ----- colors
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
        self.fig = plt.figure()
        gs = self.fig.add_gridspec(1, 2)

        # ------------------ prepare network plot
        self.net_ax = self.fig.add_subplot(gs[0, 1])

        # ------------------ aplpy properties
        self.aplstretch = "sqrt"
        self.vmin = 1e20
        self.vmax = 1e23

        # ------------------ setup aplpy figure
        import aplpy
        
        ax_ellipses = self.fig.add_subplot(gs[0, 0])
        x0 = ax_ellipses.get_position().x0
        y0 = ax_ellipses.get_position().y0
        x1 = ax_ellipses.get_position().x1
        y1 = ax_ellipses.get_position().y1
        
        dx = x1 - x0
        dy = y1 - y0
        ellipses_subplot = [x0, y0, dx, dy]
        
        ax_ellipses.remove()
        
        self.aplpyfig = aplpy.FITSFigure(self.images[-1], figure=self.fig, subplot=ellipses_subplot)
        self.aplpyfig.show_grayscale(stretch=self.aplstretch)
        plt.close(fig=self.fig)
        
        self.aplpyfig.set_auto_refresh(True)
    
        ###grid
        self.aplpyfig.add_grid()
        self.aplpyfig.grid.set_color('white')
        self.aplpyfig.grid.set_alpha(0.5)
        self.aplpyfig.grid.set_linestyle('solid')

        self.canvas = FigureCanvasTkAgg(self.fig, self.frame_plot)
        self.canvas._tkcanvas.pack()
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame_plot)
        self.toolbar.update()
    

    def setcolor(self, attribute):

        colors = askcolor(title="Tkinter Color Chooser")
        getattr(self, attribute).configure(bg=colors[1])

        idx = attribute.split("_")[-1]
        self.subset_colors[idx].set(colors[1])
        #self.aplpyfig.get_layer(f"current{idx}").set_ec(colors[1])

    def openTables(self):
        Tables("Table view", self.clumps, self.structures)

    #def updatecontrast(self, val):
    #    self.vmin = val[0]
    #    self.vmax = val[1]
    #    self.f.show_grayscale(vmin=self.vmin, vmax=self.vmax, stretch=self.stretch)
    #    self.fig.canvas.draw_idle()

    def updatesidx(self):
        print(self.sidx.get())
        self.plot("structures", int(self.sidx.get()))

    def updatenidx(self):
        self.plot("nodes", int(self.nidx.get()))

    def plot(self, type, idx): # idx is the index of the component, not the node
        if type == "nodes":
            idx = [node for node, net in enumerate(self.structures["component"]) if idx in net.nodes][0]

        self.drawNetwork(idx)
        self.drawMap(idx)
        
    def drawMap(self, idx):
        import networkx as nx

        print(f"seen component number {idx}")
        x, y = self.structures["xposition"][idx], self.structures["yposition"][idx]
        component = self.structures["component"][idx]

        #print(self.structures["size"][idx])
        radius = 1.5 * self.structures["size"][idx]

        #try:
        #    [self.aplpyfig.remove_layer(f"current{i}") for i in range(len(self.scales))]
        #except:
        #    pass

        print(f"recentering in {x,y} at radius {radius}")
        self.aplpyfig.recenter(x, y, radius=radius)

        polygons = nx.get_node_attributes(component, '_Polygon').items()

        for i, b in enumerate(sorted(self.scales, reverse=True)):
            P = [p for node, p in polygons if component.nodes[node]["_phlevel"] == b]
            PTS = []
            for poly in P:
                x, y = poly.exterior.xy
                PTS.append(np.reshape(np.ravel([x, y]), (len(x), 2), order='F'))

            alpha = 0.5
            self.aplpyfig.show_polygons(PTS,
                                        facecolor="None",
                                        edgecolor=self.subset_colors[f"{b}"].get(),
                                        lw=2,
                                        alpha=alpha, layer=f"current{i}")
        
    def drawNetwork(self, idx):
        self.net_ax.clear()

        component = self.structures["component"][idx]

        #subset_color = [color.get() for color in self.subset_colors.values()]
        color = [self.subset_colors[f"{data['_phlevel']}"].get() for v, data in component.nodes(data=True)]
        pos = nx.multipartite_layout(component, subset_key="_level", align="horizontal")
        nx.draw(component, pos, ax=self.net_ax, node_color=color, node_size=200, with_labels=True)
        plt.axis("equal")

        #self.fig.canvas.draw_idle()
        plt.close(fig=1)

    def change_image(self, event):
        data = self.input.get()
        hdu = 0
        #_, hdu = iu.OpenImage(data)

        self.aplpyfig._data, self.aplpyfig._header, self.aplpyfig._wcs, self.aplpyfig._wcsaxes_slices = \
            self.aplpyfig._get_hdu(data, hdu, north=False)

        dimensions = [0, 1]
        self.aplpyfig._wcs.nx = self.aplpyfig._header['NAXIS%i' % (dimensions[0] + 1)]
        self.aplpyfig._wcs.ny = self.aplpyfig._header['NAXIS%i' % (dimensions[1] + 1)]

        self.aplpyfig.show_grayscale(stretch=self.aplstretch)

class Tables:

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

            setattr(self.root, name, ttk.Combobox(combofr, values=[''] + sorted(set(self.df_structures[col])),
                                                  state="readonly"))

            getattr(self.root, 'label_' + col).pack(side=tk.LEFT)
            getattr(self.root, name).pack(side=tk.LEFT)
            getattr(self.root, name).bind('<<ComboboxSelected>>', self.select_from_filters)

        self.tree["columns"] = list(self.df_structures)
        self.tree.pack(expand=tk.TRUE, fill=tk.BOTH)
        self.Hbar.pack(side=tk.BOTTOM, fill='x')

        for i in sorted(self.df_structures):
            self.tree.column(i, width=40, anchor="w")
            self.tree.heading(i, text=i, anchor="w")

        for i, row in self.df_structures.iterrows():
            row['component'] = list(row['component'].nodes)
            self.tree.insert("", "end", text=i, values=list(row))

    def select_from_filters(self, event):
        # --------------- reduce the table with the filter requested
        self.tree.delete(*self.tree.get_children())

        # -------------- str x because all is str
        all_filter = lambda x: all(
            str(x[f.split('_')[-1]]) == getattr(self.root, f).get() or getattr(self.root, f).get() == ''
            for f in self.filters)

        for i, row in self.df_structures.iterrows():
            if all_filter(row):
                row['component'] = list(row['component'].nodes)
                self.tree.insert("", "end", values=list(row))
