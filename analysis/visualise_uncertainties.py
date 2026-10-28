from pathlib import Path
from io import BytesIO
from html import escape

import h5py
import numpy as np
from matplotlib import colormaps
from matplotlib.figure import Figure
from operator import index as integer_index
from matplotlib.backends.backend_agg import FigureCanvasAgg


class ProjectionData:

    def __init__(self, filename, layout="julia", sigma_missing=None, value_scale=10000):
            
        self.path = Path(filename).expanduser()
        self.layout = layout
        self.sigma_missing = sigma_missing
        self.value_scale = float(value_scale)
            
        self.axes = {}
        self.metadata = {}
        self.storage = {}
        
        with h5py.File(self.path, "r") as f:
            for name in ("phi", "scaling_ratio", "overlap"):
                values = np.asarray(f[name][()], dtype=float)
                
                if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
                    raise ValueError(f"{name} doit etre un vecteur numerique fini non vide.")
                self.axes[name] = values
                
            shape = tuple(len(self.axes[name]) for name in ("phi", "scaling_ratio", "overlap"))
            expected = shape[::-1] if layout == "julia" else shape
            
            for name in ("phi2d", "sigma_phi2d"):
                ds = f[name]
                self.storage[name] = {"dtype": ds.dtype, "shape": ds.shape, "fill": ds.fillvalue}
                
            for name in ("n_initial_parents", "hard_objects", "max_placement_attempts"):
                if name in f:
                    value = np.asarray(f[name][()])
                    self.metadata[name] = value.item() if value.size == 1 else value.tolist()

    def sample_sizes(self, target_n=None):
        source_n = self.metadata.get("n_initial_parents")
        
        if target_n is None:
            target_n = source_n
            
        result = []
        
        for name, value in (("source_n", source_n), ("target_n", target_n)):
            number = float(value)
            result.append(int(number))
            
        return tuple(result)

    def read_curve(self, scaling_index=0, overlap_index=0, *, target_n=None):
        source_n, target_n = self.sample_sizes(target_n=target_n)
        j, k = int(scaling_index), int(overlap_index)
        
        selection = (k, j, slice(None)) if self.layout == "julia" else (slice(None), j, k)
        with h5py.File(self.path, "r") as f:
            y = np.asarray(f["phi2d"][selection], dtype=float)
            sigma = np.asarray(f["sigma_phi2d"][selection], dtype=float)

        # -1 means it is not computed
        y[(y == -1) | ~np.isfinite(y)] = np.nan
        sigma[(sigma < 0) | ~np.isfinite(sigma)] = np.nan
        
        if self.sigma_missing is not None:
            sigma[sigma == self.sigma_missing] = np.nan
            
        elif (self.storage["sigma_phi2d"]["dtype"].kind == "u"
              and self.storage["sigma_phi2d"]["dtype"].itemsize == 2):
            sigma[sigma == 65535] = np.nan

        y /= self.value_scale
        sigma /= self.value_scale
        sigma *= np.sqrt(float(source_n)) / np.sqrt(float(target_n))

        x = self.axes["phi"]
        order = np.argsort(x, kind="stable")
        return x[order].copy(), y[order], sigma[order]


def _scaling_selection(data, scaling_indices=None, fallback=0):
    if scaling_indices is None:
        scaling_indices = (fallback,)
    indices = tuple(dict.fromkeys(integer_index(i) for i in scaling_indices))
    return indices


def projection_figure(data, scaling_index=0, overlap_index=0, *, target_n=None,
                      scaling_indices=None):
    
    indices = _scaling_selection(data, scaling_indices, fallback=scaling_index)
    source_n, target_n = data.sample_sizes(target_n=target_n)
    
    fig = Figure(figsize=(12, 5), dpi=130, layout="constrained")
    
    axes = fig.subplots(1, 2)
    valid_counts = [0, 0]
    n_ratios = len(data.axes["scaling_ratio"])
    cmap = colormaps["tab10" if n_ratios <= 10 else "turbo"]
    x = np.sort(data.axes["phi"])
    
    for j in indices:
        x, y, sigma = data.read_curve(j, overlap_index, target_n=target_n)
        valid_y = np.isfinite(y)
        valid_sigma = np.isfinite(sigma)
        valid_relative = valid_y & (x != 0)
        relative = np.full_like(y, np.nan)
        relative_sigma = np.full_like(sigma, np.nan)
        
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            relative[valid_relative] = (y[valid_relative] - x[valid_relative]) / x[valid_relative]
            good = valid_relative & valid_sigma
            relative_sigma[good] = sigma[good] / np.abs(x[good])
            
        relative[~np.isfinite(relative)] = np.nan
        relative_sigma[~np.isfinite(relative_sigma)] = np.nan
        
        color = cmap(j if n_ratios <= 10 else j / (n_ratios - 1))
        label = f"scaling_ratio = {data.axes['scaling_ratio'][j]:g}"
        
        for panel, (ax, values, errors) in enumerate(zip(axes, (relative, y), (relative_sigma, sigma))):
            valid = np.isfinite(values)
            with_error = valid & np.isfinite(errors)
            valid_counts[panel] += int(valid.sum())
            ax.plot(x, values, "o-", color=color, linewidth=1.3, markersize=4, label=label)
            if np.any(with_error):
                ax.errorbar(x[with_error], values[with_error], yerr=errors[with_error],
                            fmt="none", ecolor=color, capsize=3, elinewidth=1, alpha=0.8)
            missing = valid & ~with_error
            if np.any(missing):
                ax.plot(x[missing], values[missing], "x", color=color, markersize=8)

    for panel, (ax, ylabel) in enumerate(zip(
        axes,
        (r"$(\phi_{\mathrm{2D}}-\phi_{\mathrm{3D}})/\phi_{\mathrm{3D}}$", r"$\phi_{\mathrm{2D}}$"),
    )):
        ax.set(xlabel=r"$\phi_{\mathrm{3D}}$", ylabel=ylabel)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        
        if not valid_counts[panel]:
            message = ("Select at least one scaling_ratio" if not indices
                       else "No data available for this selection")
            ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes,
                    bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"})
            
    axes[0].axhline(0, color="0.45", ls="--", lw=1, label="No effect")
    axes[1].plot(x, x, color="0.45", ls="--", lw=1,
                 label=r"$\phi_{\mathrm{2D}}=\phi_{\mathrm{3D}}$")
    
    for ax in axes:
        ax.legend(fontsize=8, loc="best", ncol=1 if len(indices) < 7 else 2)
    return fig, ""


def explore_projection(filename="projection_uncertainty.h5", *, layout="julia",
                       sigma_missing=None, value_scale=10000, target_n=None,
                       scaling_indices=None):
    
    import ipywidgets as widgets

    data = ProjectionData(filename, layout=layout, sigma_missing=sigma_missing,
                          value_scale=value_scale)
    source_n, target_n = data.sample_sizes(target_n=target_n)
    initial = _scaling_selection(data, scaling_indices)
    ratio_control = widgets.SelectMultiple(
        options=[(f"{v:g} [indice {i}]", i) for i, v in enumerate(data.axes["scaling_ratio"])],
        value=initial, description="scaling_ratio", rows=min(8, len(data.axes["scaling_ratio"])),
        style={"description_width": "initial"}, layout=widgets.Layout(width="310px"),
    )
    overlap_control = widgets.Dropdown(
        options=[(f"{v:g} [indice {i}]", i) for i, v in enumerate(data.axes["overlap"])],
        value=0, description="overlap", style={"description_width": "initial"},
        layout=widgets.Layout(width="260px"),
    )
    
    select_all = widgets.Button(description="Select all")
    select_none = widgets.Button(description="Unselect all")
    
    select_all.on_click(lambda button: setattr(
        ratio_control, "value", tuple(range(len(data.axes["scaling_ratio"])))
    ))
    
    select_none.on_click(lambda button: setattr(ratio_control, "value", ()))
    target_control = widgets.BoundedIntText(
        value=target_n, min=1, max=max(2**53 - 1, target_n), step=1,
        description="sample size", continuous_update=False,
        style={"description_width": "initial"},
        layout=widgets.Layout(width="310px"),
    )
    image = widgets.Image(format="png", layout=widgets.Layout(width="100%", max_width="1250px"))
    status = widgets.HTML(layout=widgets.Layout(display="none"))
    refresh = widgets.Button(description="Refresh", icon="refresh")

    def update(change=None):
        try:
            fig, _ = projection_figure(
                data, overlap_index=overlap_control.value,
                scaling_indices=ratio_control.value, target_n=target_control.value,
            )
            buffer = BytesIO()
            FigureCanvasAgg(fig).print_png(buffer)
            image.value = buffer.getvalue()
            status.value = ""
            status.layout.display = "none"
        except Exception as exc:
            image.value = b""
            status.value = f"<b>Error :</b> {escape(str(exc))}"
            status.layout.display = ""

    for control in (ratio_control, overlap_control, target_control):
        control.observe(update, names="value")
    refresh.on_click(update)
    update()
    return widgets.VBox([
        widgets.HBox([ratio_control, overlap_control,
                      widgets.VBox([select_all, select_none, refresh])],
                     layout=widgets.Layout(flex_flow="row wrap")),
        widgets.HBox([target_control], layout=widgets.Layout(flex_flow="row wrap")),
        image, status,
    ])


def save_projection(filename, output="projection.png", *, scaling_index=0,
                    overlap_index=0, layout="julia", sigma_missing=None,
                    value_scale=10000, target_n=None, scaling_indices=None):
    """Exporte les deux courbes en PNG, PDF ou SVG, sans widgets."""
    data = ProjectionData(filename, layout=layout, sigma_missing=sigma_missing,
                          value_scale=value_scale)
    fig, _ = projection_figure(data, scaling_index, overlap_index,
                               target_n=target_n, scaling_indices=scaling_indices)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    return str(Path(output).resolve())
