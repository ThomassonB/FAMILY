"""
load.py
-------
Data loading utilities for the FAMILY (Fragmentation Analysis MultiscaLe In Young
stellar environments) pipeline.

This module provides functions to read TOML configuration files and load
astronomical source catalogs into a normalised pandas DataFrame. Three types
of sources are supported:

- **Ellipses** : Gaussian-fitted sources (e.g. output of getsf / gaussclumps),
  described by a position, semi-axes, and a position angle.
- **YSOs**     : Point-like Young Stellar Objects whose spatial extent is set
  to the observing beam.
- **Polygons** : Arbitrarily-shaped regions (e.g. clumps delimited by hand in
  DS9).

All column names in the normalised DataFrame follow the convention defined in
`standard_variables.strings_ref`:
    (_X, _Y, _A, _B, _Theta, _R, _M, _Class, _beam)

References
----------
The pipeline is designed to work with multi-wavelength continuum observations
of star-forming regions (see e.g. W43-MM1, NGC 2264).
"""

from enum import IntEnum

import os
import tomllib
import re
from shapely.geometry import Polygon, LineString

import numpy as np
import pandas as pd 
import shapely.geometry.polygon as shp

from .standard_variables import strings_ref
# strings_ref = ("_X", "_Y", "_A", "_B", "_Theta", "_R", "_M", "_Class", "_beam")


class ObjectType(IntEnum):
    """Enumeration of the astronomical source types supported by the pipeline.

    Attributes
    ----------
    ELLIPSE : int (0)
        Elliptical Gaussian sources extracted by a source-finder (e.g. getsf).
        Each source is described by a central position, semi-major axis,
        semi-minor axis, and position angle.
    YSO : int (1)
        Young Stellar Objects treated as point sources. Their angular size is
        set to the observing beam (no resolved structure assumed).
    POLYGON : int (-1)
        Arbitrarily-shaped regions defined by a polygon (e.g. hand-drawn in
        DS9 or produced by a watershed segmentation).
    """
    ELLIPSE = 0
    YSO = 1
    POLYGON = -1


def read_toml(file):
    """Read a TOML configuration file and return its contents as a dictionary.

    The configuration file controls column mappings, unit conversions,
    observation metadata, and file paths used downstream by the pipeline.

    Parameters
    ----------
    file : str or path-like
        Path to the TOML configuration file.

    Returns
    -------
    cfg : dict
        Nested dictionary of all configuration parameters.

    Examples
    --------
    >>> cfg = read_toml("config/ellipses_level_1.toml")
    >>> cfg["observation"]["beam"]
    8.4
    """
    with open(file, "rb") as f:
        cfg = tomllib.load(f)
    return cfg


def _load_ellipses(df, cfg):
    """Normalise an ellipse catalog DataFrame in-place using the TOML configuration.

    Renames catalog-specific column names to the internal standard names
    (``strings_ref``), applies unit conversions to bring all quantities to
    degrees (positions, semi-axes) and radians (position angle), and scales
    the semi-axes by the number of sigma specified in the configuration
    (``n_sigma``).  An equivalent circular radius in arcseconds is also
    computed.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw catalog DataFrame as returned by the user-supplied reader.
        Modified **in-place**.
    cfg : dict
        Configuration dictionary as returned by :func:`read_toml`.  Must
        contain the sections ``columns_names_in_catalog``,
        ``unit_conversion``, and ``observation``.

    Notes
    -----
    The position angle convention (offset and scaling) is instrument-
    dependent and must be set in the ``unit_conversion`` section of the
    TOML file.

    The equivalent radius (``_R``) is defined as the geometric mean of the
    two semi-axes, expressed in arcseconds:
        R = sqrt(A * B) * 3600  [arcsec]

    The ``Mass`` column is optional; if absent it is simply not added to
    the DataFrame.
    """
    # --- Positions (converted to degrees) ---
    df[strings_ref[0]] = df.pop(cfg["columns_names_in_catalog"]["Xposition"]) * cfg["unit_conversion"]["Xposition_to_deg"]
    df[strings_ref[1]] = df.pop(cfg["columns_names_in_catalog"]["Yposition"]) * cfg["unit_conversion"]["Yposition_to_deg"]

    # --- Ellipse geometry (semi-axes converted to degrees) ---
    df[strings_ref[2]] = df.pop(cfg["columns_names_in_catalog"]["SemiMajorAxis"]) * cfg["unit_conversion"]["SemiMajorAxis_to_deg"]
    df[strings_ref[3]] = df.pop(cfg["columns_names_in_catalog"]["SemiMinorAxis"]) * cfg["unit_conversion"]["SemiMinorAxis_to_deg"]

    # --- Position angle (converted to radians, with instrument-specific offset) ---
    df[strings_ref[4]] = df.pop(cfg["columns_names_in_catalog"]["PosAngle"]) * cfg["unit_conversion"]["PosAngle_to_rad"] 
    df[strings_ref[4]] += cfg["unit_conversion"]["PosAngle_offset"]

    # --- Scale semi-axes by n_sigma (source extent definition) ---
    df[strings_ref[2]] *= cfg["unit_conversion"]["n_sigma"]
    df[strings_ref[3]] *= cfg["unit_conversion"]["n_sigma"]

    # --- Equivalent circular radius in arcseconds: R = sqrt(A * B) * 3600 ---
    df[strings_ref[5]] = np.sqrt( df[strings_ref[2]] * df[strings_ref[3]] ) * 3600.0

    # --- Optional mass column ---
    if cfg["columns_names_in_catalog"]["Mass"] in df.columns:
        df[strings_ref[6]] = df.pop(cfg["columns_names_in_catalog"]["Mass"]) * cfg["unit_conversion"]["Mass_to_Msun"] 

    # --- Observing beam FWHM (arcsec) ---
    df[strings_ref[8]] = cfg["observation"]["beam"]


def _load_yso(df, cfg):
    """Normalise a YSO catalog DataFrame in-place using the TOML configuration.

    Young Stellar Objects are treated as unresolved point sources: their
    angular extent is set equal to the observing beam.  Positions are
    converted to degrees and the standard column names (``strings_ref``) are
    applied.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw catalog DataFrame as returned by the user-supplied reader.
        Modified **in-place**.
    cfg : dict
        Configuration dictionary as returned by :func:`read_toml`.  Must
        contain the sections ``columns_names_in_catalog``,
        ``unit_conversion``, and ``observation``.

    Notes
    -----
    Both semi-axes (``_A`` and ``_B``) are set to ``beam / 3600`` degrees
    (i.e. the beam FWHM in degrees), and the position angle (``_Theta``) is
    set to zero since YSOs are assumed circular.

    The ``Mass`` and ``Class`` columns are optional; if absent they are
    simply not added to the DataFrame.

    .. warning::
        The ``Class`` column conversion currently uses ``Mass_to_Msun`` as
        a scaling factor, which may be a placeholder.  Verify this is
        intentional for the catalog at hand.
    """
    # --- Positions (converted to degrees) ---
    df[strings_ref[0]] = df.pop(cfg["columns_names_in_catalog"]["Xposition"]) * cfg["unit_conversion"]["Xposition_to_deg"]
    df[strings_ref[1]] = df.pop(cfg["columns_names_in_catalog"]["Yposition"]) * cfg["unit_conversion"]["Yposition_to_deg"]

    # --- Semi-axes set to beam size (YSOs are unresolved point sources) ---
    df[[strings_ref[2], strings_ref[3]]] = cfg["observation"]["beam"] / 3600.0

    # --- Position angle set to zero (circular symmetry assumed) ---
    df[strings_ref[4]] = 0.0

    # --- Equivalent circular radius in arcseconds: R = sqrt(A * B) * 3600 ---
    df[strings_ref[5]] = np.sqrt( df[strings_ref[2]] * df[strings_ref[3]] ) * 3600.0

    # --- Optional mass column ---
    if cfg["columns_names_in_catalog"]["Mass"] in df.columns:
        df[strings_ref[6]] = df.pop(cfg["columns_names_in_catalog"]["Mass"]) * cfg["unit_conversion"]["Mass_to_Msun"]

    # --- Optional evolutionary class column (Class I, 0/I, II, etc.) ---
    if cfg["columns_names_in_catalog"]["Class"] in df.columns:
        df[strings_ref[7]] = df.pop(cfg["columns_names_in_catalog"]["Class"]) * cfg["unit_conversion"]["Mass_to_Msun"]

    # --- Observing beam FWHM (arcsec) ---
    df[strings_ref[8]] = cfg["observation"]["beam"]


def _load_polygons(df, cfg):
    """Normalise a polygon catalog DataFrame in-place using the TOML configuration.

    Only positions are loaded for polygon-type sources; the shape information
    is expected to be provided separately (e.g. via :func:`read_ds9_polygons`).

    Parameters
    ----------
    df : pandas.DataFrame
        Raw catalog DataFrame as returned by the user-supplied reader.
        Modified **in-place**.
    cfg : dict
        Configuration dictionary as returned by :func:`read_toml`.  Must
        contain the sections ``columns_names_in_catalog``,
        ``unit_conversion``, and ``observation``.
    """
    # --- Positions (converted to degrees) ---
    df[strings_ref[0]] = df.pop(cfg["columns_names_in_catalog"]["Xposition"]) * cfg["unit_conversion"]["Xposition_to_deg"]
    df[strings_ref[1]] = df.pop(cfg["columns_names_in_catalog"]["Yposition"]) * cfg["unit_conversion"]["Yposition_to_deg"]

    # --- Observing beam FWHM (arcsec) ---
    df[strings_ref[8]] = cfg["observation"]["beam"]


def load_data(file, reader_to_df):
    """Load a source catalog described by a TOML configuration file.

    Reads the TOML configuration, loads the catalog into a pandas DataFrame
    using the provided reader function, normalises the DataFrame columns
    (units, naming convention) according to the object type, and returns a
    metadata dictionary that bundles the DataFrame with observation metadata.

    Parameters
    ----------
    file : str or path-like
        Path to the TOML configuration file.
    reader_to_df : callable
        A function ``reader_to_df(path) -> pandas.DataFrame`` that reads
        the catalog file (CSV, FITS table, etc.) and returns a raw DataFrame.
        The user is responsible for supplying the appropriate reader (e.g.
        ``lambda x: pd.read_csv(x, delimiter=",", comment="#")``).

    Returns
    -------
    metadata : dict or None
        Dictionary containing the normalised data and observation metadata:

        - ``"df"``       : normalised :class:`pandas.DataFrame`.
        - ``"comment"``  : free-text comment from the TOML file.
        - ``"name"``     : dataset name (e.g. ``"L01"``).
        - Additional keys from ``cfg["files"]`` (e.g. ``"catalog"``,
          ``"fits_img"``).
        - Additional keys from ``cfg["observation"]`` (e.g. ``"beam"``,
          ``"wavelength"``, ``"distance"``, ``"fov_window"``).

        Returns ``None`` if the object type is not recognised.

    Raises
    ------
    KeyError
        If mandatory sections or keys are missing from the TOML file.

    Notes
    -----
    The ``object_type`` field in the TOML file must be one of the integer
    values defined in :class:`ObjectType` (0 = ELLIPSE, 1 = YSO, -1 =
    POLYGON).
    """
    cfg = read_toml(file)

    path = cfg["files"]["catalog"]
    df = reader_to_df(path)

    if cfg["objects"]["type"] == ObjectType.ELLIPSE:
        _load_ellipses(df, cfg)

    elif cfg["objects"]["type"] == ObjectType.YSO:
        _load_yso(df, cfg)

    elif cfg["objects"]["type"] == ObjectType.POLYGON:
        _load_polygons(df, cfg)

    else:
        return

    metadata = {
        "df": df,
        "comment": cfg["objects"]["comment"],
        "name": cfg["dataset"]["name"]
    }
    metadata.update(cfg["files"])
    metadata.update(cfg["observation"])
    return metadata


def read_ds9_polygons(path):
    """Parse a DS9 region file and return a list of Shapely polygons.

    Reads a DS9 ``.reg`` file (any coordinate frame) and extracts all
    ``polygon(...)`` entries, converting them into :class:`shapely.geometry.Polygon`
    objects.  Comment lines (``#``), blank lines, and coordinate-frame
    declarations are silently ignored.

    Parameters
    ----------
    path : str or path-like
        Path to the DS9 region file.

    Returns
    -------
    polygons : list of shapely.geometry.Polygon
        Ordered list of polygons as they appear in the region file.  The
        coordinate values are taken verbatim from the file (no coordinate
        transformation is applied).

    Raises
    ------
    ValueError
        If a ``polygon`` entry contains fewer than 3 vertices or an odd
        number of coordinate values.

    Notes
    -----
    Only ``polygon`` regions are extracted.  Other region types (``circle``,
    ``ellipse``, ``box``, etc.) are silently skipped.

    The function supports any coordinate frame recognised by DS9 (``image``,
    ``physical``, ``fk5``, ``icrs``, ``galactic``, ``ecliptic``); the frame
    declaration line is ignored and coordinates are returned as-is.

    Examples
    --------
    >>> polygons = read_ds9_polygons("benchmark/benchmark_structures.reg")
    >>> len(polygons)
    12
    >>> polygons[0].area
    0.00042...
    """
    path = Path(path)
    polygons = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        # Skip blank lines and comment lines
        if not line or line.startswith("#"):
            continue

        # Skip coordinate-frame declaration lines (e.g. "fk5", "image")
        if line.lower() in {"image", "physical", "fk5", "icrs", "galactic", "ecliptic"}:
            continue

        if line.lower().startswith("polygon("):
            # Extract the comma-separated coordinate string between parentheses
            inside = line[line.find("(") + 1: line.rfind(")")]
            coords = [float(x) for x in re.split(r"\s*,\s*", inside)]

            # Validate: need at least 3 vertices (6 values) and an even count
            if len(coords) < 6 or len(coords) % 2 != 0:
                raise ValueError(f"Polygone invalide: {line}")

            # Interleave x and y coordinates into (x, y) pairs
            xy = list(zip(coords[::2], coords[1::2]))
            polygons.append(Polygon(xy))

    return polygons