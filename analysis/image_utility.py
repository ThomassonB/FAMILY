

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
image_utility.py
================
Utility functions for handling astronomical FITS images in the context of the
FAMILY (Fragmentation Analysis of Multiscale hIerarchical structureLs via sYnoptic
observations) pipeline.

This module provides tools to:
  - Open and read FITS images.
  - Convert between World Coordinate System (WCS/sky) and pixel coordinates.
  - Convolve images with a Gaussian beam (e.g. to match angular resolutions).
  - Generate binary masks from sky-projected polygon footprints (e.g. source
    ellipses from multi-wavelength catalogs).
  - Compute photometric statistics (mean, median, maximum) of pixel values
    within polygon-defined regions.
  - Build logarithmically-spaced pixel histograms decomposed by the hierarchical
    fragmentation mode of each structure (HIERARCHICAL, LINEAR, ISOLATED), which
    is the primary diagnostic of the FAMILY analysis.

Typical usage in the FAMILY workflow
-------------------------------------
After extracting multi-scale structures from a network of overlapping source
catalogs (see `analyse.py`), the functions here are used to:
  1. Project polygon footprints (stored in WCS) onto a reference FITS image
     (e.g. a column-density map).
  2. Produce binary masks per structure mode.
  3. Compute per-bin fractional coverage of each structure mode as a function
     of column density, enabling the study of the fragmentation regime at each
     physical scale.

Dependencies
------------
numpy, astropy (fits, wcs, coordinates, convolution), shapely, scipy.ndimage,
tqdm

Created on Thu Nov 25 08:55:47 2021
@author: thomaben
"""

import numpy as np
from astropy.io import fits


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def OpenImage(image, path=True):
    """Open a FITS image and return its data array and header.

    Parameters
    ----------
    image : str or astropy.io.fits.HDU
        Either a file path to a FITS image (when ``path=True``) or an already
        opened HDU object (when ``path=False``).
    path : bool, optional
        If ``True`` (default), ``image`` is interpreted as a file path and the
        file is opened with :func:`astropy.io.fits.open`. If ``False``,
        ``image`` is expected to be an HDU object whose ``.header`` and
        ``.data`` attributes are read directly.

    Returns
    -------
    img : numpy.ndarray
        2-D pixel array of the primary FITS extension.
    hdr : astropy.io.fits.Header
        FITS header containing WCS and metadata.

    Raises
    ------
    ValueError
        If the file cannot be opened (path not found, corrupted file, etc.).
        The offending path is printed to stdout before the exception is raised.
    """
    if path:
        try:
            HDU = fits.open(image)[0]
        except:
            print(image)
            raise ValueError

        hdr = HDU.header
        img = HDU.data
    else:
        hdr = image.header
        img = image.data

    return img, hdr


# ---------------------------------------------------------------------------
# Coordinate conversions
# ---------------------------------------------------------------------------

def WCStoPIX(WCScoords, image):
    """Convert sky (WCS) coordinates to pixel coordinates for a given image.

    Uses the WCS solution stored in the FITS header to project equatorial
    coordinates (ICRS, decimal degrees) onto the pixel grid of ``image``.

    Parameters
    ----------
    WCScoords : tuple or list of array-like
        Sky coordinates as ``(ra, dec)`` in decimal degrees (ICRS frame).
        Each element may be a scalar or a 1-D array to convert multiple
        positions at once.
    image : str or astropy.io.fits.HDU
        Reference FITS image whose WCS solution is used for the projection.
        Accepted formats are the same as for :func:`OpenImage`.

    Returns
    -------
    PIXcoords : tuple of numpy.ndarray
        ``(xpix, ypix)`` pixel coordinates (0-based, origin at the center of
        the lower-left pixel, following the ``astropy`` convention).

    Notes
    -----
    The conversion is performed in ``'all'`` mode, which applies all WCS
    corrections (SIP distortions, etc.) if present in the header.
    """

    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    from astropy.wcs.utils import skycoord_to_pixel

    x, y = WCScoords
    img, hdr = OpenImage(image)

    wcs = WCS(hdr)

    c = SkyCoord(x, y,
                 frame="icrs",
                 unit="deg")

    PIXcoords = skycoord_to_pixel(coords=c,
                                  wcs=wcs,
                                  origin=0,
                                  mode='all')
    return PIXcoords


def PIXtoWCS(PIXcoords, image):
    """Convert pixel coordinates to sky (WCS) coordinates for a given image.

    Uses the WCS solution stored in the FITS header to convert pixel positions
    to equatorial coordinates (ICRS, decimal degrees).

    Parameters
    ----------
    PIXcoords : tuple or list of array-like
        Pixel coordinates as ``(xpix, ypix)``, 0-based. Each element may be a
        scalar or a 1-D array to convert multiple positions at once.
    image : str or astropy.io.fits.HDU
        Reference FITS image whose WCS solution is used for the deprojection.
        Accepted formats are the same as for :func:`OpenImage`.

    Returns
    -------
    WCScoords : list of numpy.ndarray
        ``[ra, dec]`` in decimal degrees (ICRS frame).

    Notes
    -----
    The conversion is performed in ``'all'`` mode (see :func:`WCStoPIX`).
    """

    from astropy.wcs import WCS
    from astropy.wcs.utils import pixel_to_skycoord

    x, y = PIXcoords
    img, hdr = OpenImage(image)

    wcs = WCS(hdr)

    coords = pixel_to_skycoord(xp=x, yp=y,
                               wcs=wcs,
                               origin=0,
                               mode='all')

    WCScoords = [coords.ra.degree, coords.dec.degree]

    return WCScoords


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

def Convolve(image, pixsize, fwhm):
    """Convolve a FITS image with a circular Gaussian beam.

    This is typically used to degrade the angular resolution of a map to match
    that of a lower-resolution observation (e.g. before cross-matching catalogs
    extracted at different wavelengths).

    Parameters
    ----------
    image : str or astropy.io.fits.HDU
        Input FITS image. Accepted formats are the same as for
        :func:`OpenImage`.
    pixsize : float
        Pixel scale of the input image in arcseconds per pixel.
    fwhm : float
        Full Width at Half Maximum (FWHM) of the target Gaussian beam in
        arcseconds. The standard deviation of the kernel is derived as
        ``sigma = fwhm / pixsize / (2 * sqrt(2 * ln(2)))``.

    Returns
    -------
    smoothed : astropy.io.fits.PrimaryHDU
        FITS HDU containing the convolved image and the original header
        (WCS metadata is preserved).

    Notes
    -----
    The smoothing is performed using
    :class:`astropy.convolution.Gaussian2DKernel` with equal standard
    deviations along both axes (circular beam). The kernel size is determined
    automatically by astropy based on the standard deviation.
    """

    from astropy.convolution import convolve, Gaussian2DKernel

    img, hdr = OpenImage(image)

    # Convert FWHM to Gaussian standard deviation in pixel units
    coef = 2 * np.sqrt(2 * np.log(2))
    x_stddev = fwhm / pixsize / coef
    y_stddev = fwhm / pixsize / coef
    print("sigma in pixel : ", x_stddev)

    smoothed = convolve(img, Gaussian2DKernel(x_stddev=x_stddev,
                                              y_stddev=y_stddev))

    # hdr["Beam"] = x_stddev
    smoothed = fits.PrimaryHDU(data=smoothed, header=hdr)

    return smoothed


# ---------------------------------------------------------------------------
# Polygon utilities
# ---------------------------------------------------------------------------

def orderingPoints(points, dist=1):
    """Order an unordered set of 2-D points into a connected path.

    Starting from the first point, the algorithm greedily appends the nearest
    remaining point within a distance threshold, producing a spatially ordered
    sequence suitable for polygon construction or contour tracing.

    Parameters
    ----------
    points : numpy.ndarray, shape (N, 2)
        Array of ``(x, y)`` coordinates to order.
    dist : float, optional
        Maximum squared distance between consecutive points for them to be
        considered neighbours. Default is ``1`` (adjacent pixels).

    Returns
    -------
    order : list of [float, float]
        Ordered list of ``[x, y]`` pairs forming a connected path through the
        input points.

    Notes
    -----
    This is an O(N²) greedy nearest-neighbour algorithm and should only be
    used on moderately sized point sets (e.g. polygon boundary pixels).
    """
    xp, yp = points[0]
    remaining_coords = points.copy().tolist()
    order = []
    k = 0

    while k < len(points):
        for i, (x, y) in enumerate(remaining_coords):
            if (x - xp) ** 2 + (y - yp) ** 2 <= dist:
                order.append([x, y])
                remaining_coords.remove(remaining_coords[i])
                xp, yp = x, y
                k += 1
    return order


def Window(image, coord="wcs"):
    """Extract the field-of-view boundary of a FITS image as a Shapely polygon.

    Detects the outer edge of valid (non-NaN) pixels in the image using binary
    morphological operations and returns the enclosing polygon either in pixel
    or WCS coordinates.

    Parameters
    ----------
    image : str or astropy.io.fits.HDU
        Input FITS image. Accepted formats are the same as for
        :func:`OpenImage`.
    coord : {'wcs', 'pix'}, optional
        Coordinate frame of the returned polygon. Use ``'wcs'`` (default) for
        equatorial coordinates in decimal degrees (ICRS) or ``'pix'`` for
        pixel coordinates.

    Returns
    -------
    shapely.geometry.Polygon
        Polygon tracing the boundary of the valid image area in the requested
        coordinate frame. This can be used as the ``fov_window`` parameter of a
        FAMILY ``Data`` object to restrict source extraction to the actual
        coverage of the map.
    """
    import scipy.ndimage.morphology as snm
    from shapely.geometry import Polygon
    from polygons import ReshapeCoordForPoly

    img, hdr = OpenImage(image)

    # Identify the filled region of valid pixels and extract its border
    filled_mask = snm.binary_fill_holes(img != np.nan)
    interior_mask = snm.binary_erosion(filled_mask)
    edges = filled_mask ^ interior_mask
    mask_idx = np.flip(np.where(edges), axis=0)

    points = ReshapeCoordForPoly(mask_idx[0], mask_idx[1])

    # Sort boundary pixels into a spatially connected sequence
    ordered = orderingPoints(points, dist=1)

    if coord == "pix":
        return Polygon(ordered)

    if coord == "wcs":
        x, y = PIXtoWCS(np.reshape(np.ravel(ordered), (2, len(mask_idx[0])), order='F'), image)
        return Polygon(ReshapeCoordForPoly(x, y))


def LinearInterpolationPolygon(xo, yo, n=100):
    """Densify a polygon boundary by linear interpolation between vertices.

    Inserts ``n`` evenly spaced points between each consecutive pair of polygon
    vertices and rounds the results to the nearest integer (pixel) coordinates.
    This ensures that all pixels enclosed by the polygon boundary are captured
    when rasterising the polygon onto a pixel grid.

    Parameters
    ----------
    xo : list or array-like of float
        x-coordinates (pixel column indices) of the polygon vertices.
    yo : list or array-like of float
        y-coordinates (pixel row indices) of the polygon vertices.
    n : int, optional
        Number of interpolated points per edge. Default is ``100``.

    Returns
    -------
    x : list of int
        x pixel coordinates of the densified (and deduplicated) boundary.
    y : list of int
        y pixel coordinates of the densified (and deduplicated) boundary.

    Notes
    -----
    Duplicate pixel positions introduced by the rounding are removed using a
    set operation, so the output order is not guaranteed to be spatially
    consecutive.
    """

    x_new = []
    y_new = []
    xlen = len(xo)

    for i, xvalue in enumerate(xo):
        # Compute displacement to the next vertex (with wrap-around)
        dx = xo[(i + 1) % xlen] - xvalue
        dy = yo[(i + 1) % xlen] - yo[i]

        xint = np.linspace(0, dx, n)
        yint = np.linspace(0, dy, n)

        [x_new.append(int(np.round(xvalue + xi))) for xi in xint]
        [y_new.append(int(np.round(yo[i] + yi))) for yi in yint]

    # Remove duplicate pixel positions arising from rounding
    x = [xx for xx, yy in list(set(zip(x_new, y_new)))]
    y = [yy for xx, yy in list(set(zip(x_new, y_new)))]

    return x, y


def PolygonToMask(polygons, image, n=100, verbose=False):
    """Create a binary FITS mask from a list of sky-projected polygons.

    For each polygon, the WCS boundary vertices are projected onto the pixel
    grid of ``image``, the boundary is densified with
    :func:`LinearInterpolationPolygon`, and the enclosed area is filled using
    binary morphology. The resulting mask can be used to select pixels
    belonging to a given source footprint.

    Parameters
    ----------
    polygons : list of shapely.geometry.Polygon
        Source footprints in WCS (equatorial) coordinates (decimal degrees,
        ICRS). Typically the ``polygon`` column of the FAMILY structures table.
    image : str or astropy.io.fits.HDU
        Reference FITS image onto which the polygons are projected. The WCS
        of this image is used for the coordinate conversion.
    n : int, optional
        Number of interpolation points per polygon edge (passed to
        :func:`LinearInterpolationPolygon`). Increase for highly elongated or
        large polygons. Default is ``100``.
    verbose : bool, optional
        If ``True``, display a progress bar via :mod:`tqdm`. Default is
        ``False``.

    Returns
    -------
    mask : astropy.io.fits.PrimaryHDU
        FITS HDU whose data array is a binary mask (0/1) of the same shape as
        ``image``. Pixels inside any of the input polygons are set to ``1``.

    Notes
    -----
    Pixel coordinates that fall outside the image boundaries are silently
    ignored.
    """

    from tqdm import tqdm
    import scipy.ndimage.morphology as snm
    import time

    img, hdr = OpenImage(image)

    xlen, ylen = np.shape(img)

    mask = np.zeros_like(img)

    if verbose:
        print("Creating mask for ellipses \n")
        time.sleep(1)

    for polygon in tqdm(polygons, disable=not verbose):
        # Project polygon boundary from WCS to pixel coordinates
        WCScoords = polygon.exterior.xy
        x, y = WCStoPIX(WCScoords, image)
        x, y = LinearInterpolationPolygon(x, y, n=n)

        # Rasterise the polygon boundary
        for xx, yy in zip(x, y):
            try:
                mask[yy, xx] = 1
            except:
                continue

    # Fill the interior of the rasterised polygon boundary
    filled_mask = snm.binary_fill_holes(mask) * 1

    mask = fits.PrimaryHDU(data=filled_mask, header=hdr)
    return mask


# ---------------------------------------------------------------------------
# Photometric statistics within polygon regions
# ---------------------------------------------------------------------------

def MeanPixelInPolygon(polygons, image, verbose=False):
    """Compute the geometric mean pixel value inside each polygon.

    For each polygon in the list, a binary mask is generated with
    :func:`PolygonToMask`, and the geometric mean (mean of log-values,
    exponentiated) of all non-zero pixels within the mask is returned.
    The geometric mean is preferred over the arithmetic mean for
    log-normally distributed quantities such as column density or flux density.

    Parameters
    ----------
    polygons : list of shapely.geometry.Polygon
        Source footprints in WCS coordinates. See :func:`PolygonToMask`.
    image : str or astropy.io.fits.HDU
        FITS image providing the pixel values. Accepted formats are the same
        as for :func:`OpenImage`.
    verbose : bool, optional
        If ``True``, display a progress bar via :mod:`tqdm`. Default is
        ``False``.

    Returns
    -------
    meanpix : list of float
        Geometric mean pixel value for each polygon. ``numpy.nan`` is
        returned for polygons that contain no valid pixels.
    """

    from tqdm import tqdm

    meanpix = []
    img, hdr = OpenImage(image)

    for polygon in tqdm(polygons, disable=not verbose):
        mask = PolygonToMask([polygon], image)

        values = img[img * mask.data != 0]

        if len(values) == 0:
            meanpix.append(np.nan)
        else:
            # Geometric mean: exp(mean(log(values)))
            log_valu = np.nanmean(np.log(values))
            meanpix.append(np.exp(log_valu))

    return meanpix


def MedPixelInPolygon(polygons, image, verbose=False):
    """Compute the median pixel value inside each polygon.

    For each polygon in the list, a binary mask is generated with
    :func:`PolygonToMask`, and the median of all non-zero masked pixels is
    returned. The median is robust to bright compact sources or artefacts
    within the polygon.

    Parameters
    ----------
    polygons : list of shapely.geometry.Polygon
        Source footprints in WCS coordinates. See :func:`PolygonToMask`.
    image : str or astropy.io.fits.HDU
        FITS image providing the pixel values. Accepted formats are the same
        as for :func:`OpenImage`.
    verbose : bool, optional
        If ``True``, display a progress bar via :mod:`tqdm`. Default is
        ``False``.

    Returns
    -------
    meanpix : list of float
        Median pixel value for each polygon. ``numpy.nan`` is returned for
        polygons that contain no valid pixels.
    """

    from tqdm import tqdm

    meanpix = []
    img, hdr = OpenImage(image)

    for polygon in tqdm(polygons, disable=not verbose):
        mask = PolygonToMask([polygon], image)

        values = img[img * mask.data != 0]

        if len(values) == 0:
            meanpix.append(np.nan)
        else:
            meanpix.append(np.nanmedian(values))

    return meanpix


def MaxPixelInPolygon(polygons, image, verbose=False):
    """Compute the maximum pixel value inside each polygon.

    For each polygon in the list, a binary mask is generated with
    :func:`PolygonToMask`, and the maximum of all non-zero masked pixels is
    returned. This can be used to estimate peak column densities or flux
    densities within source footprints.

    Parameters
    ----------
    polygons : list of shapely.geometry.Polygon
        Source footprints in WCS coordinates. See :func:`PolygonToMask`.
    image : str or astropy.io.fits.HDU
        FITS image providing the pixel values. Accepted formats are the same
        as for :func:`OpenImage`.
    verbose : bool, optional
        If ``True``, display a progress bar via :mod:`tqdm`. Default is
        ``False``.

    Returns
    -------
    maxpix : list of float
        Maximum pixel value for each polygon. ``numpy.nan`` is returned for
        polygons that contain no valid pixels.
    """

    from tqdm import tqdm

    maxpix = []
    img, hdr = OpenImage(image)

    for polygon in tqdm(polygons, disable=not verbose):
        mask = PolygonToMask([polygon], image)

        values = img[img * mask.data != 0]

        if len(values) == 0:
            maxpix.append(np.nan)
        else:
            maxpix.append(np.nanmax(values))

    return maxpix


# ---------------------------------------------------------------------------
# Column-density histogram decomposition by structure mode
# ---------------------------------------------------------------------------

def pixelBins(catalog, fitsfile, Poly_key="polygon", norm=True):
    """Decompose pixel counts per column-density bin by hierarchical structure mode.

    This is the primary diagnostic function of the FAMILY pipeline. It answers
    the question: *at a given column density, what fraction of the image area
    is dominated by hierarchical, linear, or isolated fragmentation?*

    The function builds binary masks for each structure mode (HIERARCHICAL,
    LINEAR, ISOLATED) by rasterising the corresponding polygon footprints from
    ``catalog`` onto the reference FITS image. It then loops over
    logarithmically-spaced bins of pixel values (e.g. H₂ column density) and,
    for each bin, counts how many pixels fall exclusively within each mode mask
    as well as in the pairwise overlap regions (HL, LI, HI).

    Parameters
    ----------
    catalog : pandas.DataFrame
        Structures table produced by ``Network.getStructuresTable()``. Must
        contain at least the columns ``Poly_key`` (polygon footprints in WCS)
        and ``"mode"`` (string name of the fragmentation mode, matching
        ``StructureMode`` enum names: ``"HIERARCHICAL"``, ``"LINEAR"``,
        ``"ISOLATED"``).
    fitsfile : str
        Path to the reference FITS image. Pixel values are used as the
        physical quantity along the histogram axis (typically H₂ column
        density in cm⁻²).
    Poly_key : str, optional
        Name of the column in ``catalog`` that stores the polygon footprints.
        Default is ``"polygon"``.
    norm : bool, optional
        If ``True`` (default), counts in each bin are divided by the total
        number of pixels in that bin (``N``), yielding fractional area
        coverage. If ``False``, raw pixel counts are returned.

    Returns
    -------
    Nlst : list of int
        Total number of pixels in each column-density bin.
    Hlst : list of float
        Pixel count (or fraction) exclusively inside HIERARCHICAL structures.
    Llst : list of float
        Pixel count (or fraction) exclusively inside LINEAR structures.
    Ilst : list of float
        Pixel count (or fraction) exclusively inside ISOLATED structures.
    HLlst : list of float
        Pixel count (or fraction) inside the overlap of HIERARCHICAL and
        LINEAR structures.
    LIlst : list of float
        Pixel count (or fraction) inside the overlap of LINEAR and ISOLATED
        structures.
    HIlst : list of float
        Pixel count (or fraction) inside the overlap of HIERARCHICAL and
        ISOLATED structures.
    logbins : numpy.ndarray
        Bin edges used for the histogram, logarithmically spaced between the
        minimum and maximum pixel values of the image.

    Notes
    -----
    - Pairwise overlaps (HL, LI, HI) represent sky regions covered by
      footprints of two different modes simultaneously; triple overlaps are
      not tracked separately.
    - The exclusive count for mode H is computed as
      ``|H ∩ bin| XOR |HL ∩ bin| XOR |HI ∩ bin|``, i.e. pixels inside H
      but *not* shared with L or I.
    - The bin edges span ``[min(image), max(image)]`` on a log scale with 50
      bins (numpy default for ``np.logspace``).
    """
    import scipy.ndimage.morphology as snm
    from . import multiscale_structures as ms

    image, header = OpenImage(fitsfile)

    Hlst, Llst, Ilst, HLlst, HIlst, LIlst = [], [], [], [], [], []
    Nlst = []

    # --- Build one binary mask per structure mode ----------------------------
    maskH = np.zeros_like(image)
    maskL = np.zeros_like(image)
    maskI = np.zeros_like(image)

    for idx, poly in enumerate(catalog[Poly_key]):
        # Project the polygon boundary from WCS to pixel coordinates and
        # densify it to avoid gaps when rasterising
        x, y = WCStoPIX(poly.exterior.xy, fitsfile)
        x, y = LinearInterpolationPolygon(x, y)

        if catalog["mode"][idx] == ms.StructureMode.HIERARCHICAL.name:
            for x, y in zip(x, y):
                try:
                    maskH[y, x] = 1
                except IndexError:
                    pass
        elif catalog["mode"][idx] == ms.StructureMode.LINEAR.name:
            for x, y in zip(x, y):
                try:
                    maskL[y, x] = 1
                except IndexError:
                    pass
        elif catalog["mode"][idx] == ms.StructureMode.ISOLATED.name:
            for x, y in zip(x, y):
                try:
                    maskI[y, x] = 1
                except IndexError:
                    pass

    # Fill polygon interiors (boundary rasterisation leaves only the edges set)
    filled_maskH = snm.binary_fill_holes(maskH)
    filled_maskL = snm.binary_fill_holes(maskL)
    filled_maskI = snm.binary_fill_holes(maskI)

    # --- Define logarithmic column-density bins ------------------------------
    logmin = np.log10(np.amin(image))
    logmax = np.log10(np.amax(image))
    logbins = np.logspace(logmin, logmax)
    print(f"decomposing using {len(logbins)} bins evenly log-sampled between {logmin} and {logmax} [pixel unit]")

    # --- Loop over bins and accumulate counts --------------------------------
    for k, threshold in enumerate(logbins[1:]):
        # Boolean mask selecting pixels in the current column-density bin
        im = np.logical_and(image <= threshold, image > logbins[k])
        N = np.nansum(im)

        # Pairwise overlaps between structure-mode masks within this bin
        common_HL = filled_maskH * filled_maskL * im
        common_LI = filled_maskI * filled_maskL * im
        common_HI = filled_maskH * filled_maskI * im

        HL = np.nansum(common_HL)
        LI = np.nansum(common_LI)
        HI = np.nansum(common_HI)

        # Exclusive counts: pixels inside mode X but not shared with others
        H = np.nansum((filled_maskH * im) ^ common_HL ^ common_HI)
        L = np.nansum((filled_maskL * im) ^ common_HL ^ common_LI)
        I = np.nansum((filled_maskI * im) ^ common_HI ^ common_LI)

        if norm:
            Hlst.append(H / N)
            Llst.append(L / N)
            Ilst.append(I / N)

            HLlst.append(HL / N)
            LIlst.append(LI / N)
            HIlst.append(HI / N)
        else:
            Hlst.append(H)
            Llst.append(L)
            Ilst.append(I)

            HLlst.append(HL)
            LIlst.append(LI)
            HIlst.append(HI)

        Nlst.append(N)

    return Nlst, Hlst, Llst, Ilst, HLlst, LIlst, HIlst, logbins


def _pixelBins(graph, fitsfile):
    """Deprecated graph-based pixel-bin decomposition (internal use only).

    .. deprecated::
        This function operates on a raw NetworkX graph object with node
        attributes ``"Polygon"`` and ``"Mode"``, predating the pandas-based
        catalog interface. Use :func:`pixelBins` instead.

    Parameters
    ----------
    graph : networkx.Graph
        Node graph where each node carries a ``"Polygon"`` (Shapely polygon
        in WCS) and a ``"Mode"`` string (``"Hierarchical"``, ``"Linear"``,
        or ``"Isolated"``).
    fitsfile : str
        Path to the reference FITS image.

    Returns
    -------
    Same as :func:`pixelBins` with ``norm=True``.
    """
    import scipy.ndimage.morphology as snm

    image, header = OpenImage(fitsfile)

    Hlst, Llst, Ilst, HLlst, HIlst, LIlst = [], [], [], [], [], []

    Nlst = []

    X, Y = np.shape(image)

    # --- Build one binary mask per structure mode ----------------------------
    maskH = np.zeros_like(image)
    maskL = np.zeros_like(image)
    maskI = np.zeros_like(image)

    for node, poly in graph.nodes("Polygon"):

        x, y = WCStoPIX(poly.exterior.xy, fitsfile)
        x, y = LinearInterpolationPolygon(x, y)

        if graph.nodes("Mode")[node] == "Hierarchical":
            for x, y in zip(x, y):
                try:
                    maskH[y, x] = 1
                except IndexError:
                    pass
        if graph.nodes("Mode")[node] == "Linear":
            for x, y in zip(x, y):
                try:
                    maskL[y, x] = 1
                except IndexError:
                    pass
        if graph.nodes("Mode")[node] == "Isolated":
            for x, y in zip(x, y):
                try:
                    maskI[y, x] = 1
                except IndexError:
                    pass

    # Fill polygon interiors
    filled_maskH = snm.binary_fill_holes(maskH)
    filled_maskL = snm.binary_fill_holes(maskL)
    filled_maskI = snm.binary_fill_holes(maskI)

    # --- Define logarithmic column-density bins ------------------------------
    logmin = np.log10(np.amin(image))
    logmax = np.log10(np.amax(image))
    logbins = np.logspace(logmin, logmax)

    # --- Loop over bins and accumulate normalised counts ---------------------
    for k, threshold in enumerate(logbins[1:]):
        im = np.logical_and(image <= threshold, image > logbins[k])
        N = np.nansum(im)

        # Pairwise overlaps
        common_HL = filled_maskH * filled_maskL * im
        common_LI = filled_maskI * filled_maskL * im
        common_HI = filled_maskH * filled_maskI * im

        HL = np.nansum(common_HL)
        LI = np.nansum(common_LI)
        HI = np.nansum(common_HI)

        # Exclusive counts
        H = np.nansum((filled_maskH * im) ^ common_HL ^ common_HI)
        L = np.nansum((filled_maskL * im) ^ common_HL ^ common_LI)
        I = np.nansum((filled_maskI * im) ^ common_HI ^ common_LI)

        Hlst.append(H / N)
        Llst.append(L / N)
        Ilst.append(I / N)

        HLlst.append(HL / N)
        LIlst.append(LI / N)
        HIlst.append(HI / N)

        Nlst.append(N)

    return Nlst, Hlst, Llst, Ilst, HLlst, LIlst, HIlst, logbins
