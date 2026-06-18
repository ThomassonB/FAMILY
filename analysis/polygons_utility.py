"""
polygons_utility.py
-------------------
Utility functions for building and comparing Shapely polygon representations
of astronomical sources (clumps, cores, etc.) extracted from multi-scale
observations.

Sources are stored as ellipse parameters (centre, semi-axes, position angle)
in a source catalogue and are converted here to discretised polygons so that
spatial relationships (overlap, separation) can be evaluated efficiently with
the Shapely/STRtree machinery.

Angular distances are computed with PyAstronomy.pyasl.getAngDist, which
returns great-circle separations in degrees — the natural unit for sky-plane
coordinates.

Dependencies
------------
numpy, shapely (>=2.0), PyAstronomy, tqdm
"""

from .standard_variables import ellipse_params_labels

import numpy as np
import shapely.geometry.polygon as shp
import time


# ---------------------------------------------------------------------------
# Ellipse discretisation
# ---------------------------------------------------------------------------

def ellipse(*args, N=128):
    """
    Compute the (x, y) boundary coordinates of one or several ellipses.

    The parametric equations follow the standard rotated-ellipse convention::

        x = xc + a*cos(t)*cos(phi) - b*sin(t)*sin(phi)
        y = yc + a*cos(t)*sin(phi) + b*sin(t)*cos(phi)

    where ``t`` sweeps [0, 2pi] and ``phi`` is the position angle (radians,
    east of north in the sky-plane convention used by the catalogues).

    Parameters
    ----------
    *args : array-like, shape (n_sources,) each
        Positional arguments in the order defined by
        ``standard_variables.ellipse_params_labels``: ``xc, yc, a, b, phi``.
        Each can be a scalar or a column vector of shape ``(n_sources, 1)``
        to vectorise over an entire catalogue at once.
    N : int, optional
        Number of sample points along the ellipse boundary (default 128).
        Higher values give smoother polygons at the cost of memory.

    Returns
    -------
    x : ndarray, shape (n_sources, N)
        Right-ascension (or generic x) coordinates of the boundary points.
    y : ndarray, shape (n_sources, N)
        Declination (or generic y) coordinates of the boundary points.
    """
    theta = np.linspace(0, 2 * np.pi, N)[np.newaxis, :]
    xc, yc, a, b, phi = args
    x = xc + a * np.cos(theta) * np.cos(phi) - b * np.sin(theta) * np.sin(phi)
    y = yc + a * np.cos(theta) * np.sin(phi) + b * np.sin(theta) * np.cos(phi)
    return x, y


# ---------------------------------------------------------------------------
# Coordinate reshaping helpers
# ---------------------------------------------------------------------------

def reshape_coord_for_poly(x, y):
    """
    Interleave x and y coordinates into a Shapely-compatible (N, 2) array.

    Parameters
    ----------
    x : 1D numpy array or list
        x (or RA) positions of the polygon vertices.
    y : 1D numpy array or list
        y (or Dec) positions of the polygon vertices.

    Returns
    -------
    coords : ndarray, shape (len(x), 2)
        Array of ``[x_i, y_i]`` pairs ready to be passed to
        ``shapely.geometry.polygon.Polygon``.
    """
    coords = np.reshape(np.ravel([x, y]), (len(x), 2), order='F')
    return coords

def reshape_coord_for_poly_vec(x, y):
    """
    Vectorised version of :func:`reshape_coord_for_poly` for batches of
    polygons.

    Parameters
    ----------
    x : ndarray, shape (n_sources, N)
        x (or RA) boundary coordinates for all sources.
    y : ndarray, shape (n_sources, N)
        y (or Dec) boundary coordinates for all sources.

    Returns
    -------
    coords : ndarray, shape (n_sources, N, 2)
        Stacked coordinate array where ``coords[i]`` is the ``(N, 2)``
        vertex array for source ``i``.
    """
    #coords = np.reshape(np.ravel([x, y]), (x.shape[0], 2, x.shape[1]), order='F')
    coords = np.dstack((x, y))
    return coords

# ---------------------------------------------------------------------------
# Polygon construction from a source catalogue
# ---------------------------------------------------------------------------

def buildPolygons(catalog, N=128):
    """
    Build a list of Shapely polygons from ellipse parameters stored in a
    source catalogue.

    Each row of ``catalog`` is treated as a single astronomical source whose
    spatial extent is approximated by a discretised ellipse with ``N`` vertices.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Source catalogue. Must contain the columns listed in
        ``standard_variables.ellipse_params_labels``
        (typically ``xc, yc, a, b, phi``).
    N : int, optional
        Number of boundary points used to discretise each ellipse (default 128).

    Returns
    -------
    polygons : list of shapely.geometry.polygon.Polygon
        One polygon per row in ``catalog``, in the same order.

    Notes
    -----
    A previous implementation accepted pre-computed polygon vertex lists
    directly; that path is kept below as commented-out legacy code for
    reference.
    """
    # Extract ellipse parameters as column vectors (shape: n_sources x 1)
    args = np.array([catalog[label].to_numpy()[:, np.newaxis] for label in ellipse_params_labels])
    x, y = ellipse(*args, N=N)
    coords = reshape_coord_for_poly_vec(x, y)

    """
    old script for polygons instead of ellipses parameters
    x, y = catalog[strings[0]], catalog[strings[1]]
    coords = []
    for xx, yy in zip(x, y):
        coords.append(reshape_coord_for_poly(xx, yy))
        #print(len(catalog), xx, yy)
        #pp = shp.Polygon(reshape_coord_for_poly(xx, yy))
        #if not pp.is_valid:
        #    print(len(catalog), xx, yy)
    """
    return [shp.Polygon(c) for c in coords]

# ---------------------------------------------------------------------------
# Pairwise distance / size matrices
# ---------------------------------------------------------------------------

def testMatrix(mat, size, p):
    """
    Boolean criterion to flag polygon pairs close enough to warrant an exact
    minimal-distance computation.

    A pair ``(i, j)`` is flagged when the ratio of their combined characteristic
    size to their centroid separation exceeds the tolerance ``p``::

        size[i,j] / mat[i,j] > p

    Parameters
    ----------
    mat : ndarray, shape (n, n)
        Pairwise centroid-separation matrix (degrees).
    size : ndarray, shape (n, n)
        Pairwise combined-size matrix (same units as ``mat``).
    p : float
        Proximity tolerance. Pairs with ``size/mat > p`` are recomputed using
        the exact polygon boundary distance.

    Returns
    -------
    mask : ndarray of bool, shape (n, n)
        ``True`` where the exact minimal distance should replace the centroid
        approximation.
    """
    return size / mat > p

def sizeMatrix(polygons):
    """
    Compute the pairwise combined-size matrix for a set of polygons.

    The characteristic size of a polygon is the square root of its area
    (geometric mean radius of an equivalent square). The combined size of a
    pair is the *sum* of their individual sizes, providing a conservative
    upper bound on the separation below which overlap is geometrically
    possible.

    Parameters
    ----------
    polygons : list of shapely.geometry.polygon.Polygon
        Source polygons in a common coordinate system (degrees).

    Returns
    -------
    mat : ndarray, shape (n, n)
        Symmetric matrix where ``mat[i, j] = sqrt(area_i) + sqrt(area_j)``.
        Diagonal entries are zero.
    """
    n = len(polygons)
    mat = np.zeros(shape=(n, n))
    for i, p1 in enumerate(polygons):
        for j, p2 in enumerate(polygons):
            if j > i:
                mat[i, j] = np.sqrt(p1.area) + np.sqrt(p2.area)
    mat += np.transpose(mat)
    return mat


def coordMatrix(polygons):
    """
    Extract the centroid coordinates of a list of polygons.

    Parameters
    ----------
    polygons : list of shapely.geometry.polygon.Polygon
        Source polygons.

    Returns
    -------
    lst : ndarray, shape (n, 2)
        Centroid array where ``lst[i, 0]`` is the x (RA) coordinate and
        ``lst[i, 1]`` is the y (Dec) coordinate of polygon ``i``.
    """
    n = len(polygons)
    lst = np.zeros(shape=(n, 2))
    lst[:, 0] = [p.centroid.xy[0][0] for i, p in enumerate(polygons)]
    lst[:, 1] = [p.centroid.xy[1][0] for i, p in enumerate(polygons)]
    return lst


def sepAngular(lst):
    """
    Compute the pairwise angular (great-circle) separation matrix between
    polygon centroids.

    Uses ``PyAstronomy.pyasl.getAngDist`` to account for the spherical metric
    on the sky, which matters near the celestial poles or for wide-field
    observations where the flat-sky approximation breaks down.

    Parameters
    ----------
    lst : ndarray, shape (n, 2)
        Centroid coordinates as returned by :func:`coordMatrix`.
        ``lst[:, 0]`` is RA and ``lst[:, 1]`` is Dec, both in degrees.

    Returns
    -------
    mat : ndarray, shape (n, n)
        Symmetric matrix of great-circle separations in degrees.
    """
    from PyAstronomy import pyasl
    mat = pyasl.getAngDist(lst[:, 0, np.newaxis], lst[:, 1, np.newaxis],
                           lst[np.newaxis, :, 0], lst[np.newaxis, :, 1])
    return mat

def sepDistance(lst):
    """
    Compute the pairwise Euclidean distance matrix between polygon centroids.

    This is the flat-sky approximation, suitable for small fields where
    angular distortion is negligible. For large fields use :func:`sepAngular`.

    Parameters
    ----------
    lst : ndarray, shape (n, 2)
        Centroid coordinates as returned by :func:`coordMatrix`.

    Returns
    -------
    mat : ndarray, shape (n, n)
        Symmetric matrix of Euclidean centroid separations (same units as
        ``lst``).
    """
    mat = np.sqrt(
        (lst[:, 0, np.newaxis] - lst[np.newaxis, :, 0]) ** 2 + (lst[:, 1, np.newaxis] - lst[np.newaxis, :, 1]) ** 2)
    return mat

# ---------------------------------------------------------------------------
# Exact minimal-distance computation (boundary-to-boundary)
# ---------------------------------------------------------------------------

def minDistance(polygons, mat, replace, p=0.05, verbose=False):
    """
    Replace centroid-separation entries in ``mat`` with exact
    boundary-to-boundary angular distances for pairs flagged by ``replace``.

    For pairs where the centroid separation is not a reliable proxy for the
    true source separation (e.g. elongated or overlapping sources), the
    nearest boundary points are located with Shapely's ``nearest_points`` and
    their angular separation is recomputed with ``pyasl.getAngDist``.

    Only the upper-triangular part of ``replace`` is processed; the result is
    written symmetrically into ``mat``.

    Parameters
    ----------
    polygons : list of shapely.geometry.polygon.Polygon
        Source polygons.
    mat : ndarray, shape (n, n)
        Distance matrix initialised with centroid separations (degrees).
        Modified in place.
    replace : ndarray of bool, shape (n, n)
        Mask indicating which pairs need exact recomputation. Only the upper
        triangle is used (``np.triu`` is applied internally).
    p : float, optional
        Unused; kept for API compatibility (default 0.05).
    verbose : bool, optional
        If ``True``, print progress and display a tqdm progress bar.

    Returns
    -------
    mat : ndarray, shape (n, n)
        Updated distance matrix with exact boundary distances for flagged pairs.
    """
    from PyAstronomy import pyasl
    from shapely.ops import nearest_points
    from tqdm import tqdm
    replace = np.triu(replace, k=0)
    idx = np.where(replace)

    if verbose:
        print("Processing minimal distances")
        time.sleep(0.5)

    for i, x in enumerate(tqdm(idx[0], disable=not verbose)):
        # d = polygons[x].distance(polygons[idx[1][i]])
        p1, p2 = nearest_points(polygons[x], polygons[idx[1][i]])
        d = pyasl.getAngDist(p1.x, p1.y, p2.x, p2.y)
        mat[x, idx[1][i]] = d
        mat[idx[1][i], x] = d
    return mat


# ---------------------------------------------------------------------------
# High-level distance drivers
# ---------------------------------------------------------------------------

def distancePolyst(polygons, p=0.05, verbose=False):
    """
    Compute the pairwise angular separation matrix using an adaptive two-stage
    strategy.

    Stage 1 — fast approximation
        Great-circle distances between polygon *centroids* are computed for all
        pairs.

    Stage 2 — exact refinement
        Pairs whose combined characteristic size is a significant fraction
        (``p``) of their centroid separation are recomputed using the exact
        boundary distance (:func:`minDistance`). This avoids costly
        nearest-boundary computation for well-separated sources while remaining
        accurate for close or overlapping ones.

    Parameters
    ----------
    polygons : list of shapely.geometry.polygon.Polygon
        Source polygons in equatorial coordinates (degrees).
    p : float, optional
        Proximity tolerance. A pair is refined when
        ``size[i,j] / centroid_sep[i,j] > p`` (default 0.05).
    verbose : bool, optional
        If ``True``, print the number of pairs requiring exact computation and
        display a progress bar during stage 2.

    Returns
    -------
    mat : ndarray, shape (n, n)
        Symmetric matrix of pairwise angular separations in degrees.
    """
    lst = coordMatrix(polygons)
    mat = sepAngular(lst)
    size = sizeMatrix(polygons)
    replace = testMatrix(mat, size, p)

    if verbose:
        print("There are", np.sum(replace) // 2, "paire(s) of polygons that needs minimal distance computation.")
        time.sleep(0.5)

    return minDistance(polygons, mat, replace, p, verbose)


def mindistancePolyst(polygons, verbose=False):
    """
    Compute the exact pairwise boundary-to-boundary angular separation matrix
    for *all* polygon pairs.

    Unlike :func:`distancePolyst`, no centroid-based pre-screening is applied:
    nearest boundary points are located for every pair. This is more accurate
    but scales as O(n^2) and can be slow for large catalogues.

    Parameters
    ----------
    polygons : list of shapely.geometry.polygon.Polygon
        Source polygons in equatorial coordinates (degrees).
    verbose : bool, optional
        If ``True``, display a tqdm progress bar over the outer loop.

    Returns
    -------
    mat : ndarray, shape (n, n)
        Symmetric matrix of exact boundary-to-boundary angular separations in
        degrees. Diagonal entries are zero.
    """
    from PyAstronomy import pyasl
    from shapely.ops import nearest_points
    from tqdm import tqdm

    n = len(polygons)
    mat = np.zeros(shape=(n, n))

    for i in tqdm(range(n), disable=not verbose):
        for j in range(i):
            p1, p2 = nearest_points(polygons[i], polygons[j])
            d = pyasl.getAngDist(p1.x, p1.y, p2.x, p2.y)
            mat[i, j] = d
            mat[j, i] = d

    return mat

# ---------------------------------------------------------------------------
# Spatial overlap / containment matrix
# ---------------------------------------------------------------------------

def overlapMatrix(polygons1, polygons2):
    """
    Compute the fractional spatial overlap between two sets of source polygons
    at different observational scales.

    This function is central to the multi-scale hierarchy building step of
    FAMILY: it quantifies how much of a fine-scale source (``polygons2``) is
    spatially contained within a coarse-scale source (``polygons1``). The
    result drives the construction of the hierarchical network — an edge is
    drawn when the overlap fraction exceeds the ``min_overlap`` threshold
    defined in ``analyse.Network``.

    An STRtree spatial index is built on ``polygons2`` so that only candidate
    intersecting pairs are tested, reducing complexity from O(n*m) to
    approximately O((n+m) log m).

    For each intersecting pair ``(i, j)``:

    - If ``polygons2[j]`` is fully contained within ``polygons1[i]``, the
      overlap fraction is 1.
    - Otherwise it is the intersection area divided by the area of the
      *smaller* polygon, so that a compact source mostly inside a large
      envelope scores highly even when the envelope extends far beyond it.

    Parameters
    ----------
    polygons1 : list of shapely.geometry.polygon.Polygon
        Coarse-scale (lower angular resolution) source polygons.
    polygons2 : list of shapely.geometry.polygon.Polygon
        Fine-scale (higher angular resolution) source polygons.

    Returns
    -------
    area : ndarray, shape (len(polygons1), len(polygons2))
        Fractional overlap matrix. ``area[i, j]`` is the overlap fraction
        between coarse source ``i`` and fine source ``j``:
        0 = no intersection, (0,1) = partial overlap, 1 = full containment.
    """
    # polygons1, larger scale
    # polygons2, smaller scale
    from shapely.strtree import STRtree

    area = np.zeros(shape=(len(polygons1), len(polygons2)))
    tree = STRtree(polygons2)

    arr_indices = tree.query(polygons1, predicate="intersects")    
    for idx1, idx2 in zip(*arr_indices):
        if polygons2[idx2].within(polygons1[idx1]):
            area[idx1, idx2] = 1
        else:
            intersct = polygons2[idx2].intersection(polygons1[idx1])
            area[idx1, idx2] = intersct.area / min(polygons1[idx1].area, polygons2[idx2].area)    
    return area