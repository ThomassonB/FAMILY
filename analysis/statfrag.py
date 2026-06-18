"""
statfrag.py — Statistical fragmentation utilities for the FAMILY pipeline.

This module provides tools to generate synthetic 3D Gaussian source distributions
and to sample discrete/continuous probability distributions. These are used in
the context of statistical fragmentation analysis of star-forming regions, where
synthetic clumps or cores are injected into observed maps to assess detection
completeness and fragmentation statistics.

Classes
-------
DiscretePDF : Base class for discrete probability distributions (factory pattern).
    Binary  : Samples integer fragment counts consistent with a non-integer mean.
    Poisson : Samples fragment counts from a Poisson distribution.

CoordinatesPDF : Base class for 3D spatial coordinate distributions (factory pattern).
    Uniform  : Draws positions uniformly within a box.
    Gaussian : Draws positions from a Gaussian distribution.

Functions
---------
floor, ceil                : Integer floor/ceiling helpers.
change_mat_base            : Similarity transform of a matrix under a rotation.
set_gaussian_cube          : Evaluate a rotated multivariate Gaussian on a 3D grid.
set_gaussian_cube_dep      : Axis-aligned Gaussian on a 3D grid (deprecated form).
"""

import numpy as np
import scipy.integrate


def floor(x):
    """Return the floor of x as a plain Python integer.

    Parameters
    ----------
    x : float
        Input value.

    Returns
    -------
    int
        Largest integer less than or equal to x.
    """
    return int(x)


def ceil(x):
    """Return the ceiling of x as a plain Python integer.

    Parameters
    ----------
    x : float
        Input value.

    Returns
    -------
    int
        Smallest integer greater than or equal to x, computed as int(x) + 1.

    Notes
    -----
    This is a simplified ceiling that always adds 1 to the truncated integer.
    For exact integer inputs this will differ from ``math.ceil``.
    """
    return int(x) + 1


def change_mat_base(matrix, rotation):
    """Apply a similarity (change-of-basis) transformation to a matrix.

    Computes ``R @ matrix @ R^{-1}``, which expresses ``matrix`` in the
    coordinate frame defined by ``rotation``.  This is used to rotate an
    axis-aligned covariance matrix into an arbitrary orientation.

    Parameters
    ----------
    matrix : array_like, shape (N, N)
        The matrix to transform (e.g. a diagonal covariance matrix).
    rotation : array_like, shape (N, N)
        The rotation (or more generally invertible) matrix defining the new basis.

    Returns
    -------
    numpy.ndarray, shape (N, N)
        The transformed matrix in the new basis.
    """
    from numpy.linalg import inv
    inv_rot = inv(rotation)
    return np.matmul(np.matmul(rotation, matrix), inv_rot)


def set_gaussian_cube(grid_coords, mu_xyz, sigma_xyz, rotation, amplitude=1):
    """Evaluate a rotated 3D Gaussian (multivariate normal PDF) on a grid.

    The Gaussian is defined by its centre, axis-aligned standard deviations,
    and a rotation matrix that tilts the principal axes relative to the grid
    axes.  The covariance matrix is built as ``R @ diag(sigma) @ R^{-1}``.

    Parameters
    ----------
    grid_coords : tuple of 3 array_like
        Meshgrid arrays (x, y, z) produced e.g. by ``numpy.meshgrid``.
        All three arrays must have the same shape.
    mu_xyz : array_like, shape (3,)
        Centre of the Gaussian in (x, y, z) world coordinates.
    sigma_xyz : array_like, shape (3,)
        Standard deviations along each principal axis before rotation.
    rotation : array_like, shape (3, 3)
        Rotation matrix that maps principal axes to grid axes.
    amplitude : float, optional
        Multiplicative amplitude applied to the PDF values (default: 1).

    Returns
    -------
    numpy.ndarray
        Array of the same shape as ``grid_coords[0]`` containing the
        (scaled) Gaussian evaluated at each grid point.
    """
    from scipy.stats import multivariate_normal

    # Build covariance matrix: start from axis-aligned sigma^2 diagonal,
    # then rotate into the grid frame.
    sigma_mat = np.diag(sigma_xyz)
    cov = change_mat_base(sigma_mat, rotation)

    # Flatten the grid into a (N, 3) array of positions for vectorised evaluation.
    pos = np.vstack(grid_coords).reshape(3, -1).T
    result = multivariate_normal.pdf(pos, mean=mu_xyz, cov=cov)

    return amplitude * np.reshape(result, grid_coords[0].shape, order='C')


def set_gaussian_cube_dep(grid_coords, mu_xyz, sigma_xyz, amplitude=1):
    """Evaluate an axis-aligned 3D Gaussian on a grid (deprecated).

    This is a simpler, axis-aligned version of :func:`set_gaussian_cube` that
    does not support rotation.  It is kept for backward compatibility.

    Parameters
    ----------
    grid_coords : tuple of 3 array_like
        Meshgrid arrays (x, y, z).
    mu_xyz : array_like, shape (3,)
        Centre of the Gaussian (xo, yo, zo).
    sigma_xyz : array_like, shape (3,)
        Standard deviations along x, y, z.
    amplitude : float, optional
        Multiplicative amplitude (default: 1).

    Returns
    -------
    numpy.ndarray
        Gaussian values on the grid.

    .. deprecated::
        Use :func:`set_gaussian_cube` with an identity rotation matrix instead.
    """
    xo, yo, zo = mu_xyz
    sx, sy, sz = sigma_xyz
    x, y, z = grid_coords

    # Exponent component for a single axis: -(x - x0)^2 / sigma^2
    comp = lambda x, xo, s: -(x - xo) ** 2 / (s ** 2)
    return amplitude * np.exp(comp(x, xo, sx) + comp(y, yo, sy) + comp(z, zo, sz))


class DiscretePDF:
    """Abstract base class for discrete probability distributions.

    Provides a registry-based factory pattern: subclasses register themselves
    under a string ``name`` and are instantiated via ``DiscretePDF(name, **kwargs)``.

    Subclasses must implement :meth:`_update_pdf`, :meth:`set_probabilities`,
    and :meth:`set_outcomes`.

    Examples
    --------
    >>> pdf = DiscretePDF("binary")
    >>> pdf.set_mean(2.3)
    >>> pdf.get_number(size=5)
    """

    _registry = {}

    def __init_subclass__(cls, name, **kwargs):
        """Automatically register each subclass under its ``name`` key."""
        super().__init_subclass__(**kwargs)
        cls._registry[name] = cls

    def __new__(cls, name: str, **kwargs):
        """Instantiate the appropriate subclass for the given distribution name.

        Parameters
        ----------
        name : str
            Identifier of the distribution (e.g. ``"binary"``, ``"poisson"``).
        **kwargs :
            Additional keyword arguments forwarded to the subclass ``__init__``.
        """
        subclass = cls._registry[name]
        obj = object.__new__(subclass)
        return obj

    def set_mean(self, mean):
        """Set the target mean of the distribution and refresh the PDF.

        Parameters
        ----------
        mean : float
            Desired expectation value.  Must be non-negative.
        """
        self.mean = mean
        self._update_pdf()

    def get_number(self, size=1):
        """Draw random samples from the discrete distribution.

        Parameters
        ----------
        size : int, optional
            Number of samples to draw (default: 1).

        Returns
        -------
        numpy.ndarray
            Array of integer samples drawn according to ``self.probabilities``.
        """
        return np.random.choice(self.outcomes, size=size, p=self.probabilities)


class Binary(DiscretePDF, name="binary"):
    """Binary (two-outcome) discrete distribution preserving a non-integer mean.

    Given a non-integer mean ``mu``, this distribution assigns probability
    ``ceil(mu) - mu`` to ``floor(mu)`` and ``mu - floor(mu)`` to ``ceil(mu)``,
    so that the expected value equals ``mu`` exactly.  This is useful for
    generating synthetic fragment counts whose ensemble average matches a
    prescribed fragmentation rate.

    Parameters
    ----------
    *args, **kwargs : ignored
        Accepted for interface compatibility; initialisation requires a
        subsequent call to :meth:`set_mean`.
    """

    def __init__(self, *args, **kwargs):
        pass

    def _update_pdf(self):
        """Recompute probabilities and outcomes when the mean changes."""
        self.set_probabilities()
        self.set_outcomes()

    def set_probabilities(self):
        """Compute the two probabilities such that E[X] = self.mean.

        Sets ``self.probabilities`` to
        ``[ceil(mean) - mean, mean - floor(mean)]``.
        """
        n_mean = self.mean
        self.probabilities = [
            np.ceil(n_mean) - n_mean,   # weight of the lower integer
            n_mean - np.floor(n_mean)   # weight of the upper integer
        ]

    def set_outcomes(self):
        """Set the two possible integer outcomes: floor and ceil of the mean.

        Sets ``self.outcomes`` to ``[floor(mean), ceil(mean)]``.
        """
        n_mean = self.mean
        self.outcomes = [
            np.floor(n_mean),
            np.ceil(n_mean)
        ]


class Poisson(DiscretePDF, name="poisson"):
    """Poisson-based discrete distribution for stochastic fragment counts.

    Each call to :meth:`get_number` (inherited) returns a single deterministic
    outcome drawn once from a Poisson distribution with the prescribed mean.
    This models the stochastic nature of fragmentation events where the number
    of fragments follows Poisson statistics.

    Parameters
    ----------
    mean : float
        Expected number of fragments (Poisson parameter lambda).
    """

    def __init__(self, *args, **kwargs):
        self.set_mean(kwargs.get("mean"))
        self.set_probabilities()
        self.set_outcomes()

    def set_probabilities(self):
        """Set a trivial probability vector (single certain outcome)."""
        self.probabilities = [1]

    def set_outcomes(self):
        """Draw one sample from Poisson(mean) and store it as the sole outcome."""
        self.outcomes = [np.random.poisson(self.mean)]


class CoordinatesPDF:
    """Abstract base class for 3D spatial position distributions.

    Uses the same registry-based factory pattern as :class:`DiscretePDF`.
    Subclasses implement :meth:`get_xyz` to draw (x, y, z) positions for
    synthetic source placement.

    Examples
    --------
    >>> coord = CoordinatesPDF("uniform")
    >>> xo, yo, zo = coord.get_xyz(xrange=0.1, yrange=0.1, zrange=0.05, size=10)
    """

    _registry = {}

    def __init_subclass__(cls, name, **kwargs):
        """Automatically register each subclass under its ``name`` key."""
        super().__init_subclass__(**kwargs)
        cls._registry[name] = cls

    def __new__(cls, name: str, **kwargs):
        """Instantiate the appropriate subclass for the given distribution name.

        Parameters
        ----------
        name : str
            Identifier of the spatial distribution (e.g. ``"uniform"``, ``"gaussian"``).
        **kwargs :
            Additional keyword arguments forwarded to the subclass ``__init__``.
        """
        subclass = cls._registry[name]
        obj = object.__new__(subclass)
        return obj


class Uniform(CoordinatesPDF, name='uniform'):
    """Uniform spatial distribution within a symmetric box.

    Draws positions independently and uniformly from
    ``[-xrange, xrange] x [-yrange, yrange] x [-zrange, zrange]``.

    Parameters
    ----------
    *args, **kwargs : ignored
    """

    def __init__(self, *args, **kwargs):
        pass

    def get_xyz(self, xrange, yrange=0, zrange=0, size=1):
        """Draw uniformly distributed 3D positions.

        Parameters
        ----------
        xrange : float
            Half-width of the uniform interval along x.
        yrange : float, optional
            Half-width along y (default: 0, i.e. yo = 0).
        zrange : float, optional
            Half-width along z (default: 0, i.e. zo = 0).
        size : int, optional
            Number of positions to draw (default: 1).

        Returns
        -------
        xo, yo, zo : numpy.ndarray
            Arrays of shape ``(size,)`` with drawn coordinates.
        """
        xo = np.random.uniform(low=-xrange, high=xrange, size=size)
        yo = np.random.uniform(low=-yrange, high=yrange, size=size)
        zo = np.random.uniform(low=-zrange, high=zrange, size=size)
        return xo, yo, zo


class Gaussian(CoordinatesPDF, name="gaussian"):
    """Gaussian spatial distribution centred on the origin.

    Draws x positions from a normal distribution with zero mean and standard
    deviation ``xsigma``.

    .. warning::
        The y and z components currently use ``numpy.random.uniform`` instead of
        ``numpy.random.normal``, which is inconsistent with the class name.
        This is a known issue and will be corrected in a future release.

    Parameters
    ----------
    *args, **kwargs : ignored
    """

    def __init__(self, *args, **kwargs):
        pass

    def get_xyz(self, xsigma, ysigma=0, zsigma=0, size=1):
        """Draw 3D positions from (approximately) Gaussian distributions.

        Parameters
        ----------
        xsigma : float
            Standard deviation along x.
        ysigma : float, optional
            Standard deviation along y (default: 0).
        zsigma : float, optional
            Standard deviation along z (default: 0).
        size : int, optional
            Number of positions to draw (default: 1).

        Returns
        -------
        xo, yo, zo : numpy.ndarray
            Arrays of shape ``(size,)`` with drawn coordinates.
        """
        xo = np.random.normal(low=0, high=xsigma, size=size)
        yo = np.random.uniform(low=0, high=ysigma, size=size)
        zo = np.random.uniform(low=0, high=zsigma, size=size)
        return xo, yo, zo
