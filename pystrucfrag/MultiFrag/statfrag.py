import numpy as np
import scipy.integrate

def floor(x):
    return int(x)

def ceil(x):
    return int(x) + 1

def change_mat_base(matrix, rotation):
    from numpy.linalg import inv
    inv_rot = inv(rotation)

    return np.matmul(np.matmul(rotation, matrix), inv_rot)

def set_gaussian_cube(grid_coords, mu_xyz, sigma_xyz, rotation, amplitude = 1):
    from scipy.stats import multivariate_normal

    sigma_mat = np.diag(sigma_xyz)
    cov = change_mat_base(sigma_mat, rotation)

    pos = np.vstack(grid_coords).reshape(3,-1).T
    result = multivariate_normal.pdf(pos, mean=mu_xyz, cov=cov)
    return amplitude * np.reshape(result, grid_coords[0].shape, order='C')

def set_gaussian_cube_dep(grid_coords, mu_xyz, sigma_xyz, amplitude = 1):
    xo, yo, zo = mu_xyz
    sx, sy, sz = sigma_xyz
    x, y, z = grid_coords

    comp = lambda x, xo, s: -(x - xo) ** 2 / (s ** 2)
    return amplitude * np.exp(comp(x, xo, sx) + comp(y, yo, sy) + comp(z, zo, sz))

class DiscretePDF:
    _registry = {}

    def __init_subclass__(cls, name, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[name] = cls

    def __new__(cls, name: str, **kwargs):
        subclass = cls._registry[name]
        obj = object.__new__(subclass)
        return obj

    def set_mean(self, mean):
        self.mean = mean
        self._update_pdf()

    def get_number(self, size = 1):
        return np.random.choice(self.outcomes, size=size, p=self.probabilities)

class Binary(DiscretePDF, name="binary"):

    def __init__(self, *args, **kwargs):
        pass

    def _update_pdf(self):
        self.set_probabilities()
        self.set_outcomes()

    def set_probabilities(self):
        n_mean = self.mean
        self.probabilities = [
            np.ceil(n_mean) - n_mean,
            n_mean - np.floor(n_mean)
        ]

    def set_outcomes(self):
        n_mean = self.mean
        self.outcomes = [
            np.floor(n_mean),
            np.ceil(n_mean)
        ]

class Poisson(DiscretePDF, name="poisson"):

    def __init__(self, *args, **kwargs):
        self.set_mean(kwargs.get("mean"))
        self.set_probabilities()
        self.set_outcomes()

    def set_probabilities(self):
        self.probabilities = [1]

    def set_outcomes(self):
        self.outcomes = [np.random.poisson(self.mean)]

class CoordinatesPDF:
    _registry = {}

    def __init_subclass__(cls, name, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[name] = cls

    def __new__(cls, name: str, **kwargs):
        subclass = cls._registry[name]
        obj = object.__new__(subclass)
        return obj

class Uniform(CoordinatesPDF, name='uniform'):

    def __init__(self, *args, **kwargs):
        pass

    def get_xyz(self, xrange, yrange=0, zrange=0, size=1):
        xo = np.random.uniform(low=-xrange, high=xrange, size=size)
        yo = np.random.uniform(low=-yrange, high=yrange, size=size)
        zo = np.random.uniform(low=-zrange, high=zrange, size=size)
        return xo, yo, zo

class Gaussian(CoordinatesPDF, name="gaussian"):

    def __init__(self, *args, **kwargs):
        pass

    def get_xyz(self, xsigma, ysigma=0, zsigma=0, size=1):
        xo = np.random.normal(low=0, high=xsigma, size=size)
        yo = np.random.uniform(low=0, high=ysigma, size=size)
        zo = np.random.uniform(low=0, high=zsigma, size=size)
        return xo, yo, zo
