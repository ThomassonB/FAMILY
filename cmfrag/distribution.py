import numpy as np

from scipy import integrate
from lmfit import Model, Parameters
from scipy import special
from scipy.stats import norm as scistat_norm
from scipy.stats import lognorm as scistat_lognorm

class InitialMassFunction:
    _registry = {}

    def __init_subclass__(cls, name, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[name] = cls

    def __new__(cls, name: str, **kwargs):
        subclass = cls._registry[name]
        obj = object.__new__(subclass)
        return obj

    def _fit(self, xdata, ydata, guess, dydata=None):
        fitmodel = Model(self.pdf)

        result = fitmodel.fit(ydata, x=xdata, weights=dydata, **guess)
        self.result = result

        values = {key: result.params[key].value for key in result.params}
        return self.function(xdata, **values)

    def log_pdf(self, x):
        return x * self.pdf(x)

    def survival(self, x, **kwargs):
        return 1 - self.cumulative(x, **kwargs)

    def sample(self, x, n=1000):
        y = self.pdf(x)
        ymax = max(y)
        Mfin = []
        while len(Mfin) < n:
            x_al = np.random.uniform(x.min(), x.max(), n)
            y_al = np.random.uniform(0, ymax, n)
            Mfin = np.append(Mfin, x_al[self.pdf(x_al) >= y_al])
        return Mfin[:n]


class CanonicalIMF:
    def __init__(self):
        pass

    def fit(self, xdata, ydata, guess, dydata=None):
        fitmodel = Model(self.function)

        result = fitmodel.fit(ydata, x=xdata, weights=dydata, **guess)
        self.result = result

        values = {key: result.params[key].value for key in result.params}
        return self.function(xdata, **values)

    def true(self, x, log=False, density=True, **kwargs):
        return self.function(x, log=log, density=density, **self.function_true)

    def survival(self, x, **kwargs):
        return 1 - self.cumulative(x, **kwargs)

    def sample(self, x, n=1000, **kwargs):
        y = self.function(x, **kwargs)
        ymax = max(y)
        Mfin = []
        while len(Mfin) < n:
            x_al = np.random.uniform(x.min(), x.max(), n)
            y_al = np.random.uniform(0, ymax, n)
            Mfin = np.append(Mfin, x_al[self.function(x_al, **kwargs) >= y_al])
        return Mfin[:n]

    def expected(self, bins, y, sample=1000):
        #print(sample)
        #print(y)
        #print(bins)
        return sample * y * np.diff(bins)

class Powerlaw(InitialMassFunction, name="power-law"):
    def __init__(self, gamma = 2.35, mlow=0.1, mup=100, **kwargs):
        self._gamma = gamma

        self.mlow = mlow
        self.mup = mup

        self.domain = Parameters()
        self.domain.add("gamma", value=2.35, min=1)

    def set_gamma_index(self, gamma):
        self._gamma = gamma

    def pdf(self, x):
        one_gamma = 1 - self._gamma

        y = x ** -self._gamma
        if self._gamma == 1:
            norm = np.log(np.max(x)/np.min(x)) #return np.ones_like(x)
        else:
            norm = (np.max(x) ** one_gamma - np.min(x) ** one_gamma) / one_gamma
        return y / norm

    def cumulative(self, x):
        one_gamma = 1 - self._gamma

        y = (x ** one_gamma - min(x) ** one_gamma)
        norm = (max(x) ** one_gamma - min(x) ** one_gamma)

        return y / norm


class Kroupa(InitialMassFunction, name="broken-kroupa"):
    def __init__(self, low=0.3, middle=1.3, high=2.3, low_cutoff=0.08, high_cutoff=0.5, **kwargs):
        self._indexs = (low, middle, high)
        self._cutoffs = (low_cutoff, high_cutoff)

        self.domain = Parameters()
        self.domain.add("low_index", value=0.3, max=0)
        self.domain.add("middle_index", value=1.3, max=0)
        self.domain.add("high_index", value=2.3, max=0)
        self.domain.add("low_cutoff", value=0.08, max=0)
        self.domain.add("high_cutoff", value=0.5, max=0)

    def set_low_index(self, index):
        self._indexs[0] = index

    def set_middle_index(self, index):
        self._indexs[1] = index

    def set_high_index(self, index):
        self._indexs[2] = index

    def set_low_cutoff(self, mass):
        self._cutoffs[0] = mass

    def set_high_cutoff(self, mass):
        self._cutoffs[1] = mass

    def pdf(self, x):
        y = np.zeros_like(x)
        if 1 in self._indexs:
            return y

        low     = -self._indexs[0]
        middle  = -self._indexs[1]
        high    = -self._indexs[2]

        cutoff1, cutoff2 = self._cutoffs

        k1 = cutoff1 ** (low - middle)
        k2 = cutoff2 ** (middle - high)

        first = cutoff1 >= x
        second = np.logical_and(cutoff1 < x, cutoff2 >= x)
        third = cutoff2 < x

        y[first] = x[first] ** low
        y[second] = x[second] ** middle * k1
        y[third] = x[third] ** high * k1 * k2

        norm = (cutoff1 ** (low + 1) - min(x) ** (low + 1)) / (low + 1) + (
                cutoff2 ** (middle + 1) - cutoff1 ** (middle + 1)) / (middle + 1) * k1 + (
                       max(x) ** (high + 1) - cutoff2 ** (high + 1)) / (high + 1) * k1 * k2
        return y / norm


    def cumulative(self, x):
        y = np.zeros_like(x)
        if 1 in self._indexs:
            return y

        one_low = 1 - self._indexs[0]
        one_middle = 1 - self._indexs[1]
        one_high = 1 - self._indexs[2]

        cutoff1, cutoff2 = self._cutoffs

        first = cutoff1 >= x
        second = np.logical_and(cutoff1 <= x, cutoff2 >= x)
        third = cutoff2 <= x

        y[first] = (x[first] ** one_low - min(x[first]) ** one_low) / one_low
        y[second] = y[first][-1] + (x[second] ** one_middle - min(x[second]) ** one_middle) / one_middle
        y[third] = y[second][-1] + (x[third] ** one_high - min(x[third]) ** one_high) / one_high

        norm = lambda x, one_gamma: (max(x) ** one_gamma - min(x) ** one_gamma) / one_gamma
        N = norm(x[first], one_low) + norm(x[second], one_middle) + norm(x[third], one_high)

        return y / N


class Chabrier(InitialMassFunction, name="chabrier"):
    def __init__(self, mu=0.08, sigma=0.69, cutoff=1, index=2.35, **kwargs):

        self._power_index = index
        self._mu = mu
        self._sigma = sigma
        self._cutoff = cutoff

        self.domain = Parameters()
        self.domain.add("mu", value=0.2, min=0)
        self.domain.add("sigma", value=0.55, min=0)
        self.domain.add("cutoff", value=1, min=0)
        self.domain.add("power_index", value=2.35, max=0)

    def set_power_index(self, index):
        self._power_index = index

    def set_mu(self, mu):
        self._mu = mu

    def set_sigma(self, sigma):
        self._sigma = sigma

    def set_cutoff(self, mass):
        self._cutoff = mass

    def _function(self, x, slope, cutoff, mu, sigma):
        y = np.zeros_like(x)
        if slope == 1:
            return y

        slope = -slope

        def lognormal(x, mu, sigma):
            return np.exp(- (np.log10(x) - np.log10(mu)) ** 2 / (2 * sigma ** 2)) / x / sigma

        k = lognormal(cutoff, mu, sigma) * cutoff ** -slope

        y[x < cutoff] = lognormal(x[x < cutoff], mu, sigma)
        y[x >= cutoff] = k * x[x >= cutoff] ** slope

        norm = (max(x) ** (slope + 1) - cutoff ** (slope + 1)) / (slope + 1) * k
        if len(x[x < cutoff]) != 0:
            norm += integrate.simpson(
                lognormal(x[x < cutoff], mu, sigma), x[x < cutoff])

        return y / norm

    def pdf(self, x):
        dict = {
            "slope": self._power_index,
            "mu": self._mu,
            "sigma": self._sigma,
            "cutoff": self._cutoff
        }

        return self._function(x, **dict)

    def _cumulative(self, x, mu, sigma, cutoff, slope):
        y = np.zeros_like(x)
        if slope == 1:
            return y

        slope = -slope

        def cdf_lognormal(x, mu, sigma):
            return (1 + special.erf((np.log(x) - mu) / (np.sqrt(2) * sigma))) / 2

        N = cdf_lognormal(cutoff, mu, sigma) + (
                max(x[x >= cutoff]) ** (slope + 1) - min(x[x >= cutoff]) ** (slope + 1)) / (slope + 1)

        y[x <= cutoff] = cdf_lognormal(x[x <= cutoff], mu, sigma)
        y[x >= cutoff] = y[x <= cutoff][-1] + (x[x >= cutoff] ** (slope + 1) - min(x[x >= cutoff]) ** (slope + 1)) / (
                slope + 1)

        return y / N

    def cumulative(self, x):
        dict = {
            "slope": self._power_index,
            "mu": self._mu,
            "sigma": self._sigma,
            "cutoff": self._cutoff
        }

        return self._cumulative(x, **dict)



class Logistic3(InitialMassFunction, name="L3"):
    def __init__(self, alpha=2.3, beta=1.4, mu=0.2, mlow=0.01, mup=150, **kwargs):

        self._alpha = alpha
        self._beta = beta
        self._mu = mu

        self._mlow = mlow
        self._mup = mup

        self.domain = Parameters()
        self.domain.add("alpha", value=2.3, min=0)
        self.domain.add("beta", value=1.4, min=0)
        self.domain.add("mu", value=0.2, min=0)

    def get_boundaries(self):
        return self._mlow, self._mup

    def set_alpha(self, alpha):
        self._alpha = alpha

    def set_beta(self, beta):
        self._beta = beta

    def set_mu(self, mu):
        self._mu = mu

    def function(self, x, **kwargs):
        return self.pdf(x)

    def get_peak_mp(self):
        return self._mu * (self._beta - 1) ** (1 / (self._alpha - 1))

    def get_low_slope(self):
        return self._alpha + self._beta * (1 - self._alpha)

    def set_beta_from_slope(self, low_slope):
        self._beta = (low_slope - self._alpha) / (1 - self._alpha)

    def set_mu_from_peak(self, mp):
        self._mu = mp / (self._beta - 1)**(1 / (self._alpha - 1))

    def auxiliary(self, x):
        return (1 + (x / self._mu) ** (1 - self._alpha)) ** (1 - self._beta)

    def pdf(self, x):
        if 1 in (self._alpha, self._beta):
            return np.ones_like(x)

        Gmax = self.auxiliary(self._mup)
        Gmin = self.auxiliary(self._mlow)

        A = (1 - self._alpha) * (1 - self._beta) / self._mu / (Gmax - Gmin)
        return A * (x / self._mu) ** -self._alpha * (1 + (x / self._mu) ** (1 - self._alpha)) ** (- self._beta)

    def cumulative(self, x):
        if 1 in (self._alpha, self._beta):
            return np.ones_like(x)

        Gx = self.auxiliary(x)

        Gup = self.auxiliary(self._mup)
        Glow = self.auxiliary(self._mlow)

        return (Gx - Glow) / (Gup - Glow)

    def interquartile(self, x, u):
        expb = 1 / (1 - self._beta)
        expa = 1 / (1 - self._alpha)
        auxup = self.auxiliary(max(x))
        auxlow = self.auxiliary(min(x))

        return self._mu * ( (u * (auxup - auxlow) + auxlow) ** expb ) ** expa

class LogNormal(InitialMassFunction, name="Lognormal"):
    def __init__(self, mu=-2.5, sigma=1.6, mup = 150, mlow = 0.01, **kwargs):
        self.sigma = sigma
        self.mu = mu

        self._mup = mup
        self._mlow = mlow

        self.mean, self.var, _, _ = scistat_lognorm.stats(sigma, scale=np.exp(mu), moments='mvsk')
        self.median = scistat_lognorm.median(sigma, scale=np.exp(mu))

    def _pdf(self, x, sigma, mu, A):
        return A * scistat_lognorm.pdf(x, s=sigma, scale=np.exp(mu))

    def pdf(self, x):
        return self._pdf(x, self.sigma, self.mu, 1)
        # return np.exp(- (np.log(x) - self.mu) ** 2 / (2 * self.sigma ** 2)) / x / self.sigma / np.sqrt(2  * np.pi)

    def renorm_pdf(self, x):
        x_norm = np.logspace(np.log10(self._mlow), np.log10(self._mup), 1_000)
        y = self.pdf(x_norm)
        return self.pdf(x) / integrate.trapezoid(y, x_norm)

    def _logpdf(self, x, sigma, mu, A):
        """
        :param x: in log scale
        :return: dN/dlogx (logx)
        """
        return A * scistat_norm.pdf(x, loc=mu, scale=sigma)

    def logpdf(self, x):
        """
        :param x: in log scale
        :return: dN/dlogx (logx)
        """
        return self._logpdf(x, self.sigma, self.mu, 1)
        #creturn np.exp(- (x - np.log(self.mu)) ** 2 / (2 * self.sigma ** 2)) / self.sigma / np.sqrt(2  * np.pi)

    def _cumulative(self, x):
        return scistat_lognorm.cdf(x, self.sigma, scale=np.exp(self.mu))
        #from scipy import special
        #return (1 + special.erf((np.log(x) - np.log(self.mu)) / (np.sqrt(2) * self.sigma))) / 2

    def cumulative(self, x):
        gx = self._cumulative(x)
        gup = self._cumulative(self._mup)
        glow = self._cumulative(self._mlow)
        return (gx - glow)/(gup - glow)

    def fit(self, data, guess, log=True):
        if log:
            fitmodel = Model(self._logpdf)
            bins = np.linspace(np.min(data), np.max(data), 25)
        else:
            fitmodel = Model(self._pdf)
            bins = np.logspace(np.min(data), np.max(data), 25)

        x = (bins[:-1] + bins[1:]) / 2
        heights, _ = np.histogram(data, density=True, bins=bins)
        print(x, heights)

        result = fitmodel.fit(heights, x=x, **guess)

        return {key: result.params[key].value for key in result.params}, result


if __name__=='__main__':
    import matplotlib.pyplot as plt

    powerlaw = InitialMassFunction(name="power-law")

    kroupa = InitialMassFunction(name="broken-kroupa")

    chabrier = InitialMassFunction(name="chabrier")

    L3IMF = InitialMassFunction(name="L3")

    x = np.logspace(-2, 2, 64)

    y = powerlaw.pdf(x)
    plt.plot(x, y, "-")

    y = kroupa.pdf(x)
    plt.plot(x, y, "-")

    y = chabrier.pdf(x)
    plt.plot(x, y, "-")

    y = L3IMF.pdf(x)
    plt.plot(x, y, "-k")

    plt.xscale('log')
    plt.yscale('log')

    plt.show()