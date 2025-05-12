import itertools

import numpy as np
import scipy.special
from scipy import integrate
from tqdm import tqdm


def indiv_chi2(y, ymod):
    return (y - ymod) ** 2 / ymod ** 2


def chi2_func(y, ymod):
    return np.sum(indiv_chi2(y, ymod), axis=0)


def Gammafunction(x):
    return scipy.special.gamma(x)


def Gammafunction_incomplete(a, x):
    return scipy.special.gammainc(a, x)


def CDF_chi2(x, k):
    return Gammafunction_incomplete(k, x) / Gammafunction(x)


def SDF_chi2(x, k):
    return 1 - CDF_chi2(x, k)


def pvalue(y, ymod, ddof=1):
    """return log of p-value of chi2 test"""
    from scipy import stats
    chi2 = chi2_func(y, ymod)
    p = stats.chi2.logsf(x=chi2, df=len(y) - ddof)  # proba d'avoir un chi2 plus grand
    # print(f"scipy khi = {round(chi2, 2)} with {len(y) - ddof} degree of freedom")
    # print(f"scipy = {round(p, 2)}")
    return p


def pvalue_interval(f, x, y, xmin, xmax, ddof=1, **func_kwargs):
    idxs = np.logical_and(x >= xmin, x <= xmax)
    y_exp = f(x[idxs], **func_kwargs)
    return pvalue(y[idxs], y_exp, ddof)


def compare_pvalue(x, f, g, f_args, g_args):
    y = f(x, **f_args)
    yexp = g(x, **g_args)
    return chi2_func(y, yexp), pvalue(y, yexp, ddof=len(f_args))


def compare_powerlaw(x, gammas, **kwargs):
    K = []
    P = []
    for gamma in gammas:
        kwargs["f_args"]["alpha"] = gamma
        k, p = compare_pvalue(x, **kwargs)
        K.append(k)
        P.append(p)
    return K, P


def cumulative(x, y, **kwargs):
    for idx in range(len(x)):
        yield integrate.trapezoid(y[:idx], x[:idx])


def cumulative2(x, y):
    return scipy.integrate.cumulative_trapezoid(y, x, initial=0) / scipy.integrate.trapezoid(y, x)


def ks_1sample(xx, x, y, **kwargs):
    def _cdf(x, y):
        return y

    xx = np.sort(xx)
    y = cumulative2(x, y)
    yinterp = np.interp(xx, x, y)
    return scipy.stats.ks_1samp(xx, cdf=_cdf, args=([yinterp]), **kwargs)


def ks_distance(y1, y2, index=False):
    if index:
        y = np.abs(y1 - y2)
        return y.max(), np.argmax(y)
    return np.max(np.abs(y1 - y2))


def compare_cumulative(x, f, g, f_args, g_args):
    cumul_f = np.array(list(cumulative(x, f(x, **f_args))))
    cumul_g = np.array(list(cumulative(x, g(x, **g_args))))
    return ks_distance(cumul_f, cumul_g)


def sample(x, y_func, y_kwargs, n=1000):
    y = y_func(x, **y_kwargs)
    ymax = max(y)
    sampling = []
    while len(sampling) < n:
        x_al = np.random.uniform(x.min(), x.max(), n)
        y_al = np.random.uniform(0, ymax, n)
        sampling = np.append(sampling, x_al[y_func(x_al, **y_kwargs) >= y_al])
    return sampling[:n]


##################

def _ks_test(x, cdf, **kwargs):
    print(cdf)
    ks = scipy.stats.ks_1samp(x, cdf=cdf[0], **kwargs)
    return ks.pvalue


def _ad_test(s1, s2, **kwargs):
    ad = scipy.stats.anderson_ksamp([s1, s2], **kwargs)
    return ad.significance_level


def mww_test(sample1, sample2, **kwargs):
    mww = scipy.stats.mannwhitneyu(sample1, sample2, **kwargs)
    return mww.pvalue


def ks_test(x, cdf, **kwargs):
    f = np.vectorize(_ks_test, signature='(n,m),(k)->(n)')
    return f(x, cdf, **kwargs)


def ad_test(s1, s2, **kwargs):
    f = np.vectorize(_ad_test, signature='(n),(n)->()')
    return f(s1, s2, **kwargs)


##################

def sampler(x, y_func, n_run, n_sample, verbose=False, **y_kwargs):
    size = (n_run, n_sample)
    prod_size = np.prod(size)
    sampling = np.zeros(shape=prod_size)
    # prod_size = min(np.prod(size) // 100, np.prod(size))

    if callable(y_func):
        y = y_func(x, **y_kwargs)
        ymax = max(y)

        while np.any(sampling == 0):
            x_al = np.random.uniform(x.min(), x.max(), prod_size)
            y_al = np.random.uniform(0, ymax, prod_size)

            xt = x_al[y_func(x_al, **y_kwargs) >= y_al]

            nonzeros = sampling[sampling != 0].size
            to_fill = min(xt.size, sampling[sampling == 0].size)

            sampling[nonzeros:nonzeros + to_fill] = xt[:to_fill]

            # print(to_fill, prod_size)

    else:
        y_func = y_func / integrate.trapezoid(y_func, x)
        ymax = max(y_func)

        while 0 in sampling:
            x_al = np.random.uniform(x.min(), x.max(), prod_size)

            y_interp = np.interp(x_al, x, y_func)

            y_al = np.random.uniform(0, ymax, prod_size)

            # idx = np.searchsorted(x, x_al, side="left") - 1
            # print(max(idx), len(x))
            # lininterp =

            # xt = x_al[ y_al <= lininterp(x_al, x[idx], x[idx + 1], y_func[idx], y_func[idx + 1]) ]
            xt = x_al[y_interp >= y_al]

            nonzeros = sampling[sampling != 0].size
            to_fill = min(xt.size, sampling[sampling == 0].size)

            sampling[nonzeros:nonzeros + to_fill] = xt[:to_fill]

    if n_run == 1:
        return sampling
    return np.reshape(sampling, size)


def lininterp(x, x1, x2, y1, y2):
    return (y1 - y2) / (x1 - x2) * (x - x1) + y1


def log_sampler(x, y_func, n_sample, **y_kwargs):
    size = n_sample
    sampling = np.zeros(shape=size)
    # prod_size = min(np.prod(size) // 100, np.prod(size))

    y_func = y_func / integrate.trapezoid(y_func, x)

    log_x = np.log(x)
    y_logx = y_func * x
    ymax = max(y_logx)

    while 0 in sampling:
        x_al = np.random.uniform(log_x.min(), log_x.max(), size)

        y_interp = np.interp(x_al, log_x, y_logx)

        y_al = np.random.uniform(0, ymax, size)

        xt = x_al[y_interp >= y_al]

        nonzeros = sampling[sampling != 0].size
        to_fill = min(xt.size, sampling[sampling == 0].size)

        sampling[nonzeros:nonzeros + to_fill] = xt[:to_fill]

    return np.exp(sampling)


def exploreSpace_old(function, *args):
    paramlist = itertools.product(*args)
    test = list(map(function, paramlist))
    return np.reshape(test, tuple([len(arg) for arg in args]))


def exploreSpace(function, *args):
    length, *PKP = args
    paramlist = itertools.product(*PKP)
    # test = list(map(function, paramlist))
    arr = [test for t in tqdm(map(function, paramlist)) for test in t]
    return np.reshape(arr, tuple([len(arg) for arg in PKP] + [length]))


def compute_AD(x, y, y0, n_run=1000, n_sample=1000):
    Xsample = sampler(x, y, n_run=n_run, n_sample=n_sample)
    IMFsample = sampler(x, y0, n_run=n_run, n_sample=n_sample)

    xsample = np.random.choice(Xsample, size=(n_run, n_sample), replace=False)
    imfsample = np.random.choice(IMFsample, size=(n_run, n_sample), replace=False)

    return np.median(ad_test(xsample, imfsample))


def ADalt(x, cdf):
    x = np.sort(x)
    N = len(x)

    i = np.arange(1, N + 1)
    return - N - np.sum((2 * i - 1) * (np.log(cdf(x)) + np.log(1 - cdf(x[::-1])))) / N

def ADtest(x, cdf):
    sign_levels = np.array([0.250, 0.150, 0.100, 0.050, 0.025, 0.010, 0.005, 0.001])
    sign_stat = np.array([1.246636, 1.623813, 1.937456, 2.500057, 3.092781, 3.90363, 4.53244, 5.957617])

    score = ADalt(x, cdf)
    return np.interp(score, sign_stat, sign_levels)

def cumulative_from_sample(sample):
    k = sample.size
    return np.sort(sample), np.arange(k) / k
