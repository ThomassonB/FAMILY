import time

import numpy as np
import usefulfunc as uf
from collections.abc import Iterable

import sys
#sys.path.insert(0, '/home/thomab/Documents/Fragmentation_code/')
from fragmentation import solve_fragODE

class ModelsGenerator:
    def __init__(self):
        self.kwargs = {}

    def __str__(self):
        return f" ----------------- Model {type(self).__name__} ----------------- \n" \
               f"\t o Scales {list(self.scales.values())} \n\t   with {len(self.scales) - 1} levels \n" \
               f"\t o Fragmentation rate $\\alpha$ = {self.get_phi(1)} \n" \
               f"\t o Mass loss rate $\ksi$ = {self.get_ksi(1)} \n" \
               f"\t o Mass partition $\omega$ = {self.get_partition(1)} \n" \
               f" ---------------------------------------------------" \
            # pass

    def __iter__(self):
        pass

    def getDicts(self):
        return self.Pmod, self.Mmod, self.scales

    def get_phi(self, R_range):
        return self.alpha(R_range, **self.kwargs)

    def get_ksi(self, R_range):
        return self.ksi(R_range, **self.kwargs)

    def get_partition(self, R_range):
        return self.partition(R_range, **self.kwargs)


class ScaleFree(ModelsGenerator):
    def __init__(self, alpha, ksi, partition, scales_ratio, Ro=1):
        super().__init__()
        self.alpha = lambda x, **kwargs: alpha
        self.ksi = lambda x, **kwargs: ksi
        self.partition = lambda x, **kwargs: partition

        self.Pmod = {}
        self.Mmod = {}
        self.scales = {0: Ro}

        scales = np.append([Ro], Ro / np.cumprod(scales_ratio))

        for level in range(1, len(scales_ratio) + 1):
            self.Pmod[level] = uf.Pfunc_binary
            self.Mmod[level] = uf.Mfunc
            self.scales[level] = scales[level]

class GravoTurbulent(ModelsGenerator):
    def __init__(self, Rtab, phis, ksi, omega, scales_ratio, Ro = 0):
        """

        :param CI: tuple (Ro, rho0) or (Rlist, phi_list)
        :param omega:
        :param scales_ratio:
        :param args:
        :param kwargs:
        """
        super().__init__()

        self.Rtab = Rtab
        self.phis = phis

        self.alpha = lambda R, **kwargs: self._interp_phi(R)
        self.ksi = lambda R, **kwargs: ksi
        self.partition = lambda x, **kwargs: omega

        self.Pmod = {}
        self.Mmod = {}

        if Ro != 0:
            self.scales = {0: Ro}
            scales = np.append([Ro], Ro / np.cumprod(scales_ratio))
        else:
            self.scales = {0: max(Rtab)}
            scales = np.append([max(Rtab)], max(Rtab) / np.cumprod(scales_ratio))

        for level in range(1, len(scales_ratio) + 1):
            self.Pmod[level] = uf.Pfunc_binary
            self.Mmod[level] = uf.Mfunc
            self.scales[level] = scales[level]

    def _interp_phi(self, R):
        phi_used = self.phis[np.logical_and(max(R) > self.Rtab, min(R) < self.Rtab)]
        R_used = self.Rtab[np.logical_and(max(R) > self.Rtab, min(R) < self.Rtab)]
        # print(R, R_used, phi_used)
        return np.interp(R, R_used[::-1], phi_used[::-1]) # need to be increasing


class Poisson(ModelsGenerator):
    def __init__(self, alpha, ksi, partition, scales_ratio, Ro=1):
        self.alpha = alpha
        self.ksi = ksi
        self.partition = partition

        self.Pmod = {}
        self.Mmod = {}
        self.scales = {0: Ro}

        scales = np.append([Ro], Ro / np.cumprod(scales_ratio))

        for level in range(1, len(scales_ratio) + 1):
            # self.Pmod[level] = uf.Pfunc_poisson
            self.Pmod[level] = uf.pseudoPoisson
            self.Mmod[level] = uf.Mfunc
            self.scales[level] = scales[level]


class NetworkModel:
    def __init__(self, ModelGenerated, nlevels=None, **extra):
        """ create an instance model that contains for each level either dictionaries or single function """

        self.model = ModelGenerated

        [setattr(self, f"extra{key}", value) for key, value in extra.items()]

        if isinstance(self.model.Pmod, dict):
            self.nlevels = len(self.model.scales) - 1

        elif nlevels is None:
            print("Specify number of levels")
        else:
            self.nlevels = nlevels

        # total pdf = sum(pdfs)
        # pdfs[level][i] = multiplicity fraction fi for i = 1, n
        # self.pdfs = {0:{1:None}}
        # {R_idx : {level : {multiplicity : pdf at starting R
        # } } }
        # print(self.pdfs)
        self.pdfs = {}

        self.labels = {}
        self.masses = {} #l: None for l in range(0, self.nlevels + 1)

    def save(self):
        """ save the different levels of the network as nlevel arrays in a .fits file with models in header """
        pass

    def genNormandMass_levels(self, *tpl, **kwargs):
        m = 0
        Pfunc, Mfunc = tpl
        for nl, pl in Pfunc(**kwargs):
            m += pl * nl
        for nl, pl in Pfunc(**kwargs):
            for frag_number, mass_partition in Mfunc(nl, **kwargs):
                if nl == 0:
                    yield (0, 0, 0, 1)
                else:
                    yield (nl, pl, pl / m * frag_number, mass_partition)

    def update_params(self, R):  # add here
        N = uf.number_produced(R, self.model.get_phi(R))
        #print(N)
        eff = uf.effective_efficiency(R, self.model.get_ksi(R))
        part = uf.partition(R, self.model.get_partition(R)) #self.model.partition
        return N, eff, part

    def computePDF(self, xo, initial_func, level=1, verbose=False, **func_kwargs):

        if level <= self.nlevels:
            if verbose:
                print(f"level {level} out of {self.nlevels}")
            if callable(initial_func):
                initial_func_copy = initial_func(xo, **func_kwargs)
            else:
                initial_func_copy = initial_func[:]

            if level == 1:
                self.pdfs.update({level-1: (xo, initial_func_copy)})

            y = np.zeros_like(initial_func_copy)
            x_end = xo[:]

            # --------- update multiplicity and total efficiency
            minR = np.log10(self.model.scales[level])
            maxR = np.log10(self.model.scales[level - 1])
            R_transit = np.logspace(minR, maxR, 1_000)
            Nprod, efficiency, partition = self.update_params(R_transit)

            scaling_ratio = self.model.scales[level - 1] / self.model.scales[level]

            if verbose:
                print(f"\n\tNumber produced = {Nprod}"
                      f"\tEfficiency = {efficiency}"
                      f"\tPartition = {partition}\n")

            nmod, mmod = self.model.Pmod[level], self.model.Mmod[level]
            # --------- prepare final x axis, x_end
            for _, _, _, meff in self.genNormandMass_levels(nmod, mmod,
                                                            Nprod=Nprod, efficiency=efficiency, partition=partition,
                                                            r=scaling_ratio):
                xmin = np.amin([xo * meff, x_end])
                xmax = np.amax([xo * meff, x_end])
                x_end = np.logspace(np.log10(xmin), np.log10(xmax), len(xo))

            iteration = 0
            for _, _, nu, meff in self.genNormandMass_levels(nmod, mmod,
                                                            Nprod=Nprod, efficiency=efficiency, partition=partition,
                                                            r=scaling_ratio):
                iteration += 1
                x1 = xo * meff
                idx_fin = np.logical_and(x_end >= x1.min(),
                                         x1.max() >= x_end)  # get the mask of final index in x_end axis

                # project the original f onto the new axis
                f = np.interp(x_end[idx_fin], x1, initial_func_copy)
                y[idx_fin] += f * nu / meff

            if verbose:
                print(f"#iterations {iteration}")
            self.pdfs.update({level: (x_end, y)})
            return self.computePDF(x_end, y, level=level + 1, verbose=verbose)
        return xo, initial_func

    def get_multiplicity(self, multiplicity, level=1, verbose=False):
        if level <= self.nlevels:
            self.labels[level - 1] = np.arange(1, len(multiplicity) + 1, 1)

            # --------- update multiplicity
            minR = np.log10(self.model.scales[level])
            maxR = np.log10(self.model.scales[level - 1])
            R_transit = np.logspace(minR, maxR, 1_000)
            Nprod, *_ = self.update_params(R_transit)

            nmod, _ = self.model.Pmod[level], self.model.Mmod[level]

            path_chosen = np.random.choice([nl for nl, _ in nmod(Nprod)], size=len(multiplicity),
                                           p=[pl for _, pl in nmod(Nprod)])

            for init_scale in range(0, level):
                self.labels[init_scale] = np.repeat(self.labels[init_scale], path_chosen)

            return self.get_multiplicity(self.labels[0], level=level + 1, verbose=verbose)

        if verbose:
            print(f"{level} > {len(self.model.Pmod)}")

    def samplemass(self, M, level=1, verbose=False, time_it=False):
        """ Compute final masses of a finite sample. Same as assess PDF but slower and correct for small samples """
        # self.multiplicity[init_scale - 1] = System
        if time_it:
            to = time.time()

        self.labels[level - 1] = np.arange(1, len(M) + 1, 1)
        if level <= self.nlevels:

            # --------- update multiplicity and total efficiency
            r = self.model.scales[level - 1] / self.model.scales[level]

            minR = np.log10(self.model.scales[level])
            maxR = np.log10(self.model.scales[level - 1])
            R_transit = np.logspace(minR, maxR, 1_000)
            Nprod, efficiency, partition = self.update_params(R_transit)

            nmod, mmod = self.model.Pmod[level], self.model.Mmod[level]

            if verbose:
                print(f"{level} <= {len(self.model.Pmod)},\n\tN={Nprod},\n\teps={efficiency},\n\tomega={partition}")
            # --------------- get random number of fragment produced depending on the probabilities
            path_chosen = np.random.choice([nl for nl, _ in nmod(Nprod, r=r)], size=len(M),
                                           p=[pl for _, pl in nmod(Nprod, r=r)])

            # --------------- generates multiplicative efficiencies and fragments to add
            efficiency_table = np.zeros(shape=(len(M), np.max(path_chosen)))
            n_possible = np.unique(path_chosen)
            for nf in n_possible:
                populations = uf.Mfunc(nf, efficiency, partition)
                multip, tot_efficiency = np.reshape(np.ravel(populations), (2, len(populations)), order='F')
                multip = multip.astype(int)

                efficiency_table[path_chosen == nf, :np.sum(multip)] = np.repeat(tot_efficiency, multip)

            M = np.ravel(M[:, np.newaxis] * efficiency_table)
            M = M[M != 0]
            self.masses[level] = M

            for init_scale in range(0, level):
                self.labels[init_scale] = np.repeat(self.labels[init_scale], path_chosen)

            if time_it:
                print(f'level done in\t{time.time() - to} seconds'
                      f"\n\t= {(time.time() - to) / 3600} hours")

            return self.samplemass(M, level=level + 1, verbose=verbose, time_it=time_it)

        if verbose:
            print(f"{level} > {len(self.model.Pmod)}")
        return M


    def multiplicity_in_massbin(self, Mini, bins_number, bin_arr = False):
        if len(self.masses) == 0:
            self.samplemass(Mini, verbose=False)
        level = self.nlevels

        if bin_arr:
            if np.amax(self.masses[level]) * (1 + 0.1) > max(bins_number):
                bins = np.append(bins_number, np.amax(self.masses[level]) * (1 + 0.1))
            else:
                bins = bins_number
        else:
            bins = np.logspace(np.log10(np.amin(self.masses[level]) * (1 - 0.1)),
                               np.log10(np.amax(self.masses[level]) * (1 + 0.1)), bins_number)

        res_low = self.masses[level] >= bins[:-1, np.newaxis]
        res_high = self.masses[level] < bins[1:, np.newaxis]
        res = res_low & res_high

        _, number_of_fragments = np.unique(self.labels[0], return_counts=True)

        weight_counts = {
            n: [0 for _ in bins[:-1]] for n in np.unique(number_of_fragments)
        }
        dweight_counts = {
            n: [0 for _ in bins[:-1]] for n in np.unique(number_of_fragments)
        }

        for idx, bs in enumerate(res):
            masses_selection = number_of_fragments[self.labels[0][bs] - 1]
            # pour chaque masse dans le bin tu as un nb de fragments dans la structure

            multiplicity, frequency = np.unique(masses_selection, return_counts=True)
            # proba qu'un fragment dans ce bin possède n voisins

            for m, f in zip(multiplicity, frequency):
                weight_counts[m][idx] = f / np.sum(frequency)
                dweight_counts[m][idx] = np.sqrt(f) / np.sum(frequency)

        return weight_counts, bins, dweight_counts

    def primary_mass_vs_multiplicity(self, Mini):
        if len(self.masses) == 0:
            self.samplemass(Mini, verbose=False)
            
        levelmax = self.nlevels

        labels = np.unique(self.labels[0])

        #print(len(self.labels[0]), len(self.masses[levelmax]))

        multiplicity = []
        primary_mass = []
        
        for label in labels:
            idxs = self.labels[0] == label
            #print(idxs)
            primary_mass.append(max(self.masses[levelmax][idxs]))
            multiplicity.append(np.sum(idxs))
        #print(multiplicity)

        return np.array(multiplicity), np.array(primary_mass)

    def primary_mass_vs_multiplicityfraction(self, Mini, bins_number, bin_arr = False):
        multiplicity, primary_mass = self.primary_mass_vs_multiplicity(Mini)

        #print(primary_mass)
        
        levelmax = self.nlevels

        if bin_arr:
            if np.amax(primary_mass) * (1 + 0.1) > max(bins_number):
                bins = np.append(bins_number, np.amax(primary_mass) * (1 + 0.1))
            else:
                bins = bins_number
        else:
            bins = np.logspace(np.log10(np.amin(primary_mass) * (1 - 0.1)),
                               np.log10(np.amax(primary_mass) * (1 + 0.1)), bins_number)

        res_low = primary_mass >= bins[:-1, np.newaxis]
        res_high = primary_mass < bins[1:, np.newaxis]
        res = res_low & res_high

        multiplicity_fraction = []

        for idx, bs in enumerate(res):
            #print(bs)
            multiplicity_fraction.append(sum(multiplicity[bs] > 1)/len(multiplicity[bs]))

        return multiplicity_fraction, bins


    def mass_ratio(self, Mini):
        from scipy.sparse import csr_matrix
        if len(self.masses) == 0:
            self.samplemass(Mini, verbose=False)
        level = self.nlevels

        mat = csr_matrix(
            (self.masses[level], (self.labels[0] - 1, np.arange(0, len(self.labels[0]), 1)))
                         )

        maxmasses = mat.max(axis=1).toarray().flatten()
        inv_minmasses = mat.power(-1).max(axis=1).toarray().flatten()

        return maxmasses * inv_minmasses, maxmasses

    """
        def _multiplicity(self, R0=0):
            return np.unique(self.multiplicity[R0], return_counts=True)

        def multiplicity_fraction(self, R0=0):
            _, multi = self._multiplicity(R0=R0)
            numbers, count = np.unique(multi, return_counts=True)
            return 1 - count[numbers == 1][0] / np.sum(count)

        def _fraction(self, n, numbers, counts):  # fraction of systems of size R0 containing n stars
            if n in numbers:
                return counts[numbers == n][0] / np.sum(counts)  # / n
            return 0
            # fraction of stars within n system / n

        def companion_frequency(self, R0=0):
            _, multi = self._multiplicity(R0=R0)
            numbers, count = np.unique(multi, return_counts=True)
            return sum((n - 1) * self._fraction(n, numbers, count) for n in numbers)
        """

def pad_array(x, arr, xmin, xmax):
    dx = np.log10(x[1]) - np.log10(x[0])
    origin_length = len(x)

    nleft = int(np.log10(x.min()) - np.log10(xmin))
    nright = int(np.log10(xmax) - np.log10(x.max()))
    if nleft < 0:
        nleft = 0
    if nright < 0:
        nright = 0
    # print(x)
    # print(nleft, nright)
    x = np.pad(x, (nleft, nright), 'constant', constant_values=(0, 0))
    end = nleft + origin_length
    # print(end, dx)
    # print(x)
    # x[:nleft+1] = np.logspace(np.log10(xmin), np.log10(x[nleft]), nleft+1)
    # print(10**np.arange(np.log10(xmin), np.log10(x[nleft]), step=10**dx))
    x[:nleft] = 10 ** np.arange(np.log10(xmin), np.log10(x[nleft]), step=10 ** dx)
    # x[:nleft] = 10 ** np.arange(np.log10(x[nleft]) - dx * nleft, np.log10(x[nleft]), step=10 ** dx)
    # print(x)
    # print("ok", 10 ** np.arange(np.log10(x[end-1]), np.log10(xmax), step=10**dx))
    x[end:] = 10 ** np.arange(np.log10(x[end - 1]), np.log10(xmax), step=10 ** dx)
    # x[end:] = 10 ** np.arange(np.log10(x[end - 1]), np.log10(x[end - 1]), step=10 ** dx)
    # print(x)
    y = np.pad(arr, (nleft, nright), 'constant', constant_values=(0, 0))
    return x, y


def medianBins(x, y, nevents=10000, nbins=15, boundaries=None):
    # print("before", y)

    if boundaries is None:
        xmin = x.min()
        xmax = x.max()
    else:
        xmin, xmax = boundaries
        x, y = pad_array(x, y, xmin, xmax)

    bins = np.logspace(np.log10(xmin), np.log10(xmax), nbins)

    idx = np.searchsorted(x, bins)

    ybin = np.array([np.median(y[idx[i]:idx[i + 1]]) for i in range(nbins - 1)])

    # ------------ normalise the dy coordinate
    binwidth = (bins[1:] - bins[:-1])
    dybin = 1 / np.sqrt(ybin * nevents * binwidth) / nevents / binwidth

    # ------------ compute the x coordinate as the middle
    xbin = (bins[1:] + bins[:-1]) / 2

    # ------------ dxbin as the width of the bin
    dxbin = (bins[1:] - bins[:-1]) / 2
    # print("after", ybin)

    return xbin, ybin, dybin, dxbin

"""
def computePDF2(self, xo, initial_func, level=1, verbose=False, **func_kwargs):
    if level <= self.nlevels:
        if verbose:
            print(f"level {level} out of {self.nlevels}")

        if callable(initial_func):
            initial_func_copy = initial_func(xo, **func_kwargs)
        else:
            initial_func_copy = initial_func[:]
        y = np.zeros_like(initial_func_copy)
        x_end = xo[:]

        # --------- update multiplicity and total efficiency
        minR = np.log10(self.model.scales[level])
        maxR = np.log10(self.model.scales[level - 1])
        R_transit = np.logspace(minR, maxR, 1_000)
        Nprod, efficiency, partition = self.update_params(R_transit)

        if verbose:
            print(f"\n\tNumber produced = {Nprod}"
                  f"\tEfficiency = {efficiency}"
                  f"\tPartition = {partition}\n")

        nmod, mmod = self.model.Pmod[level], self.model.Mmod[level]
        # --------- prepare final x axis, x_end
        for _, _, _, meff in self.genNormandMass_levels2(nmod, mmod,
                                                        Nprod=Nprod, efficiency=efficiency, partition=partition):
            xmin = np.amin([xo * meff, x_end])
            xmax = np.amax([xo * meff, x_end])
            # print(xo, "x", meff, "=", xo * meff, "\n", x_end)
            x_end = np.logspace(np.log10(xmin), np.log10(xmax), len(xo))

        iteration = 0
        # print(x_end)
        for _, pl, w, meff in self.genNormandMass_levels2(nmod, mmod,
                                                         Nprod=Nprod, efficiency=efficiency, partition=partition):
            #print(w, meff)
            iteration += 1
            x1 = xo * meff
            idx_fin = np.logical_and(x_end >= x1.min(),
                                     x1.max() >= x_end)  # get the mask of final index in x_end axis

            # project the original f onto the new axis
            f = np.interp(x_end[idx_fin], x1, initial_func_copy)
            f = f / integrate.trapezoid(f, x_end[idx_fin])  # normalise the function

            y[idx_fin] += f * pl / efficiency

            # print(f)

        norm = integrate.trapezoid(y, x_end)
        #print('norm', norm)
        # self.x_pdfs[level] = x_end
        # self.y_pdfs[level] = y / norm
        # norm = 1/efficiency
        # print("norm = eff ?", integrate.trapezoid(y / efficiency, x_end), efficiency)
        # print(integrate.trapezoid(y, x_end))
        y = y / norm

        if verbose:
            print(f"#iterations {iteration}")
        return self.computePDF(x_end, y, level=level + 1, verbose=verbose)
    return xo, initial_func
"""
