import numpy as np
import itertools
from tqdm import tqdm

import distribution
import fragmentCMF
import stat_tests as sts
import time

from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astropy.io import fits

phi_xi = {0:3.5, 1:3.5, 2:3.5, 3:2.35, 4:1.74, 5:1.40, 6:1.15, 7:0.98, 8:0.83}

class AndersonDarling_Test:
    def __init__(self):
        self.sign_levels = np.array([0.250, 0.150, 0.100, 0.050, 0.025, 0.010, 0.005, 0.001])
        self.sign_stat = np.array([1.246636, 1.623813, 1.937456, 2.500057, 3.092781, 3.90363, 4.53244, 5.957617])

    def get_tabulated_AD(self):
        return self.sign_levels, self.sign_stat

    def get_pvalue(self, score):
        return np.interp(score, self.sign_stat, self.sign_levels)

    def testAD(self, x, cdf):
        x = np.sort(x)
        N = len(x)

        i = np.arange(1, N + 1)
        return - N - np.sum((2 * i - 1) * (np.log(cdf(x)) + np.log(1 - cdf(x[::-1])))) / N

class Mapping:
    def __init__(self, x0, initial_pdf, final_level, fragmentation_rate, mass_transfer_rate, mass_partition,
                 target_distribution, number_distribution="Binary", scaling_ratio=1.5, test="AD",
                 **kwargs):

        self.x0 = x0
        self.initial_pdf = initial_pdf
        self.target_distribution = target_distribution

        self.scaling_ratio = scaling_ratio
        self.final_level = final_level

        self.number_distribution = number_distribution

        self.fragmentation_rate = fragmentation_rate
        self.mass_transfer_rate = mass_transfer_rate
        #print("XI of Mapping", self.mass_transfer_rate)
        #self.phi_xi = [
        #    np.linspace(px - 0.2, px + 0.2, 100) for _, px in phi_xi.items()
        #]

        #phi_xi #list
        self.mass_partition = mass_partition

        self.test = AndersonDarling_Test()

    def _compute_AD(self, cmfs, n_sample = 1_000):
        pdf_xmin, pdf_xmax = self.target_distribution.get_boundaries()

        for cmf in cmfs:
            #print(cmf.model)
            for level, (x, y) in cmf.pdfs.items():
                idxs = np.logical_and(x >= pdf_xmin,
                                      x <= pdf_xmax)

                if np.any(y[idxs]):
                    sample = sts.sampler(x[idxs], y[idxs], n_run=1, n_sample=n_sample)
                    score = self.test.testAD(sample, self.target_distribution.cumulative)
                    yield self.test.get_pvalue(score)
                else:
                    yield np.nan

    def compute_AD(self, n_sample = 1_000, verbose=True, **kwargs):
        if verbose:
            print("AD test with sample:", n_sample)
            to = time.time()

        cmfs = self.map_parameter_space(
            self.compute_distributions,
            self.fragmentation_rate,
            self.mass_transfer_rate,
            #self.phi_xi,
            self.mass_partition
        )

        shape = (len(self.fragmentation_rate), len(self.mass_transfer_rate), len(self.mass_partition), self.final_level + 1)
        ad_scores_arr = list(
            tqdm(self._compute_AD(cmfs, n_sample = n_sample),
                 total= np.prod(shape),
                 desc ="Scoring AD test on distributions")
                             )
        ad_scores = np.reshape(ad_scores_arr, shape)

        if verbose:
            print(f"End of computation of "
                  f"{final_level} levels "
                  f"x {len(fragmentation_rate)} "
                  f"x {len(mass_transfer_rate)} "
                  f"x {len(mass_partition)} sized data"
                  f"\nin {time.time() - to} seconds"
                  f"\n= {(time.time() - to) / 3600} hours")

        return np.ma.masked_invalid(ad_scores)

    def map_parameter_space(self, function, *args):
        paramlist = itertools.product(*args)
        return map(function, paramlist)

    def compute_distributions(self, params):
        phi, xi, omega = params
        #print("XI, compute distribution", xi)
        #xi = phi - phi_xi

        scaling_ratios = [self.scaling_ratio for _ in range(self.final_level)]

        if self.number_distribution == "Poisson":
            model = fragmentCMF.Poisson(phi, xi, omega, scaling_ratios)

        elif self.number_distribution == "Binary":
            model = fragmentCMF.ScaleFree(phi, xi, omega, scaling_ratios)

        else:
            model = fragmentCMF.ScaleFree(phi, xi, omega, scaling_ratios)

        cmf = fragmentCMF.NetworkModel(model)
        cmf.computePDF(x0, self.initial_pdf.pdf)

        return cmf

    def compute_distributions_from_degen(self, params):
        phi, phi_xi, omega = params
        xi = phi - phi_xi
        params = phi, xi, omega
        return self.compute_distributions(params)

    def compute_AD_from_degen(self, n_sample = 1_000, verbose=True, px_pts=100, dx=0.2, **kwargs):
        if verbose:
            print("AD test with sample:", n_sample)
            to = time.time()

        ad_shape = (len(self.fragmentation_rate),
                    px_pts,
                    len(self.mass_partition),
                    self.final_level + 1)

        ad_scores = np.empty(shape=ad_shape)

        for level in range(self.final_level + 1):
            px = np.linspace(phi_xi[level] - dx, phi_xi[level] + dx, px_pts)

            cmfs = self.map_parameter_space(
                self.compute_distributions_from_degen,
                self.fragmentation_rate,
                px,
                self.mass_partition
            )

            shape = ad_shape[:-1]

            ad_scores_arr = list(
                tqdm(self._compute_AD(cmfs, n_sample = n_sample),
                     total= np.prod(shape),
                     desc ="Scoring AD test on distributions")
                                 )

            ad_scores[:, :, :, level] = np.reshape(ad_scores_arr, shape)

        if verbose:
            print(f"End of computation of "
                  f"{final_level} levels "
                  f"x {len(fragmentation_rate)} "
                  f"x {len(mass_transfer_rate)} "
                  f"x {len(mass_partition)} sized data"
                  f"\nin {time.time() - to} seconds"
                  f"\n= {(time.time() - to) / 3600} hours")

        return np.ma.masked_invalid(ad_scores)

    def save_cube(self, cube, path, filename, **metadata):
        primary_hdu = fits.PrimaryHDU(data=cube)

        for key, value in metadata.items():
           primary_hdu.header[key] = value

        #levels = np.arange(0, self.final_level + 1, 1)
        #col1 = fits.Column(name="Larr", format='E', array=levels)
        #col2 = fits.Column(name="Farr", format='E', array=self.fragmentation_rate)
        #col3 = fits.Column(name="Marr", format='E', array=self.mass_transfer_rate)
        #col4 = fits.Column(name="Parr", format='E', array=self.mass_partition)

        #table_hdu = fits.BinTableHDU.from_columns([col1, col2, col3, col4])

        #hdu_tot = fits.HDUList([primary_hdu, table_hdu])

        primary_hdu.writeto(path + f"{filename}.fits")

def plot_cubes(cube, parameters, Ro=1, omega_proj=0):
    fragmentation_rate, mass_transfer_rate, mass_partition, scaling_ratio = parameters


    # exterior of blue = can reject H0 at 1 sigma
    # exterior of red = can reject H0 at 2 sigma
    # exterior of green = can reject H0 at 3 sigma

    def initialise(ax, khi, Marr, Farr, level, part):
        M, F = np.meshgrid(Marr, Farr)

        vmin = 1e-3
        vmax = np.nanmax(khi)
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)

        if level < 4:
            khi[:, :, part, level][np.isnan(khi[:, :, part, level])] = 0.001

        im = ax.pcolor(M, F, khi[:, :, part, level],
                       norm=norm,
                       shading='auto', cmap='Greens')

        ax.set_ylim([np.amin(Farr), np.amax(Farr)])
        ax.set_xlim([np.amin(Marr), np.amax(Marr)])
        return im

    levels = cube.shape[-1] - 1
    side_length = int(np.sqrt(levels)) + 1

    fig, axs = plt.subplots(ncols=side_length, nrows=side_length, figsize=(8, 5), sharex=True, sharey=True, dpi=150)

    sigmas = np.array([0.01, 0.05, 0.25])

    yaxis = ([1] + [0] * (side_length - 1)) * side_length
    xaxis = [0] * (side_length - 1) + [1] * side_length

    seen_ax = []

    # ---------- PLOT IN EVERY AX ----------
    for ax, level, y, x in zip(np.ravel(axs), range(levels + 1), yaxis, xaxis):
        seen_ax.append(ax)

        # if not y:
        #    ax.get_yaxis().set_visible(False)
        # if not x:
        #    ax.get_xaxis().set_visible(False)

        Rcurrent = Ro / scaling_ratio ** (level)
        Rint = int(np.round(Rcurrent, decimals=0))

        im = initialise(ax, ad_scores, mass_transfer_rate, fragmentation_rate, level, omega_proj)

        textstr = f'{Rint} UA'

        colors_slopes = ["x", "x", "x", "x", "b", "orange", "g", "r", "m"]
        if colors_slopes[level] != "x":
            props1 = dict(boxstyle='round', facecolor="white", edgecolor=colors_slopes[level], alpha=1)
            props2 = dict(boxstyle='round', facecolor=colors_slopes[level], edgecolor="none", alpha=0.25)

            ax.text(1 - 0.95, 1 - 0.05, textstr, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=props1)

            ax.text(1 - 0.95, 1 - 0.05, textstr, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=props2)

        else:
            props = dict(boxstyle='round', facecolor="white", alpha=1)

            ax.text(1 - 0.95, 1 - 0.05, textstr, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)

        # major_tick = -np.arange(-1, 0.501, 0.5)
        # ax.set_xticks(major_tick)
        # minor_ticks = -np.arange(-1, 0.5, 0.1)
        # ax.set_xticks(minor_ticks, minor=True)

        # major_ticks = -np.arange(-1.5, 0.1, 0.3)
        # ax.set_yticks(major_ticks)
        # minor_ticks = -np.arange(-1.5, 0, 0.1)
        # ax.set_yticks(minor_ticks, minor=True)

        ax.tick_params(labelsize=9, labelrotation=45)

    # ---------- DELETE AXIS ----------
    for ax in np.ravel(axs):
        if ax not in seen_ax:
            # ax.get_yaxis().set_visible(False)
            # ax.get_xaxis().set_visible(False)
            plt.axis('off')

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.1,
                        hspace=0.1)

    # And a corresponding grid
    # plt.grid(which='both')


    # ---------- ADD COLORBAR ----------
    cb = fig.colorbar(im, ax=np.ravel(axs), orientation='vertical', extend="min", aspect=40)
    cb.ax.set_ylabel('AD p-value', rotation=270, va="center", ha="right")

    fig.text(0.02, 0.5, 'Fragmentation rate $\phi$', va='center', rotation='vertical')
    fig.text(0.5, -0.005, "Mass transfer rate $\\xi$", ha='center')

    return fig


def open_data(filename):
    HDU = fits.open(filename)

    khi = HDU[0].data
    [print(key, " : ", HDU[0].header[key]) for key in HDU[0].header]

    Farr = np.linspace(HDU[0].header["MINPHI"],
                       HDU[0].header["MAXPHI"],
                       int(HDU[0].header["NPHI"]))

    Larr = np.arange(1,
                     int(HDU[0].header["MAXL"]) + 1,
                     1)

    Marr = np.linspace(HDU[0].header["MINXI"],
                       HDU[0].header["MAXXI"],
                       int(HDU[0].header["NXI"]))

    Parr = np.linspace(HDU[0].header["MINOMEGA"],
                       HDU[0].header["MAXOMEGA"],
                       int(HDU[0].header["NOMEGA"]))

    khi = np.ma.masked_invalid(khi)

    initial_func = getattr(distribution, HDU[0].header["FUNCTION"]).pdf

    func_kwargs = {}
    for key, value in HDU[0].header.items():
        if "KW" in key:
            func_kwargs[key[2:].lower()] = value

    xstart = np.linspace(HDU[0].header["XMIN"],
                         HDU[0].header["XMAX"],
                         1000)

    return Farr, Larr, Marr, Parr, khi, xstart, initial_func, func_kwargs, scaling_ratio

if __name__ == '__main__':
    x0 = np.logspace(0, 2, 256)

    gamma = 2.35
    print("Initial distribution with slope", gamma)
    final_level = 8
    scaling_ratio = 1.5

    initial_func = distribution.InitialMassFunction(name="power-law", gamma=gamma)
    xmin, xmax = 0.01, 150
    target_distribution = distribution.InitialMassFunction(name='L3', mlow = xmin, mup = xmax)

    pdf = initial_func.pdf(x0)
    number_distribution = "Binary"

    factor = 50/1.5

    max_frag_rate = np.log(2)/np.log(scaling_ratio)
    fmax = max_frag_rate - 0.2
    print(f"Frag. rate from 0 to {fmax}")
    fragmentation_rate = np.linspace(0, fmax, int(fmax * factor))

    lowxi = -2
    highxi = 0.5
    mass_transfer_rate = np.linspace(-2, 0.5, int((highxi-lowxi) * factor))

    mass_partition = np.linspace(1, 5, 5)

    #print("XI", mass_transfer_rate)
    #print("PHI", fragmentation_rate)
    #print("Q", mass_partition)

    lengths = (len(fragmentation_rate), len(mass_transfer_rate), len(mass_partition), final_level)
    print("LENGTH FRAG\t", len(fragmentation_rate),
          "\nLENGTH XI\t", len(mass_transfer_rate),
          "\nLENGTH PARTITION\t", len(mass_partition),
          "\nLENGTH LEVELS\t", final_level)
    print("TOTAL LENGTH\t", np.prod(lengths))

    filename = f"full_ADmap_r{scaling_ratio}_l{final_level}_0_alpha{gamma}_degen"

    mapper = Mapping(x0, initial_func, final_level, fragmentation_rate, mass_transfer_rate, mass_partition,
                        target_distribution,
                        number_distribution=number_distribution,
                        scaling_ratio=scaling_ratio, test="AD")

    ad_scores = mapper.compute_AD()
    #ad_scores = mapper.compute_AD_from_degen()
    #print(ad_scores)
    #print(mass_partition)

    metadata = {
        "MAXPHI": max(fragmentation_rate),
        "MINPHI": min(fragmentation_rate),
        "NPHI": len(fragmentation_rate),
        "MAXXI": max(mass_transfer_rate),
        "MINXI": min(mass_transfer_rate),
        "NXI": len(mass_transfer_rate),
        "MAXOMEGA": max(mass_partition),
        "MINOMEGA": min(mass_partition),
        "NOMEGA": len(mass_partition),
        "MAXL": final_level,
        "R": scaling_ratio,
        "FUNCTION": str(initial_func.__class__).split('.')[-1][:-2],
        "PDISTRIB": "Binary",
        "KWGAMMA": gamma,
        "XMIN": xmin,
        "XMAX": xmax,
        "MODE":"Degen"
    }

    mapper.save_cube(ad_scores.data,
                     path="/Users/thomaben/Documents/GitHub/CMFragmentation/ADtest_cubes/",
                     filename=filename, **metadata)

    #Farr, Larr, Marr, Parr, khi, xstart, initial_func,
    # func_kwargs = open_data('/Users/thomaben/Documents/GitHub/CMFragmentation/ADtest_cubes/testsave.fits')
    #print(initial_func, func_kwargs)

    # args = (fragmentation_rate, mass_transfer_rate, mass_partition, scaling_ratio)
    # fig = plot_cubes(ad_scores, args, Ro=2500)
    # fig.suptitle(f"Solutions pour $\\psi = {np.sort(np.unique(Parr))[part]}$", va="top", ha="right")
    # plt.show()
