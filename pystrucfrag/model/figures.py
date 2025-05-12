import matplotlib.pyplot as plt
from fragmentation import *
from scipy.signal import argrelextrema

display_options(EquationOfState)
display_options(TurbulentCascade)

dpi = 150

def print_initial_conditions(CI, Ro, Mo, *args, name=""):
    xi, TurbCascade, EOS = args
    print("\n" + name + ":\n\tInitial density [g/cm3]", CI[-1] * CONVERSION.Msol_to_g / CONVERSION.pc_to_cm ** 3)
    print("\tInitial size [pc]", Ro)
    print("\tInitial mass [Msol]", Mo)
    print("\tMass transfer rate", xi)
    print("\tEOS", EOS)
    print("\tTurbulent Cascade", TurbCascade)

####### PLOT FRAGMENTATION RATE FOR DIFFERENT XI
Ro = 10  # pc
Mo = 1e4  # Msol
CI = (1, mean_density(Ro, Mo))  # initial conditions

Rfin = 10 * CONVERSION.au_to_pc  # smallest scale to go in pc
R = np.logspace(np.log10(Ro), np.log10(Rfin), 10_000, endpoint=True)

TurbCascade = TurbulentCascade(name='scale-free', Vo=1_000, eta=0.38)
EOS = EquationOfState(name='isothermal', temperature=10)

ksis = [-1, -0.5, 0, 0.5, 1][::-1]
colors = ["b", "b", "k", "r", 'r']  # [::-1]
lss = [":", "-.", "-", "-.", ":"][::-1]

args = (ksis, TurbCascade, EOS)  # xi, TurbCascade, EOS
print_initial_conditions(CI, Ro, Mo, *args, name="PLOT FRAGMENTATION RATE FOR DIFFERENT XI")

fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)

fs = 14
fs_contour = 11
ticksize = 12
lw = 2
lsize = 11
padding = 10

ax.set_xscale('log')
ax.set_ylabel('Fragmentation rate $\phi$', fontsize=fs, labelpad=padding)
ax.set_xlabel('Size R [pc]', fontsize=fs, labelpad=padding)

ax2 = ax.twiny()  # instantiate a second axes that shares the same x-axis

ax2.set_xscale('log')
ax2.set_xlabel('R [AU]', fontsize=fs, labelpad=padding)

ax2.set_xlim([Rfin * CONVERSION.pc_to_au, Ro * CONVERSION.pc_to_au])
ax.set_xlim([Rfin, Ro])

for ksi, color, ls in zip(ksis, colors, lss):
    xi = MassTransferRate(name='constant', ksi0=ksi)  # mass transfer rate
    args = (xi, TurbCascade, EOS)  # xi, TurbCascade, EOS
    phi, density = solve_fragODE(CI, R, *args, filling_factor=-3, additional_support=0)
    ax.plot(R, phi, ls=ls, color=color, label=f"{ksi}")

##### esthetics

ax.fill_between([Rfin, Ro], [-3.5, -3.5], color="k", alpha=0.2, hatch="//")

ax.set_ylim([-0.5, 3])

major_tick = np.arange(-0.5, 3.1, 0.5)
ax.set_yticks(major_tick)
minor_ticks = np.arange(-0.5, 3.1, 0.25)
ax.set_yticks(minor_ticks, minor=True)

ax.tick_params(labelsize=ticksize)
ax2.tick_params(labelsize=ticksize)

ax.legend(loc=4, title="$\\xi$", fontsize=lsize, framealpha=1)  # , bbox_to_anchor=(0.05,1.))
plt.tight_layout()

####### PLOT COLLAPSE THRESHOLD
Ro = 10  # pc
Mo = 1e4  # Msol
CI = (1, mean_density(Ro, Mo))  # initial conditions

Rfin = 10 * CONVERSION.au_to_pc  # smallest scale to go in pc
R = np.logspace(np.log10(Ro), np.log10(Rfin), 10_000, endpoint=True)

TurbCascade = TurbulentCascade(name='scale-free', Vo=1_000, eta=0.38)
EOS = EquationOfState(name='isothermal', temperature=10)

ksis = [-1, -0.5, 0, 0.5, 1][::-1]
colors = ["b", "b", "k", "r", 'r']  # [::-1]
lss = [":", "-.", "-", "-.", ":"][::-1]

args = (ksis, TurbCascade, EOS)  # xi, TurbCascade, EOS
print_initial_conditions(CI, Ro, Mo, *args, name="PLOT COLLAPSE THRESHOLD")

fs = 14
fs_contour = 11
ticksize = 12
lw = 2
lsize = 11
padding = 10

fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)

ax.set_xscale('log')
ax.set_ylabel('Threshold $\delta_c$', fontsize=fs, labelpad=padding)
ax.set_xlabel('Size R [pc]', fontsize=fs, labelpad=padding)

ax2 = ax.twiny()  # instantiate a second axes that shares the same x-axis

ax2.set_xscale('log')
ax2.set_xlabel('R [AU]', fontsize=fs, labelpad=padding)

ax2.set_xlim([Rfin * CONVERSION.pc_to_au, Ro * CONVERSION.pc_to_au])
ax.set_xlim([Rfin, Ro])

PHI = []
for ksi, color, ls in zip(ksis, colors, lss):
    xi = MassTransferRate(name='constant', ksi0=ksi)  # mass transfer rate
    args = (xi, TurbCascade, EOS)  # xi, TurbCascade, EOS
    phi, density = solve_fragODE(CI, R, *args, filling_factor=-3, additional_support=0)
    delta_c = get_threshold(R, density, EOS, TurbCascade)

    ax.plot(R, delta_c, ls=ls, color=color, label=f"{ksi}")

    ddelta_dlogR = np.gradient(delta_c, R) / R

    #idx = argrelextrema(-delta_c, np.less)[0][0]
    #PHI.append(phi[idx])
    #ax.plot(R[idx], delta_c[idx], "x", color=color)

    # fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    # ax.plot(ksis, PHI, "+")
    # ax.plot(ksis, np.array(ksis) + 1.3, '-k')

ax.legend(loc=2, title="$\\xi$", fontsize=lsize, framealpha=1, frameon=False)  # , bbox_to_anchor=(0.05,1.))
ax.tick_params(labelsize=ticksize)
ax2.tick_params(labelsize=ticksize)
plt.tight_layout()

####### PDF PLOTS
Ro = 10  # pc
Mo = 1e4  # Msol
CI = (1, mean_density(Ro, Mo))  # initial conditions

Rfin = 10 * CONVERSION.au_to_pc  # smallest scale to go in pc
R = np.logspace(np.log10(Ro), np.log10(Rfin), 10_000, endpoint=True)

TurbCascade = TurbulentCascade(name='scale-free', Vo=1_000, eta=0.38)
EOS = EquationOfState(name='isothermal', temperature=10)

ksis = [-1, -0.5, 0, 0.5, 1][::-1]
colors = ["b", "b", "k", "r", 'r']  # [::-1]
lss = [":", "-.", "-", "-.", ":"][::-1]

args = (ksis, TurbCascade, EOS)  # xi, TurbCascade, EOS
print_initial_conditions(CI, Ro, Mo, *args, name="PDF PLOTS")

# --------------- Figures
fig, axs = plt.subplots(figsize=(6, 5), dpi=dpi)

fontsize = 14
ticksize = 12
padding = 10

axs.set_xlabel('Size R [pc]', fontsize=fontsize, labelpad=padding)
axs.set_ylabel('P($\delta > \delta_c$', fontsize=fontsize, labelpad=padding)

for ksi, color, ls in zip(ksis, colors, lss):
    xi = MassTransferRate(name='constant', ksi0=ksi)  # mass transfer rate
    args = (xi, TurbCascade, EOS)  # xi, TurbCascade, EOS
    phi, density = solve_fragODE(CI, R, *args, filling_factor=-3, additional_support=0)

    pdf = get_cumulative_pdf(R, density, EOS, TurbCascade, additional_support=0)
    axs.plot(R, pdf, ls=ls, color=color, label=f"{ksi}")

axs.set_xscale('log')
axs.set_yscale('log')

axs.legend(loc=2, title="$\\xi$", fontsize=lsize, framealpha=1, frameon=False)  # , bbox_to_anchor=(0.05,1.))
axs.tick_params(labelsize=ticksize)

####### NESTS
Ro = 10  # pc
Mos = np.array([1e0, 1e3, 1e4, 1e5])

CIs = (1, mean_density(Ro, Mos))  # initial conditions

Rfin = 10 * CONVERSION.au_to_pc  # smallest scale to go in pc
R = np.logspace(np.log10(Ro), np.log10(Rfin), 10_000, endpoint=True)

TurbCascade = TurbulentCascade(name='scale-free', Vo=1000, eta=0.38)
EOS = EquationOfState(name='adiabatic', temperature=10)

ksis = [-0.75, -0.5, -0.25, 0]  # [::-1]
colors = ["k", "k", "k", "k", 'r', "r", 'r']  # [::-1]
lss = [":", "-.", "--", "-", "--", "-.", ":"]  # [::-1]
mstyle = "<"
markers = [mstyle] * len(lss)

args = (ksis, TurbCascade, EOS)  # xi, TurbCascade, EOS
print_initial_conditions(CI, Ro, Mo, *args, name="NESTS")

Rfin = 10 * CONVERSION.au_to_pc  # smallest scale to go in pc
R = np.logspace(np.log10(Ro), np.log10(Rfin), 10_000, endpoint=True)

fig, axs = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True, figsize=(5, 5), dpi=200)

NESTy = [15, 23, 4, 10, 5, 4, 5, 6, 8, 7, 6, 4, 4, 13, 4, 4, 5, 6, 5, 13]
NESTx = [53.8, 81, 10.1, 40.5, 5.3, 11.1, 29.9, 15,
         40.9, 46.1, 27.8, 19.9, 12.9, 46.4, 11, 8.3,
         22.8, 41.1, 11.5, 61.5]

fs = 14
fs_contour = 11
ticksize = 10
lw = 2
lsize = 11
msize = 5

idxs_r = R > 10 * CONVERSION.au_to_pc  # 270 ksi = 0.5

for Mo, ax in zip(Mos, np.ravel(axs)):
    ax.plot(np.array(NESTx) * 1000 * CONVERSION.au_to_pc, NESTy, '+', color='r', ms=14)
    rho0 = mean_density(Ro, Mo)
    CI = (1, rho0)  # initial conditions

    for ksi, color, ls, marker in zip(ksis, colors, lss, markers):
        xi = MassTransferRate(name='constant', ksi0=ksi)  # mass transfer rate
        args = (xi, TurbCascade, EOS)
        phi, density = solve_fragODE(CI, R, *args, filling_factor = -3, additional_support=0)

        rstop = Rstop(R, phi)
        if rstop < 0.1:
            ax.plot([rstop], [1], color=color, marker=marker, ms=msize,
                     linewidth=lw * 4)

            ax.plot(R[idxs_r][R[idxs_r] > rstop][::-1], get_NESTs(R[idxs_r], rstop, phi[idxs_r]),
                     ls=ls, color=color, label=f"{np.round(ksi, 2)}", lw=lw, alpha=0.8)

    rho0 = int(np.log10(rho0 * CONVERSION.Msol_to_g / CONVERSION.pc_to_cm ** 3))
    if ax == axs[0, 0]:
        ax.text(0.05, 0.9, f"$\\rho_0$ = 10$^{{{rho0}}}$ g/cm$^{3}$", transform=ax.transAxes, weight="bold", fontsize=10)
    else:
        ax.text(0.05, 0.9, f"10$^{{{rho0}}}$ g/cm$^{3}$", transform=ax.transAxes, weight="bold",
                fontsize=10)

ax = axs[0, 0]
ax.set_xscale('log')
#ax.set_xlabel('Size R [pc]', fontsize=fs)
#ax.set_ylabel('$< N_* >$', fontsize=fs)

Rmin = 10 * CONVERSION.au_to_pc
ax.set_xlim([Rmin, Ro])

##### esthetics
ax.set_yscale('log')
ax.set_ylim([0.5, 1e3])
# ax.legend(loc=0, frameon=False, title="$\\xi$", fontsize=lsize, framealpha=1)

#major_tick = np.logspace(0, 3, 4)
#ax.set_yticks(major_tick)

#major_tick = np.logspace(-4, 1, 6)
#ax.set_xticks(major_tick)

for ax in np.ravel(axs):
    ax.tick_params(labelsize=ticksize)
    ax.axhline(4, ls='-', alpha=0.5, color="k", xmax=0.70)
    plt.setp(ax.get_yticklabels()[1:][::2], visible=False)


plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.1)
plt.tight_layout()

####### INITIAL CONDITIONS PLOT
Ro = 10 # pc
xi = MassTransferRate(name='constant', ksi0=0)  # mass transfer rate
TurbCascade = TurbulentCascade(name='scale-free', Vo=1000, eta=0.38)
#TurbCascade = TurbulentCascade(name='saturated scale-free', Vo=1000, eta=0.38, Vadd=500)
EOS = EquationOfState(name='isothermal', temperature=10)

args = (xi, TurbCascade, EOS)  # xi, TurbCascade, EOS

Mos = [1e3, 1e4, 1e5, 1e6][::-1]
lss = [":", "-.", "--", "-"][::-1]  # (0, (3, 5, 1, 5, 1, 5)),

CIs = (1, mean_density(Ro, np.array(Mos)))
print_initial_conditions(CIs, Ro, Mos, *args, name="INITIAL CONDITIONS PLOT")

Rfin = 10 * CONVERSION.au_to_pc  # smallest scale to go in pc
R = np.logspace(np.log10(Ro), np.log10(Rfin), 10_000, endpoint=True)

# --------------- Figures
fig, axs = plt.subplots(ncols=3, figsize=(12, 4.5), dpi=dpi)

fontsize = 14
ticksize = 12
padding = 10

for ax in axs:
    ax.set_xscale('log')
for ax in axs[1:]:
    ax.set_yscale('log')

axs[0].set_xlabel('Size R [pc]', fontsize=fontsize, labelpad=padding)
axs[0].set_ylabel('Fragmentation rate $\phi$', fontsize=fontsize, labelpad=padding)

axs[1].set_xlabel('R [pc]', fontsize=fontsize, labelpad=padding)
axs[1].set_ylabel('$<M>$ [$M_\odot$]', fontsize=fontsize, labelpad=padding)

axs[2].set_xlabel('R [pc]', fontsize=fontsize, labelpad=padding)
axs[2].set_ylabel('$<\\rho>$ [g/cm$^{-3}$]', fontsize=fontsize, labelpad=padding)

for Mo, initial_density, ls in zip(Mos, CIs[-1], lss):
    CI = (1, initial_density)  # initial conditions

    rrho = CI[-1] / CONVERSION.pc_to_cm ** 3 * CONVERSION.Msol_to_g

    phi, density = solve_fragODE(CI, R, *args, filling_factor = -3, additional_support=0)
    rhos_cgs = density / CONVERSION.pc_to_cm ** 3 * CONVERSION.Msol_to_g

    axs[0].plot(R, phi, ls=ls, color="k", label=f"{rrho:.1e}")
    axs[1].plot(R, get_mass(Mo, R, phi, xi.get_xi(0)), ls=ls, color="k")
    axs[2].plot(R, rhos_cgs, ls=ls, color="k")

    # number = get_number(1, R, phi)
    # print("rho:", rrho, "N:", number[R > 100 * CONVERSION.au_to_pc][0])

#Mbe = BonnorEbert(R / 2 * CONVERSION.pc_to_m, sound_speed2(10)) * CONVERSION.kg_to_Msol
#axs[1].plot(R, Mbe, "--", alpha=0.5)
#critical_mass = get_mass_threshold(R, density, EOS, TurbCascade, additional_support = 0) * CONVERSION.kg_to_Msol
#axs[1].plot(R, critical_mass, "--r")

#deltac = get_threshold(R, density, EOS, TurbCascade)
#critical_density = density * np.exp(deltac) * CONVERSION.Msol_to_g / CONVERSION.pc_to_cm ** 3
#axs[2].plot(R, critical_density, ls=ls, color="r")

axs[0].legend(loc=2, frameon=False, title="$\\rho_0$ [g/cm$^{-3}$]",
              fontsize=ticksize, title_fontsize='x-large')

##### esthetics
ax2 = axs[0].twiny()  # instantiate a second axes that shares the same x-axis

ax2.set_xscale('log')
ax2.set_xlabel('R [AU]', fontsize=fontsize, labelpad=padding)
ax2.set_xlim([Rfin * CONVERSION.pc_to_au, Ro * CONVERSION.pc_to_au])

axs[0].set_xlim([Rfin, Ro])
axs[0].set_ylim([0, 3])

axs[1].set_xlim([Rfin, Ro])
axs[1].set_ylim([1e-3, 1e5])

axs[2].set_xlim([Rfin, Ro])
axs[2].set_ylim([1e-20, 1e-12])


x = np.array([1e-3, 1e-4])
y = 5e1 * x ** (1)
axs[1].plot(x, y, color='k', alpha=0.5)
axs[1].text(1e-4, 5e-2, "$M \propto R$", fontsize=ticksize + 1, color="k", alpha=0.75)

#x = np.array([1e-1, 1e-2])
#y = 5e3 * x ** (2)
#axs[1].plot(x, y, color='k', alpha=0.5)
#axs[1].text(1e-2, 5e0, "$M \propto R^2$", fontsize=ticksize + 1, color="k", alpha=0.75)

x = np.array([1e-2, 1e-3])
y = 5e-19 * x ** (-2)
axs[2].plot(x, y, color='k', alpha=0.5)
axs[2].text(5e-3, 1e-13, "$\\rho \propto R^{-2}$", fontsize=ticksize + 1, color="k", alpha=0.75)


major_tick = np.logspace(-4, 1, 6)
for ax in axs:
    ax.set_xticks(major_tick)

major_tick = np.logspace(1, 6, 6)
ax2.set_xticks(major_tick)

major_tick = np.logspace(-20, -12, 9)
axs[2].set_yticks(major_tick)
# minor_ticks = 10**np.arange(0, 12, 1)
# axs[0].set_xticks(minor_ticks, minor=True)

axs[0].tick_params(labelsize=ticksize)
axs[1].tick_params(labelsize=ticksize)
axs[2].tick_params(labelsize=ticksize)
ax2.tick_params(labelsize=ticksize)

axs[0].text(-0.2, 1.15, "(a)", transform=axs[0].transAxes, weight="bold")
axs[1].text(-0.2, 1.15, "(b)", transform=axs[1].transAxes, weight="bold")
axs[2].text(-0.2, 1.15, "(c)", transform=axs[2].transAxes, weight="bold")

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.1)
plt.tight_layout()

####### MEAN FRAGMENTATION RATE
Ro = 10  # pc
Mo = 1e4  # Msol
CI = (1, mean_density(Ro, Mo))  # initial conditions

Rfin = 10 * CONVERSION.au_to_pc  # smallest scale to go in pc
R = np.logspace(np.log10(Ro), np.log10(Rfin), 10_000, endpoint=True)

TurbCascade = TurbulentCascade(name='scale-free', Vo=1000, eta=0.38)
EOS = EquationOfState(name='adiabatic', temperature=10)

args = ("variable between -1 and 1", TurbCascade, EOS)
print_initial_conditions(CI, Ro, Mo, *args, name="MEAN FRAGMENTATION RATE")

xmin = -0.6
xmax = 0.1

ymin = 0.5
ymax = 1.1

phi_mes = 0.77
dphi_mes = 0.20

ksis = np.linspace(xmin, xmax, 64)
phi_moy = []

idxs_r = R > 10 * CONVERSION.au_to_pc  # 270 ksi = 0.5
Rstops = []
Vadd = 0
for ksi in ksis:
    xi = MassTransferRate(name='constant', ksi0=ksi)  # mass transfer rate
    args = (xi, TurbCascade, EOS)
    phi, density = solve_fragODE(CI, R, *args, filling_factor=-3, additional_support=Vadd)

    phi_moy.append(mean_fragmentation_rate(R * CONVERSION.pc_to_au, phi, R1=1_400, R2=26_000))

    rstop = Rstop(R, phi)
    Rstops.append( rstop )

"""
with open(f"low_boundary_xi_vs_R_Vadd_{Vadd}mps", "w") as file:
    file.write("# Rstop $\\xi$ \n")        # column names
    ksis[ksis < -0.65] = -0.65
    np.savetxt(file, np.array([Rstops[::-1], ksis[::-1]]).T)

fig, axs = plt.subplots(figsize=(5, 4), dpi=150)
Rstops, ksis = np.loadtxt(f"low_boundary_xi_vs_R_Vadd_{Vadd}mps", unpack=True)
axs.plot(Rstops, ksis, "+r")
axs.set_xlabel("$R/R_0$")
axs.set_ylabel("$\\xi(R)$")
axs.set_xscale('log')
"""

fig, axs = plt.subplots(figsize=(6, 5), dpi=dpi)

fs = 16
fs_contour = 14
ticksize = 14
lw = 2
tickl = 5
tickw = 1.1

axs.plot(ksis, phi_moy, "-k", lw=lw)

axs.set_ylabel("$\\overline{\phi_{3D}}^{model}$", fontsize=fs)
axs.set_xlabel("$\\xi$", fontsize=fs)

axs.set_xlim([xmin, xmax])
axs.set_ylim([ymin, ymax])

major_tick = np.arange(xmin, xmax + 0.01, 0.1)
axs.set_xticks(major_tick)
minor_ticks = np.arange(xmin, xmax + 0.01, 0.01)
axs.set_xticks(minor_ticks, minor=True)

minor_ticks = np.arange(ymin, ymax+0.01, 0.1)
axs.set_yticks(minor_ticks)
minor_ticks = np.arange(ymin, ymax+0.01, 0.05)
axs.set_yticks(minor_ticks, minor=True)

axs.fill_between(ksis, y1=phi_mes - dphi_mes, y2=phi_mes + dphi_mes, color='grey', alpha=0.3)

xi1, xi2 = np.interp([phi_mes - dphi_mes, phi_mes + dphi_mes],
                     phi_moy, ksis)

axs.axvline(xi1, ymax=(phi_mes - dphi_mes - ymin)/(ymax - ymin), ls="--", lw=lw)
axs.axvline(xi2, ymax=(phi_mes + dphi_mes - ymin)/(ymax - ymin), ls="--", lw=lw)

decimals = 2
axs.text(xi1+0.01, ymin+0.025, "$\mathbf{\\xi \\approx}$"+f"{np.round(xi1, decimals=decimals)}", color='C0', fontsize=fs_contour)
axs.text(xi2+0.01, ymin+0.025, "$\mathbf{\\xi \\approx}$"+f"{np.round(xi2, decimals=decimals)}", color='C0', fontsize=fs_contour)

axs.tick_params(labelsize=ticksize, length=tickl, width=tickw)

plt.tight_layout()

####### ADDITIONAL SUPPORT PLOT
"""
xi = MassTransferRate(name='constant', ksi0=0)  # mass transfer rate
# TurbCascade = TurbulentCascade(name='scale-free', Vo=1000, eta=0.38)
EOS = EquationOfState(name='adiabatic', temperature=10)

args = (xi, TurbCascade, EOS)  # xi, TurbCascade, EOS

fig, ax = plt.subplots(figsize=(7, 5), dpi=200)

fs = 14
fs_contour = 11
ticksize = 12
lw = 2
lsize = 11

ax.set_xscale('log')
ax.set_ylabel('Fragmentation rate $\phi$', fontsize=fs)
ax.set_xlabel('Size R [pc]', fontsize=fs)

ax2 = ax.twiny()  # instantiate a second axes that shares the same x-axis

ax2.set_xscale('log')
ax2.set_xlabel('Size R [AU]', fontsize=fs)

ax.set_xlim([Rfin, Ro])
ax2.set_xlim([Rfin * CONVERSION.pc_to_au, Ro * CONVERSION.pc_to_au])

Vadd_list = [0, 100, 300, 500]
colors = ["k", 'none', 'r', 'none']
lss = ['-', 'none', '--', 'none']
fill = []

for Vadd, color, ls in zip(Vadd_list, colors, lss):
    args = (xi, TurbCascade, EOS)  # xi, TurbCascade, EOS
    phi, density = solve_fragODE(CI, R, *args, filling_factor = -3, additional_support = Vadd)

    #ax.plot(R, phi, color=color, ls=ls)
    if color == 'none':
        fill.append(phi)

ax.fill_between(R, fill[0], fill[1], color="r", alpha=0.2)

##### esthetics
ax.fill_between([Rfin, Ro], [-3.5, -3.5], color="k", alpha=0.2, hatch="//")

ax.set_ylim([-.5, 3])

major_tick = np.arange(-0.5, 3.1, 0.5)
ax.set_yticks(major_tick)
minor_ticks = np.arange(-0.5, 3.1, 0.25)
ax.set_yticks(minor_ticks, minor=True)

ax.tick_params(labelsize=ticksize)
ax2.tick_params(labelsize=ticksize)

#ax.legend(loc=4, title=r"$\eta$", fontsize=lsize, framealpha=1)  # , bbox_to_anchor=(0.05,1.))
plt.tight_layout()
"""
####### RSTOP AS A FUNCTION OF VADD

Ro = 10  # pc
Mo = 1e4  # Msol
CI = (1, mean_density(Ro, Mo))  # initial conditions

Rfin = 10 * CONVERSION.au_to_pc  # smallest scale to go in pc
R = np.logspace(np.log10(Ro), np.log10(Rfin), 10_000, endpoint=True)

xi = MassTransferRate(name='constant', ksi0=0)  # mass transfer rate
TurbCascade = TurbulentCascade(name='scale-free', Vo=1000, eta=0.38)
EOS = EquationOfState(name='adiabatic', temperature=10)

args = (xi, TurbCascade, EOS)  # xi, TurbCascade, EOS

Vadd_list = np.linspace(0, 500, 64)
Rstops = []
densities = []

idxs_r = R > 40 * CONVERSION.au_to_pc  # 270 ksi = 0.5
fig, ax = plt.subplots(figsize=(7, 5), dpi=200)

for Vadd in Vadd_list:
    args = (xi, TurbCascade, EOS)  # xi, TurbCascade, EOS
    phi, density = solve_fragODE(CI, R, *args, filling_factor = -3, additional_support = Vadd)

    rstop = Rstop(R, phi)
    Rstops.append(rstop * CONVERSION.pc_to_au)
    densities.append(density[R == rstop] * CONVERSION.Msol_to_g / CONVERSION.pc_to_cm ** 3)

    ax.plot(R[R > rstop], density[R > rstop])
    ax.plot(rstop, density[R == rstop], "o")

ax.set_xscale("log")
ax.set_yscale("log")

fig, ax = plt.subplots(figsize=(7, 5), dpi=200)

fs = 14
fs_contour = 11
ticksize = 12
lw = 2
lsize = 11

ax.set_xlabel('$V_{add}$ [m/s]', fontsize=fs)
ax.set_ylabel('$R_{stop}$ [AU]', fontsize=fs, color="b")

ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('$< \\rho(R_{stop}) >$ [g/cm$^{-3}$]', fontsize=fs, color="r")

#major_tick = np.arange(-0.5, 3.1, 0.5)
#ax.set_yticks(major_tick)
#minor_ticks = np.arange(-0.5, 3.1, 0.25)
#ax.set_yticks(minor_ticks, minor=True)

ax.plot(Vadd_list, Rstops, color="b")
ax2.plot(Vadd_list, densities, color="r")

ax.tick_params(labelsize=ticksize)
ax2.tick_params(labelsize=ticksize)

#ax.legend(loc=4, title=r"$\eta$", fontsize=lsize, framealpha=1)  # , bbox_to_anchor=(0.05,1.))
plt.tight_layout()

####### DIFFERENT XI ADIABATIC

Ro = 10  # pc
Mo = 1e4  # Msol
CI = (1, mean_density(Ro, Mo))  # initial conditions

TurbCascade = TurbulentCascade(name='scale-free', Vo=1000, eta=0.38)
EOS = EquationOfState(name='adiabatic', temperature=10)

ksis = [1, 0.5, 0, -0.5, -1]
colors = ["r", "r", "k", "b", "b"]
lss = ["-.", "--", "-", "--", "-."]
markers = ["s", "^", "o", "^", "s"]

args = (ksis, TurbCascade, EOS)
print_initial_conditions(CI, Ro, Mo, *args, name="DIFFERENT XI ADIABATIC")

Rfin = 10 * CONVERSION.au_to_pc  # smallest scale to go in pc
R = np.logspace(np.log10(Ro), np.log10(Rfin), 10_000, endpoint=True)

# --------------- Figures
fig = plt.figure(figsize=(12, 4), dpi=50)

gs = fig.add_gridspec(nrows=2, ncols=3)

ax0 = fig.add_subplot(gs[:, :-1])
ax1 = fig.add_subplot(gs[0, -1])
ax2 = fig.add_subplot(gs[1, -1])

axs = [ax0, ax1, ax2]

fontsize = 14
ticksize = 12
padding = 10

for ax in axs:
    ax.set_xscale('log')
for ax in axs[1:]:
    ax.set_yscale('log')

axs[0].set_xlabel('Size R [pc]', fontsize=fontsize, labelpad=padding)
axs[0].set_ylabel('Fragmentation rate $\phi$', fontsize=fontsize, labelpad=padding)

axs[1].set_xlabel('R [pc]', fontsize=fontsize, labelpad=padding)
axs[1].set_ylabel('$<\\rho>$ [g/cm$^{-3}$]', fontsize=fontsize, labelpad=padding)

axs[2].set_xlabel('$<\\rho>$ [g/cm$^{-3}$]', fontsize=fontsize, labelpad=padding)
axs[2].set_ylabel('T [K]', fontsize=fontsize, labelpad=padding)

idxs_r = R > 10 * CONVERSION.au_to_pc  # 270 ksi = 0.5
for ksi, color, ls, marker in zip(ksis, colors, lss, markers):
    xi = MassTransferRate(name='constant', ksi0=ksi)  # mass transfer rate
    args = (xi, TurbCascade, EOS)  # xi, TurbCascade, EOS

    phi, density = solve_fragODE(CI, R, *args, filling_factor=-3, additional_support=0)

    rhos_cgs = density / CONVERSION.pc_to_cm ** 3 * CONVERSION.Msol_to_g

    axs[0].plot(R, phi, ls=ls, color=color)

    rstop = Rstop(R, phi)
    #print(ksi, get_number(1, R, phi)[R == Rstop])
    idxs = R > rstop
    axs[1].plot(R[idxs], rhos_cgs[idxs], ls=ls, color=color, label=f"{np.round(ksi, 1)}")

    temperature = EOS.get_temperature(density)
    if ksi == ksis[0]:
        axs[2].plot(rhos_cgs, temperature, ls="-", color="k")

    # plot Rstops
    mkfc = color
    mkec = "k"

    axs[0].plot(rstop, phi[R == rstop], marker=marker, color=mkec, ms=8)
    axs[1].plot(rstop, rhos_cgs[R == rstop], marker=marker, color=mkec, ms=8)
    axs[2].plot(rhos_cgs[R == rstop], temperature[R == rstop], marker=marker, color=mkec, ms=8)

    axs[0].plot(rstop, phi[R == rstop], marker=marker, color=mkfc, ms=6)
    axs[1].plot(rstop, rhos_cgs[R == rstop], marker=marker, color=mkfc, ms=6)
    axs[2].plot(rhos_cgs[R == rstop], temperature[R == rstop], marker=marker, color=mkfc, ms=6, ls="none",
                label=f"{np.round(ksi, 1)}")

    ## customisation for sub/supersonic
    delta_c = get_threshold(R, density, EOS, TurbCascade)

    idx = argrelextrema(-delta_c, np.less)[0][0]
    axs[0].plot(R[idx], phi[idx], "x", color=color)

## customisation for isothermal
EOS = EquationOfState(name='isothermal', temperature=10)
xi = MassTransferRate(name='constant', ksi0=0)  # mass transfer rate
args = (xi, TurbCascade, EOS)  # xi, TurbCascade, EOS

phi, density = solve_fragODE(CI, R, *args, filling_factor=-3, additional_support=0)
axs[0].plot(R, phi, ls=":", color="k")

#axs[1].legend(loc=1, frameon=False, title="$\\xi$", fontsize=fontsize, title_fontsize='x-large')
#axs[2].legend(loc=2, frameon=False, title="$\\xi$", fontsize=fontsize, title_fontsize='x-large')

##### esthetics

ax2 = axs[0].twiny()  # instantiate a second axes that shares the same x-axis

ax2.set_xscale('log')
ax2.set_xlabel('R [AU]', fontsize=fontsize, labelpad=padding)
ax2.set_xlim([Rfin * CONVERSION.pc_to_au, Ro * CONVERSION.pc_to_au])

axs[0].set_xlim([Rfin, Ro])
axs[0].set_ylim([-0.5, 3])

axs[1].set_xlim([Rfin, Ro])
axs[1].set_ylim([1e-22, 1e-12])

axs[2].set_xlim([1e-18, 1e-12])
axs[2].set_ylim([7, 100])

axs[0].fill_between([Rfin, Ro],
                    [-3.5, -3.5], color="k", alpha=0.2, hatch="//")

for idx in argrelextrema((1 / 3 - np.gradient(np.log(temperature), np.log(density))) ** 2, np.less)[0]:
#if idx < 5000:
    axs[2].axvline(rhos_cgs[idx],
                   ls="--", color="k", alpha=0.8)

major_tick = np.logspace(-20, -12, 9)
axs[1].set_yticks(major_tick)

x = np.array([5e-4, 5e-3])
y = 5e-20 * x ** (-2)
axs[1].plot(x, y, color='k', alpha=0.5)
axs[1].text(2e-3, 5e-14, "$\\rho \propto R^{-2}$", fontsize=ticksize + 1, color="k", alpha=0.75)

axs[0].tick_params(labelsize=ticksize)
axs[1].tick_params(labelsize=ticksize)
axs[2].tick_params(labelsize=ticksize)
ax2.tick_params(labelsize=ticksize)

axs[0].text(-0.2, 1.15, "(a)", transform=axs[0].transAxes, weight="bold")
axs[1].text(-0.2, 1.15, "(b)", transform=axs[1].transAxes, weight="bold")
axs[2].text(-0.2, 1.15, "(c)", transform=axs[2].transAxes, weight="bold")

#print("density 3", mean_density(10, 1e3) * CONVERSION.Msol_to_g / CONVERSION.pc_to_cm**3)
#print("density 4", mean_density(10, 1e4) * CONVERSION.Msol_to_g / CONVERSION.pc_to_cm**3)
#print("density 5", mean_density(10, 1e5) * CONVERSION.Msol_to_g / CONVERSION.pc_to_cm**3)

#plt.tight_layout()

plt.show()