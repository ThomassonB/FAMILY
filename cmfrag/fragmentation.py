import scipy
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt

class CONSTANTS:
    KB = 1.380649e-23  # Boltzmann constante in J K−1
    MMW = 2.33  # mean molecular weight
    MH = 1.67265e-27  # hydrogen mass in kg
    G = 6.67430e-11  # Gravity constant in m3kg−1s−2

    bj = np.sqrt(np.pi) / 2
    aj = 4 / 3 * np.pi * bj ** 3
    cj = np.pi

class CONVERSION:
    pc_to_m = 3.0857E16 # 1 pc = 3e16 m
    pc_to_cm = 3.0857E18 # 1 pc = 3e18 cm
    Msol_to_kg = 1.98847E30 # 1Msol = 2e30 kg
    Msol_to_g = 1.98847E33 # 1Msol = 2e33 g
    kg_to_g = 1e3
    pc_to_au = 206264.806  # 1 Parsec = 206264.806 Unité astronomique
    yr_to_sec = 60 * 60 * 24 * 365.25

    m_to_pc = 1 / pc_to_m
    cm_to_pc = 1 / pc_to_cm
    kg_to_Msol = 1 / Msol_to_kg
    g_to_Msol = 1 / Msol_to_g
    g_to_kg = 1e-3
    au_to_pc = 1 / pc_to_au
    sec_to_yr = 1 / yr_to_sec

class EquationOfState:
    _registry = {}

    def __init_subclass__(cls, name, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[name] = cls

    def __new__(cls, name: str, **kwargs):
        subclass = cls._registry[name]
        obj = object.__new__(subclass)
        return obj

    def set_initial_temperature(self, temperature):
        self._initial_temperature = temperature

class Isothermal(EquationOfState, name="isothermal"):

    def __init__(self, *args, **kwargs):
        self.set_initial_temperature(kwargs.get("temperature"))

    def get_temperature(self, rho):
        return self._initial_temperature

    def get_logT_logR_derivative(self, rho):
        return 0

class Adiabatic(EquationOfState, name="adiabatic"):

    def __init__(self, gamma=5 / 3, rho_ad=1e-13, *args, **kwargs):  # g/cm^-3
        self.set_initial_temperature(kwargs.get("temperature"))

        self.rho_ad = rho_ad * CONVERSION.g_to_Msol / CONVERSION.cm_to_pc ** 3

        self.gamma = gamma

    def get_temperature(self, rho):
        """
        :param rho: density in Msol/pc**3
        :return:
        """
        To = self._initial_temperature
        return To * (1 + (rho / self.rho_ad) ** (self.gamma - 1))

    def get_logT_logR_derivative(self, rho):
        idx = self.gamma - 1
        return idx * (rho / self.rho_ad) ** idx / (1 + (rho / self.rho_ad) ** idx)


class CompositeLH2018(EquationOfState, name="composite_Lee_Hennebelle2018"):

    def __init__(self, gamma1=5 / 3, rho_ad1=1e-13,
                 gamma2=7 / 5, rho_ad2=3e-12,
                 gamma3=1.1, rho_ad3=3e-9, *args, **kwargs):
        self.set_initial_temperature(kwargs.get("temperature"))

        self.rho_ad1 = rho_ad1 * CONVERSION.g_to_Msol / CONVERSION.cm_to_pc ** 3
        self.rho_ad2 = rho_ad2 * CONVERSION.g_to_Msol / CONVERSION.cm_to_pc ** 3
        self.rho_ad3 = rho_ad3 * CONVERSION.g_to_Msol / CONVERSION.cm_to_pc ** 3

        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3

    def get_temperature(self, rho, n=3):
        To = self._initial_temperature

        num = (rho / self.rho_ad1) ** (self.gamma1 - 1)
        term_denom1 = (rho / self.rho_ad2) ** ((self.gamma1 - self.gamma2) * n)
        term_denom2 = (1 + (rho / self.rho_ad3) ** ((self.gamma2 - self.gamma3) * n))
        denom = (1 + term_denom1 * term_denom2) ** (1 / n)

        return To * (1 + num / denom)

    def get_logT_logR_derivative(self, rho, n=3):
        dT_drho = self._dT_drho(rho, n)
        return rho / self.get_temperature(rho) * dT_drho

    def _dT_drho(self, rho, n=3):
        To = self._initial_temperature

        prefactor = lambda rho_ad, gamma1, gamma2, n: (gamma1 - gamma2) / rho_ad * n
        root_func = lambda rho, rho_ad, gamma1, gamma2, n: (rho / rho_ad) ** ((gamma1 - gamma2) * n)
        droot_func = lambda rho, rho_ad, gamma1, gamma2, n: (rho / rho_ad) ** ((gamma1 - gamma2) * n - 1)

        gamma1, gamma2, gamma3 = self.gamma1, self.gamma2, self.gamma3
        rho_ad1, rho_ad2, rho_ad3 = self.rho_ad1, self.rho_ad2, self.rho_ad3

        def _Y(rho, n=3):
            term1 = root_func(rho, rho_ad2, gamma1, gamma2, n)
            term2 = root_func(rho, rho_ad3, gamma2, gamma3, n)

            return term1 * (1 + term2)

        def _dY_drho(rho, n=3):
            term1_1 = prefactor(rho_ad2, gamma1, gamma2, n) * droot_func(rho, rho_ad2, gamma1, gamma2, n)
            term1_2 = 1 + root_func(rho, rho_ad3, gamma2, gamma3, n)

            term2_1 = prefactor(rho_ad3, gamma2, gamma3, n) * droot_func(rho, rho_ad2, gamma1, gamma2, n)
            term2_2 = droot_func(rho, rho_ad3, gamma2, gamma3, n)

            return term1_1 * term1_2 + term2_1 * term2_2

        X = root_func(rho, rho_ad1, gamma1, 1, 1)
        dX_drho = prefactor(rho, gamma1, 1, X)

        Y = _Y(rho, n=n)
        dY_drho = _dY_drho(rho, n=n)

        term1 = dX_drho / (1 + Y) ** (1 / n)
        term2 = - X / n * dY_drho / (1 + Y) ** (1 / n + 1)
        return To * (term1 + term2)

class TurbulentCascade:
    _registry = {}

    def __init_subclass__(cls, name, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[name] = cls

    def __new__(cls, name: str, **kwargs):
        subclass = cls._registry[name]
        obj = object.__new__(subclass)
        return obj

class ScaleFree(TurbulentCascade, name="scale-free"):

    def __init__(self, *args, **kwargs):
        """ **kwargs should contain Vo and eta """
        self.set_initial_velocity(kwargs.get("Vo"))
        self.set_power(kwargs.get("eta"))

    def set_initial_velocity(self, velocity):
        self._initial_velocity = velocity

    def set_power(self, power):
        self._power = power

    def get_velocity(self, R):
        return self._initial_velocity * R ** self._power

    def get_dlnVt_dlnR_derivative(self, R):
        return self._power

class SaturatedScaleFree(TurbulentCascade, name="saturated scale-free"):

    def __init__(self, *args, **kwargs):
        """ **kwargs should contain Vo and eta """
        self.set_initial_velocity(kwargs.get("Vo"))
        self.set_power(kwargs.get("eta"))
        self.set_offset_velocity(kwargs.get("Vadd"))

    def set_initial_velocity(self, velocity):
        self._initial_velocity = velocity

    def set_power(self, power):
        self._power = power

    def set_offset_velocity(self, Vadd):
        self._offset_velocity = Vadd

    def get_velocity(self, R):
        return self._initial_velocity * R ** self._power + self._offset_velocity

    def get_dlnVt_dlnR_derivative(self, R):
        return self._power

class MassTransferRate:
    _registry = {}

    def __init_subclass__(cls, name, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[name] = cls

    def __new__(cls, name: str, **kwargs):
        subclass = cls._registry[name]
        obj = object.__new__(subclass)
        return obj

class ConstantXI(MassTransferRate, name="constant"):
    def __init__(self, ksi0, *args, **kwargs):
        self.set_xi(ksi0)

    def set_xi(self, ksi0):
        self._ksi0 = ksi0

    def get_xi(self, R):
        return self._ksi0


class AdHocXI(MassTransferRate, name="adhoc"):
    def __init__(self, *args, **kwargs):
        pass
        #self.set_xi(ksi0)
        #self._Ro = kwargs.get('Ro')
        #self._ksimax= kwargs.get('ksimax')

    def get_xi(self, R):
        Rstops, ksis = np.loadtxt("low_boundary_xi_vs_R", unpack=True)
        #print(np.interp(R, Rstops, ksis))
        #print(np.interp(R, Rstops[::-1], ksis[::-1]))
        return np.interp(R, Rstops, ksis)
        #return self._ksi0 + (self._ksimax - self._ksi0) * np.exp(-R / self._Ro)

def mass_transfer_rate_adhoc(R, ksi0, Ro=0.01, gamma = -1.5):
    return ksi0 * (1 - (R / Ro) ** gamma)

def quadratic_velocity(weights, squared_velocities):
    func = lambda x: x[0] * x[1]
    return sum(map(func, zip(weights, squared_velocities))) # faster than numpy ???

def mean_density(R, M):
    geom = np.pi / 6 #4 / 3 * np.pi
    return M / R ** 3 / geom

def sound_speed2(T):
    return CONSTANTS.KB * T / CONSTANTS.MMW / CONSTANTS.MH

def critical_delta(R, rho, V2):
    R_m = R * CONVERSION.pc_to_m
    rho_mkg = rho * CONVERSION.Msol_to_kg / CONVERSION.pc_to_m ** 3
    return np.log(CONSTANTS.cj * V2 / CONSTANTS.G / R_m ** 2 / rho_mkg)

def variance(Mach, b=0.25):
    sigma_0 = np.log(1 + b * Mach ** 2)
    return sigma_0

#def variance_powerspectrum(Mach, b=0.25, n = 11/3, Ro = 10):
#    sigma_0 = np.log(1 + b * Mach ** 2)
#    return sigma_0 * (1 - (R/Ro)**(n - 3))

#def variance_powerspectrum_additional(R, Mach, b=0.25, n = 11/3, Ro = 100):
#    sigma_0 = np.log(1 + b * Mach ** 2)
#    return - sigma_0 / 2 * (n - 3) * (R / Ro)**(n - 3) / (1 - (R / Ro)**(n - 3))

def critical_mass(R, V2):
    par = np.pi**2 / 6
    R_m = R * CONVERSION.pc_to_m
    return R_m / CONSTANTS.G * V2 * par

####################### AUXILIARY FUNCTIONS

def _u(delta_c, sigma):
    return (delta_c + sigma ** 2 / 2) / sigma / np.sqrt(2)

def _v(delta_c, sigma):
    return (delta_c - sigma ** 2 / 2) / sigma / np.sqrt(2)

def _Vt2_chap(Cs2, Vt2):
    return Vt2 / (3 * Cs2 + Vt2)

def _Cs2_chap(Cs2, Vt2):
    return 3 * Cs2 / (3 * Cs2 + Vt2)

def _M2_chap(mach, b=0.25):
    return b * mach ** 2 / (1 + b * mach ** 2)

def _probability_repartition(u):
    return 1 / 2 * (1 - scipy.special.erf(u))

########################

def _Ap(u, sigma):
    return - np.exp(-u ** 2) / _probability_repartition(u) / sigma / np.sqrt(2 * np.pi)

def _Ath(Ap, Cs2_chap, dlnT_dlnrho, comp):
    return Ap * (1 - dlnT_dlnrho * (Cs2_chap + comp))

#def _Ath_raw_powerspectrum(Ap, Cs2_chap, dlnT_dlnrho, comp):
#    return Ap * (1 - dlnT_dlnrho * (Cs2_chap))

def _Aturb(Ap, Vt2_chap, eta, comp):
    return Ap * (1 - eta * (Vt2_chap - comp))

#def _Aturb_powerspectrum(Ap, Vt2_chap, eta, comp, v, additional):
#    return Ap * (1 - eta * (Vt2_chap - comp)) - Ap * np.sqrt(2) * v * additional

#def _Aturb_raw_powerspectrum(Ap, Vt2_chap, eta, comp, v, additional):
#    return Ap * (1 - eta * (Vt2_chap)) - Ap * np.sqrt(2) * v * additional

def _Aalt(Cs2_chap, dlnT_dlnrho, comp, Vt2_chap, eta):
    return (1 - eta * (Vt2_chap - comp)) / (1 - dlnT_dlnrho * (Cs2_chap + comp))

#def _Aalt_raw_powerspectrum(Cs2_chap, dlnT_dlnrho, comp, Vt2_chap, eta, v, additional):
#    return ((1 - eta * (Vt2_chap)) - np.sqrt(2) * v * additional) / (1 - dlnT_dlnrho * (Cs2_chap))

######################## COMPUTE THE FRAGMENTATION RATE

def _frag_condition(phi, xi, delta_c):
    #if phi - xi - 3 >= 0:# and delta_c <= 0:
    #    return 0
    return phi

def _phi(ksi, delta_c, Ap, Ath, Aturb, A_alt=0, filling_factor=-3, **kwargs):
    if np.isinf(Ap) or np.isnan(Ap):

        term2 = ksi + 3
        term3 = - 2 * A_alt

        phi = term2 + term3
        return _frag_condition(phi, ksi, delta_c)

    denom = 1 - Ath

    term1 = - filling_factor
    term2 = - (ksi + 3) * Ath
    term3 = 2 * Aturb

    phi = (term1 + term2 + term3) / denom
    return _frag_condition(phi, ksi, delta_c)

def _perso_fragment(CI, R, *args, filling_factor = -3, additional_support = 0):
    xi, TurbCascade, EOS = args
    N, rho = CI
    Mo = rho * max(R)**3 * np.pi/6
    M = Mo

    M_arr = []
    T_arr, rho_arr, Mach_arr = [], [], []
    phis = []

    PR = [[], []]

    for r in R:
        idx = np.where(R == r)[0][0]

        rho = mean_density(r, M)
        T = EOS.get_temperature(rho)
        dlnT_dlnrho = EOS.get_logT_logR_derivative(rho)

        Cs2 = sound_speed2(T)
        Vt2 = TurbCascade.get_velocity(r) ** 2
        eta = TurbCascade.get_dlnVt_dlnR_derivative(r)

        weights = (1, 1 / 3, 1)
        squared_velocities = (Cs2, Vt2, additional_support ** 2)
        V2 = quadratic_velocity(weights, squared_velocities)

        Mach = np.sqrt(Vt2 / Cs2)

        Vt2_chap = _Vt2_chap(Cs2, Vt2)
        Cs2_chap = _Cs2_chap(Cs2, Vt2)
        M2_chap = _M2_chap(Mach)

        delta_c = critical_delta(r, rho, V2)
        sigma = np.sqrt(variance(Mach))

        u = _u(delta_c, sigma)
        v = _v(delta_c, sigma)

        comp = v * M2_chap / np.sqrt(2) / sigma

        Ap = _Ap(u, sigma)
        Ath = _Ath(Ap, Cs2_chap, dlnT_dlnrho, comp)
        Aturb = _Aturb(Ap, Vt2_chap, eta, comp)

        A_alt = _Aalt(Cs2_chap, dlnT_dlnrho, comp, Vt2_chap, eta)

        ksi = xi.get_xi(r)
        phi = _phi(ksi, delta_c, Ap, Ath, Aturb, A_alt=A_alt, filling_factor=filling_factor)

        if r == max(R):
            M = Mo
        else:
            neff = np.exp(scipy.integrate.trapezoid((np.array(phis) - ksi)/ R[:idx], R[:idx]))
            M = Mo * neff

        M_arr.append(M)
        T_arr.append(T)
        rho_arr.append(rho)
        Mach_arr.append(Mach)
        phis.append(phi)

        PR[0].append(delta_c)
        PR[1].append(sigma)

    M_arr = np.array(M_arr)
    T_arr = np.array(T_arr)
    rho_arr = np.array(rho_arr)
    Mach_arr = np.array(Mach_arr)
    phis = np.array(phis)

    return M_arr, T_arr, rho_arr, Mach_arr, phis, PR

def fragment(CI, R, *args):
    xi, TurbCascade, EOS, filling_factor, additional_support = args

    N, rho = CI

    T = EOS.get_temperature(rho)
    dlnT_dlnrho = EOS.get_logT_logR_derivative(rho)

    Cs2 = sound_speed2(T)
    Vt2 = TurbCascade.get_velocity(R)**2
    eta = TurbCascade.get_dlnVt_dlnR_derivative(R)

    weights = (1, 1/3, 1)
    squared_velocities = (Cs2, Vt2, additional_support ** 2)
    V2 = quadratic_velocity(weights, squared_velocities)

    delta_c = critical_delta(R, rho, V2)

    Mach = np.sqrt(Vt2 / Cs2)
    sigma = np.sqrt(variance(Mach))

    u = _u(delta_c, sigma)
    v = _v(delta_c, sigma)

    Vt2_chap = _Vt2_chap(Cs2, Vt2)
    Cs2_chap = _Cs2_chap(Cs2, Vt2)
    M2_chap = _M2_chap(Mach)

    comp = v * M2_chap / np.sqrt(2) / sigma

    Ap = _Ap(u, sigma)
    Ath = _Ath(Ap, Cs2_chap, dlnT_dlnrho, comp)
    Aturb = _Aturb(Ap, Vt2_chap, eta, comp)
    A_alt = _Aalt(Cs2_chap, dlnT_dlnrho, comp, Vt2_chap, eta)

    #additional = variance_powerspectrum_additional(R, Mach)
    #Ath = _Ath_raw_powerspectrum(Ap, Cs2_chap, dlnT_dlnrho, comp)
    #Aturb = _Aturb_raw_powerspectrum(Ap, Vt2_chap, eta, comp, v, additional)
    #A_alt = _Aalt_raw_powerspectrum(Cs2_chap, dlnT_dlnrho, comp, Vt2_chap, eta, v, additional)
    ksi = xi.get_xi(R)

    phi = _phi(ksi, delta_c, Ap, Ath, Aturb, A_alt=A_alt, filling_factor=filling_factor)
    drhodR = (phi - ksi - 3) * rho / R
    #print(np.array([- phi * N / R, drhodR]))

    return np.array([- phi * N / R, drhodR])

def phi_derivative(R, N):
    return - np.gradient(np.log(N), R) * R

######################## FUNCTIONS TO COMPUTE PHYSICAL PROPERTIES

def solve_fragODE(initial_condition, R, *args, filling_factor = -3, additional_support = 0):
    args = *args, filling_factor, additional_support
    sol = integrate.odeint(fragment, initial_condition, R, args=args)
    return phi_derivative(R, sol[:, 0]), sol[:, 1]

def get_mass(M0, R, fragmentation_rate, masstransfer_rate):
    integrand = (fragmentation_rate - masstransfer_rate) / R
    integral = integrate.cumulative_trapezoid(integrand, R, initial=0)
    return M0 * np.exp(integral)

def get_efficiency(R, masstransfer_rate):
    integral = integrate.cumulative_trapezoid(masstransfer_rate / R, R, initial=0)
    return np.exp(-integral)

def get_number(N0, R, fragmentation_rate):
    integral = integrate.cumulative_trapezoid(fragmentation_rate / R, R, initial=0)
    return N0 * np.exp(-integral)

def get_number_scalefree(N0, R, R0, fragmentation_rate):
    return N0 * (R/R0) ** -fragmentation_rate

def get_efficiency_scalefree(R, R0, masstransfer_rate):
    return (R/R0) ** -masstransfer_rate

def get_threshold(R, rho, EOS, TurbCascade, additional_support = 0):
    T = EOS.get_temperature(rho)
    Cs2 = sound_speed2(T)

    Vt2 = TurbCascade.get_velocity(R) ** 2

    weights = (1, 1 / 3, 1)
    squared_velocities = (Cs2, Vt2, np.ones_like(R) * additional_support ** 2)
    V2 = quadratic_velocity(weights, squared_velocities)

    return critical_delta(R, rho, V2)

def get_mass_threshold(R, rho, EOS, TurbCascade, additional_support = 0):
    T = EOS.get_temperature(rho)
    Cs2 = sound_speed2(T)

    Vt2 = TurbCascade.get_velocity(R) ** 2

    weights = (1, 1 / 3, 1)
    squared_velocities = (Cs2, Vt2, np.ones_like(R) * additional_support ** 2)
    V2 = quadratic_velocity(weights, squared_velocities)

    return critical_mass(R, V2)

def get_cumulative_pdf(R, rho, EOS, TurbCascade, additional_support = 0):
    delta_c = get_threshold(R, rho, EOS, TurbCascade, additional_support)

    T = EOS.get_temperature(rho)
    Cs2 = sound_speed2(T)
    Vt2 = TurbCascade.get_velocity(R) ** 2

    Mach = np.sqrt(Vt2 / Cs2)
    sigma = np.sqrt(variance(Mach))

    u = _u(delta_c, sigma)
    return _probability_repartition(u)

def get_NESTs(R, Rstop, fragmentation_rate):
    idxs = R > Rstop
    integrand = fragmentation_rate / R
    integral = integrate.cumulative_trapezoid(integrand[idxs][::-1], R[idxs][::-1], initial=0)
    return np.exp(integral)

def get_BondiHoyle(rho, mass, temperature):
    rho = rho * CONVERSION.Msol_to_kg / CONVERSION.pc_to_m**3
    mass = mass * CONVERSION.Msol_to_kg
    return np.pi * rho * CONSTANTS.G**2 * mass**2 / sound_speed2(temperature)**(3/2)

def BonnorEbert(R, Cs2):
    return 2.4 * R * Cs2 / CONSTANTS.G

def mean_fragmentation_rate(R, phi, R1, R2):
    idxs = np.logical_and(R >= R1, R <= R2)
    integral = integrate.trapezoid(phi[idxs] / R[idxs], R[idxs])
    denom = 1 / (np.log(R1) - np.log(R2))
    return integral * denom

def Rstop(R, phi):
    phi_sign = np.sign(phi)
    #print(phi_sign)
    last_positiv = (np.roll(phi_sign, -1) - phi_sign) != 0

    if np.any(last_positiv):
        return max(R[last_positiv])
    elif np.all(phi_sign == -1):
        return max(R)
    return -1

def display_options(cls):
    out = f"Choose one {cls.__name__}:\n"
    for key in cls._registry.keys():
        out += f"\t{key}\n"
    print(out)

if __name__=='__main__':
    display_options(EquationOfState)
    display_options(TurbulentCascade)
    display_options(MassTransferRate)

    Ro = 10 #* CONVERSION.au_to_pc
    Mo = 1e4  #Msol
    CI = (1, mean_density(Ro, Mo)) #initial conditions
    #print("cm-3", mean_density(Ro, Mo) * CONVERSION.Msol_to_kg / CONVERSION.pc_to_cm**3 / CONSTANTS.MH )
    #print("g.cm-3", mean_density(Ro, Mo) * CONVERSION.Msol_to_g / CONVERSION.pc_to_cm**3 )

    #print("cm-3", 1e-16 / CONSTANTS.MH * 1e10)
    #print("g.cm-3", 1e-13)

    #print("cm", 1e17 * CONVERSION.cm_to_pc)
    #print("g/cm3", 1e-20)

    #print("1 Msol", 1e31 * CONVERSION.g_to_Msol)
    #print("1 pc", 6e13 * CONVERSION.cm_to_pc)

    #print(mean_density(6e14, 1e31), mean_density(8.68e-5, 5.1e-4) * CONVERSION.Msol_to_g / CONVERSION.pc_to_cm**3)
    Rfin = 10 * CONVERSION.au_to_pc #smallest scale to go in pc
    R = np.logspace(np.log10(Ro), np.log10(Rfin), 10_000, endpoint=True)

    xi = MassTransferRate(name='constant', ksi0 = 0) #mass transfer rate

    TurbCascade = TurbulentCascade(name='scale-free', Vo = 1000, eta = 0.38)
    EOS = EquationOfState(name='adiabatic', temperature=10)

    args = (xi, TurbCascade, EOS)  # xi, TurbCascade, EOS

    ####### COMPUTE PHI AND DENSITY
    vadd = 0
    phi, density = solve_fragODE(CI, R, *args, filling_factor = -3, additional_support=vadd)
    mass = get_mass(Mo, R, phi, xi.get_xi(R))
    plt.plot(mass, phi, "-k")

    #plt.plot(R, mass, "-r")

    #M_arr, T_arr, rho_arr, Mach_arr, phis, PR = _perso_fragment(CI, R, *args, filling_factor=-3, additional_support=vadd)
    #plt.plot(R, phis, "--r")

    #plt.yscale('log')
    plt.xscale('log')
    #plt.plot(R, np.ones_like(R) * xi.get_xi(R), "--k")
    #plt.plot(R, phi - xi.get_xi(R), "-.k")

    # xi = MassTransferRate(name='adhoc') #mass transfer rate
    # args = (xi, TurbCascade, EOS)  # xi, TurbCascade, EOS
    # phi, density = solve_fragODE(CI, R, *args, filling_factor=-3, additional_support=0)
    # mass = get_mass(Mo, R, phi, xi.get_xi(R))
    # plt.plot(R, phi, "-r")
    #plt.plot(R, xi.get_xi(R), "--r")

    #plt.plot(R, phi - xi.get_xi(R), "-.r")

    plt.xscale('log')
    #plt.yscale('log')

    ####### GET PHYSICAL PROPERTIES
    #       mass = get_mass(Mo, R, phi, xi)
    #       number = get_number(CI[0], R, phi)
    #       delta_c = get_threshold(R, density, EOS, TurbCascade)

    ####### GET PHYSICAL PROPERTIES FOR SCALE-FREE FRAGMENTATION
    Rmin = 1000  # AU
    Rmax = 6400 # AU
    scale_free_number = get_number_scalefree(1, Rmin, Rmax, fragmentation_rate = 1)
    print(scale_free_number)

    scale_free_eff = get_efficiency_scalefree(Rmin, Rmax, -0.7)
    print(scale_free_eff)

    rstop = Rstop(R, phi)
    rhostop = density[R == rstop]
    mc = get_mass_threshold(rstop, rhostop, EOS, TurbCascade)
    print("critical mass at the end:", mc * CONVERSION.kg_to_Msol)
    print("density at the end:", rhostop * CONVERSION.Msol_to_g / CONVERSION.pc_to_cm ** 3)

    print(critical_mass(rstop, sound_speed2(EOS.get_temperature(rhostop))) * CONVERSION.kg_to_Msol)
    plt.show()