import numpy as np
import scipy.integrate

def SelectFragmentNumber(x, size, probabilities):
    return np.random.choice(x, size=size, p=probabilities)

def SelectFragmentMass(M, n, eff, psi, N = None):
    p_mass = 1 / (psi + n[n != 1] - 1)

    M_next = M[:] * eff
    M_next[n != 1] = M_next[n != 1] * p_mass * psi
    M_next = np.append(M_next, np.repeat([M_next[n != 1] / psi], (n[n != 1] - 1)))

    if N is None:
        return M_next

    N_next = N + n - 1
    N_next = np.append(N_next, np.repeat([N[n != 1] + n[n != 1] - 1], (n[n != 1] - 1)))
    return M_next, N_next

def floor(x):
    return int(x)

def ceil(x):
    return int(x) + 1

def Pfunc_binary(Nprod, **kwargs):
    return [
        (floor(Nprod), 1 - (Nprod - int(Nprod))),
        (ceil(Nprod), Nprod - int(Nprod))
            ]

def Mfunc(nl, efficiency, partition, **kwargs):
    if nl == 0:
        return [(0, 1)]

    elif partition in (0, 1) or nl==1:
        return [(nl, efficiency / nl)]

    #return [(1, efficiency * partition), (nl - 1, efficiency * (1 - partition) / (nl - 1))]
    #return [(1, efficiency * partition / (nl + 1)), (nl - 1, efficiency / (nl + 1))]
    return [
        (1, efficiency * partition / (partition + nl - 1)),
        (nl - 1, efficiency / (partition + nl - 1))
    ]

def Pfunc_poisson(Nprod, xmin=0, xmax=4, **kwargs):
    from scipy.special import factorial

    def Poisson(mu, xmax):
        k = np.arange(0, xmax + 1, 1)
        p = np.exp(-mu) * mu ** k / factorial(k)
        return p / np.sum(p)

    p_k = Poisson(Nprod, xmax)
    return [(k, p_k[i])
            for i, k in enumerate(range(xmin, xmax + 1))
            ]


def _lambda_0(mu, *args):
    Nprod, xmin, xmax = args[0]
    m = 0
    for number, proba in Pfunc_poisson(mu, xmin, xmax):
        m += number * proba
    return Nprod - m


def pseudoPoisson(Nprod, r, kmin=1,  **kwargs):
    kmax = min( int(r ** 3), 100) + 10
    #print(Nprod)
    #print(r, kmax)
    #print("r, kmax, Nprod", r, kmax, Nprod)
    from scipy import optimize

    sol = optimize.root(_lambda_0, x0=[0], args=[Nprod, kmin, kmax], method='lm')

    """
    m = 100
    n_virt = 0
    for n in np.linspace(1, min(kmax, 100), 10_000):
        poiss = Pfunc_poisson(n, kmin, min(kmax, 100))
        norm = np.sum(y for _, y in poiss)
        mean = np.sum([k * pk/norm for k, pk in poiss])
        m = min(m, Nprod - mean)
        
        if m < Nprod - mean:
            n_virt
        
    print(m, sol.x)
    """

    if kmax == 2:
        pdf = Pfunc_binary(Nprod)
    elif Nprod < 15:
        pdf = Pfunc_poisson(sol.x, kmin, kmax)
    else:
        pdf = Pfunc_poisson(Nprod, kmin, kmax)

    norm = np.sum(y for _, y in pdf)

    #print(Nprod, kmax)
    #print( Nprod, np.sum([(k * pk / norm) for k, pk in pdf]) )

    #poiss = Pfunc_poisson(m, kmin, min(kmax, 100))
    #norm = np.sum(y for _, y in poiss)
    #print( Nprod, np.sum([(k * pk / norm) for k, pk in poiss]) )
    #print("2", _lambda_0(2, [Nprod, kmin, kmax]))
    #print("Nprod", _lambda_0(Nprod, [Nprod, kmin, kmax]), kmin, kmax )
    #if Nprod > 15:
        #print(Pfunc_poisson(Nprod, kmin, min(kmax, 100)))
        #print(np.sum([(k * pk) for k, pk in Pfunc_poisson(Nprod, kmin, min(kmax, 100))]))
        #print(norm)
        #print([y for _, y in Pfunc_poisson(Nprod, kmin, min(kmax, 100))])
        #print([k * pk for k, pk in Pfunc_poisson(Nprod, kmin, min(kmax, 100))])
    return [(k, pk/norm)
            for k, pk in pdf
            ]

def gravoturb(Nprod, mass, **kwargs):
    return pseudoPoisson(Nprod, **kwargs)

def number_produced(R, alphas, No=1):
    neff = scipy.integrate.trapezoid(alphas / R, R) # R increases
    return No * np.exp(neff)

def effective_efficiency(R, ksi):
    return np.exp(scipy.integrate.trapezoid(ksi / R, R)) # R increases

def partition(R, partition):
    return partition