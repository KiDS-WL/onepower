import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from scipy.integrate import simpson, solve_ivp, quad
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='colossus')

# TO-DO: Move these to the main class

def get_halo_collapse_redshifts(M, z, dc, g, cosmo, mf):
    """
    Calculate halo collapse redshifts according to the Bullock et al. (2001) prescription.
    """
    gamma = 0.01
    a = cosmo.scale_factor(z)
    zf = np.zeros_like(M)
    for iM, _M in enumerate(M):
        Mc = gamma * _M
        Rc = mf.filter.mass_to_radius(Mc, mf.mean_density0)
        sigma = mf.normalised_filter.sigma(Rc)
        fac = g(a) * dc / sigma
        if fac >= g(a):
            af = a  # These haloes formed 'in the future'
        else:
            af_root = lambda af: g(af) - fac
            af = root_scalar(af_root, bracket=(1e-3, 1.)).root
        zf[iM] = -1.0 + 1.0 / af
    return zf

def _Omega_m(a, Om, Ode, Ok, cosmo):
    """
    Evolution of Omega_m with scale-factor ignoring radiation
    Massive neutrinos are counted as 'matter'.
    """
    return Om * a**-3 / _Hubble2(a, Om, Ode, Ok, cosmo)

def _Hubble2(a, Om, Ode, Ok, cosmo):
    """
    Squared Hubble parameter ignoring radiation.
    Massive neutrinos are counted as 'matter'.
    """
    z = -1.0 + 1.0 / a
    H2 = Om * a**-3 + Ode * cosmo.de_density_scale(z) + Ok * a**-2
    return H2

def _AH(a, Om, Ode, cosmo):
    """
    Acceleration parameter ignoring radiation.
    Massive neutrinos are counted as 'matter'.
    """
    z = -1.0 + 1.0 / a
    AH = -0.5 * (Om * a**-3 + (1.0 + 3.0 * cosmo.w(z)) * Ode * cosmo.de_density_scale(z))
    return AH

def get_growth_interpolator(cosmo):
    """
    Solve the linear growth ODE and returns an interpolating function for the solution.
    TODO: w dependence for initial conditions; f here is correct for w=0 only.
    TODO: Could use d_init = a(1+(w-1)/(w(6w-5))*(Om_w/Om_m)*a**-3w) at early times with w = w(a<<1).
    """
    
    # TODO: Add check if w0 and wa exists / if astropy w0waCDM cosmology class is used
    a_init = 1e-4
    Om = cosmo.Om0 + cosmo.Onu0
    Ode = 1.0 - Om
    Ok = cosmo.Ok0
    na = 129  # Number of scale factors used to construct interpolator
    a = np.linspace(a_init, 1., na)

    f = 1.0 - _Omega_m(a_init, Om, Ode, Ok, cosmo)  # Early mass density
    d_init = a_init**(1.0 - 3.0 * f / 5.0)  # Initial condition (~ a_init; but f factor accounts for EDE-ish)
    v_init = (1.0 - 3.0 * f / 5.0) * a_init**(-3.0 * f / 5.0)  # Initial condition (~ 1; but f factor accounts for EDE-ish)
    y0 = (d_init, v_init)

    def fun(a, y):
        d, v = y[0], y[1]
        dxda = v
        fv = -(2.0 + _AH(a, Om, Ode, cosmo) / _Hubble2(a, Om, Ode, Ok, cosmo)) * v / a
        fd = 1.5 * _Omega_m(a, Om, Ode, Ok, cosmo) * d / a**2
        dvda = fv + fd
        return dxda, dvda

    g = solve_ivp(fun, (a[0], a[-1]), y0, t_eval=a).y[0]
    return interp1d(a, g, kind='cubic', assume_sorted=True)


def get_accumulated_growth(a, g):
    """
    Calculates the accumulated growth at scale factor 'a'.
    """
    a_init = 1e-4

    # Eq A5 of Mead et al. 2021 (2009.01858).
    # We approximate the integral as g(a_init) for 0 to a_init<<0.
    missing = g(a_init)
    G, _ = quad(lambda a: g(a) / a, a_init, a) + missing
    return G

def f_Mead(x, y, p0, p1, p2, p3):
    # eq A3 of 2009.01858
    return p0 + p1 * (1.0 - x) + p2 * (1.0 - x)**2.0 + p3 * (1.0 - y)

def dc_Mead(a, Om, f_nu, g, G):
    """
    delta_c fitting function from Mead et al. 2021 (2009.01858).
    All input parameters should be evaluated as functions of a/z.
    """
    # See Table A.1 of 2009.01858 for naming convention
    p1 = [-0.0069, -0.0208, 0.0312, 0.0021]
    p2 = [0.0001, -0.0647, -0.0417, 0.0646]
    a1, a2 = 1, 0
    # Linear collapse threshold
    # Eq A1 of 2009.01858
    dc_Mead = 1.0 + f_Mead(g/a, G/a, *p1) * np.log10(Om)**a1 + f_Mead(g/a, G/a, *p2) * np.log10(Om)**a2
    # delta_c = ~1.686' EdS linear collapse threshold
    dc0 = (3.0 / 20.0) * (12.0 * np.pi)**(2.0 / 3.0)
    return dc_Mead * dc0 * (1.0 - 0.041 * f_nu)

def Dv_Mead(a, Om, f_nu, g, G):
    """
    Delta_v fitting function from Mead et al. 2021 (2009.01858), eq A.2.
    All input parameters should be evaluated as functions of a/z.
    """
    # See Table A.1 of 2009.01858 for naming convention
    p3 = [-0.79, -10.17, 2.51, 6.51]
    p4 = [-1.89, 0.38, 18.8, -15.87]
    a3, a4 = 1, 2

    # Halo virial overdensity
    # Eq A2 of 2009.01858
    Dv_Mead = 1.0 + f_Mead(g/a, G/a, *p3) * np.log10(Om)**a3 + f_Mead(g/a, G/a, *p4) * np.log10(Om)**a4
    Dv0 = 18.0 * np.pi**2.0  # Delta_v = ~178, EdS halo virial overdensity
    return Dv_Mead * Dv0 * (1.0 + 0.763 * f_nu)

def Tk_cold_ratio(k, g, ommh2, h, f_nu, N_nu, T_CMB=2.7255):
    """
    Ratio of cold to matter transfer function from Eisenstein & Hu (1999).
    This can be used to get the cold-matter spectrum approximately from the matter spectrum.
    Captures the scale-dependent growth with neutrino free-streaming scale.
    """
    if f_nu == 0.0:  # Fix to unity if there are no neutrinos
        return 1.0

    pcb = (5.0 - np.sqrt(1.0 + 24. * (1.0 - f_nu))) / 4.0  # Growth exponent for unclustered neutrinos completely
    BigT = T_CMB / 2.7  # Big Theta for temperature
    zeq = 2.5e4 * ommh2 * BigT**(-4)  # Matter-radiation equality redshift
    D = (1.0 + zeq) * g  # Growth normalized such that D=(1.+z_eq)/(1+z) at early times
    q = k * h * BigT**2 / ommh2  # Wave number relative to the horizon scale at equality (equation 5)
    yfs = 17.2 * f_nu * (1.0 + 0.488 * f_nu**(-7.0 / 6.0)) * (N_nu * q / f_nu)**2  # Free streaming scale (equation 14)
    Dcb = (1.0 + (D / (1. + yfs))**0.7)**(pcb / 0.7)  # Cold growth function
    Dcbnu = ((1.0 - f_nu)**(0.7 / pcb) + (D / (1.0 + yfs))**0.7)**(pcb / 0.7)  # Cold and neutrino growth function
    return Dcb / Dcbnu  # Finally, the ratio

def sigmaR_cc(power, k, r):
    rk = np.outer(r, k)
    dlnk = np.log(k[1] / k[0])

    k_space = (3 / rk**3) * (np.sin(rk) - rk * np.cos(rk))
    # we multiply by k because our steps are in logk.
    rest = power * k**3
    integ = rest * k_space**2
    sigma = (0.5 / np.pi**2) * simpson(integ, dx=dlnk, axis=-1)
    return np.sqrt(sigma)


