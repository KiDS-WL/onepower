import numpy as np
import warnings
from astropy.cosmology import Planck15
from scipy.integrate import quad, simpson, solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

from hmf.cosmology.cosmo import astropy_to_colossus
from hmf.halos.mass_definitions import SphericalOverdensity

warnings.filterwarnings('ignore', category=UserWarning, module='colossus')

# TODO: unused
def concentration_halomod(cosmo, mass, z, model, mdef, overdensity, mf, delta_c):
    """
    Calculate concentration given halo mass using the halomod model provided in config.
    """
    mdef = getattr(md, mdef)() if mdef in ['SOVirial'] else getattr(md, mdef)(overdensity=overdensity)
    cm = getattr(conc_func, model)(cosmo=mf, filter0=mf.filter, delta_c=delta_c, mdef=mdef)
    return cm.cm(mass, z)

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

def acceleration_parameter(cosmo, z):
    """
    Calculate the acceleration parameter.
    """
    return -0.5 * (cosmo.Om(z) + (1.0 + 3.0 * cosmo.w(z)) * cosmo.Ode(z))

def _w(a, w0, wa):
    """
    Dark energy equation of state for w0, wa models.
    """
    return w0 + (1. - a) * wa

def _X_w(a, w0, wa):
    """
    Cosmological dark energy density for w0, wa models.
    """
    return a ** (-3. * (1. + w0 + wa)) * np.exp(-3. * wa * (1. - a))

def _Omega_m(a, Om, Ode, Ok, w0, wa):
    """
    Evolution of Omega_m with scale-factor ignoring radiation.
    """
    return Om * a**-3 / _Hubble2(a, Om, Ode, Ok, w0, wa)

def _Hubble2(a, Om, Ode, Ok, w0, wa):
    """
    Squared Hubble parameter ignoring radiation.
    Massive neutrinos are counted as 'matter'.
    """
    H2 = Om * a**-3 + Ode * _X_w(a, w0, wa) + Ok * a**-2
    return H2

def _AH(a, Om, Ode, w0, wa):
    """
    Acceleration parameter ignoring radiation.
    Massive neutrinos are counted as 'matter'.
    """
    AH = -0.5 * (Om * a**-3 + (1. + 3. * _w(a, w0, wa)) * Ode * _X_w(a, w0, wa))
    return AH

def get_growth_interpolator(cosmo):
    """
    Solve the linear growth ODE and returns an interpolating function for the solution.
    TODO: w dependence for initial conditions; f here is correct for w=0 only.
    TODO: Could use d_init = a(1+(w-1)/(w(6w-5))*(Om_w/Om_m)*a**-3w) at early times with w = w(a<<1).
    """
    a_init = 1e-4
    Om = cosmo.Om0 + cosmo.Onu0
    Ode = 1.0 - Om
    Ok = cosmo.Ok0
    w0 = getattr(cosmo, 'w0', -1.0)
    wa = getattr(cosmo, 'wa', 0.0)
    na = 129  # Number of scale factors used to construct interpolator
    a = np.linspace(a_init, 1., na)
    f = 1. - _Omega_m(a_init, Om, Ode, Ok, w0, wa)  # Early mass density
    d_init = a_init**(1. - 3. * f / 5.)  # Initial condition (~ a_init; but f factor accounts for EDE-ish)
    v_init = (1. - 3. * f / 5.) * a_init**(-3. * f / 5.)  # Initial condition (~ 1; but f factor accounts for EDE-ish)
    y0 = (d_init, v_init)

    def fun(a, y):
        d, v = y[0], y[1]
        dxda = v
        fv = -(2. + _AH(a, Om, Ode, w0, wa) / _Hubble2(a, Om, Ode, Ok, w0, wa)) * v / a
        fd = 1.5 * _Omega_m(a, Om, Ode, Ok, w0, wa) * d / a**2
        dvda = fv + fd
        return dxda, dvda

    g = solve_ivp(fun, (a[0], a[-1]), y0, t_eval=a).y[0]
    return interp1d(a, g, kind='cubic', assume_sorted=True)

#def get_growth_interpolator(cosmo):
#    """
#    Solve the linear growth ODE and returns an interpolating function for the solution
#    LCDM = True forces w = -1 and imposes flatness by modifying the dark-energy density
#    TODO: w dependence for initial conditions; f here is correct for w=0 only
#    TODO: Could use d_init = a(1+(w-1)/(w(6w-5))*(Om_w/Om_m)*a**-3w) at early times with w = w(a<<1)
#    """
#    a_init = 1e-3
#    z_init = (1.0/a_init) - 1.0
#    print(z_init)
#    #z_init = 500.0
#    #a_init = 1.0/(1.0+z_init)
#
#    na = 129 # Number of scale factors used to construct interpolator
#    a = np.linspace(a_init, 1.0, na)
#    f = 1.0 - cosmo.Om(z_init) # Early mass density
#    d_init = a_init**(1.0 - ((3.0/5.0) * f))            # Initial condition (~ a_init; but f factor accounts for EDE-ish)
#    v_init = (1.0 - ((3.0/5.0) * f))*a_init**(-((3.0/5.0) * f)) # Initial condition (~ 1; but f factor accounts for EDE-ish)
#
#    y0 = (d_init, v_init)
#    def fun(ax, y):
#        d, v = y[0], y[1]
#        dxda = v
#        zx = (1.0/ax) - 1.0
#        fv = -(2.0 + acceleration_parameter(cosmo, zx)*cosmo.inv_efunc(zx)**2.0)*v/ax
#        fd = 1.5*cosmo.Om(zx)*d/ax**2
#        dvda = fv+fd
#        return dxda, dvda
#
#    #g = solve_ivp(fun, (a[0], a[-1]), y0, t_eval=a, atol=1e-8, rtol=1e-8, vectorized=True).y[0]
#    g = solve_ivp(fun, (a[0], a[-1]), y0, t_eval=a).y[0]
#    g_interp = interp1d(a, g, kind='linear', assume_sorted=True)
#    return g_interp

# Mead corrections: See appendix A of 2009.01858
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
    dc_Mead = 1.0 + f_Mead(g / a, G / a, *p1) * np.log10(Om) ** a1 + f_Mead(g / a, G / a, *p2) * np.log10(Om) ** a2
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
    Dv_Mead = 1.0 + f_Mead(g / a, G / a, *p3) * np.log10(Om) ** a3 + f_Mead(g / a, G / a, *p4) * np.log10(Om) ** a4
    Dv0 = 18.0 * np.pi**2.0  # Delta_v = ~178, EdS halo virial overdensity
    return Dv_Mead * Dv0 * (1.0 + 0.763 * f_nu)

def Tk_cold_ratio(k, g, ommh2, h, f_nu, N_nu, T_CMB=2.7255):
    """
    Ratio of cold to matter transfer function from Eisenstein & Hu (1999).
    This can be used to get the cold-matter spectrum approximately from the matter spectrum.
    Captures the scale-dependent growth with neutrino free-streaming scale.
    """
    if f_nu == 0.:  # Fix to unity if there are no neutrinos
        return 1.

    pcb = (5. - np.sqrt(1. + 24. * (1. - f_nu))) / 4.  # Growth exponent for unclustered neutrinos completely
    BigT = T_CMB / 2.7  # Big Theta for temperature
    zeq = 2.5e4 * ommh2 * BigT**(-4)  # Matter-radiation equality redshift
    D = (1. + zeq) * g  # Growth normalized such that D=(1.+z_eq)/(1+z) at early times
    q = k * h * BigT**2 / ommh2  # Wave number relative to the horizon scale at equality (equation 5)
    yfs = 17.2 * f_nu * (1. + 0.488 * f_nu**(-7. / 6.)) * (N_nu * q / f_nu)**2  # Free streaming scale (equation 14)
    Dcb = (1. + (D / (1. + yfs))**0.7)**(pcb / 0.7)  # Cold growth function
    Dcbnu = ((1. - f_nu)**(0.7 / pcb) + (D / (1. + yfs))**0.7)**(pcb / 0.7)  # Cold and neutrino growth function
    return Dcb / Dcbnu  # Finally, the ratio

def sigmaR_cc(power, k, r):
    rk = np.outer(r, k)
    dlnk = np.log(k[1] / k[0])

    k_space = (3 / rk**3) * (np.sin(rk) - rk * np.cos(rk))
    # we multiply by k because our steps are in logk.
    rest = power * k ** 3
    integ = rest * k_space ** 2
    sigma = (0.5 / np.pi**2) * simpson(integ, dx=dlnk, axis=-1)
    return np.sqrt(sigma)

# To be maybe incorporated in hmf
class SOVirial_Mead(SphericalOverdensity):
    """
    SOVirial overdensity definition from Mead et al. (2021).
    """
    _defaults = {"overdensity": 200}

    def halo_density(self, z=0, cosmo=Planck15):
        """The density of haloes under this definition."""
        return self.params["overdensity"] * self.mean_density(z, cosmo)

    @property
    def colossus_name(self):
        return "200c"

    def __str__(self):
        """Describe the halo definition in standard notation."""
        return "SOVirial"


# def concentration_colossus(block, cosmo, mass, z, model, mdef, overdensity):
#     """
#     calculates concentration given halo mass, using the halomod model provided in config
#     furthermore it converts to halomod instance to be used with the halomodel,
#     consistenly with halo mass function and halo bias function
#     """

#     with warnings.catch_warnings():
#         warnings.filterwarnings('ignore', category=UserWarning)
#         # This dissables the warning from colossus that is just telling us what we know
#         # colossus warns us about massive neutrinos, but that is also ok
#         # as we do not use cosmology from it, but it requires it to setup the instance!
#         this_cosmo = colossus_cosmology.fromAstropy(astropy_cosmo=cosmo, cosmo_name='custom',
#                      sigma8=block[cosmo_params, 'sigma_8'], ns=block[cosmo_params, 'n_s'])


#     mdef = getattr(md, mdef)() if mdef in ['SOVirial'] else getattr(md, mdef)(overdensity=overdensity)

#     # This is the slow part: 0.4-0.5 seconds per call, called separately for each redshift.
#     # MA: Possible solution: See if we can get away with a smaller numbr of redshifts and interpolate.
#     #tic = time.perf_counter()
#     c, ms = colossus_concentration.concentration(M=mass, z=z, mdef=mdef.colossus_name, model=model,
#             range_return=True, range_warning=False)
#     #toc = time.perf_counter()
#     #print(" colossus_concentration.concentration: "+'%.4f' %(toc - tic)+ "s")
#     if len(c[c>0]) == 0:
#         c_interp = lambda x: np.ones_like(x)
#     else:
#         c_interp = interp1d(mass[c>0], c[c>0], kind='linear', bounds_error=False, fill_value=1.0)
#     return c_interp(mass)
