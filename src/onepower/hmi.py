"""
A module for computing various cosmological quantities and halo model ingredients.
This module provides classes and functions to calculate properties of dark matter halos,
cosmological parameters, and related quantities using different models and corrections.
"""

import halomod.concentration as concentration_classes
import halomod.profiles as profile_classes
import numpy as np
import warnings
from astropy.cosmology import Flatw0waCDM, Planck15
from astropy import units as u
from functools import cached_property
from halomod.concentration import interp_concentration, make_colossus_cm
from halomod.halo_model import DMHaloModel, TracerHaloModel
from scipy.integrate import quad, simpson, solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

from hmf._internals._cache import cached_quantity, parameter, subframework
from hmf._internals._framework import Framework
from hmf.halos.mass_definitions import SphericalOverdensity

# Silencing a warning from hmf for which the nonlinear mass is still correctly calculated
warnings.filterwarnings('ignore', message='Nonlinear mass outside mass range')

# This disables the warning from hmf. hmf is just telling us what we know
# hmf's internal way of calculating the overdensity and the collapse threshold are fixed.
# When we use the mead correction we want to define the haloes using the virial definition.
# To avoid conflicts we manually pass the overdensity and the collapse threshold,
# but for that we need to set the mass definition to be "mean",
# so that it is compared to the mean density of the Universe rather than critical density.
# hmf warns us that the value is not a native definition for the given halo mass function,
# but will interpolate between the known ones (this is happening when one uses Tinker hmf for instance).
warnings.filterwarnings('ignore', category=UserWarning)

DMHaloModel.ERROR_ON_BAD_MDEF = False
valid_corrections = ['feedback', 'nofeedback', 'fit', None]


class SOVirial_Mead(SphericalOverdensity):
    """
    SOVirial overdensity definition from Mead et al. (2021).
    """

    _defaults = {'overdensity': 200}

    def halo_density(self, z=0, cosmo=Planck15):
        """The density of haloes under this definition."""
        # return self.params["overdensity"].reshape(z.shape) * self.mean_density(z, cosmo)
        return self.params['overdensity'] * self.mean_density(z, cosmo)

    @property
    def colossus_name(self):
        return '200c'

    def __str__(self):
        """Describe the halo definition in standard notation."""
        return 'SOVirial'


class CosmologyBase(Framework):
    """
    A cosmology base class.

    Parameters:
    -----------
    z_vec : array_like, optional
        Array of redshifts.
    h0 : float, optional
        Hubble parameter (small h).
    omega_c : float, optional
        Cold dark matter density parameter.
    omega_b : float, optional
        Baryon density parameter.
    w0 : float, optional
        Dark energy equation of state parameter.
    wa : float, optional
        Dark energy equation of state parameter.
    n_s : float, optional
        Spectral index.
    tcmb : float, optional
        Temperature of the CMB.
    Neff : float, optional
        Effective number of neutrino species.
    m_nu : float, optional
        Neutrino mass.
    sigma_8 : float, optional
        Amplitude of matter fluctuations on 8 Mpc scales.
    log10T_AGN : float, optional
        Log10 of AGN temperature.
    """

    def __init__(
        self,
        z_vec=np.linspace(0.0, 3.0, 15),
        h0=0.7,
        omega_c=0.25,
        omega_b=0.05,
        w0=-1.0,
        wa=0.0,
        n_s=0.9,
        tcmb=2.7255,
        Neff=3.044,
        m_nu=0.06,
        sigma_8=0.8,
        log10T_AGN=7.8,
    ):
        self.z_vec = z_vec
        self.h0 = h0
        self.omega_c = omega_c
        self.omega_b = omega_b
        self.w0 = w0
        self.wa = wa
        self.n_s = n_s
        self.tcmb = tcmb
        self.Neff = Neff
        self.m_nu = m_nu
        self.sigma_8 = sigma_8
        self.log10T_AGN = log10T_AGN

    @parameter('param')
    def z_vec(self, val):
        """
        Array of redshifts.

        :type: array_like
        """
        return val

    @parameter('param')
    def h0(self, val):
        """
        Hubble parameter (small h).

        :type: float
        """
        return val

    @parameter('param')
    def omega_c(self, val):
        """
        Cold dark matter density parameter.

        :type: float
        """
        return val

    @parameter('param')
    def omega_b(self, val):
        """
        Baryon density parameter.

        :type: float
        """
        return val

    @parameter('param')
    def w0(self, val):
        """
        Dark energy equation of state parameter.

        :type: float
        """
        return val

    @parameter('param')
    def wa(self, val):
        """
        Dark energy equation of state parameter.

        :type: float
        """
        return val

    @parameter('param')
    def n_s(self, val):
        """
        Spectral index.

        :type: float
        """
        return val

    @parameter('param')
    def tcmb(self, val):
        """
        Temperature of the CMB.

        :type: float
        """
        return val

    @parameter('param')
    def Neff(self, val):
        """
        Effective number of neutrino species.

        :type: float
        """
        return val

    @parameter('param')
    def m_nu(self, val):
        """
        Neutrino mass.

        :type: float
        """
        return val

    @parameter('param')
    def sigma_8(self, val):
        """
        Amplitude of matter fluctuations on 8 Mpc scales.

        :type: float
        """
        return val

    @parameter('param')
    def log10T_AGN(self, val):
        """
        Log10 of AGN temperature.

        :type: float
        """
        return val

    @cached_quantity
    def cosmo_model(self):
        """
        Return the astropy cosmology object assuming Flatw0waCDM model.

        Returns:
        --------
        object
            astropy cosmology object
        """
        return Flatw0waCDM(
            H0=self.h0 * 100.0,
            Ob0=self.omega_b,
            Om0=self.omega_c + self.omega_b,
            Neff=self.Neff,
            m_nu=[0.0, 0.0, self.m_nu] * u.eV,
            Tcmb0=self.tcmb,
            w0=self.w0,
            wa=self.wa,
        )

    @cached_property
    def omega_m(self):
        """Matter density parameter."""
        return self.cosmo_model.Om0

    @cached_quantity
    def scale_factor(self):
        """
        Return the scale factor.

        Returns:
        --------
        array_like
            scale factor array
        """
        return 1.0 / (1.0 + self.z_vec)

    def _Omega_m(self, a, Om, Ode, Ok):
        """
        Evolution of Omega_m with scale-factor ignoring radiation.
        Massive neutrinos are counted as 'matter'.

        Parameters:
        -----------
        a : array_like
            Scale factor.
        Om : float
            Matter density parameter.
        Ode : float
            Dark energy density parameter.
        Ok : float
            Curvature density parameter.

        Returns:
        --------
        array_like
            Omega_m at scale factor 'a'.
        """
        return Om * a**-3 / self._Hubble2(a, Om, Ode, Ok)

    def _Hubble2(self, a, Om, Ode, Ok):
        """
        Squared Hubble parameter ignoring radiation.
        Massive neutrinos are counted as 'matter'.

        Parameters:
        -----------
        a : array_like
            Scale factor.
        Om : float
            Matter density parameter.
        Ode : float
            Dark energy density parameter.
        Ok : float
            Curvature density parameter.

        Returns:
        --------
        array_like
            Squared Hubble parameter at scale factor 'a'.
        """
        z = -1.0 + 1.0 / a
        H2 = Om * a**-3 + Ode * self.cosmo_model.de_density_scale(z) + Ok * a**-2
        return H2

    def _AH(self, a, Om, Ode):
        """
        Squared Hubble parameter ignoring radiation.
        Massive neutrinos are counted as 'matter'.

        Parameters:
        -----------
        a : array_like
            Scale factor.
        Om : float
            Matter density parameter.
        Ode : float
            Dark energy density parameter.

        Returns:
        --------
        array_like
            Squared Hubble parameter at scale factor 'a'.
        """
        z = -1.0 + 1.0 / a
        AH = -0.5 * (
            Om * a**-3
            + (1.0 + 3.0 * self.cosmo_model.w(z))
            * Ode
            * self.cosmo_model.de_density_scale(z)
        )
        return AH

    @cached_quantity
    def get_mead_growth_fnc(self):
        """
        Solve the linear growth ODE and returns an interpolating function for the solution.
        TODO: w dependence for initial conditions; f here is correct for w=0 only.
        TODO: Could use d_init = a(1+(w-1)/(w(6w-5))*(Om_w/Om_m)*a**-3w) at early times with w = w(a<<1).

        Returns:
        --------
        object
            interpolating function g(a)
        """
        a_init = 1e-4
        Om = self.cosmo_model.Om0 + self.cosmo_model.Onu0
        Ode = 1.0 - Om
        Ok = self.cosmo_model.Ok0
        na = 129  # Number of scale factors used to construct interpolator
        a = np.linspace(a_init, 1.0, na)

        f = 1.0 - self._Omega_m(a_init, Om, Ode, Ok)  # Early mass density
        d_init = a_init ** (
            1.0 - 3.0 * f / 5.0
        )  # Initial condition (~ a_init; but f factor accounts for EDE-ish)
        v_init = (1.0 - 3.0 * f / 5.0) * a_init ** (
            -3.0 * f / 5.0
        )  # Initial condition (~ 1; but f factor accounts for EDE-ish)
        y0 = (d_init, v_init)

        def fun(a, y):
            d, v = y[0], y[1]
            dxda = v
            fv = -(2.0 + self._AH(a, Om, Ode) / self._Hubble2(a, Om, Ode, Ok)) * v / a
            fd = 1.5 * self._Omega_m(a, Om, Ode, Ok) * d / a**2
            dvda = fv + fd
            return dxda, dvda

        g = solve_ivp(fun, (a[0], a[-1]), y0, t_eval=a).y[0]
        return interp1d(a, g, kind='cubic', assume_sorted=True)

    @cached_quantity
    def get_mead_growth(self):
        """
        Return the Mead growth factor at the scale factors.

        Returns:
        --------
        array_like
            Mead 2020 growth factor as a function of scale factor 'a'
        """
        return self.get_mead_growth_fnc(self.scale_factor)

    @cached_quantity
    def get_mead_accumulated_growth(self):
        """
        Calculates the accumulated growth at scale factor 'a'.

        Returns:
        --------
        array_like
            accumulated growth at scale factor 'a'
        """
        a_init = 1e-4

        # Eq A5 of Mead et al. 2021 (2009.01858).
        # We approximate the integral as g(a_init) for 0 to a_init<<0.
        missing = self.get_mead_growth_fnc(a_init)
        G = np.array(
            [
                quad(lambda a: self.get_mead_growth_fnc(a) / a, a_init, ai)[0] + missing
                for ai in self.scale_factor
            ]
        )
        return G

    def f_Mead(self, x, y, p0, p1, p2, p3):
        r"""
        Fitting function from Mead et al. 2021 (2009.01858), eq A3,
        used in :math:`\delta_c` and :math:`\Delta_{\rm v}` calculations.

        Parameters:
        -----------
        x : float
            First variable.
        y : float
            Second variable.
        p0, p1, p2, p3 : float
            Fitting parameters.

        Returns:
        --------
        float
            Value of the fitting function.
        """
        return p0 + p1 * (1.0 - x) + p2 * (1.0 - x) ** 2.0 + p3 * (1.0 - y)

    @cached_quantity
    def dc_Mead(self):
        r"""
        The critical overdensity for collapse :math:`\delta_c`
        fitting function from Mead et al. 2021 (2009.01858).
        All input parameters should be evaluated as functions of a/z.

        Returns:
        --------
        array_like
            Delta_c at redshifrs z
        """
        a = self.scale_factor
        Om = self.cosmo_model.Om(self.z_vec) + self.cosmo_model.Onu(self.z_vec)
        f_nu = self.cosmo_model.Onu0 / (self.cosmo_model.Om0 + self.cosmo_model.Onu0)
        g = self.get_mead_growth
        G = self.get_mead_accumulated_growth

        # See Table A.1 of 2009.01858 for naming convention
        p1 = [-0.0069, -0.0208, 0.0312, 0.0021]
        p2 = [0.0001, -0.0647, -0.0417, 0.0646]
        a1, a2 = 1, 0
        # Linear collapse threshold
        # Eq A1 of 2009.01858
        dc_Mead = (
            1.0
            + self.f_Mead(g / a, G / a, *p1) * np.log10(Om) ** a1
            + self.f_Mead(g / a, G / a, *p2) * np.log10(Om) ** a2
        )
        # delta_c = ~1.686' EdS linear collapse threshold
        dc0 = (3.0 / 20.0) * (12.0 * np.pi) ** (2.0 / 3.0)
        return dc_Mead * dc0 * (1.0 - 0.041 * f_nu)

    @cached_quantity
    def Dv_Mead(self):
        r"""
        Overdensity :math:`\Delta_{\rm v}` fitting function from Mead et al. 2021 (2009.01858), eq A.2.
        All input parameters should be evaluated as functions of a/z.

        Returns:
        --------
        array_like
            Overdensities at given redshifs
        """
        a = self.scale_factor
        Om = self.cosmo_model.Om(self.z_vec) + self.cosmo_model.Onu(self.z_vec)
        f_nu = self.cosmo_model.Onu0 / (self.cosmo_model.Om0 + self.cosmo_model.Onu0)
        g = self.get_mead_growth
        G = self.get_mead_accumulated_growth

        # See Table A.1 of 2009.01858 for naming convention
        p3 = [-0.79, -10.17, 2.51, 6.51]
        p4 = [-1.89, 0.38, 18.8, -15.87]
        a3, a4 = 1, 2

        # Halo virial overdensity
        # Eq A2 of 2009.01858
        Dv_Mead = (
            1.0
            + self.f_Mead(g / a, G / a, *p3) * np.log10(Om) ** a3
            + self.f_Mead(g / a, G / a, *p4) * np.log10(Om) ** a4
        )
        Dv0 = 18.0 * np.pi**2.0  # Delta_v = ~178, EdS halo virial overdensity
        return Dv_Mead * Dv0 * (1.0 + 0.763 * f_nu)


class HaloModelIngredients(CosmologyBase):
    """
    A class to compute various ingredients for the halo model.
    This includes halo mass functions, bias models, halo profiles, and concentration models.
    Based on the hmf and halomod packages.

    Parameters:
    -----------
    k_vec : array_like, optional
        Array of wavenumbers.
    lnk_min : float, optional
        Minimum natural log of wavenumber (for hmf).
    lnk_max : float, optional
        Maximum natural log of wavenumber (for hmf).
    dlnk : float, optional
        Spacing in natural log of wavenumber (for hmf).
    Mmin : float, optional
        Minimum halo mass (for hmf).
    Mmax : float, optional
        Maximum halo mass (for hmf).
    dlog10m : float, optional
        Spacing in log10 of halo mass (for hmf).
    mdef_model : str, optional
        Mass definition model (for hmf).
    hmf_model : str, optional
        Halo mass function model (for hmf).
    bias_model : str, optional
        Halo bias model (for halomod).
    halo_profile_model_dm : str, optional
        Halo profile model for dark matter / central galaxies (for halomod).
    halo_profile_model_sat : str, optional
        Halo profile model for satellite galaxies (for halomod).
    halo_concentration_model_dm : str, optional
        Halo concentration model for dark matter / central galaxies (for halomod).
    halo_concentration_model_sat : str, optional
        Halo concentration model for satellite galaxies (for halomod).
    transfer_model : str, optional
        Transfer function model (for hmf).
    transfer_params : dict, optional
        Parameters for the transfer function (for hmf).
    growth_model : str, optional
        Growth function model (for hmf).
    growth_params : dict, optional
        Parameters for the growth function (for hmf).
    norm_cen : float, optional
        Normalization of c(M) relation for central galaxies.
    norm_sat : float, optional
        Normalization of c(M) relation for satellite galaxies.
    eta_cen : float, optional
        Bloating parameter for central galaxies.
    eta_sat : float, optional
        Bloating parameter for satellite galaxies.
    overdensity : float, optional
        Overdensity parameter.
    delta_c : float, optional
        Critical density threshold for collapse.
    mead_correction : str, optional
        Correction model from Mead et al.

    """

    def __init__(
        self,
        k_vec=np.logspace(-4, 4, 100),
        lnk_min=np.log(10 ** (-4.0)),
        lnk_max=np.log(10 ** (4.0)),
        dlnk=(np.log(10 ** (4.0)) - np.log(10 ** (-4.0))) / 100,
        Mmin=9.0,
        Mmax=16.0,
        dlog10m=0.05,
        mdef_model='SOMean',
        hmf_model='Tinker10',
        bias_model='Tinker10',
        halo_profile_model_dm='NFW',
        halo_concentration_model_dm='Duffy08',
        halo_profile_model_sat='NFW',
        halo_concentration_model_sat='Duffy08',
        transfer_model='CAMB',
        transfer_params=None,
        growth_model='CambGrowth',
        growth_params=None,
        norm_cen=1.0,
        norm_sat=1.0,
        eta_cen=0.0,
        eta_sat=0.0,
        overdensity=200,
        delta_c=1.686,
        mead_correction: str | None = None,
        **cosmology_kwargs,
    ):
        super().__init__(**cosmology_kwargs)

        self.mead_correction = mead_correction

        self.k_vec = k_vec
        self.lnk_min = lnk_min
        self.lnk_max = lnk_max
        self.dlnk = dlnk
        self.Mmin = Mmin
        self.Mmax = Mmax
        self.dlog10m = dlog10m
        self.mdef_model = mdef_model
        self.hmf_model = hmf_model
        self.bias_model = bias_model
        self.halo_concentration_model_dm = halo_concentration_model_dm
        self.halo_concentration_model_sat = halo_concentration_model_sat
        self.halo_profile_model_dm = halo_profile_model_dm
        self.halo_profile_model_sat = halo_profile_model_sat
        self.transfer_model = transfer_model
        self.transfer_params = transfer_params or {}
        self.growth_model = growth_model
        self.growth_params = growth_params or {}

        self.norm_cen = norm_cen
        self.norm_sat = norm_sat

        self.eta_cen = eta_cen
        self.eta_sat = eta_sat

        self.overdensity = overdensity
        self.delta_c = delta_c

    def validate(self):
        if self.mead_correction not in valid_corrections:
            raise ValueError(
                f'Desired Mead correction is not supported. You have provided {self.mead_correction}, valid options are {valid_corrections}!'
            )

    @parameter('param')
    def mead_correction(self, val):
        """
        Correction model from Mead et al.

        :type: str
        """
        return val

    @parameter('param')
    def k_vec(self, val):
        """
        Array of wavenumbers.

        :type: array_like
        """
        return val

    @parameter('param')
    def lnk_min(self, val):
        """
        Minimum natural log of wavenumber (for hmf).

        :type: float
        """
        return val

    @parameter('param')
    def lnk_max(self, val):
        """
        Maximum natural log of wavenumber (for hmf).

        :type: float
        """
        return val

    @parameter('param')
    def dlnk(self, val):
        """
        Spacing in natural log of wavenumber (for hmf).

        :type: float
        """
        return val

    @parameter('param')
    def Mmin(self, val):
        """
        Minimum halo mass (for hmf).

        :type: float
        """
        return val

    @parameter('param')
    def Mmax(self, val):
        """
        Maximum halo mass (for hmf).

        :type: float
        """
        return val

    @parameter('param')
    def dlog10m(self, val):
        """
        Spacing in log10 of halo mass (for hmf).

        :type: float
        """
        return val

    @parameter('param')
    def mdef_model(self, val):
        """
        Mass definition model (for hmf).

        :type: str
        """
        return val

    @parameter('param')
    def hmf_model(self, val):
        """
        Halo mass function model (for hmf).

        :type: str
        """
        return val

    @parameter('param')
    def bias_model(self, val):
        """
        Halo bias model (for halomod).

        :type: str
        """
        return val

    @parameter('param')
    def halo_concentration_model_dm(self, val):
        """
        Halo concentration model of dark matter / central galaxies (for halomod).

        :type: str
        """
        return val

    @parameter('param')
    def halo_concentration_model_sat(self, val):
        """
        Halo concentration model of satellite galaxies (for halomod).

        :type: str
        """
        return val

    @parameter('param')
    def halo_profile_model_dm(self, val):
        """
        Halo profile model of dark matter / central galaxies (for halomod).

        :type: str
        """
        return val

    @parameter('param')
    def halo_profile_model_sat(self, val):
        """
        Halo profile model of satellite galaxies (for halomod).

        :type: str
        """
        return val

    @parameter('param')
    def transfer_model(self, val):
        """
        Transfer function model (for hmf).

        :type: str
        """
        return val

    @parameter('param')
    def transfer_params(self, val):
        """
        Parameters for the transfer function (for hmf).

        :type: dict
        """
        return val

    @parameter('param')
    def growth_model(self, val):
        """
        Growth function model (for hmf).

        :type: str
        """
        return val

    @parameter('param')
    def growth_params(self, val):
        """
        Parameters for the growth function (for hmf).

        :type: dict
        """
        return val

    @parameter('param')
    def norm_cen(self, val):
        """
        Normalization of c(M) relation for central galaxies.

        :type: float
        """
        return np.atleast_1d(val)

    @parameter('param')
    def norm_sat(self, val):
        """
        Normalization of c(M) relation for satellite galaxies.

        :type: float
        """
        return np.atleast_1d(val)

    @parameter('param')
    def eta_cen(self, val):
        """
        Bloating parameter for central galaxies.

        :type: float
        """
        return np.atleast_1d(val)

    @parameter('param')
    def eta_sat(self, val):
        """
        eta_sat : float
            Bloating parameter for satellite galaxies.
        """
        return np.atleast_1d(val)

    @parameter('param')
    def delta_c(self, val):
        r"""
        Critical density threshold for collapse :math:`\delta_c`.

        :type: float
        """
        return val

    @parameter('param')
    def overdensity(self, val):
        """
        Overdensity parameter.

        :type: float
        """
        return val

    @cached_quantity
    def _norm_c(self):
        """
        Sets the norm_cen parameter to the shape of redshift vector.

        Returns:
        --------
        array_like
            norm_cen array
        """
        return self.norm_cen * np.ones_like(self.z_vec)

    @cached_quantity
    def _norm_s(self):
        """
        Sets the norm_sat parameter to the shape of redshift vector.

        Returns:
        --------
        array_like
            norm_sat array
        """
        return self.norm_sat * np.ones_like(self.z_vec)

    @cached_quantity
    def _eta_c(self):
        """
        Sets the eta_cen parameter to the shape of redshift vector.

        Returns:
        --------
        array_like
            eta_cen array

        """
        return self.eta_cen * np.ones_like(self.z_vec)

    @cached_quantity
    def _eta_s(self):
        """
        Sets the eta_sat parameter to the shape of redshift vector.

        Returns:
        --------
        array_like
            eta_sat array
        """
        return self.eta_sat * np.ones_like(self.z_vec)

    @cached_quantity
    def _delta_c_mod(self):
        r"""
        Sets the :math:`\delta_c` parameter to the one used in HMCode or to the one for virial collapse.
        Overrides the default passed value in those two cases.

        Returns:
        --------
        array_like
            delta_c array
        """
        if self.mead_correction in ['feedback', 'nofeedback']:
            val = self.dc_Mead
        else:
            val = (
                (3.0 / 20.0)
                * (12.0 * np.pi) ** (2.0 / 3.0)
                * (1.0 + 0.0123 * np.log10(self.cosmo_model.Om(self.z_vec)))
                if self.mdef_model == 'SOVirial'
                else self.delta_c * np.ones_like(self.z_vec)
            )
        return val

    @cached_quantity
    def _mdef_mod(self):
        """
        Sets the mass definition to the one used in HMCode.
        Overrides the default passed value in this case.

        Returns:
        --------
        object
            SOVirial_Mead mass definition class if mead_correction is True, otherwise the input mass definition class
        """
        if self.mead_correction in ['feedback', 'nofeedback']:
            return SOVirial_Mead
        return self.mdef_model

    @cached_quantity
    def _hmf_mod(self):
        """
        Sets the mass function to the one used in HMCode.
        Overrides the default passed value in this case.

        Returns:
        --------
        object
            hmf_model class, Sheth-Thormen model in case if meed_correction is True
        """
        if self.mead_correction in ['feedback', 'nofeedback']:
            return 'ST'
        return self.hmf_model

    @cached_quantity
    def _bias_mod(self):
        """
        Sets the halo bias function to the one used in HMCode.
        Overrides the default passed value in this case.

        Returns:
        --------
        object
            bias_model class, Sheth-Thormen model in case if meed_correction is True
        """
        if self.mead_correction in ['feedback', 'nofeedback']:
            return 'ST99'
        return self.bias_model

    @cached_quantity
    def _halo_concentration_mod_dm(self):
        """
        Sets the c(M) relation of dark matter / centrals to the one used in HMCode.
        Overrides the default passed value in this case.

        Returns:
        --------
        object
            halo_concentration class, Bullock 2001 model in case if meed_correction is True
        """
        if self.mead_correction in ['feedback', 'nofeedback']:
            val = interp_concentration(getattr(concentration_classes, 'Bullock01'))
        else:
            try:
                val = interp_concentration(
                    getattr(concentration_classes, self.halo_concentration_model_dm)
                )
            except AttributeError:
                val = interp_concentration(
                    make_colossus_cm(self.halo_concentration_model_dm)
                )
        return val

    @cached_quantity
    def _halo_concentration_mod_sat(self):
        """
        Sets the c(M) relation of satellite galaxies.

        Returns:
        --------
        object
            halo_concentration class
        """
        if self.mead_correction in ['feedback', 'nofeedback']:
            val = interp_concentration(getattr(concentration_classes, 'Bullock01'))
        else:
            try:
                val = interp_concentration(
                    getattr(concentration_classes, self.halo_concentration_model_sat)
                )
            except AttributeError:
                val = interp_concentration(
                    make_colossus_cm(self.halo_concentration_model_sat)
                )
        return val

    @cached_quantity
    def mdef_params(self):
        """
        Sets the overdensity parameter to the one used in HMCode or to the one for virial collapse.
        Overrides the default passed value in those two cases.

        Returns:
        --------
        array_like
            array of mass mass definition dictionaries, one for each redshift
        """
        if self.mead_correction in ['feedback', 'nofeedback']:
            val = [{'overdensity': overdensity} for overdensity in self.Dv_Mead]
        else:
            val = [
                (
                    {}
                    if self.mdef_model == 'SOVirial'
                    else {'overdensity': self.overdensity}
                )
                for _ in self.z_vec
            ]
        return val

    @cached_quantity
    def halo_profile_params(self):
        """
        Packs the cosmology to a dictionary to be passed to the hmf.
        Can in principle have more parameters that need to be passed to hmf that are not explicitly set.

        Returns:
        --------
        dict
            astropy cosmology object for hmf package
        """
        return {'cosmo': self.cosmo_model}

    @cached_quantity
    def disable_mass_conversion(self):
        """
        Dissables the mass conversion for the mass definitions defined in HMCode.

        Returns:
        --------
        bool
        """
        return self.mead_correction in ['feedback', 'nofeedback']

    @cached_quantity
    def K(self):
        """
        Sets the c(M) normalisations to the one used in HMCode.
        Overrides the default passed value in this case.

        Returns:
        --------
        array_like
            normalisation of the Bullock 2001 c(M) relation, if mead_corrections is True
        """
        if self.mead_correction == 'nofeedback':
            k = 5.196 * np.ones_like(self.z_vec)
        elif self.mead_correction == 'feedback':
            theta_agn = self.log10T_AGN - 7.8
            k = (5.196 / 4.0) * (
                (3.44 - 0.496 * theta_agn)
                * np.power(10.0, self.z_vec * (-0.0671 - 0.0371 * theta_agn))
            )
        else:
            k = np.zeros_like(self.z_vec)
        return k

    @cached_quantity
    def _hmf_generator(self):
        """
        Generate halo mass function models for central and satellite galaxies at different redshifts.
        Setups the hmf and halomod classes at desired cosmology and uses the "update" functionality
        to calculate the models at different redshifts.

        Returns:
        --------
        tuple
            tuple of lists of DMHaloModel objects for centrals and satellite galaxies at different redshifts
        """
        x = DMHaloModel(
            z=0.0,
            lnk_min=self.lnk_min,
            lnk_max=self.lnk_max,
            dlnk=self.dlnk,
            Mmin=self.Mmin,
            Mmax=self.Mmax,
            dlog10m=self.dlog10m,
            hmf_model=self._hmf_mod,
            mdef_model=self._mdef_mod,
            disable_mass_conversion=self.disable_mass_conversion,
            bias_model=self._bias_mod,
            halo_profile_model=self.halo_profile_model_dm,
            halo_profile_params=self.halo_profile_params,
            halo_concentration_model=self._halo_concentration_mod_dm,
            cosmo_model=self.cosmo_model,
            sigma_8=self.sigma_8,
            n=self.n_s,
            transfer_model=self.transfer_model,
            transfer_params=self.transfer_params,
            growth_model=self.growth_model,
            growth_params=self.growth_params,
            mdef_params=self.mdef_params[0],
            delta_c=self._delta_c_mod[0],
        )
        y = x.clone()
        x_out, y_out = [], []

        if self.mead_correction in ['feedback', 'nofeedback']:
            # For centrals
            for z, mdef_par, dc, norm_cen, k in zip(
                self.z_vec,
                self.mdef_params,
                self._delta_c_mod,
                self._norm_c,
                self.K,
                strict=False,
            ):
                x.update(
                    z=z,
                    mdef_params=mdef_par,
                    delta_c=dc,
                )
                eta_cen = 0.1281 * x.sigma8_z ** (-0.3644)
                x.update(
                    halo_profile_params={'eta_bloat': eta_cen},
                    halo_concentration_params={'norm': norm_cen, 'K': k},
                )
                # yield x.clone()
                x_out.append(x.clone())

            # For satellites
            for z, mdef_par, dc, norm_sat, k in zip(
                self.z_vec,
                self.mdef_params,
                self._delta_c_mod,
                self._norm_s,
                self.K,
                strict=False,
            ):
                y.update(
                    z=z,
                    mdef_params=mdef_par,
                    delta_c=dc,
                    halo_profile_model=self.halo_profile_model_sat,
                    halo_concentration_model=self._halo_concentration_mod_sat,
                )
                eta_sat = 0.1281 * y.sigma8_z ** (-0.3644)
                y.update(
                    halo_profile_params={'eta_bloat': eta_sat},
                    halo_concentration_params={'norm': norm_sat, 'K': k},
                )
                # yield y.clone()
                y_out.append(y.clone())
        else:
            # For centrals
            for z, mdef_par, dc, eta_cen, norm_cen in zip(
                self.z_vec,
                self.mdef_params,
                self._delta_c_mod,
                self._eta_c,
                self._norm_c,
                strict=False,
            ):
                x.update(
                    z=z,
                    mdef_params=mdef_par,
                    delta_c=dc,
                    halo_profile_params={'eta_bloat': eta_cen},
                    halo_concentration_params={'norm': norm_cen},
                )
                # yield x.clone()
                x_out.append(x.clone())

            # For satellites
            for z, mdef_par, dc, eta_sat, norm_sat in zip(
                self.z_vec,
                self.mdef_params,
                self._delta_c_mod,
                self._eta_s,
                self._norm_s,
                strict=False,
            ):
                y.update(
                    z=z,
                    mdef_params=mdef_par,
                    delta_c=dc,
                    halo_profile_model=self.halo_profile_model_sat,
                    halo_concentration_model=self._halo_concentration_mod_sat,
                    halo_profile_params={'eta_bloat': eta_sat},
                    halo_concentration_params={'norm': norm_sat},
                )
                # yield y.clone()
                y_out.append(y.clone())
        return x_out, y_out

    @cached_quantity
    def _hmf_cen(self):
        """
        Return the halo mass function for central galaxies.
        """
        return self._hmf_generator[0]

    @cached_quantity
    def _hmf_sat(self):
        """
        Return the halo mass function for satellite galaxies.
        """
        return self._hmf_generator[1]

    @cached_quantity
    def mass(self):
        """
        Return the masses.

        Returns:
        --------
        ndarray
            halo masses
        """
        return self._hmf_cen[0].m

    @cached_quantity
    def power(self):
        """
        Return the linear power spectrum at z.

        Returns:
        --------
        ndarray
            linear power spectrum at z
        """
        return np.array([x.power for x in self._hmf_cen])

    @cached_quantity
    def nonlinear_power(self):
        """
        Return the non-linear power spectrum at z (if options passed).

        Returns:
        --------
        ndarray
            non-linear power spectrum at z
        """
        return np.array([x.nonlinear_power for x in self._hmf_cen])

    @cached_quantity
    def kh(self):
        """
        Return the k vector defined using lnk in hmf.

        Returns:
        --------
        array_like
            k vector from hmf
        """
        return self._hmf_cen[0].k

    @cached_quantity
    def halo_overdensity_mean(self):
        """
        Return the mean halo overdensity.

        Returns:
        --------
        ndarray
            meah halo overdensity
        """
        return np.array([x.halo_overdensity_mean for x in self._hmf_cen])

    @cached_quantity
    def nu(self):
        """
        Return the peak height parameter.

        Returns:
        --------
        ndarray
            peak heights
        """
        return np.array([x.nu**0.5 for x in self._hmf_cen])

    @cached_quantity
    def dndlnm(self):
        """
        Return the differential mass function.

        Returns:
        --------
        ndarray
            dndlnm
        """
        return np.array([x.dndlnm for x in self._hmf_cen])

    @cached_quantity
    def mean_density0(self):
        """
        Return the mean density at redshift zero.

        Returns:
        --------
        array_like
            mean density at z=0
        """
        return np.array([x.mean_density0 for x in self._hmf_cen])

    @cached_quantity
    def mean_density_z(self):
        """
        Return the mean density at the given redshifts.

        Returns:
        --------
        array_like
            mean density at z
        """
        return np.array([x.mean_density for x in self._hmf_cen])

    @cached_quantity
    def rho_halo(self):
        """
        Return the halo density.

        Returns:
        --------
        array_like
            halo density at z
        """
        return np.array(
            [x.halo_overdensity_mean * x.mean_density0 for x in self._hmf_cen]
        )

    @cached_quantity
    def halo_bias(self):
        """
        Return the halo bias.

        Returns:
        --------
        ndarray
            halo bias function
        """
        return np.array([x.halo_bias for x in self._hmf_cen])

    @cached_quantity
    def neff(self):
        """
        Return the effective spectral index.

        Returns:
        --------
        array_like
            effective spectral index
        """
        return np.array([x.n_eff_at_collapse for x in self._hmf_cen])

    @cached_quantity
    def sigma8_z(self):
        """
        Return the amplitude of matter fluctuations on 8 Mpc scales at the given redshifts.

        Returns:
        --------
        array_like
            sigma8(z)
        """
        return np.array([x.sigma8_z[0] for x in self._hmf_cen])

    @cached_quantity
    def fnu(self):
        """
        Return the neutrino density fraction.

        Returns:
        --------
        array_like
            neutrino density fraction
        """
        return np.array(
            [self.cosmo_model.Onu0 / self.cosmo_model.Om0 for _ in self.z_vec]
        )

    @cached_quantity
    def conc_cen(self):
        """
        Return the concentration for matter/central galaxies.

        Returns:
        --------
        ndarray
            concentration for matter/central galaxies
        """
        return np.array([x.cmz_relation for x in self._hmf_cen])

    @cached_quantity
    def nfw_cen(self):
        """
        Return the density profile for matter/central galaxies.

        Returns:
        --------
        ndarray
            density profile for matter/central galaxies.
        """
        return np.array([x.halo_profile.u(self.k_vec, x.m) for x in self._hmf_cen])

    @cached_quantity
    def u_dm(self):
        """
        Return the normalized density profile for dark matter.

        Returns:
        --------
        ndarray
            normalised density profile for matter/central galaxies.
        """
        return self.nfw_cen / np.expand_dims(self.nfw_cen[:, 0, :], 1)

    @cached_quantity
    def r_s_cen(self):
        """
        Return the scale radius for matter/central galaxies.

        Returns:
        --------
        ndarray
            scale radius for matter/central galaxies
        """
        return np.array([x.halo_profile._rs_from_m(x.m) for x in self._hmf_cen])

    @cached_quantity
    def rvir_cen(self):
        """
        Return the virial radius for matter/central galaxies.

        Returns:
        --------
        ndarray
            virial radius for matter/central galaxies
        """
        return np.array(
            [x.halo_profile.halo_mass_to_radius(x.m) for x in self._hmf_cen]
        )

    @cached_quantity
    def conc_sat(self):
        """
        Return the concentration for satellite galaxies.

        Returns:
        --------
        ndarray
            concenctration for satellite galaxies
        """
        return np.array([x.cmz_relation for x in self._hmf_sat])

    @cached_quantity
    def nfw_sat(self):
        """
        Return the density profile for satellite galaxies.

        Returns:
        --------
        ndarray
            density profile for satellite galaxies
        """
        return np.array([x.halo_profile.u(self.k_vec, x.m) for x in self._hmf_sat])

    @cached_quantity
    def u_sat(self):
        """
        Return the normalized density profile for satellite galaxies.

        Returns:
        --------
        ndarray
            normalised density profile for satellite galaxies
        """
        return self.nfw_sat / np.expand_dims(self.nfw_sat[:, 0, :], 1)

    @cached_quantity
    def r_s_sat(self):
        """
        Return the scale radius for satellite galaxies.

        Returns:
        --------
        ndarray
            scale radius for satellite galaxies
        """
        return np.array([x.halo_profile._rs_from_m(x.m) for x in self._hmf_sat])

    @cached_quantity
    def rvir_sat(self):
        """
        Return the virial radius for satellite galaxies.

        Returns:
        --------
        ndarray
            virial radius for satellite galaxies
        """
        return np.array(
            [x.halo_profile.halo_mass_to_radius(x.m) for x in self._hmf_sat]
        )

    @cached_quantity
    def growth_factor(self):
        """
        Return the growth factor.

        Returns:
        --------
        array_like
            growth factor at z
        """
        return self._hmf_cen[0]._growth_factor_fn(self.z_vec)

    # Maybe implement at some point?
    # Rnl = DM_hmf.filter.mass_to_radius(DM_hmf.mass_nonlinear, DM_hmf.mean_density0)
    # neff[jz] = -3.0 - 2.0*DM_hmf.normalised_filter.dlnss_dlnm(Rnl)

    # Only used for mead_corrections
    # pk_cold = DM_hmf.power * hmu.Tk_cold_ratio(DM_hmf.k, g, block[cosmo_params, 'ommh2'], block[cosmo_params, 'h0'], this_cosmo_run.Onu0/this_cosmo_run.Om0, this_cosmo_run.Neff, T_CMB=tcmb)**2.0
    # sigma8_z[jz] = hmu.sigmaR_cc(pk_cold, DM_hmf.k, 8.0)

    # Currently unused
    def Tk_cold_ratio(
        self, k, g, ommh2, h, f_nu, N_nu, T_CMB=2.7255
    ):  # pragma: no cover
        """
        Ratio of cold to matter transfer function from Eisenstein & Hu (1999).
        This can be used to get the cold-matter spectrum approximately from the matter spectrum.
        Captures the scale-dependent growth with neutrino free-streaming scale.

        Parameters:
        -----------
        k : float
            Wavenumber.
        g : float
            Growth factor.
        ommh2 : float
            Omega_m * h^2.
        h : float
            Hubble parameter.
        f_nu : float
            Fraction of neutrino density.
        N_nu : float
            Number of neutrino species.
        T_CMB : float, optional
            Temperature of the CMB.

        Returns:
        --------
        float
            Ratio of cold to matter transfer function.
        """
        if f_nu == 0.0:  # Fix to unity if there are no neutrinos
            return 1.0

        pcb = (
            5.0 - np.sqrt(1.0 + 24.0 * (1.0 - f_nu))
        ) / 4.0  # Growth exponent for unclustered neutrinos completely
        BigT = T_CMB / 2.7  # Big Theta for temperature
        zeq = 2.5e4 * ommh2 * BigT ** (-4)  # Matter-radiation equality redshift
        D = (
            1.0 + zeq
        ) * g  # Growth normalized such that D=(1.+z_eq)/(1+z) at early times
        q = (
            k * h * BigT**2 / ommh2
        )  # Wave number relative to the horizon scale at equality (equation 5)
        yfs = (
            17.2 * f_nu * (1.0 + 0.488 * f_nu ** (-7.0 / 6.0)) * (N_nu * q / f_nu) ** 2
        )  # Free streaming scale (equation 14)
        Dcb = (1.0 + (D / (1.0 + yfs)) ** 0.7) ** (pcb / 0.7)  # Cold growth function
        Dcbnu = ((1.0 - f_nu) ** (0.7 / pcb) + (D / (1.0 + yfs)) ** 0.7) ** (
            pcb / 0.7
        )  # Cold and neutrino growth function
        return Dcb / Dcbnu  # Finally, the ratio

    # Currently unused
    def sigmaR_cc(self, power, k, r):  # pragma: no cover
        """
        Calculate the variance of the cold matter density field smoothed on scale R.

        Parameters:
        -----------
        power : array_like
            Power spectrum.
        k : array_like
            Wavenumber.
        r : float
            Scale.

        Returns:
        --------
        float
            Variance of the density field.
        """
        rk = np.outer(r, k)
        dlnk = np.log(k[1] / k[0])

        k_space = (3 / rk**3) * (np.sin(rk) - rk * np.cos(rk))
        # we multiply by k because our steps are in logk.
        rest = power * k**3
        integ = rest * k_space**2
        sigma = (0.5 / np.pi**2) * simpson(integ, dx=dlnk, axis=-1)
        return np.sqrt(sigma)

    # Currently unused, we use the halomod calculation directly
    # as it returns the same results, but being much faster
    def get_halo_collapse_redshifts(self, M, z, dc, g, cosmo, mf):  # pragma: no cover
        """
        Calculate halo collapse redshifts according to the Bullock et al. (2001) prescription.

        Parameters:
        -----------
        M : array_like
            Halo mass.
        z : float
            Redshift.
        dc : float
            Critical density threshold for collapse.
        g : float
            Growth factor.
        cosmo : astropy cosmology model
            Cosmology model.
        mf : hmf Mass function object
            Mass function object.

        Returns:
        --------
        array_like
            Collapse redshifts for halos.
        """
        gamma = 0.01
        a = cosmo.scale_factor(z)
        zf = np.zeros_like(M)
        for iM, _M in enumerate(M):
            Mc = gamma * _M
            Rc = mf.filter.mass_to_radius(Mc, mf.mean_density0)
            sigma = mf.normalised_filter.sigma(Rc)
            current_growth_factor = g(a)
            fac = current_growth_factor * dc / sigma
            if fac >= current_growth_factor:
                af = a  # These haloes formed 'in the future'
            else:
                af_root = lambda af: g(af) - fac
                af = root_scalar(af_root, bracket=(1e-3, 1.0)).root
            zf[iM] = -1.0 + 1.0 / af
        return zf
