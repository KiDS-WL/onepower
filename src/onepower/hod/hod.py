from functools import cached_property
import numpy as np
from scipy.integrate import simpson
from scipy.special import erf
from scipy.interpolate import interp1d
from hmf._internals._cache import cached_quantity, parameter
from hmf._internals._framework import Framework

"""
A module for computing Halo Occupation Distribution (HOD) models.
This module provides classes and functions to calculate properties of galaxies within dark matter halos,
using various HOD models and conditional observable functions (COFs).
"""

_defaults = {
    'observables_file': None,
    'obs_min': np.array([8.0]),
    'obs_max': np.array([12.0]),
    'zmin': np.array([0.0]),
    'zmax': np.array([0.2]),
    'nz': 15,
    'nobs': 300,
    'observable_h_unit': '1/h^2',
}

def load_data(file_name):
    """
    Load data from a file.

    Parameters:
    -----------
    file_name : str
        Path to the file to load data from.

    Returns:
    --------
    tuple
        Redshift data, minimum observable data, and maximum observable data.
    """
    z_data, obs_min, obs_max = np.loadtxt(
        file_name, usecols=(0, 1, 2), unpack=True, dtype=float
    )
    return z_data, obs_min, obs_max


class HOD(Framework):
    """
    Base class for Halo Occupation Distribution (HOD) models.
    This class provides the framework for computing various properties of galaxies within dark matter halos.

    Parameters:
    -----------
    mass : array_like
        Array of halo masses.
    dndlnm : array_like
        Halo mass function.
    halo_bias : array_like
        Halo bias.
    z_vec : array_like
        Array of redshifts.
    hod_settings : dict
        Dictionary of HOD settings.
    """
    def __init__(
            self,
            mass=None,
            dndlnm=None,
            halo_bias=None,
            z_vec=None,
            hod_settings: dict =_defaults
        ):
        self.mass = mass
        self.z_vec = z_vec
        self.hod_settings = hod_settings
        self.dndlnm = dndlnm
        self.halo_bias = halo_bias
        
    @parameter("param")
    def z_vec(self, val):
        """
        z_vec : array_like
            Array of redshifts.
        """
        return val

    @parameter("param")
    def mass(self, val):
        """
        mass : array_like
            Array of halo masses.
        """
        if val is None:
            raise ValueError("Mass needs to be specified!")
        # With newaxis we make sure the HOD shape is (nb, nz, nmass)
        return val[np.newaxis, np.newaxis, :]

    @parameter("param")
    def dndlnm(self, val):
        """
        dndlnm : array_like
            Halo mass function.
        """
        if val is None:
            raise ValueError("Halo mass function needs to be specified!")
        return val

    @parameter("param")
    def halo_bias(self, val):
        """
        halo_bias : array_like
            Halo bias.
        """
        if val is None:
            raise ValueError("Halo bias function needs to be specified!")
        return val
        
    @parameter("param")
    def hod_settings(self, val):
        """
        hod_settings : dict
            Dictionary of HOD settings.
        """
        return val
        
    @cached_quantity
    def obs(self):
        """
        Returns the array of observables at which to calculate the CSMF/CLF and integrate over to get the HOD

        Returns:
        --------
        array_like
            array of observables
        """
        obs = np.array([[np.logspace(self.log_obs_min[nb, jz], self.log_obs_max[nb, jz], self.nobs) for jz in range(self.nz)] for nb in range(self.nbins)])
        # With newaxis we make sure the COF shape is (nb, nz, nmass, nobs)
        return obs[:, :, np.newaxis, :]
        
    @cached_quantity
    def dndlnm_int(self):
        """
        Returns the interpolated halo mass function at HOD specific redshifts.

        Returns:
        --------
        array_like
            interpolated halo mass function at self.z
        """
        dndlnm_fnc = interp1d(
            self.z_vec, self.dndlnm, kind='linear', fill_value='extrapolate',
            bounds_error=False, axis=0
        )
        return dndlnm_fnc(self.z)

    @cached_quantity
    def halo_bias_int(self):
        """
        Returns the interpolated halo bias function at HOD specific redshifts.

        Returns:
        --------
        array_like
            interpolated halo bias function at self.z
        """
        halo_bias_fnc = interp1d(
            self.z_vec, self.halo_bias, kind='linear', fill_value='extrapolate',
            bounds_error=False, axis=0
        )
        return halo_bias_fnc(self.z)
        
    @cached_quantity
    def data(self):
        """
        Returns the z_bins, obs_min and obs_max quantities from tabulated observables file.

        Returns:
        --------
        tupple or None
        """
        if self.hod_settings['observables_file'] is not None:
            z_bins, obs_min, obs_max = load_data(self.hod_settings['observables_file'])
            return z_bins, obs_min, obs_max
        else:
            return None

    @cached_quantity
    def nobs(self):
        """
        Sets the number of observables in the observable array.
        
        Returns:
        --------
        array_like
            number of obserables in the observable array
        """
        return self.hod_settings['nobs']

    @cached_quantity
    def nbins(self):
        """
        Returns the number of HOD bins.

        Returns:
        --------
        int
            number of HOD bins
        """
        if self.hod_settings['observables_file'] is not None:
            return 1
        else:
            return len(self.hod_settings['obs_min'])

    @cached_quantity
    def nz(self):
        """
        Sets the number of redshift bins.

        Returns:
        --------
        int
            number of redshift bins
        """
        if self.hod_settings['observables_file'] is not None:
            return len(self.data[0])
        else:
            return self.hod_settings['nz']
        
    @cached_quantity
    def z(self):
        """
        Sets the HOD specific redshift array.

        Returns:
        --------
        array_like
            HOD specific redshifts
        """
        if self.hod_settings['observables_file'] is not None:
            return self.data[0][np.newaxis, :]
        else:
            zmin = self.hod_settings['zmin']
            zmax = self.hod_settings['zmax']
            return np.array([np.linspace(zmin_i, zmax_i, self.nz) for zmin_i, zmax_i in zip(zmin, zmax)])
        
    @cached_quantity
    def log_obs_min(self):
        """
        Sets the min observable limits.

        Returns:
        --------
        array_like
            min observable limits
        """
        if self.hod_settings['observables_file'] is not None:
            return np.log10(self.data[1])[np.newaxis, :]
        else:
            obs_min = self.hod_settings['obs_min']
            return np.array([np.repeat(obs_min_i, self.nz) for obs_min_i in obs_min])

    @cached_quantity
    def log_obs_max(self):
        """
        Sets the max observable limits.

        Returns:
        --------
        array_like
            max observable limits
        """
        if self.hod_settings['observables_file'] is not None:
            return np.log10(self.data[2])[np.newaxis, :]
        else:
            obs_max = self.hod_settings['obs_max']
            return np.array([np.repeat(obs_max_i, self.nz) for obs_max_i in obs_max])

    def _mass_integral(self, hod):
        """
        Compute the mass integral for a given HOD.

        Parameters:
        -----------
        hod : array_like
            Halo Occupation Distribution.

        Returns:
        --------
        array_like
            Integral of the HOD weighted by the halo mass function.
        """
        integrand = hod * self.dndlnm_int / self.mass
        return simpson(integrand, self.mass, axis=-1)

    def _mean_mass_integral(self, hod):
        """
        Compute the mean mass integral for a given HOD.

        Parameters:
        -----------
        hod : array_like
            Halo Occupation Distribution.

        Returns:
        --------
        array_like
            Integral of the HOD weighted by the halo mass function and halo mass.
        """
        integrand = hod * self.dndlnm_int
        return simpson(integrand, self.mass, axis=-1)

    def _bias_integral(self, hod):
        """
        Compute the bias integral for a given HOD.

        Parameters:
        -----------
        hod : array_like
            Halo Occupation Distribution.

        Returns:
        --------
        array_like
            Integral of the HOD weighted by the halo bias and halo mass function.
        """
        bg_integrand = hod * self.halo_bias_int * self.dndlnm_int / self.mass
        return simpson(bg_integrand, self.mass, axis=-1) / self.ntot

    def _interpolate(self, data, fill_value='extrapolate', axis=-1):
        """
        Helper function to interpolate data along a given axis in an array.

        Parameters:
        -----------
        data : array_like
            Data to interpolate.
        fill_value : str, optional
            Value to use for points outside the data range.
        axis : int, optional
            Axis along which to interpolate.

        Returns:
        --------
        array_like
            Interpolated data.
        """
        n_int = [interp1d(self.z[i], data[i], fill_value=fill_value, bounds_error=False, axis=axis) for i in range(self.z.shape[0])]
        return np.array([f(self.z_vec) for f in n_int])

    @cached_quantity
    def _ncen(self):
        """
        Total number density of galaxies with the given HOD, e.g. central and satellite galaxies
        This is an integral over the HOD and the halo mass function to remove the halo mass dependence.
        Nx = int ⟨Nx|M⟩ n(M) dM
        """
        return self._mass_integral(self._compute_hod_cen)

    @cached_quantity
    def _nsat(self):
        """
        Total number density of galaxies with the given HOD, e.g. central and satellite galaxies
        This is an integral over the HOD and the halo mass function to remove the halo mass dependence.
        Nx = int ⟨Nx|M⟩ n(M) dM
        """
        return self._mass_integral(self._compute_hod_sat)

    @cached_quantity
    def _ntot(self):
        """
        Total number density of galaxies with the given HOD, e.g. central and satellite galaxies
        This is an integral over the HOD and the halo mass function to remove the halo mass dependence.
        Nx = int ⟨Nx|M⟩ n(M) dM
        """
        return self._mass_integral(self._compute_hod)

    @cached_quantity
    def _mass_avg_cen(self):
        """
        The mean halo mass for the given population of galaxies, e.g. central and satellite galaxies
        M_mean = int ⟨Nx|M⟩ M n(M) dM
        """
        return self._mean_mass_integral(self._compute_hod_cen)

    @cached_quantity
    def _mass_avg_sat(self):
        """
        The mean halo mass for the given population of galaxies, e.g. central and satellite galaxies
        M_mean = int ⟨Nx|M⟩ M n(M) dM
        """
        return self._mean_mass_integral(self._compute_hod_sat)

    @cached_quantity
    def _mass_avg_tot(self):
        """
        The mean halo mass for the given population of galaxies, e.g. central and satellite galaxies
        M_mean = int ⟨Nx|M⟩ M n(M) dM
        """
        return self._mean_mass_integral(self._compute_hod)

    @cached_quantity
    def _bg_cen(self):
        """
        Mean linear halo bias for the given population of galaxies.
        b_lin_x = int ⟨Nx|M⟩ b_h(M) n(M) dM
        """
        return self._bias_integral(self._compute_hod_cen)

    @cached_quantity
    def _bg_sat(self):
        """
        Mean linear halo bias for the given population of galaxies.
        b_lin_x = int ⟨Nx|M⟩ b_h(M) n(M) dM
        """
        return self._bias_integral(self._compute_hod_sat)

    @cached_quantity
    def _bg_tot(self):
        """
        Mean linear halo bias for the given population of galaxies.
        b_lin_x = int ⟨Nx|M⟩ b_h(M) n(M) dM
        """
        return self._bias_integral(self._compute_hod)

    @cached_quantity
    def number_density_cen(self):
        """
        Compute the number density of central galaxies.

        Returns:
        --------
        array_like
            number density of central galaxies
        """
        return self._interpolate(self._ncen)

    @cached_quantity
    def number_density_sat(self):
        """
        Compute the number density of satellite galaxies.

        Returns:
        --------
        array_like
            number density of satellite galaxies
        """
        return self._interpolate(self._nsat)

    @cached_quantity
    def number_density(self):
        """
        Compute the number density of galaxies.
        
        Returns:
        --------
        array_like 
            number density of all galaxies
        """
        return self._interpolate(self._ntot)

    @cached_quantity
    def f_c(self):
        """
        Fraction of central galaxies.

        Returns:
        --------
        array_like
            fraction of central galaxies
        """
        f_c = self._ncen / self._ntot
        return self._interpolate(f_c, fill_value=0.0)

    @cached_quantity
    def f_s(self):
        """
        Fraction of satellite galaxies.

        Returns:
        --------
        array_like
            fraction of satellite galaxies
        """
        f_s = self._nsat / self._ntot
        return self._interpolate(f_s, fill_value=0.0)

    @cached_quantity
    def avg_halo_mass_cen(self):
        """
        Compute the average halo mass for central galaxies.

        Returns:
        --------
        array_like
            average halo mass for central galaxies
        """
        return self._interpolate(self._mass_avg_cen, fill_value=0.0)

    @cached_quantity
    def avg_halo_mass_sat(self):
        """
        Compute the average halo mass for satellite galaxies.

        Returns:
        --------
        array_like
            average halo mass for satellite galaxies
        """
        return self._interpolate(self._mass_avg_sat, fill_value=0.0)

    @cached_quantity
    def avg_halo_mass(self):
        """
        Compute the average halo mass for galaxies.

        Returns:
        --------
        array_like
            average halo mass for all galaxies
        """
        return self._interpolate(self._mass_avg_tot, fill_value=0.0)

    @cached_quantity
    def galaxy_linear_bias_cen(self):
        """
        Compute the galaxy linear bias for central galaxies.

        Returns:
        --------
        array_like
            galaxy linear bias for central galaxies
        """
        return self._interpolate(self._bg_cen)

    @cached_quantity
    def galaxy_linear_bias_sat(self):
        """
        Compute the galaxy linear bias for satellite galaxies.

        Returns:
        --------
        array_like
            galaxy linear bias for satellite galaxies
        """
        return self._interpolate(self._bg_sat)

    @cached_quantity
    def galaxy_linear_bias(self):
        """
        Compute the galaxy linear bias for galaxies.

        Returns:
        --------
        array_like
            galaxy linear bias for all galaxies
        """
        return self._interpolate(self._bg_tot)

    @cached_quantity
    def hod_cen(self):
        """
        Compute the HOD for central galaxies.

        Returns:
        --------
        array_like
            HOD of central galaxies
        """
        return self._interpolate(self._compute_hod_cen, axis=0)

    @cached_quantity
    def hod_sat(self):
        """
        Compute the HOD for satellite galaxies.

        Returns:
        --------
        array_like
            HOD of satellite galaxies
        """
        return self._interpolate(self._compute_hod_sat, axis=0)

    @cached_quantity
    def hod(self):
        """
        Compute the HOD for galaxies.

        Returns:
        --------
        array_like
            HOD of all galaxies
        """
        return self._interpolate(self._compute_hod, axis=0)

    @cached_quantity
    def stellar_fraction_cen(self):
        """
        Compute the stellar fraction for central galaxies.

        Returns:
        --------
        array_like
            stellar fraction of central galaxies
        """
        if self._compute_stellar_fraction_cen is None:
            return np.zeros((self.nbins, self.z_vec.size, self.mass.shape[-1]))
        else:
            return self._interpolate(self._compute_stellar_fraction_cen, axis=0)

    @cached_quantity
    def stellar_fraction_sat(self):
        """
        Compute the stellar fraction for satellite galaxies.

        Returns:
        --------
        array_like
            stellar fraction of satellite galaxies
        """
        if self._compute_stellar_fraction_sat is None:
            return np.zeros((self.nbins, self.z_vec.size, self.mass.shape[-1]))
        else:
            return self._interpolate(self._compute_stellar_fraction_sat, axis=0)

    @cached_quantity
    def stellar_fraction(self):
        """
        Compute the stellar fraction for galaxies.

        Returns:
        --------
        array_like
            stellar fraction of all galaxies
        """
        if self._compute_stellar_fraction is None:
            return np.zeros((self.nbins, self.z_vec.size, self.mass.shape[-1]))
        else:
            return self._interpolate(self._compute_stellar_fraction, axis=0)

    @property
    def _compute_hod_cen(self):
        """
        Compute the HOD for central galaxies.
        """
        raise NotImplementedError

    @property
    def _compute_hod_sat(self):
        """
        Compute the HOD for satellite galaxies.
        """
        raise NotImplementedError

    @property
    def _compute_hod(self):
        """
        Compute the total HOD by summing central and satellite HODs.
        """
        return self._compute_hod_cen + self._compute_hod_sat

    @property
    def _compute_stellar_fraction_cen(self):
        return None

    @property
    def _compute_stellar_fraction_sat(self):
        return None

    @property
    def _compute_stellar_fraction(self):
        return None

class Cacciato(HOD):
    """
    The conditional observable functions (COFs) tell us how many galaxies with the observed property O, exist in haloes of
    mass M: Φ(O|M).
    Integrating over the observable will give us the total number of galaxies in haloes of a given mass, the so-called
    Halo Occupation Distribution (HOD).
    The observable can be galaxy stellar mass or galaxy luminosity or possibly other properties of galaxies.
    Note that the general mathematical form of the COFs might not hold for other observables.
    COF is different for central and satellite galaxies. The total COF can be written as the sum of the two:
    Φ(O|M) = Φc(O|M) + Φs(O|M)
    The halo mass dependence comes in through pivot observable values denoted by *, e.g. O∗c, O∗s
    
    Parameters:
    -----------
    log10_obs_norm_c : float
        Log10 of the normalization for central galaxies.
    log10_m_ch : float
        Log10 of the characteristic mass.
    g1 : float
        Low mass slope parameter for central galaxies.
    g2 : float
        High mass slope parameter for central galaxies.
    sigma_log10_O_c : float
        Scatter in log10 of the observable for central galaxies.
    norm_s : float
        Normalization for satellite galaxies.
    pivot : float
        Pivot mass for the normalization of the stellar mass function.
    alpha_s : float
        Slope parameter for satellite galaxies.
    beta_s : float
        Exponent parameter for satellite galaxies.
    b0 : float
        Parameter for the conditional stellar mass function.
    b1 : float
        Parameter for the conditional stellar mass function.
    b2 : float
        Parameter for the conditional stellar mass function.
    A_cen : float, optional
        Decorated HOD assembly bias parameter for central galaxies.
    A_sat : float, optional
        Decorated HOD assembly bias parameter for satellite galaxies.
    hod_kwargs : dict
        Additional keyword arguments for the HOD class.
    """
    def __init__(
            self,
            log10_obs_norm_c=9.95,
            log10_m_ch=11.24,
            g1=3.18,
            g2=0.245,
            sigma_log10_O_c=0.157,
            norm_s=0.562,
            pivot=12.0,
            alpha_s=-1.18,
            beta_s=2,
            b0=-1.17,
            b1=1.53,
            b2=-0.217,
            A_cen=None,
            A_sat=None,
            **hod_kwargs
        ):

        # Call super init MUST BE DONE FIRST.
        super().__init__(**hod_kwargs)
        # Set all given parameters.

        # centrals
        # all observable masses in units of log10(M_sun h^-2)
        self.log10_m_ch = log10_m_ch  # log10_m_ch
        self.g1 = g1  # gamma_1
        self.g2 = g2  # gamma_2
        self.log10_obs_norm_c = log10_obs_norm_c  # O_0, O_norm_c
        self.sigma_log10_O_c = sigma_log10_O_c  # sigma_log10_O_c
        # satellites
        self.norm_s = norm_s  # extra normalisation factor for satellites
        self.pivot = pivot  # pivot mass for the normalisation of the stellar mass function: ϕ∗s
        self.alpha_s = alpha_s  # goes into the conditional stellar mass function COF_sat(M*|M)
        self.beta_s = beta_s  # goes into the conditional stellar mass function COF_sat(M*|M)
        # log10[ϕ∗s(M)] = b0 + b1(log10 m_p)+ b2(log10 m_p)^2, m_p = M/pivot
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        # Decorated HOD assembly bias parameters
        self.A_cen = A_cen
        self.A_sat = A_sat

    @parameter("param")
    def log10_m_ch(self, val):
        """
        log10_m_ch : float
            Log10 of the characteristic mass.
        """
        return val
        
    @parameter("param")
    def g1(self, val):
        """
        g1 : float
            Low mass slope parameter for central galaxies.
        """
        return val
        
    @parameter("param")
    def g2(self, val):
        """
        g2 : float
            High mass slope parameter for central galaxies.
        """
        return val
        
    @parameter("param")
    def log10_obs_norm_c(self, val):
        """
        log10_obs_norm_c : float
            Log10 of the normalization for central galaxies.
        """
        return val
        
    @parameter("param")
    def sigma_log10_O_c(self, val):
        """
        sigma_log10_O_c : float
            Scatter in log10 of the observable for central galaxies.
        """
        return val
    
    @parameter("param")
    def norm_s(self, val):
        """
        norm_s : float
            Normalization for satellite galaxies.
        """
        return val
        
    @parameter("param")
    def pivot(self, val):
        """
        pivot : float
            Pivot mass for the normalization of the stellar mass function.
        """
        return val
        
    @parameter("param")
    def alpha_s(self, val):
        """
        alpha_s : float
            Slope parameter for satellite galaxies.
        """
        return val
        
    @parameter("param")
    def beta_s(self, val):
        """
        beta_s : float
            Exponent parameter for satellite galaxies.
        """
        return val
        
    @parameter("param")
    def b0(self, val):
        """
        b0 : float
            Parameter for the conditional stellar mass function.
        """
        return val
        
    @parameter("param")
    def b1(self, val):
        """
        b1 : float
            Parameter for the conditional stellar mass function.
        """
        return val
        
    @parameter("param")
    def b2(self, val):
        """
        b2 : float
            Parameter for the conditional stellar mass function.
        """
        return val
        
    @parameter("param")
    def A_cen(self, val):
        """
        A_cen : float
            Decorated HOD assembly bias parameter for central galaxies.
        """
        return val
        
    @parameter("param")
    def A_sat(self, val):
        """
        A_sat : float
            Decorated HOD assembly bias parameter for satellite galaxies.
        """
        return val
        
    @cached_quantity
    def M_char(self):
        """
        Return the 10.**log10_m_ch

        Returns:
        --------
        array_like
            10.**log10_m_ch
        """
        return 10.0**self.log10_m_ch

    @cached_quantity
    def Obs_norm_c(self):
        """
        Return the 10.**log10_obs_norm_c

        Returns:
        --------
        array_like
            10.**log10_obs_norm_c
        """
        return 10.0**self.log10_obs_norm_c

    @cached_quantity
    def COF_cen(self):
        """
        COF for Central galaxies.
        eq 17 of D23: 2210.03110:
        Φc(O|M) = 1/[√(2π) ln(10) σ_c O] exp[ -log(O/O∗c)^2/ (2 σ_c^2) ]
        Note Φc(O|M) is unitless.

        Returns:
        --------
        array_like
            COF of central galaxies
        """
        mean_obs_c = self.cal_mean_obs_c[:, :, :, np.newaxis]  # O∗c
        COF_c = (1.0 / (np.sqrt(2.0 * np.pi) * np.log(10.0) * self.sigma_log10_O_c * self.obs) *
                 np.exp(-(np.log10(self.obs / mean_obs_c))**2 / (2.0 * self.sigma_log10_O_c**2)))
        return COF_c

    @cached_quantity
    def COF_sat(self):
        """
        COF for satellite galaxies.
        eq 18 of D23: 2210.03110:
        Φs(O|M) = ϕ∗s/O∗s (O/O∗s)^α_s exp [−(O/O∗s)^2], O*s is O∗s(M) = 0.56 O∗c(M)
        Note Φs(O|M) is unitless.

        Returns:
        --------
        array_like
            COF of satellite galaxies
        """
        obs_s_star = self.norm_s * self.cal_mean_obs_c[:, :, :, np.newaxis]
        obs_tilde = self.obs / obs_s_star
        phi_star_val = self.phi_star_s[:, :, :, np.newaxis]
        COF_s = (phi_star_val / obs_s_star) * (obs_tilde**self.alpha_s) * np.exp(-obs_tilde**self.beta_s)
        return COF_s

    @cached_quantity
    def COF(self):
        """
        Total COF, sum of central and satellite COFs.

        Returns:
        --------
        array_like
            COF of all galaxies
        """
        return self.COF_cen + self.COF_sat

    @cached_quantity
    def obs_func_cen(self):
        """
        The Observable function, this is Φs(O|M), Φc(O|M) integrated over the halo mass weighted
        with the Halo Mass Function (HMF) to give:  Φs(O),Φc(O)
        Φx(O) =int Φx(O|M) n(M) dM,
        dndlnm is basically n(M) x mass, it is the output of hmf
        The differential mass function in terms of natural log of m,
        len=len(m) [units \(h^3 Mpc^{-3}\)]
        dn(m)/ dln m eq1 of 1306.6721
        obs_func unit is h^3 Mpc^{-3} dex^-1

        Returns:
        --------
        array_like
            observable function of central galaxies
        """
        integrand = self.COF_cen * self.dndlnm_int[:, :, :, np.newaxis] / self.mass[:, :, :, np.newaxis]
        obs_function = simpson(integrand, self.mass[:, :, :, np.newaxis], axis=-2)
        return obs_function

    @cached_quantity
    def obs_func_sat(self):
        """
        The Observable function, this is Φs(O|M), Φc(O|M) integrated over the halo mass weighted
        with the Halo Mass Function (HMF) to give:  Φs(O),Φc(O)
        Φx(O) =int Φx(O|M) n(M) dM,
        dndlnm is basically n(M) x mass, it is the output of hmf
        The differential mass function in terms of natural log of m,
        len=len(m) [units \(h^3 Mpc^{-3}\)]
        dn(m)/ dln m eq1 of 1306.6721
        obs_func unit is h^3 Mpc^{-3} dex^-1

        Returns:
        --------
        array_like
            observable function of satellite galaxies
        """
        integrand = self.COF_sat * self.dndlnm_int[:, :, :, np.newaxis] / self.mass[:, :, :, np.newaxis]
        obs_function = simpson(integrand, self.mass[:, :, :, np.newaxis], axis=-2)
        return obs_function

    @cached_quantity
    def obs_func(self):
        """
        The Observable function, this is Φs(O|M), Φc(O|M) integrated over the halo mass weighted
        with the Halo Mass Function (HMF) to give:  Φs(O),Φc(O)
        Φx(O) =int Φx(O|M) n(M) dM,
        dndlnm is basically n(M) x mass, it is the output of hmf
        The differential mass function in terms of natural log of m,
        len=len(m) [units \(h^3 Mpc^{-3}\)]
        dn(m)/ dln m eq1 of 1306.6721
        obs_func unit is h^3 Mpc^{-3} dex^-1

        Returns:
        --------
        array_like
            observable function of all galaxies
        """
        integrand = self.COF * self.dndlnm_int[:, :, :, np.newaxis] / self.mass[:, :, :, np.newaxis]
        obs_function = simpson(integrand, self.mass[:, :, :, np.newaxis], axis=-2)
        return obs_function

    @cached_quantity
    def cal_mean_obs_c(self):
        """
        eqs 19 of D23: 2210.03110
        O∗c(M) = O_0 (M/M1)^γ1 / [1 + (M/M1)]^(γ1−γ2)
        To get the values for the satellite call this * hod_par.norm_s
        O∗s(M) = 0.56 O∗c(M)
        Here M1 is a characteristic mass scale, and O_0 is the normalization.
        used to be mor

        (observable can be galaxy luminosity or stellar mass)
        returns the observable given halo mass. Assumed to be a double power law with characteristic
        scale m_1, normalisation m_0 and slopes g_1 and g_2

        Returns:
        --------
        array_like
            observable given halo mass - SMHM relation
        """
        mean_obs_c = (self.Obs_norm_c * (self.mass / self.M_char)**self.g1 /
                      (1.0 + (self.mass / self.M_char))**(self.g1 - self.g2))
        return mean_obs_c

    @cached_quantity
    def phi_star_s(self):
        """
        pivot COF used in eq 21 of D23: 2210.03110
        using a bias expansion around the pivot mass
        eq 22 of D23: 2210.03110
        log[ϕ∗s(M)] = b0 + b1(log m13) , m13 is logM_pivot, m13 = M/(hod_par.pivot M⊙ h−1)

        Returns:
        --------
        array_like
            normalisation of satellite COF function
        """
        logM_pivot = np.log10(self.mass) - self.pivot
        log_phi_s = self.b0 + self.b1 * logM_pivot + self.b2 * (logM_pivot**2.0)
        return 10.0**log_phi_s

    @property
    def _compute_hod_cen(self):
        """
        The HOD is computed by integrating over the COFs
        eq 23 of D23: 2210.03110
        ⟨Nx|M⟩ =int_{O_low}^{O_high} Φx(O|M) dO
        """
        N_cen = simpson(self.COF_cen, self.obs)
        if self.A_cen is not None:
            delta_pop_c = self.A_cen * np.fmin(N_cen, 1.0 - N_cen)
            N_cen = N_cen + delta_pop_c
        return N_cen

    @property
    def _compute_hod_sat(self):
        """
        The HOD is computed by integrating over the COFs
        eq 23 of D23: 2210.03110
        ⟨Nx|M⟩ =int_{O_low}^{O_high} Φx(O|M) dO
        """
        N_sat = simpson(self.COF_sat, self.obs)
        if self.A_sat is not None:
            delta_pop_s = self.A_sat * N_sat
            N_sat = N_sat + delta_pop_s
        return N_sat

    @cached_quantity
    def _compute_stellar_fraction_cen(self):
        """
        The mean value of the observable for the given galaxy population for a given halo mass.
        O is weighted by the number of galaxies with the property O for each halo mass: Φx(O|M)
        f_star = int_{O_low}^{O_high} Φx(O|M) O dO
        """
        return simpson(self.COF_cen * self.obs, self.obs) / self.mass

    @cached_quantity
    def _compute_stellar_fraction_sat(self):
        """
        The mean value of the observable for the given galaxy population for a given halo mass.
        O is weighted by the number of galaxies with the property O for each halo mass: Φx(O|M)
        f_star = int_{O_low}^{O_high} Φx(O|M) O dO
        """
        return simpson(self.COF_sat * self.obs, self.obs) / self.mass

    @cached_quantity
    def _compute_stellar_fraction(self):
        """
        The mean value of the observable for the given galaxy population for a given halo mass.
        O is weighted by the number of galaxies with the property O for each halo mass: Φx(O|M)
        f_star = int_{O_low}^{O_high} Φx(O|M) O dO
        """
        return simpson(self.COF * self.obs, self.obs) / self.mass

class Simple(HOD):
    """
    Simple HOD model
    
    Parameters:
    -----------
    log10_Mmin : array_like
        Log10 of the minimum mass for central galaxies.
    log10_Msat : array_like
        Log10 of the minimum mass for satellite galaxies.
    alpha : array_like
        Slope parameter for satellite galaxies.
    A_cen : float, optional
        Decorated HOD assembly bias parameter for central galaxies.
    A_sat : float, optional
        Decorated HOD assembly bias parameter for satellite galaxies.
    hod_kwargs : dict
        Additional keyword arguments for the HOD class.
    """
    def __init__(
            self,
            log10_Mmin=12.0,
            log10_Msat=13.0,
            alpha=1.0,
            A_cen=None,
            A_sat=None,
            **hod_kwargs
        ):
        # Call super init MUST BE DONE FIRST.
        super().__init__(**hod_kwargs)
        self.log10_Mmin = log10_Mmin
        self.log10_Msat = log10_Msat
        self.alpha = alpha
        # Decorated HOD assembly bias parameters
        self.A_cen = A_cen
        self.A_sat = A_sat
        
    @parameter("param")
    def log10_Mmin(self, val):
        """
        log10_Mmin : array_like
            Log10 of the minimum mass for central galaxies.
        """
        if not hasattr(val, "__len__"):
            val = [val]
        return np.array(val)
        
    @parameter("param")
    def log10_Msat(self, val):
        """
        og10_Msat : array_like
            Log10 of the minimum mass for satellite galaxies.
        """
        if not hasattr(val, "__len__"):
            val = [val]
        return np.array(val)
        
    @parameter("param")
    def alpha(self, val):
        """
        alpha : array_like
            Slope parameter for satellite galaxies.
        """
        if not hasattr(val, "__len__"):
            val = [val]
        return np.array(val)[:, np.newaxis, np.newaxis]
        
    @parameter("param")
    def A_cen(self, val):
        """
        A_cen : float
            Decorated HOD assembly bias parameter for central galaxies.
        """
        return val
        
    @parameter("param")
    def A_sat(self, val):
        """
        A_sat : float
            Decorated HOD assembly bias parameter for satellite galaxies.
        """
        return val
        
    @cached_quantity
    def Mmin(self):
        """
        Return the Mmin parameter in the correct shape and units.

        Returns:
        --------
        array_like
            Mmin
        """
        return 10.0**self.log10_Mmin[:, np.newaxis, np.newaxis]
        
    @cached_quantity
    def Msat(self):
        """
        Return the Msat parameter in the correct shape and units.

        Returns:
        --------
        array_like
            Msat
        """
        return 10.0**self.log10_Msat[:, np.newaxis, np.newaxis]

    @cached_quantity
    def _compute_hod_cen(self):
        """
        Compute the HOD for central galaxies.
        """
        N_cen = np.heaviside(self.mass - self.Mmin, 1.0)
        if self.A_cen is not None:
            delta_pop_c = self.A_cen * np.fmin(N_cen, 1.0 - N_cen)
            N_cen = N_cen + delta_pop_c
        return np.tile(N_cen, (self.nz, 1))

    @cached_quantity
    def _compute_hod_sat(self):
        """
        Compute the HOD for satellite galaxies.
        """
        N_sat = self.compute_hod_cen * (self.mass / self.Msat)**self.alpha
        if self.A_sat is not None:
            delta_pop_s = self.A_sat * N_sat
            N_sat = N_sat + delta_pop_s
        return np.tile(N_sat, (self.nz, 1))
        

class Zehavi(HOD):
    """
    HOD model from Zehavi et al. (2004; https://arxiv.org/abs/astro-ph/0703457)
    Same as Zheng model in the limit that sigma=0 and M0=0
    Mean number of central galaxies is only ever 0 or 1 in this HOD
    
    Parameters:
    -----------
    log10_Mmin : array_like
        Log10 of the minimum mass for central galaxies.
    log10_Msat : array_like
        Log10 of the minimum mass for satellite galaxies.
    alpha : array_like
        Slope parameter for satellite galaxies.
    A_cen : float, optional
        Decorated HOD assembly bias parameter for central galaxies.
    A_sat : float, optional
        Decorated HOD assembly bias parameter for satellite galaxies.
    hod_kwargs : dict
        Additional keyword arguments for the HOD class.
    """
    def __init__(
            self,
            log10_Mmin=12.0,
            log10_Msat=13.0,
            alpha=1.0,
            A_cen=None,
            A_sat=None,
            **hod_kwargs
        ):
        # Call super init MUST BE DONE FIRST.
        super().__init__(**hod_kwargs)
        self.log10_Mmin = log10_Mmin
        self.log10_Msat = log10_Msat
        self.alpha = alpha
        # Decorated HOD assembly bias parameters
        self.A_cen = A_cen
        self.A_sat = A_sat
        
    @parameter("param")
    def log10_Mmin(self, val):
        """
        log10_Mmin : array_like
            Log10 of the minimum mass for central galaxies.
        """
        if not hasattr(val, "__len__"):
            val = [val]
        return np.array(val)
        
    @parameter("param")
    def log10_Msat(self, val):
        """
        log10_Msat : array_like
            Log10 of the minimum mass for satellite galaxies.
        """
        if not hasattr(val, "__len__"):
            val = [val]
        return np.array(val)
        
    @parameter("param")
    def alpha(self, val):
        """
        alpha : array_like
            Slope parameter for satellite galaxies.
        """
        if not hasattr(val, "__len__"):
            val = [val]
        return np.array(val)[:, np.newaxis, np.newaxis]
        
    @parameter("param")
    def A_cen(self, val):
        """
        A_cen : float
            Decorated HOD assembly bias parameter for central galaxies.
        """
        return val
        
    @parameter("param")
    def A_sat(self, val):
        """
        A_sat : float
            Decorated HOD assembly bias parameter for satellite galaxies.
        """
        return val
        
    @cached_quantity
    def Mmin(self):
        """
        Return the Mmin parameter in the correct shape and units.

        Returns:
        --------
        array_like
            Mmin
        """
        return 10.0**self.log10_Mmin[:, np.newaxis, np.newaxis]
        
    @cached_quantity
    def Msat(self):
        """
        Return the Msat parameter in the correct shape and units.

        Returns:
        --------
        array_like
            Msat
        """
        return 10.0**self.log10_Msat[:, np.newaxis, np.newaxis]

    @cached_quantity
    def _compute_hod_cen(self):
        """
        Compute the HOD for central galaxies.
        """
        N_cen = np.heaviside(self.mass - self.Mmin, 1.0)
        if self.A_cen is not None:
            delta_pop_c = self.A_cen * np.fmin(N_cen, 1.0 - N_cen)
            N_cen = N_cen + delta_pop_c
        return np.tile(N_cen, (self.nz, 1))

    @cached_quantity
    def _compute_hod_sat(self):
        """
        Compute the HOD for satellite galaxies.
        """
        N_sat = (self.mass / self.Msat)**self.alpha
        if self.A_sat is not None:
            delta_pop_s = self.A_sat * N_sat
            N_sat = N_sat + delta_pop_s
        return np.tile(N_sat, (self.nz, 1))


class Zheng(HOD):
    """
    Zheng et al. (2005; https://arxiv.org/abs/astro-ph/0408564) HOD model
    
    Parameters:
    -----------
    log10_Mmin : array_like
        Log10 of the minimum mass for central galaxies.
    log10_M0 : array_like
        Log10 of the cutoff mass for satellite galaxies.
    log10_M1 : array_like
        Log10 of the normalization mass for satellite galaxies.
    sigma : array_like
        Scatter in the central galaxy occupation.
    alpha : array_like
        Slope parameter for satellite galaxies.
    A_cen : float, optional
        Decorated HOD assembly bias parameter for central galaxies.
    A_sat : float, optional
        Decorated HOD assembly bias parameter for satellite galaxies.
    hod_kwargs : dict
        Additional keyword arguments for the HOD class.
    """
    def __init__(
            self,
            log10_Mmin=12.0,
            log10_M0=12.0,
            log10_M1=13.0,
            sigma=0.15,
            alpha=1.0,
            A_cen=None,
            A_sat=None,
            **hod_kwargs
        ):
        # Call super init MUST BE DONE FIRST.
        super().__init__(**hod_kwargs)
        self.log10_Mmin = log10_Mmin
        self.log10_M0 = log10_M0
        self.log10_M1 = log10_M1
        self.sigma = sigma
        self.alpha = alpha
        # Decorated HOD assembly bias parameters
        self.A_cen = A_cen
        self.A_sat = A_sat

    @parameter("param")
    def log10_Mmin(self, val):
        """
        log10_Mmin : array_like
            Log10 of the minimum mass for central galaxies.
        """
        if not hasattr(val, "__len__"):
            val = [val]
        return np.array(val)
        
    @parameter("param")
    def log10_M0(self, val):
        """
        log10_M0 : array_like
            Log10 of the cutoff mass for satellite galaxies.
        """
        if not hasattr(val, "__len__"):
            val = [val]
        return np.array(val)
    
    @parameter("param")
    def log10_M1(self, val):
        """
        log10_M1 : array_like
            Log10 of the normalization mass for satellite galaxies.
        """
        if not hasattr(val, "__len__"):
            val = [val]
        return np.array(val)
        
    @parameter("param")
    def alpha(self, val):
        """
        alpha : array_like
            Slope parameter for satellite galaxies.
        """
        if not hasattr(val, "__len__"):
            val = [val]
        return np.array(val)[:, np.newaxis, np.newaxis]
        
    @parameter("param")
    def sigma(self, val):
        """
        sigma : array_like
            Scatter in the central galaxy occupation.
        """
        if not hasattr(val, "__len__"):
            val = [val]
        return np.array(val)[:, np.newaxis, np.newaxis]
        
    @parameter("param")
    def A_cen(self, val):
        """
        A_cen : float
            Decorated HOD assembly bias parameter for central galaxies.
        """
        return val
        
    @parameter("param")
    def A_sat(self, val):
        """
        A_sat : float
            Decorated HOD assembly bias parameter for satellite galaxies.
        """
        return val
        
    @cached_quantity
    def Mmin(self):
        """
        Return the Mmin parameter in the correct shape and units.

        Returns:
        --------
        array_like
            Mmin
        """
        return 10.0**self.log10_Mmin[:, np.newaxis, np.newaxis]
        
    @cached_quantity
    def M0(self):
        """
        Return the M0 parameter in the correct shape and units.

        Returns:
        --------
        array_like
            M0
        """
        return 10.0**self.log10_M0[:, np.newaxis, np.newaxis]
        
    @cached_quantity
    def M1(self):
        """
        Return the M1 parameter in the correct shape and units.ž

        Returns:
        --------
        array_like
            M1
        """
        return 10.0**self.log10_M1[:, np.newaxis, np.newaxis]

    @cached_quantity
    def _compute_hod_cen(self):
        """
        Compute the HOD for central galaxies.
        """
        N_cen = np.where(self.sigma == 0.0, np.heaviside(self.mass - self.Mmin, 1.0), 0.5 * (1.0 + erf(np.log10(self.mass / self.Mmin) / self.sigma)))
        if self.A_cen is not None:
            delta_pop_c = self.A_cen * np.fmin(N_cen, 1.0 - N_cen)
            N_cen = N_cen + delta_pop_c
        return np.tile(N_cen, (self.nz, 1))

    @cached_quantity
    def _compute_hod_sat(self):
        """
        Compute the HOD for satellite galaxies.
        """
        N_sat = (np.heaviside(self.mass - self.M0, 1.0) * (self.mass - self.M0) / self.M1)**self.alpha
        if self.A_sat is not None:
            delta_pop_s = self.A_sat * N_sat
            N_sat = N_sat + delta_pop_s
        return np.tile(N_sat, (self.nz, 1))


class Zhai(HOD):
    """
    HOD model from Zhai et al. (2017; https://arxiv.org/abs/1607.05383)
    
    Parameters:
    -----------
    log10_Mmin : array_like
        Log10 of the minimum mass for central galaxies.
    log10_Msat : array_like
        Log10 of the minimum mass for satellite galaxies.
    log10_Mcut : array_like
        Log10 of the cutoff mass for satellite galaxies.
    sigma : array_like
        Scatter in the central galaxy occupation.
    alpha : array_like
        Slope parameter for satellite galaxies.
    A_cen : float, optional
        Decorated HOD assembly bias parameter for central galaxies.
    A_sat : float, optional
        Decorated HOD assembly bias parameter for satellite galaxies.
    hod_kwargs : dict
        Additional keyword arguments for the HOD model.
    """
    def __init__(
            self,
            log10_Mmin=13.68,
            log10_Msat=14.87,
            log10_Mcut=12.32,
            sigma=0.82,
            alpha=0.41,
            A_cen=None,
            A_sat=None,
            **hod_kwargs
        ):
        # Call super init MUST BE DONE FIRST.
        super().__init__(**hod_kwargs)
        self.log10_Mmin = log10_Mmin
        self.log10_Msat = log10_Msat
        self.log10_Mcut = log10_Mcut
        self.sigma = sigma
        self.alpha = alpha
        # Decorated HOD assembly bias parameters
        self.A_cen = A_cen
        self.A_sat = A_sat
        
    @parameter("param")
    def log10_Mmin(self, val):
        """
        log10_Mmin : array_like
            Log10 of the minimum mass for central galaxies.
        """
        if not hasattr(val, "__len__"):
            val = [val]
        return np.array(val)
        
    @parameter("param")
    def log10_Msat(self, val):
        """
        log10_Msat : array_like
            Log10 of the minimum mass for satellite galaxies.
        """
        if not hasattr(val, "__len__"):
            val = [val]
        return np.array(val)
    
    @parameter("param")
    def log10_Mcut(self, val):
        """
        log10_Mcut : array_like
            Log10 of the cutoff mass for satellite galaxies.
        """
        if not hasattr(val, "__len__"):
            val = [val]
        return np.array(val)
        
    @parameter("param")
    def alpha(self, val):
        """
        alpha : array_like
            Slope parameter for satellite galaxies.
        """
        if not hasattr(val, "__len__"):
            val = [val]
        return np.array(val)[:, np.newaxis, np.newaxis]
        
    @parameter("param")
    def sigma(self, val):
        """
        sigma : array_like
            Scatter in the central galaxy occupation.
        """
        if not hasattr(val, "__len__"):
            val = [val]
        return np.array(val)[:, np.newaxis, np.newaxis]
        
    @parameter("param")
    def A_cen(self, val):
        """
        A_cen : float, optional
            Decorated HOD assembly bias parameter for central galaxies.
        """
        return val
        
    @parameter("param")
    def A_sat(self, val):
        """
        A_sat : float, optional
            Decorated HOD assembly bias parameter for satellite galaxies.
        """
        return val
        
    @cached_quantity
    def Mmin(self):
        """
        Return the Mmin parameter in the correct shape and units.

        Returns:
        --------
        array_like
            Mmin
        """
        return 10.0**self.log10_Mmin[:, np.newaxis, np.newaxis]
        
    @cached_quantity
    def Msat(self):
        """
        Return the Msat parameter in the correct shape and units.

        Returns:
        --------
        array_like
            Msat
        """
        return 10.0**self.log10_Msat[:, np.newaxis, np.newaxis]
        
    @cached_quantity
    def Mcut(self):
        """
        Return the Mcut parameter in the correct shape and units.

        Returns:
        --------
        array_like
            Mcut
        """
        return 10.0**self.log10_Mcut[:, np.newaxis, np.newaxis]

    @cached_quantity
    def _compute_hod_cen(self):
        """
        Compute the HOD for central galaxies.
        """
        N_cen = np.where(self.sigma == 0.0, np.heaviside(self.mass - self.Mmin, 1.0), 0.5 * (1.0 + erf(np.log10(self.mass / self.Mmin) / self.sigma)))
        if self.A_cen is not None:
            delta_pop_c = self.A_cen * np.fmin(N_cen, 1.0 - N_cen)
            N_cen = N_cen + delta_pop_c
        return np.tile(N_cen, (self.nz, 1))

    @cached_quantity
    def _compute_hod_sat(self):
        """
        Compute the HOD for satellite galaxies.
        """
        # Paper has a Nc(M) multiplication, but I think the central condition covers this
        N_sat = ((self.mass / self.Msat)**self.alpha) * np.exp(-self.Mcut / self.mass)
        if self.A_sat is not None:
            delta_pop_s = self.A_sat * N_sat
            N_sat = N_sat + delta_pop_s
        return np.tile(N_sat, (self.nz, 1))
