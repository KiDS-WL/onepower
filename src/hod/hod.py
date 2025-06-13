from functools import cached_property
import numpy as np
from scipy.integrate import simpson
from scipy.special import erf
from scipy.interpolate import interp1d

"""
A module for computing Halo Occupation Distribution (HOD) models.
This module provides classes and functions to calculate properties of galaxies within dark matter halos,
using various HOD models and conditional observable functions (COFs).
"""

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


class HOD:
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
            mass = None,
            dndlnm = None,
            halo_bias = None,
            z_vec = None,
            hod_settings = {}
        ):
        if mass is None or dndlnm is None:
            raise ValueError("Mass and halo mass function need to be specified!")

        # Set all given parameters.
        self._process_hod_settings(hod_settings)

        obs = np.array([[np.logspace(self.log_obs_min[nb, jz], self.log_obs_max[nb, jz], self.nobs) for jz in range(self.nz)] for nb in range(self.nbins)])
        # With newaxis we make sure the COF shape is (nb, nz, nmass, nobs)
        self.obs = obs[:, :, np.newaxis, :]
        # With newaxis we make sure the HOD shape is (nb, nz, nmass)
        self.mass = mass[np.newaxis, np.newaxis, :]
        self.z_vec = z_vec
        
        dndlnm_int = interp1d(
            self.z_vec, dndlnm, kind='linear', fill_value='extrapolate',
            bounds_error=False, axis=0
        )
        halo_bias_int = interp1d(
            self.z_vec, halo_bias, kind='linear', fill_value='extrapolate',
            bounds_error=False, axis=0
        )
        self.dndlnm = dndlnm_int(self.z)
        self.halo_bias = halo_bias_int(self.z)
        
    def _process_hod_settings(self, hod_settings):
        """
        Process the HOD settings.

        Parameters:
        -----------
        hod_settings : dict
            Dictionary of HOD settings.
        """
        self.nobs = hod_settings['nobs']
        if hod_settings['observables_file'] is not None:
            z_bins, obs_min, obs_max = load_data(hod_settings['observables_file'])
            self.nz = len(z_bins)
            self.z = z_bins[np.newaxis, :]
            self.nbins = 1
            self.log_obs_min = np.log10(obs_min)[np.newaxis, :]
            self.log_obs_max = np.log10(obs_max)[np.newaxis, :]
            hod_settings['obs_min'] = np.log10(obs_min)
            hod_settings['obs_max'] = np.log10(obs_max)
            hod_settings['zmin'] = np.array([z_bins.min()])
            hod_settings['zmax'] = np.array([z_bins.max()])
        else:
            self.nz = hod_settings['nz']
            obs_min = hod_settings['obs_min']
            obs_max = hod_settings['obs_max']
            zmin = hod_settings['zmin']
            zmax = hod_settings['zmax']
            if not np.all(np.array([obs_min.size, obs_max.size, zmin.size, zmax.size]) == obs_min.size):
                raise ValueError('obs_min, obs_max, zmin, and zmax need to be of the same length.')
            self.nbins = len(obs_min)
            self.z = np.array([np.linspace(zmin_i, zmax_i, self.nz) for zmin_i, zmax_i in zip(zmin, zmax)])
            self.log_obs_min = np.array([np.repeat(obs_min_i, self.nz) for obs_min_i in obs_min])
            self.log_obs_max = np.array([np.repeat(obs_max_i, self.nz) for obs_max_i in obs_max])
        
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
        """        integrand = hod * self.dndlnm / self.mass
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
        integrand = hod * self.dndlnm
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
        """        bg_integrand = hod * self.halo_bias * self.dndlnm / self.mass
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

    @property
    def ncen(self):
        """
        Total number density of galaxies with the given HOD, e.g. central and satellite galaxies
        This is an integral over the HOD and the halo mass function to remove the halo mass dependence.
        Nx = int ⟨Nx|M⟩ n(M) dM
        """
        return self._mass_integral(self._compute_hod_cen)

    @property
    def nsat(self):
        """
        Total number density of galaxies with the given HOD, e.g. central and satellite galaxies
        This is an integral over the HOD and the halo mass function to remove the halo mass dependence.
        Nx = int ⟨Nx|M⟩ n(M) dM
        """
        return self._mass_integral(self._compute_hod_sat)

    @property
    def ntot(self):
        """
        Total number density of galaxies with the given HOD, e.g. central and satellite galaxies
        This is an integral over the HOD and the halo mass function to remove the halo mass dependence.
        Nx = int ⟨Nx|M⟩ n(M) dM
        """
        return self._mass_integral(self._compute_hod)

    @property
    def mass_avg_cen(self):
        """
        The mean halo mass for the given population of galaxies, e.g. central and satellite galaxies
        M_mean = int ⟨Nx|M⟩ M n(M) dM
        """
        return self._mean_mass_integral(self._compute_hod_cen)

    @property
    def mass_avg_sat(self):
        """
        The mean halo mass for the given population of galaxies, e.g. central and satellite galaxies
        M_mean = int ⟨Nx|M⟩ M n(M) dM
        """
        return self._mean_mass_integral(self._compute_hod_sat)

    @property
    def mass_avg_tot(self):
        """
        The mean halo mass for the given population of galaxies, e.g. central and satellite galaxies
        M_mean = int ⟨Nx|M⟩ M n(M) dM
        """
        return self._mean_mass_integral(self._compute_hod)

    @property
    def bg_cen(self):
        """
        Mean linear halo bias for the given population of galaxies.
        b_lin_x = int ⟨Nx|M⟩ b_h(M) n(M) dM
        """
        return self._bias_integral(self._compute_hod_cen)

    @property
    def bg_sat(self):
        """
        Mean linear halo bias for the given population of galaxies.
        b_lin_x = int ⟨Nx|M⟩ b_h(M) n(M) dM
        """
        return self._bias_integral(self._compute_hod_sat)

    @property
    def bg_tot(self):
        """
        Mean linear halo bias for the given population of galaxies.
        b_lin_x = int ⟨Nx|M⟩ b_h(M) n(M) dM
        """
        return self._bias_integral(self._compute_hod)

    @property
    def compute_number_density_cen(self):
        """
        Compute the number density of central galaxies.
        """
        return self._interpolate(self.ncen)

    @property
    def compute_number_density_sat(self):
        """
        Compute the number density of satellite galaxies.
        """
        return self._interpolate(self.nsat)

    @property
    def compute_number_density(self):
        """
        Compute the number density of galaxies.
        """
        return self._interpolate(self.ntot)

    @property
    def f_c(self):
        """
        Fraction of central galaxies.
        """
        f_c = self.ncen / self.ntot
        return self._interpolate(f_c, fill_value=0.0)

    @property
    def f_s(self):
        """
        Fraction of satellite galaxies.
        """
        f_s = self.nsat / self.ntot
        return self._interpolate(f_s, fill_value=0.0)

    @property
    def compute_avg_halo_mass_cen(self):
        """
        Compute the average halo mass for central galaxies.
        """
        return self._interpolate(self.mass_avg_cen, fill_value=0.0)

    @property
    def compute_avg_halo_mass_sat(self):
        """
        Compute the average halo mass for satellite galaxies.
        """
        return self._interpolate(self.mass_avg_sat, fill_value=0.0)

    @property
    def compute_avg_halo_mass(self):
        """
        Compute the average halo mass for galaxies.
        """
        return self._interpolate(self.mass_avg_tot, fill_value=0.0)

    @property
    def compute_galaxy_linear_bias_cen(self):
        """
        Compute the galaxy linear bias for central galaxies.
        """
        return self._interpolate(self.bg_cen)

    @property
    def compute_galaxy_linear_bias_sat(self):
        """
        Compute the galaxy linear bias for satellite galaxies.
        """
        return self._interpolate(self.bg_sat)

    @property
    def compute_galaxy_linear_bias(self):
        """
        Compute the galaxy linear bias for galaxies.
        """
        return self._interpolate(self.bg_tot)

    @property
    def compute_hod_cen(self):
        """
        Compute the HOD for central galaxies.
        """
        return self._interpolate(self._compute_hod_cen, axis=0)

    @property
    def compute_hod_sat(self):
        """
        Compute the HOD for satellite galaxies.
        """
        return self._interpolate(self._compute_hod_sat, axis=0)

    @property
    def compute_hod(self):
        """
        Compute the HOD for galaxies.
        """
        return self._interpolate(self._compute_hod, axis=0)

    @property
    def compute_stellar_fraction_cen(self):
        """
        Compute the stellar fraction for central galaxies.
        """
        if self._compute_stellar_fraction_cen is None:
            return np.zeros((self.nbins, self.z_vec.size, self.mass.shape[-1]))
        else:
            return self._interpolate(self._compute_stellar_fraction_cen, axis=0)

    @property
    def compute_stellar_fraction_sat(self):
        """
        Compute the stellar fraction for satellite galaxies.
        """
        if self._compute_stellar_fraction_sat is None:
            return np.zeros((self.nbins, self.z_vec.size, self.mass.shape[-1]))
        else:
            return self._interpolate(self._compute_stellar_fraction_sat, axis=0)

    @property
    def compute_stellar_fraction(self):
        """
        Compute the stellar fraction for galaxies.
        """
        if self._compute_stellar_fraction is None:
            return np.zeros((self.nbins, self.z_vec.size, self.mass.shape[-1]))
        else:
            return self._interpolate(self._compute_stellar_fraction, axis=0)


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
            log10_obs_norm_c = 9.95,
            log10_m_ch = 11.24,
            g1 = 3.18,
            g2 = 0.245,
            sigma_log10_O_c = 0.157,
            norm_s = 0.562,
            pivot = 12.0,
            alpha_s = -1.18,
            beta_s = 2,
            b0 = -1.17,
            b1 = 1.53,
            b2 = -0.217,
            A_cen = None,
            A_sat = None,
            **hod_kwargs
        ):

        # Call super init MUST BE DONE FIRST.
        super().__init__(**hod_kwargs)
        # Set all given parameters.

        # centrals
        # all observable masses in units of log10(M_sun h^-2)
        self.M_char = 10.0**log10_m_ch  # M_char
        self.g_1 = g1  # gamma_1
        self.g_2 = g2  # gamma_2
        self.Obs_norm_c = 10.0**log10_obs_norm_c  # O_0, O_norm_c
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

    @property
    def COF_cen(self):
        """
        COF for Central galaxies.
        eq 17 of D23: 2210.03110:
        Φc(O|M) = 1/[√(2π) ln(10) σ_c O] exp[ -log(O/O∗c)^2/ (2 σ_c^2) ]
        Note Φc(O|M) is unitless.
        """
        mean_obs_c = self.cal_mean_obs_c[:, :, :, np.newaxis]  # O∗c
        COF_c = (1.0 / (np.sqrt(2.0 * np.pi) * np.log(10.0) * self.sigma_log10_O_c * self.obs) *
                 np.exp(-(np.log10(self.obs / mean_obs_c))**2 / (2.0 * self.sigma_log10_O_c**2)))
        return COF_c

    @property
    def COF_sat(self):
        """
        COF for satellite galaxies.
        eq 18 of D23: 2210.03110:
        Φs(O|M) = ϕ∗s/O∗s (O/O∗s)^α_s exp [−(O/O∗s)^2], O*s is O∗s(M) = 0.56 O∗c(M)
        Note Φs(O|M) is unitless.
        """
        obs_s_star = self.norm_s * self.cal_mean_obs_c[:, :, :, np.newaxis]
        obs_tilde = self.obs / obs_s_star
        phi_star_val = self.phi_star_s[:, :, :, np.newaxis]
        COF_s = (phi_star_val / obs_s_star) * (obs_tilde**self.alpha_s) * np.exp(-obs_tilde**self.beta_s)
        return COF_s

    @property
    def COF(self):
        """
        Total COF, sum of central and satellite COFs.
        """
        return self.COF_cen + self.COF_sat

    @cached_property
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
        """
        integrand = self.COF_cen * self.dndlnm[:, :, :, np.newaxis] / self.mass[:, :, :, np.newaxis]
        obs_function = simpson(integrand, self.mass[:, :, :, np.newaxis], axis=-2)
        return obs_function

    @cached_property
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
        """
        integrand = self.COF_sat * self.dndlnm[:, :, :, np.newaxis] / self.mass[:, :, :, np.newaxis]
        obs_function = simpson(integrand, self.mass[:, :, :, np.newaxis], axis=-2)
        return obs_function

    @cached_property
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
        """
        integrand = self.COF * self.dndlnm[:, :, :, np.newaxis] / self.mass[:, :, :, np.newaxis]
        obs_function = simpson(integrand, self.mass[:, :, :, np.newaxis], axis=-2)
        return obs_function

    @cached_property
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
        """
        mean_obs_c = (self.Obs_norm_c * (self.mass / self.M_char)**self.g_1 /
                      (1.0 + (self.mass / self.M_char))**(self.g_1 - self.g_2))
        return mean_obs_c

    @cached_property
    def phi_star_s(self):
        """
        pivot COF used in eq 21 of D23: 2210.03110
        using a bias expansion around the pivot mass
        eq 22 of D23: 2210.03110
        log[ϕ∗s(M)] = b0 + b1(log m13) , m13 is logM_pivot, m13 = M/(hod_par.pivot M⊙ h−1)
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

    @property
    def _compute_hod(self):
        """
        Compute the total HOD by summing central and satellite HODs.
        """
        return self._compute_hod_cen + self._compute_hod_sat

    @property
    def _compute_stellar_fraction_cen(self):
        """
        The mean value of the observable for the given galaxy population for a given halo mass.
        O is weighted by the number of galaxies with the property O for each halo mass: Φx(O|M)
        f_star = int_{O_low}^{O_high} Φx(O|M) O dO
        """
        return simpson(self.COF_cen * self.obs, self.obs) / self.mass

    @property
    def _compute_stellar_fraction_sat(self):
        """
        The mean value of the observable for the given galaxy population for a given halo mass.
        O is weighted by the number of galaxies with the property O for each halo mass: Φx(O|M)
        f_star = int_{O_low}^{O_high} Φx(O|M) O dO
        """
        return simpson(self.COF_sat * self.obs, self.obs) / self.mass

    @property
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
    obs : array_like, optional
        Observable.
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
            obs = None,
            log10_Mmin = np.array([12.0]),
            log10_Msat = np.array([13.0]),
            alpha = np.array([1.0]),
            A_cen = None,
            A_sat = None,
            **hod_kwargs
        ):
        # Call super init MUST BE DONE FIRST.
        super().__init__(**hod_kwargs)
        self.Mmin = 10.0**log10_Mmin[:, np.newaxis, np.newaxis]
        self.Msat = 10.0**log10_Msat[:, np.newaxis, np.newaxis]
        self.alpha = alpha[:, np.newaxis, np.newaxis]
        # Decorated HOD assembly bias parameters
        self.A_cen = A_cen
        self.A_sat = A_sat

    @cached_property
    def _compute_hod_cen(self):
        """
        Compute the HOD for central galaxies.
        """
        N_cen = np.heaviside(self.mass - self.Mmin, 1.0)
        if self.A_cen is not None:
            delta_pop_c = self.A_cen * np.fmin(N_cen, 1.0 - N_cen)
            N_cen = N_cen + delta_pop_c
        return np.tile(N_cen, (self.nz, 1))

    @cached_property
    def _compute_hod_sat(self):
        """
        Compute the HOD for satellite galaxies.
        """
        N_sat = self.compute_hod_cen * (self.mass / self.Msat)**self.alpha
        if self.A_sat is not None:
            delta_pop_s = self.A_sat * N_sat
            N_sat = N_sat + delta_pop_s
        return np.tile(N_sat, (self.nz, 1))

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
        

class Zehavi(HOD):
    """
    HOD model from Zehavi et al. (2004; https://arxiv.org/abs/astro-ph/0703457)
    Same as Zheng model in the limit that sigma=0 and M0=0
    Mean number of central galaxies is only ever 0 or 1 in this HOD
    
    Parameters:
    -----------
    obs : array_like, optional
        Observable.
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
            obs = None,
            log10_Mmin = np.array([12.0]),
            log10_Msat = np.array([13.0]),
            alpha = np.array([1.0]),
            A_cen = None,
            A_sat = None,
            **hod_kwargs
        ):
        # Call super init MUST BE DONE FIRST.
        super().__init__(**hod_kwargs)
        self.Mmin = 10.0**log10_Mmin[:, np.newaxis, np.newaxis]
        self.Msat = 10.0**log10_Msat[:, np.newaxis, np.newaxis]
        self.alpha = alpha[:, np.newaxis, np.newaxis]
        # Decorated HOD assembly bias parameters
        self.A_cen = A_cen
        self.A_sat = A_sat

    @cached_property
    def _compute_hod_cen(self):
        """
        Compute the HOD for central galaxies.
        """
        N_cen = np.heaviside(self.mass - self.Mmin, 1.0)
        if self.A_cen is not None:
            delta_pop_c = self.A_cen * np.fmin(N_cen, 1.0 - N_cen)
            N_cen = N_cen + delta_pop_c
        return np.tile(N_cen, (self.nz, 1))

    @cached_property
    def _compute_hod_sat(self):
        """
        Compute the HOD for satellite galaxies.
        """
        N_sat = (self.mass / self.Msat)**self.alpha
        if self.A_sat is not None:
            delta_pop_s = self.A_sat * N_sat
            N_sat = N_sat + delta_pop_s
        return np.tile(N_sat, (self.nz, 1))

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

class Zheng(HOD):
    """
    Zheng et al. (2005; https://arxiv.org/abs/astro-ph/0408564) HOD model
    
    Parameters:
    -----------
    obs : array_like, optional
        Observable.
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
            obs = None,
            log10_Mmin = np.array([12.0]),
            log10_M0 = np.array([12.0]),
            log10_M1 = np.array([13.0]),
            sigma = np.array([0.15]),
            alpha = np.array([1.0]),
            A_cen = None,
            A_sat = None,
            **hod_kwargs
        ):
        # Call super init MUST BE DONE FIRST.
        super().__init__(**hod_kwargs)
        self.Mmin = 10.0**log10_Mmin[:, np.newaxis, np.newaxis]
        self.M0 = 10.0**log10_M0[:, np.newaxis, np.newaxis]
        self.M1 = 10.0**log10_M1[:, np.newaxis, np.newaxis]
        self.sigma = sigma[:, np.newaxis, np.newaxis]
        self.alpha = alpha[:, np.newaxis, np.newaxis]
        # Decorated HOD assembly bias parameters
        self.A_cen = A_cen
        self.A_sat = A_sat

    @cached_property
    def _compute_hod_cen(self):
        """
        Compute the HOD for central galaxies.
        """
        N_cen = np.where(self.sigma == 0.0, np.heaviside(self.mass - self.Mmin, 1.0), 0.5 * (1.0 + erf(np.log10(self.mass / self.Mmin) / self.sigma)))
        if self.A_cen is not None:
            delta_pop_c = self.A_cen * np.fmin(N_cen, 1.0 - N_cen)
            N_cen = N_cen + delta_pop_c
        return np.tile(N_cen, (self.nz, 1))

    @cached_property
    def _compute_hod_sat(self):
        """
        Compute the HOD for satellite galaxies.
        """
        N_sat = (np.heaviside(self.mass - self.M0, 1.0) * (self.mass - self.M0) / self.M1)**self.alpha
        if self.A_sat is not None:
            delta_pop_s = self.A_sat * N_sat
            N_sat = N_sat + delta_pop_s
        return np.tile(N_sat, (self.nz, 1))

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

class Zhai(HOD):
    """
    HOD model from Zhai et al. (2017; https://arxiv.org/abs/1607.05383)
    
    Parameters:
    -----------
    obs : array_like, optional
        Observable.
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
            obs = None
            log10_Mmin = np.array([13.68]),
            log10_Msat = np.array([14.87]),
            log10_Mcut = np.array([12.32]),
            sigma = np.array([0.82]),
            alpha = np.array([0.41]),
            A_cen = None,
            A_sat = None,
            **hod_kwargs
        ):
        # Call super init MUST BE DONE FIRST.
        super().__init__(**hod_kwargs)
        self.obs = obs
        self.Mmin = 10.0**log10_Mmin[:, np.newaxis, np.newaxis]
        self.Msat = 10.0**log10_Msat[:, np.newaxis, np.newaxis]
        self.Mcut = 10.0**log10_Mcut[:, np.newaxis, np.newaxis]
        self.sigma = sigma[:, np.newaxis, np.newaxis]
        self.alpha = alpha[:, np.newaxis, np.newaxis]
        # Decorated HOD assembly bias parameters
        self.A_cen = A_cen
        self.A_sat = A_sat

    @cached_property
    def _compute_hod_cen(self):
        """
        Compute the HOD for central galaxies.
        """
        N_cen = np.where(self.sigma == 0.0, np.heaviside(self.mass - self.Mmin, 1.0), 0.5 * (1.0 + erf(np.log10(self.mass / self.Mmin) / self.sigma)))
        if self.A_cen is not None:
            delta_pop_c = self.A_cen * np.fmin(N_cen, 1.0 - N_cen)
            N_cen = N_cen + delta_pop_c
        return np.tile(N_cen, (self.nz, 1))

    @cached_property
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
