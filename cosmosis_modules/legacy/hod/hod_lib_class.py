import numpy as np
from scipy.integrate import simpson
from scipy.special import erf

# TO-DO: Add nbins as another dimension to all quantities to remove looping!

class HOD:
    def __init__(
            self,
            mass=None,
            dndlnm=None,
            nz=1
        ):
        if mass is None or dndlnm is None:
            raise ValueError("Mass and halo mass function need to be specified!")

        # Set all given parameters.
        # With newaxis we make sure the HOD shape is (nz, nmass)
        self.mass = mass[np.newaxis, :]
        self.dndlnm = dndlnm
        self.nz = nz

    @property
    def compute_number_density_cen(self):
        """
        Total number density of galaxies with the given HOD, e.g. central and satellite galaxies
        This is an integral over the HOD and the halo mass function to remove the halo mass dependence.
        Nx = int ⟨Nx|M⟩ n(M) dM
        """
        integrand = self.compute_hod_cen * self.dndlnm / self.mass
        return simpson(integrand, self.mass)

    @property
    def compute_number_density_sat(self):
        """
        Total number density of galaxies with the given HOD, e.g. central and satellite galaxies
        This is an integral over the HOD and the halo mass function to remove the halo mass dependence.
        Nx = int ⟨Nx|M⟩ n(M) dM
        """
        integrand = self.compute_hod_sat * self.dndlnm / self.mass
        return simpson(integrand, self.mass)

    @property
    def compute_number_density(self):
        """
        Total number density of galaxies with the given HOD, e.g. central and satellite galaxies
        This is an integral over the HOD and the halo mass function to remove the halo mass dependence.
        Nx = int ⟨Nx|M⟩ n(M) dM
        """
        integrand = self.compute_hod * self.dndlnm / self.mass
        return simpson(integrand, self.mass)

    @property
    def compute_avg_halo_mass_cen(self):
        """
        The mean halo mass for the given population of galaxies, e.g. central and satellite galaxies
        M_mean = int ⟨Nx|M⟩ M n(M) dM
        """
        integrand = self.compute_hod_cen * self.dndlnm
        return simpson(integrand, self.mass)

    @property
    def compute_avg_halo_mass_sat(self):
        """
        The mean halo mass for the given population of galaxies, e.g. central and satellite galaxies
        M_mean = int ⟨Nx|M⟩ M n(M) dM
        """
        integrand = self.compute_hod_sat * self.dndlnm
        return simpson(integrand, self.mass)

    @property
    def compute_avg_halo_mass(self):
        """
        The mean halo mass for the given population of galaxies, e.g. central and satellite galaxies
        M_mean = int ⟨Nx|M⟩ M n(M) dM
        """
        integrand = self.compute_hod * self.dndlnm
        return simpson(integrand, self.mass)

    def compute_galaxy_linear_bias_cen(self, halo_bias):
        """
        Mean linear halo bias for the given population of galaxies.
        b_lin_x = int ⟨Nx|M⟩ b_h(M) n(M) dM
        """
        bg_integrand = self.compute_hod_cen * halo_bias * self.dndlnm / self.mass
        return simpson(bg_integrand, self.mass)

    def compute_galaxy_linear_bias_sat(self, halo_bias):
        """
        Mean linear halo bias for the given population of galaxies.
        b_lin_x = int ⟨Nx|M⟩ b_h(M) n(M) dM
        """
        bg_integrand = self.compute_hod_sat * halo_bias * self.dndlnm / self.mass
        return simpson(bg_integrand, self.mass)

    def compute_galaxy_linear_bias(self, halo_bias):
        """
        Mean linear halo bias for the given population of galaxies.
        b_lin_x = int ⟨Nx|M⟩ b_h(M) n(M) dM
        """
        bg_integrand = self.compute_hod * halo_bias * self.dndlnm / self.mass
        return simpson(bg_integrand, self.mass)

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
    """
    def __init__(
            self,
            obs = np.array([np.logspace(9, 15, 100)]),
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
        # With newaxis we make sure the COF shape is (nz, nmass, nobs)
        self.obs = obs[:, np.newaxis]

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
        mean_obs_c = self.cal_mean_obs_c[:, :, np.newaxis]  # O∗c
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
        obs_s_star = self.norm_s * self.cal_mean_obs_c[:, :, np.newaxis]
        obs_tilde = self.obs / obs_s_star
        phi_star_val = self.phi_star_s[:, :, np.newaxis]
        COF_s = (phi_star_val / obs_s_star) * (obs_tilde**self.alpha_s) * np.exp(-obs_tilde**self.beta_s)
        return COF_s

    @property
    def COF(self):
        return self.COF_cen + self.COF_sat

    def obs_func_cen(self, axis=-2):
        r"""
        The Observable function, this is Φs(O|M), Φc(O|M) integrated over the halo mass weighted
        with the Halo Mass Function (HMF) to give:  Φs(O),Φc(O)
        Φx(O) =int Φx(O|M) n(M) dM,
        dndlnm is basically n(M) x mass, it is the output of hmf
        The differential mass function in terms of natural log of m,
        len=len(m) [units \(h^3 Mpc^{-3}\)]
        dn(m)/ dln m eq1 of 1306.6721
        obs_func unit is h^3 Mpc^{-3} dex^-1
        """
        integrand = self.COF_cen * self.dndlnm[:, :, np.newaxis] / self.mass[:, :, np.newaxis]
        obs_function = simpson(integrand, self.mass[0, :], axis=axis)
        return obs_function

    def obs_func_sat(self, axis=-2):
        r"""
        The Observable function, this is Φs(O|M), Φc(O|M) integrated over the halo mass weighted
        with the Halo Mass Function (HMF) to give:  Φs(O),Φc(O)
        Φx(O) =int Φx(O|M) n(M) dM,
        dndlnm is basically n(M) x mass, it is the output of hmf
        The differential mass function in terms of natural log of m,
        len=len(m) [units \(h^3 Mpc^{-3}\)]
        dn(m)/ dln m eq1 of 1306.6721
        obs_func unit is h^3 Mpc^{-3} dex^-1
        """
        integrand = self.COF_sat * self.dndlnm[:, :, np.newaxis] / self.mass[:, :, np.newaxis]
        obs_function = simpson(integrand, self.mass[0, :], axis=axis)
        return obs_function

    def obs_func(self, axis=-2):
        r"""
        The Observable function, this is Φs(O|M), Φc(O|M) integrated over the halo mass weighted
        with the Halo Mass Function (HMF) to give:  Φs(O),Φc(O)
        Φx(O) =int Φx(O|M) n(M) dM,
        dndlnm is basically n(M) x mass, it is the output of hmf
        The differential mass function in terms of natural log of m,
        len=len(m) [units \(h^3 Mpc^{-3}\)]
        dn(m)/ dln m eq1 of 1306.6721
        obs_func unit is h^3 Mpc^{-3} dex^-1
        """
        integrand = self.COF * self.dndlnm[:, :, np.newaxis] / self.mass[:, :, np.newaxis]
        obs_function = simpson(integrand, self.mass[0, :], axis=axis)
        return obs_function

    @property
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

    @property
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
    def compute_hod_cen(self):
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
    def compute_hod_sat(self):
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
    def compute_hod(self):
        return self.compute_hod_cen + self.compute_hod_sat

    @property
    def compute_stellar_fraction_cen(self):
        """
        The mean value of the observable for the given galaxy population for a given halo mass.
        O is weighted by the number of galaxies with the property O for each halo mass: Φx(O|M)
        f_star = int_{O_low}^{O_high} Φx(O|M) O dO
        """
        return simpson(self.COF_cen * self.obs, self.obs) / self.mass

    @property
    def compute_stellar_fraction_sat(self):
        """
        The mean value of the observable for the given galaxy population for a given halo mass.
        O is weighted by the number of galaxies with the property O for each halo mass: Φx(O|M)
        f_star = int_{O_low}^{O_high} Φx(O|M) O dO
        """
        return simpson(self.COF_sat * self.obs, self.obs) / self.mass

    @property
    def compute_stellar_fraction(self):
        """
        The mean value of the observable for the given galaxy population for a given halo mass.
        O is weighted by the number of galaxies with the property O for each halo mass: Φx(O|M)
        f_star = int_{O_low}^{O_high} Φx(O|M) O dO
        """
        return simpson(self.COF * self.obs, self.obs) / self.mass

class Simple(HOD):
    """
    Simple HOD model
    """
    def __init__(
            self,
            obs = None,
            log10_Mmin = 12.0,
            log10_Msat = 13.0,
            alpha = 1.0,
            A_cen = None,
            A_sat = None,
            **hod_kwargs
        ):
        # Call super init MUST BE DONE FIRST.
        super().__init__(**hod_kwargs)
        self.obs = obs
        self.Mmin = 10.0**log10_Mmin
        self.Msat = 10.0**log10_Msat
        self.alpha = alpha
        # Decorated HOD assembly bias parameters
        self.A_cen = A_cen
        self.A_sat = A_sat

    @property
    def compute_hod_cen(self):
        """
        Centrals
        """
        N_cen = np.heaviside(self.mass - self.Mmin, 1.0)
        if self.A_cen is not None:
            delta_pop_c = self.A_cen * np.fmin(N_cen, 1.0 - N_cen)
            N_cen = N_cen + delta_pop_c
        return np.tile(N_cen, (self.nz, 1))

    @property
    def compute_hod_sat(self):
        """
        Satellites
        """
        N_sat = self.compute_hod_cen * (self.mass / self.Msat)**self.alpha
        if self.A_sat is not None:
            delta_pop_s = self.A_sat * N_sat
            N_sat = N_sat + delta_pop_s
        return np.tile(N_sat, (self.nz, 1))

    @property
    def compute_hod(self):
        return self.compute_hod_cen + self.compute_hod_sat

class Zehavi(HOD):
    """
    HOD model from Zehavi et al. (2004; https://arxiv.org/abs/astro-ph/0703457)
    Same as Zheng model in the limit that sigma=0 and M0=0
    Mean number of central galaxies is only ever 0 or 1 in this HOD
    """
    def __init__(
            self,
            obs = None,
            log10_Mmin = 12.0,
            log10_Msat = 13.0,
            alpha = 1.0,
            A_cen = None,
            A_sat = None,
            **hod_kwargs
        ):
        # Call super init MUST BE DONE FIRST.
        super().__init__(**hod_kwargs)
        self.obs = obs
        self.Mmin = 10.0**log10_Mmin
        self.Msat = 10.0**log10_Msat
        self.alpha = alpha
        # Decorated HOD assembly bias parameters
        self.A_cen = A_cen
        self.A_sat = A_sat

    @property
    def compute_hod_cen(self):
        """
        Centrals
        """
        N_cen = np.heaviside(self.mass - self.Mmin, 1.0)
        if self.A_cen is not None:
            delta_pop_c = self.A_cen * np.fmin(N_cen, 1.0 - N_cen)
            N_cen = N_cen + delta_pop_c
        return np.tile(N_cen, (self.nz, 1))

    @property
    def compute_hod_sat(self):
        """
        Satellites
        """
        N_sat = (self.mass / self.Msat)**self.alpha
        if self.A_sat is not None:
            delta_pop_s = self.A_sat * N_sat
            N_sat = N_sat + delta_pop_s
        return np.tile(N_sat, (self.nz, 1))

    @property
    def compute_hod(self):
        return self.compute_hod_cen + self.compute_hod_sat

class Zheng(HOD):
    """
    Zheng et al. (2005; https://arxiv.org/abs/astro-ph/0408564) HOD model
    """
    def __init__(
            self,
            obs = None,
            log10_Mmin = 12.0,
            log10_M0 = 12.0,
            log10_M1 = 13.0,
            sigma = 0.15,
            alpha = 1.0,
            A_cen = None,
            A_sat = None,
            **hod_kwargs
        ):
        # Call super init MUST BE DONE FIRST.
        super().__init__(**hod_kwargs)
        self.obs = obs
        self.Mmin = 10.0**log10_Mmin
        self.M0 = 10.0**log10_M0
        self.M1 = 10.0**log10_M1
        self.sigma = sigma
        self.alpha = alpha
        # Decorated HOD assembly bias parameters
        self.A_cen = A_cen
        self.A_sat = A_sat

    @property
    def compute_hod_cen(self):
        """
        Centrals
        """
        if self.sigma == 0.0:
            N_cen = np.heaviside(self.mass - self.Mmin, 1.0)
        else:
            N_cen = 0.5 * (1.0 + erf(np.log10(self.mass / self.Mmin) / self.sigma))
        if self.A_cen is not None:
            delta_pop_c = self.A_cen * np.fmin(N_cen, 1.0 - N_cen)
            N_cen = N_cen + delta_pop_c
        return np.tile(N_cen, (self.nz, 1))

    @property
    def compute_hod_sat(self):
        """
        Satellites
        """
        N_sat = (np.heaviside(self.mass - self.M0, 1.0) * (self.mass - self.M0) / self.M1)**self.alpha
        if self.A_sat is not None:
            delta_pop_s = self.A_sat * N_sat
            N_sat = N_sat + delta_pop_s
        return np.tile(N_sat, (self.nz, 1))

    @property
    def compute_hod(self):
        return self.compute_hod_cen + self.compute_hod_sat

class Zhai(HOD):
    """
    HOD model from Zhai et al. (2017; https://arxiv.org/abs/1607.05383)
    """
    def __init__(
            self,
            obs = None,
            log10_Mmin = 13.68,
            log10_Msat = 14.87,
            log10_Mcut = 12.32,
            sigma = 0.82,
            alpha = 0.41,
            A_cen = None,
            A_sat = None,
            **hod_kwargs
        ):
        # Call super init MUST BE DONE FIRST.
        super().__init__(**hod_kwargs)
        self.obs = obs
        self.Mmin = 10.0**log10_Mmin
        self.Msat = 10.0**log10_Msat
        self.Mcut = 10.0**log10_Mcut
        self.sigma = sigma
        self.alpha = alpha
        # Decorated HOD assembly bias parameters
        self.A_cen = A_cen
        self.A_sat = A_sat

    @property
    def compute_hod_cen(self):
        """
        Centrals
        """
        if self.sigma == 0.0:
            N_cen = np.heaviside(self.mass - self.Mmin, 1.0)
        else:
            N_cen = 0.5 * (1.0 + erf(np.log10(self.mass / self.Mmin) / self.sigma))
        if self.A_cen is not None:
            delta_pop_c = self.A_cen * np.fmin(N_cen, 1.0 - N_cen)
            N_cen = N_cen + delta_pop_c
        return np.tile(N_cen, (self.nz, 1))

    @property
    def compute_hod_sat(self):
        """
        Satellites
        """
        # Paper has a Nc(M) multiplication, but I think the central condition covers this
        N_sat = ((self.mass / self.Msat)**self.alpha) * np.exp(-self.Mcut / self.mass)
        if self.A_sat is not None:
            delta_pop_s = self.A_sat * N_sat
            N_sat = N_sat + delta_pop_s
        return np.tile(N_sat, (self.nz, 1))

    @property
    def compute_hod(self):
        return self.compute_hod_cen + self.compute_hod_sat
