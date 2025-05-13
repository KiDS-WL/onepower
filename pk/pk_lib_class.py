# Library of the power spectrum module
"""
Calculates 3D power spectra using the halo model approach:
See section 2 of https://arxiv.org/pdf/2303.08752.pdf for details

P_uv = P^2h_uv + P^1h_uv  (1)

P^1h_uv (k) = int_0^infty dM Wu(M, k) Wv(M, k) n(M)  (2)

P^2h_uv (k) = int_0^infty int_0^infty dM1 dM2 Phh(M1, M2, k) Wu(M1, k) Wv(M2, k) n(M1) n(M2)  (3)

Wx are the profile of the fields, u and v, showing how they fit into haloes.
n(M) is the halo mass function, quantifying the number of haloes of each mass, M.
Integrals are taken over halo mass.

The halo-halo power spectrum can be written as,

Phh(M1,M2,k) = b(M1) b(M2) P^lin_mm(k) (1 + beta_nl(M1,M2,k)) (4)

In the vanilla halo model the 2-halo term is usually simplified by assuming that haloes are linearly biased with respect to matter.
This sets beta_nl to zero and effectively decouples the integrals in (3). Here we allow for both options to be calculated.
If you want the option with beta_nl the beta_nl modules has to be run before this module.

Equation (3) then becomes:

P^2h_uv (k) = P^lin_mm(k) * [I_u * I_v + I^NL_uv] (5)

where I_u and I_v are defined as:

I_x = int_0^infty dM b(M)  Wx(M, k) n(M) (6)

and the integral over beta_nl is

I^NL_uv = int_0^infty int_0^infty dM1 dM2 b(M1) b(M2) beta_nl(M1,M2,k) Wu(M1, k) Wv(M2, k) n(M1) n(M2)  (7)

---------------------------------------------------------------------------------------------------------------------

We truncate the 1-halo term so that it doesn't dominate at large scales.

The linear matter power spectrum needs to be provided.
The halo_model_ingredients and hod modules (for everything but mm, unless you run 'stellar_fraction_from_observable_feedback' option)
need to be run before this.

Current power spectra that we predict are
mm: matter-matter
gg: galaxy-galaxy
gm: galaxy-matter

II: intrinsic-intrinsic alignments
gI: galaxy-intrinsic alignment
mI: matter-intrinsic alignment
"""
import numpy as np
import numexpr as ne
from scipy.interpolate import interp1d, RegularGridInterpolator, UnivariateSpline
from scipy.integrate import simpson, quad, trapezoid
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import warnings

# Helper functions borrowed from Alex Mead
def Tk_EH_nowiggle(k, h, ombh2, ommh2, T_CMB=2.7255):
    """
    No-wiggle transfer function from astro-ph:9709112
    """
    rb = ombh2 / ommh2 # Baryon ratio
    s = 44.5 * np.log(9.83 / ommh2) / np.sqrt(1.0 + 10.0 * ombh2**0.75) # Equation (26)
    alpha = 1.0 - 0.328 * np.log(431.0 * ommh2) * rb + 0.38 * np.log(22.3 * ommh2) * rb**2.0 # Equation (31)

    Gamma = (ommh2 / h) * (alpha + (1. - alpha) / (1. + (0.43 * k * s * h)**4)) # Equation (30)
    q = k * (T_CMB / 2.7)**2.0 / Gamma # Equation (28)
    L = np.log(2.0 * np.e + 1.8 * q) # Equation (29)
    C = 14.2 + 731. / (1. + 62.5 * q) # Equation (29)
    Tk_nw = L / (L + C * q**2.0) # Equation (29)
    return Tk_nw

def sigmaV(k, power):
    # In the limit where r -> 0
    dlnk = np.log(k[1] / k[0])
    # we multiply by k because our steps are in logk.
    integ = power * k
    sigma = (0.5 / np.pi**2.0) * simpson(integ, dx=dlnk, axis=-1)
    return np.sqrt(sigma / 3.0)

def get_Pk_wiggle(k, Pk_lin, h, ombh2, ommh2, ns, T_CMB=2.7255, sigma_dlnk=0.25):
    """
    Extract the wiggle from the linear power spectrum
    TODO: Should get to work for uneven log(k) spacing
    NOTE: https://stackoverflow.com/questions/24143320/gaussian-sum-filter-for-irregular-spaced-points
    """
    if not np.isclose(np.all(np.diff(k) - np.diff(k)[0]), 0.):
        raise ValueError('Dewiggle only works with linearly-spaced k array')

    dlnk = np.log(k[1] / k[0])
    sigma = sigma_dlnk / dlnk

    Pk_nowiggle = (k**ns) * Tk_EH_nowiggle(k, h, ombh2, ommh2, T_CMB)**2
    Pk_ratio = Pk_lin / Pk_nowiggle
    Pk_ratio = gaussian_filter1d(Pk_ratio, sigma)
    Pk_smooth = Pk_ratio * Pk_nowiggle
    Pk_wiggle = Pk_lin - Pk_smooth
    return Pk_wiggle


class MatterSpectra:
    def __init__(self,
            z_vec = None,
            mass = None,
            k_vec = None,
            mean_density0 = None,
            dndlnm = None,
            halobias = None,
            matter_power_lin = None,
            u_dm = None,
            omega_c = 0.25,
            omega_m = 0.3,
            omega_b = 0.05,
            h0 = 0.7,
            n_s = 1.0,
            tcmb = 2.7255,
            log10T_AGN = 7.8,
            fnu = 0.01,
            mb = 13.0,
            fstar_mm = None,
            dewiggle = False,
            bnl = False,
            beta_nl = None,
            mead_correction = None
        ):
        
        self.z_vec = z_vec
        self.k_vec = k_vec
        self.mass = mass
        self.mean_density0 = mean_density0
        self.omega_c = omega_c
        self.omega_m = omega_m
        self.omega_b = omega_b
        self.h0 = h0
        self.n_s = n_s
        self.tcmb = tcmb
        self.log10T_AGN = log10T_AGN
        self.fnu = fnu
        self.mb = mb
        self.dndlnm = dndlnm
        self.halobias = halobias
        self.u_dm = u_dm
        self.mead_correction = mead_correction
        self.bnl = bnl

        if self.mead_correction in ['feedback', 'nofeedback'] or dewiggle:
            self.matter_power_lin = self.dewiggle_plin(matter_power_lin)
        else:
            self.matter_power_lin = matter_power_lin
        
        if self.bnl:
            self.beta_nl = beta_nl
            self.I12 = self.prepare_I12_integrand(self.halobias, self.halobias, self.dndlnm, self.dndlnm, self.beta_nl)
            self.I21 = self.prepare_I21_integrand(self.halobias, self.halobias, self.dndlnm, self.dndlnm, self.beta_nl)
            self.I22 = self.prepare_I22_integrand(self.halobias, self.halobias, self.dndlnm, self.dndlnm, self.beta_nl)
        
    def dewiggle_plin(self, plin):
        sigma = sigmaV(self.k_vec, plin)
        ombh2 = self.omega_b * self.h0**2.0
        ommh2 = self.omega_m * self.h0**2.0
        pk_wig = get_Pk_wiggle(self.k_vec, plin, self.h0, ombh2, ommh2, self.n_s, self.tcmb)
        plin_dw = plin - (1.0 - np.exp(-(self.k_vec[np.newaxis, :] * sigma[:, np.newaxis])**2.0)) * pk_wig
        return plin_dw

    def compute_matter_profile(self, mass, mean_density0, u_dm, fnu):
        """
        Compute the matter halo profile with a correction for neutrino mass fraction.
        Feedback can be included through u_dm
        We lower the amplitude of W(M, k,z) in the one-halo term by the factor 1− fν ,
        where fν = Ων /Ωm is the neutrino mass fraction, to account for the fact that
        we assume that hot neutrinos cannot cluster in haloes and therefore
        do not contribute power to the one-halo term. Therefore W(M, k → 0,z) = (1− fν )M/ρ¯ and has units of volume
        This is the same as Mead et al. 2021
        """
        Wm_0 = mass / mean_density0
        return Wm_0 * u_dm * (1.0 - fnu)

    @property
    def matter_profile(self):
        """
        Compute the matter profile grid in z, k, and M.
        """
        return self.compute_matter_profile(
            self.mass[np.newaxis, np.newaxis, :],
            self.mean_density0[:, np.newaxis, np.newaxis],
            self.u_dm,
            self.fnu[:, np.newaxis, np.newaxis]
        )

    def compute_matter_profile_with_feedback(self, mass, mean_density0, u_dm, z, fnu):
        """
        Compute the matter profile including feedback as modelled by hmcode2020.

        eq 25 of 2009.01858
        W(M, k) = [Ω_c/Ω_m+ fg(M)]W(M, k) + f∗ M/ρ¯
        The parameter 0 < f∗ < Ω_b/Ω_m can be thought of as an effective halo stellar mass fraction.

        Total matter profile from Mead2020 for baryonic feedback model
        Table 4 and eq 26 of 2009.01858
        f*(z) = f*_0 10^(z f*_z)

        This profile does not have 1-fnu correction as that is already accounted for in  dm_to_matter_frac
        """
        fstar = self.fs(z)
        dm_to_matter_frac = self.omega_c / self.omega_m
        f_gas = self.fg(mass, z, fstar)

        Wm_0 = mass / mean_density0
        Wm = (dm_to_matter_frac + f_gas) * Wm_0 * u_dm * (1.0 - fnu) + fstar * Wm_0
        #Wm = (dm_to_matter_frac + f_gas) * Wm_0 * u_dm + fstar * Wm_0
        return Wm

    @property
    def matter_profile_with_feedback(self):
        """
        Compute the matter profile grid with feedback.
        """
        return self.compute_matter_profile_with_feedback(
            self.mass[np.newaxis, np.newaxis, :],
            self.mean_density0[:, np.newaxis, np.newaxis],
            self.u_dm,
            self.z_vec[:, np.newaxis, np.newaxis],
            self.fnu[:, np.newaxis, np.newaxis]
        )

    def compute_matter_profile_with_feedback_stellar_fraction_from_obs(self, mass, mean_density0, u_dm, z, fnu, mb, fstar):
        """
        Compute the matter profile using stellar fraction from observations.

        Using f* from HOD/CSMF/CLF that also provides for point mass estimate when used in the
        GGL power spectra

        This profile does not have 1-fnu correction as that is already accounted for in  dm_to_matter_frac
        """
        dm_to_matter_frac = self.omega_c / self.omega_m
        Wm_0 = mass / mean_density0
        f_gas_fit = self.fg_fit(mass, mb, fstar)

        Wm = (dm_to_matter_frac + f_gas_fit) * Wm_0 * u_dm * (1.0 - fnu) + fstar * Wm_0
        return Wm

    def matter_profile_with_feedback_stellar_fraction_from_obs(self, fstar):
        """
        Compute the matter profile grid using stellar fraction from observations.
        """
        return self.compute_matter_profile_with_feedback_stellar_fraction_from_obs(
            self.mass[np.newaxis, np.newaxis, :],
            self.mean_density0[:, np.newaxis, np.newaxis],
            self.u_dm,
            self.z_vec[:, np.newaxis, np.newaxis],
            self.fnu[:, np.newaxis, np.newaxis],
            self.mb,
            fstar[:, np.newaxis, :]
        )

    def one_halo_truncation(self, k_trunc=0.1):
        """
        1-halo term truncation at large scales (small k)
        """
        if k_trunc is None:
            return np.ones_like(self.k_vec)
        k_frac = self.k_vec / k_trunc
        return (k_frac**4.0) / (1.0 + k_frac**4.0)

    def two_halo_truncation(self, k_trunc=2.0):
        """
        2-halo term truncation at larger k-values (large k)
        """
        #k_frac = k_vec/k_trunc
        #return 1.0 - f * (k_frac**nd)/(1.0 + k_frac**nd)
        if k_trunc is None:
            return np.ones_like(self.k_vec)
        k_d = 0.05699#0.07
        nd = 2.853
        k_frac = self.k_vec / k_d
        return 1.0 - 0.05 * (k_frac**nd)/(1.0 + k_frac**nd)
        #return 0.5*(1.0+(erf(-(k_vec-k_trunc))))

    def one_halo_truncation_ia(self, k_trunc=4.0):
        """
        1-halo term truncation for IA.
        """
        if k_trunc is None:
            return np.ones_like(self.k_vec)
        return 1.0 - np.exp(-(self.k_vec / k_trunc)**2.0)

    def two_halo_truncation_ia(self, k_trunc=6.0):
        """
        2-halo term truncation for IA.
        """
        if k_trunc is None:
            return np.ones_like(self.k_vec)
        return np.exp(-(self.k_vec / k_trunc)**2.0)

    def one_halo_truncation_mead(self, sigma8_in):
        """
        1-halo term truncation in 2009.01858
        eq 17 and table 2
        """
        sigma8_z = sigma8_in[:, np.newaxis]
        # One-halo term damping wavenumber
        k_star = 0.05618 * sigma8_z**(-1.013) # h/Mpc
        k_frac = self.k_vec / k_star
        return (k_frac**4.0) / (1.0 + k_frac**4.0)

    def two_halo_truncation_mead(self, sigma8_in):
        """
        eq 16 of 2009.01858
        As long as nd > 0, the multiplicative term in square brackets is
        unity for k << kd and (1 − f) for k >> kd.
        This damping is used instead of the regular 2-halo term integrals
        """
        sigma8_z = sigma8_in[:, np.newaxis]
        f = 0.2696 * sigma8_z**(0.9403)
        k_d = 0.05699 * sigma8_z**(-1.089)
        nd = 2.853
        k_frac = self.k_vec / k_d
        return 1.0 - f * (k_frac**nd) / (1.0 + k_frac**nd)

    def transition_smoothing(self, neff, p_1h, p_2h):
        """
        eq 23 and table 2 of 2009.01858
        This smooths the transition between 1 and 2 halo terms.
        α = 1 would correspond to a standard transition.
        α < 1 smooths the transition while α > 1 sharpens it.
        Delta^2(k) = k^3/(2 pi^2) P(k)
        ∆^2_hmcode(k,z) = {[∆^2_2h(k,z)]^α +[∆^2_1h(k,z)]^α}^1/α
        """
        delta_prefac = (self.k_vec**3.0) / (2.0 * np.pi**2.0)
        alpha = (1.875 * (1.603**neff[:, np.newaxis]))
        Delta_1h = delta_prefac * p_1h
        Delta_2h = delta_prefac * p_2h
        Delta_hmcode = (Delta_1h**alpha + Delta_2h**alpha)**(1.0 / alpha)
        return Delta_hmcode / delta_prefac

    def compute_1h_term(self, profile_u, profile_v, mass, dndlnm):
        """
        For two fields u,v e.g. matter, galaxy, intrinsic alignment, we calculate the 1 halo term.
        P^1h_uv(k)= int W_u(k,z,M) W_v(k,z,M) n(M) dM
        If the fields are the same and they correspond to discrete tracers (e.g. satellite galaxies):
        P^1h_uv(k)= 1/n_x^2 int <N_x(M)[N_x(M)-1]> U_x(k,z,M)^2 n(M) dM + 1/n_x
        n_x = int N_x(M) n(M) dM
        The shot noise term is removed as we do our measurements in real space where it only shows up
        at zero lag which is not measured.
        See eq 22 of Asgari, Mead, Heymans 2023 review paper.
        But for satellite galaxis we use:
        <N_sat(N_sat-1)> = P_oisson <N_sat>^2:
        P^1h_ss(k)= 1/n_s^2 int P_oisson <N_sat>^2 U_s(k,z,M)^2 n(M) dM
        and write profile_u = profile_v = <N_sat> U_s(k,z,M) * sqrt(P_oisson)/n_s
        for matter halo profile is: W_m = (M/rho_m) U_m(z,k,M)
        for galaxies: W_g = (N_g(M)/n_g) U_g(z,k,M)
        """
        integrand = profile_u * profile_v * dndlnm / mass
        return simpson(integrand, mass)

    def compute_A_term(self, mass, b_dm, dndlnm, mean_density0):
        """
        Integral over the missing haloes.
        This term is used to compensate for low mass haloes that are missing from the integral in the matter 2-halo term.
        Equation A.5 of Mead and Verde 2021, 2011.08858
        A(M_min) = 1−[1/ρ¯ int_M_min^infty dM M b(M) n(M)]
        Here all missing mass is assumed to be in halos of minimum mass M_min = min(mass)
        This equation arises from
        int_0^infty M b(M) n(M) dM = ρ¯ .
        and
        int_0^infty M n(M) dM = ρ¯ .
        This ρ¯ is the mean matter density at that redshift.
        """
        integrand_m1 = b_dm * dndlnm * (1.0 / mean_density0)
        A = 1.0 - simpson(integrand_m1, mass)
        if (A < 0.0).any():
            warnings.warn('Warning: Mass function/bias correction is negative!', RuntimeWarning)
        return A

    @property
    def missing_mass_integral(self):
        return self.compute_A_term(
            self.mass[np.newaxis, np.newaxis, :],
            self.halobias[:, np.newaxis, :],
            self.dndlnm[:, np.newaxis, :],
            self.mean_density0[:, np.newaxis, np.newaxis]
        )

    @property
    def A_term(self):
        return self.missing_mass_integral

    @property
    def Im_term(self):
        """
        eq 35 of Asgari, Mead, Heymans 2023: 2303.08752
        2-halo term integral for matter, I_m = int_0^infty dM b(M) W_m(M,k) n(M) = int_0^infty dM b(M) M/rho_bar U_m(M,k) n(M)
        """
        I_m_term = self.compute_Im_term(
            self.mass[np.newaxis, np.newaxis, :],
            self.u_dm,
            self.halobias[:, np.newaxis, :],
            self.dndlnm[:, np.newaxis, :],
            self.mean_density0[:, np.newaxis, np.newaxis]
        )
        return I_m_term + self.A_term

    def compute_Im_term(self, mass, u_dm, b_dm, dndlnm, mean_density0):
        integrand_m = b_dm * dndlnm * u_dm * (1. / mean_density0)
        return simpson(integrand_m, mass)

    def prepare_I22_integrand(self, b_1, b_2, dndlnm_1, dndlnm_2, B_NL_k_z):
        """
        integrand_22 = B_NL_k_z * b_1[:,:,np.newaxis,np.newaxis] * b_2[:,np.newaxis,:,np.newaxis] \
            * dn_dlnm_z_1[:,:,np.newaxis,np.newaxis] \
            * dn_dlnm_z_2[:,np.newaxis,:,np.newaxis] \
            / (mass_1[np.newaxis,:,np.newaxis,np.newaxis] * mass_2[np.newaxis,np.newaxis,:,np.newaxis])
        """
        b_1e = b_1[:, :, np.newaxis, np.newaxis]
        b_2e = b_2[:, np.newaxis, :, np.newaxis]
        dndlnm_1e = dndlnm_1[:, :, np.newaxis, np.newaxis]
        dndlnm_2e = dndlnm_2[:, np.newaxis, :, np.newaxis]
        mass_1e = self.mass[np.newaxis, :, np.newaxis, np.newaxis]
        mass_2e = self.mass[np.newaxis, np.newaxis, :, np.newaxis]

        integrand_22 = ne.evaluate('B_NL_k_z * b_1e * b_2e * dndlnm_1e * dndlnm_2e / (mass_1e * mass_2e)')
        return integrand_22

    def prepare_I12_integrand(self, b_1, b_2, dndlnm_1, dndlnm_2, B_NL_k_z):
        """
        integrand_12 = B_NL_k_z[:,:,0,:] * b_2[:,:,np.newaxis] \
            * dn_dlnm_z_2[:,:,np.newaxis] / mass_2[np.newaxis,:,np.newaxis]
        """
        B_NL_k_z_e = B_NL_k_z[:, :, 0, :]
        b_2e = b_2[:, :, np.newaxis]
        dndlnm_2e = dndlnm_2[:, :, np.newaxis]
        mass_2e = self.mass[np.newaxis, :, np.newaxis]

        integrand_12 = ne.evaluate('B_NL_k_z_e * b_2e * dndlnm_2e / mass_2e')
        return integrand_12

    def prepare_I21_integrand(self, b_1, b_2, dndlnm_1, dndlnm_2, B_NL_k_z):
        """
        integrand_21 = B_NL_k_z[:,0,:,:] * b_1[:,:,np.newaxis] \
            * dn_dlnm_z_1[:,:,np.newaxis] / mass_1[np.newaxis,:,np.newaxis]
        """
        B_NL_k_z_e = B_NL_k_z[:, 0, :, :]
        b_1e = b_1[:, :, np.newaxis]
        dndlnm_1e = dndlnm_1[:, :, np.newaxis]
        mass_1e = self.mass[np.newaxis, :, np.newaxis]

        integrand_21 = ne.evaluate('B_NL_k_z_e * b_1e * dndlnm_1e / mass_1e')
        return integrand_21

    def I_NL(self, W_1, W_2, b_1, b_2, dndlnm_1, dndlnm_2, A, rho_mean, B_NL_k_z, integrand_12_part, integrand_21_part, integrand_22_part):
        """
        uses eqs A.7 to A.10 fo Mead and Verde 2021, 2011.08858 to calculate the integral over beta_nl
        """
        # TODO: check if we need this now the profile_c is the same format as profile_s
        # check the format of c_align_profile and s_align_profile
        # if not also combine with I_NL.
        if len(W_1.shape) < 3:
            W_1 = W_1[:, np.newaxis, :]
        if len(W_2.shape) < 3:
            W_2 = W_2[:, np.newaxis, :]

        W_1 = np.transpose(W_1, [0, 2, 1])
        W_2 = np.transpose(W_2, [0, 2, 1])

        # Takes the integral over mass_1
        # TODO: check that these integrals do the correct thing, keep this TODO
        
        #integrand_22 = integrand_22_part * W_1[:,:,np.newaxis,:] * W_2[:,np.newaxis,:,:]
        W_1e = W_1[:, :, np.newaxis, :]
        W_2e = W_2[:, np.newaxis, :, :]
        integrand_22 = ne.evaluate('integrand_22_part * W_1e * W_2e')

        integral_M1 = trapezoid(integrand_22, self.mass, axis=1)
        integral_M2 = trapezoid(integral_M1, self.mass, axis=1)
        I_22 = integral_M2

        I_11 = B_NL_k_z[:, 0, 0, :] * ((A * A) * W_1[:, 0, :] * W_2[:, 0, :] * (rho_mean[:, np.newaxis] * rho_mean[:, np.newaxis])) / (self.mass[0] * self.mass[0])

        integrand_12 = integrand_12_part * W_2[:, :, :]
        integral_12 = trapezoid(integrand_12, self.mass, axis=1)
        I_12 = A * W_1[:, 0, :] * integral_12 * rho_mean[:, np.newaxis] / self.mass[0]

        integrand_21 = integrand_21_part * W_1[:, :, :]
        integral_21 = trapezoid(integrand_21, self.mass, axis=1)
        I_21 = A * W_2[:, 0, :] * integral_21 * rho_mean[:, np.newaxis] / self.mass[0]

        I_NL = I_11 + I_12 + I_21 + I_22

        return I_NL

    def fg(self, mass, z_vec, fstar, beta=2):
        """
        Gas fraction
        Eq 24 of 2009.01858
        fg(M) = [Ωb/Ωm− f∗] (M/Mb)^β/ (1 + (M/Mb)^β)
        where fg is the halo gas fraction, the pre-factor in parenthesis is the
        available gas reservoir, while Mb > 0 and β > 0 are fitted parameters.
        Haloes of M >> Mb are unaffected while those of M < Mb have
        lost more than half of their gas

        Gas fraction from Mead2020 for baryonic feedback model
            theta_agn = log10_TAGN - 7.8
        table 4 of 2009.01858, units of M_sun/h
        """
        theta_agn = self.log10T_AGN - 7.8
        mb = np.power(10.0, 13.87 + 1.81 * theta_agn) * np.power(10.0, z_vec * (0.195 * theta_agn - 0.108))
        baryon_to_matter_fraction = self.omega_b / self.omega_m
        return (baryon_to_matter_fraction - fstar) * (mass / mb)**beta / (1.0 + (mass / mb)**beta)

    def fg_fit(self, mass, mb, fstar, beta=2):
        """
        Gas fraction
        Eq 24 of 2009.01858
        fg(M) = [Ωb/Ωm - f_*] (M/Mb)^β/ (1 + (M/Mb)^β)
        where fg is the halo gas fraction, the pre-factor in parenthesis is the
        available gas reservoir, while Mb > 0 and β > 0 are fitted parameters.
        Haloes of M >> Mb are unaffected while those of M < Mb have
        lost more than half of their gas

        Gas fraction for a general baryonic feedback model
        """
        baryon_to_matter_fraction = self.omega_b / self.omega_m
        return (baryon_to_matter_fraction - fstar) * (mass / mb)**beta / (1.0 + (mass / mb)**beta)

    def fs(self, z_vec):
        """
        Stellar fraction from table 4 and eq 26 of 2009.01858 (Mead et al. 2021)
        f*(z) = f*_0 10^(z f*_z)
        """
        theta_agn = self.log10T_AGN - 7.8
        fstar_0 = (2.01 - 0.3 * theta_agn) * 0.01
        fstar_z = 0.409 + 0.0224 * theta_agn
        return fstar_0 * np.power(10.0, z_vec * fstar_z)

    def poisson_func(self, mass, **kwargs):
        """
        Calculates the Poisson parameter for use in Pgg integrals.
        Can be either a scalar (P = poisson) or a power law (P = poisson x (M/M_0)^slope).
        Further models can be added to this function if necessary.
    
        :param mass: halo mass array
        :param kwargs: keyword arguments for different options
        :return: poisson_num, same shape as mass
        """
        poisson_type = kwargs.get('poisson_type', '')
        if poisson_type == 'scalar':
            poisson = kwargs.get('poisson', 1.0)
            return poisson * np.ones_like(mass)
    
        if poisson_type == 'power_law':
            poisson = kwargs.get('poisson', 1.0)
            M_0 = kwargs.get('M_0', None)
            slope = kwargs.get('slope', None)
            if M_0 is None or slope is None:
                raise ValueError("M_0 and slope must be provided for 'power_law' poisson_type.")
            return poisson * (mass / (10.0**M_0))**slope
    
        return np.ones_like(mass)
        
    def compute_power_spectrum_mm(
            self,
            one_halo_ktrunc=None,
            two_halo_ktrunc=None,
            sigma8_z=None,
            neff=None,
            fstar=None
        ):
    
        galaxy_linear_bias = None

        
        if self.mead_correction == 'feedback':
            self.matter_profile_1h_mm = self.matter_profile_with_feedback
        elif self.mead_correction == 'fit':
            self.matter_profile_1h = self.matter_profile_with_feedback_stellar_fraction_from_obs(fstar)[0]
        else:
            self.matter_profile_1h = self.matter_profile
        if self.bnl:
            I_NL = self.I_NL(self.matter_profile, self.matter_profile, self.halobias, self.halobias, self.dndlnm, self.dndlnm, self.A_term, self.mean_density0, self.beta_nl, self.I12, self.I21, self.I22)
            pk_2h = (self.matter_power_lin * self.Im_term * self.Im_term + self.matter_power_lin * I_NL) #* self.two_halo_truncation()[np.newaxis, :]
            pk_1h = self.compute_1h_term(self.matter_profile_1h, self.matter_profile_1h, self.mass[np.newaxis, np.newaxis, :], self.dndlnm[:, np.newaxis, :]) * self.one_halo_truncation(one_halo_ktrunc)
            pk_tot = pk_1h + pk_2h
        else:
            if self.mead_correction in ['feedback', 'no_feedback']:
                pk_2h = self.matter_power_lin * self.two_halo_truncation_mead(sigma8_z)
                pk_1h = self.compute_1h_term(self.matter_profile_1h, self.matter_profile_1h, self.mass[np.newaxis, np.newaxis, :], self.dndlnm[:, np.newaxis, :]) * self.one_halo_truncation_mead(sigma8_z)
                pk_tot = self.transition_smoothing(neff, pk_1h, pk_2h)
            else :
                pk_2h = self.matter_power_lin * self.Im_term * self.Im_term * self.two_halo_truncation(two_halo_ktrunc)[np.newaxis, :]
                pk_1h = self.compute_1h_term(self.matter_profile_1h, self.matter_profile_1h, self.mass[np.newaxis, np.newaxis, :], self.dndlnm[:, np.newaxis, :]) * self.one_halo_truncation(one_halo_ktrunc)
                pk_tot = pk_1h + pk_2h
    
        return [pk_1h], [pk_2h], [pk_tot], galaxy_linear_bias



class GalaxySpectra(MatterSpectra):
    def __init__(self,
            u_sat = None,
            Ncen = None,
            Nsat = None,
            numdencen = None,
            numdensat = None,
            f_c = None,
            f_s = None,
            nbins = None,
            pointmass = None,
            **matter_spectra_kwargs
        ):
        
        # Call super init MUST BE DONE FIRST.
        super().__init__(**matter_spectra_kwargs)

        self.u_sat = u_sat
        self.Ncen = Ncen
        self.Nsat = Nsat
        self.numdencen = numdencen
        self.numdensat = numdensat
        self.f_c = f_c
        self.f_s = f_s
        self.nbins = nbins
        self.pointmass = pointmass
        
    @property
    def central_galaxy_profile(self):
        """
        galaxy profile for a sample of centrals galaxies.
        set u_sample to ones if centrals are in the centre of the halo
        """
        profile = [
            self.f_c[i][:, np.newaxis, np.newaxis] * self.Ncen[i][:, np.newaxis, :] * np.ones_like(self.u_sat[i]) / self.numdencen[i][:, np.newaxis, np.newaxis]
            for i in range(self.nbins)
        ]
        return profile

    @property
    def satellite_galaxy_profile(self):
        """
        galaxy profile for a sample of satellite galaxies.
        """
        profile = [
            self.f_s[i][:, np.newaxis, np.newaxis] * self.Nsat[i][:, np.newaxis, :] * self.u_sat[i] / self.numdensat[i][:, np.newaxis, np.newaxis]
            for i in range(self.nbins)
        ]
        return profile

    def compute_Ig_term(self, profile, mass, dndlnm, b_m):
        integrand = profile * b_m * dndlnm / mass
        return simpson(integrand, mass)

    @property
    def Ic_term(self):
        term = [
            self.compute_Ig_term(
                self.central_galaxy_profile[i],
                self.mass[np.newaxis, np.newaxis, :],
                self.dndlnm[:, np.newaxis, :],
                self.halobias[:, np.newaxis, :]
            ) for i in range(self.nbins)
        ]
        return term

    @property
    def Is_term(self):
        term = [
            self.compute_Ig_term(
                self.satellite_galaxy_profile[i],
                self.mass[np.newaxis, np.newaxis, :],
                self.dndlnm[:, np.newaxis, :],
                self.halobias[:, np.newaxis, :]
            ) for i in range(self.nbins)
        ]
        return term

    def compute_power_spectrum_gg(
            self,
            one_halo_ktrunc=None,
            two_halo_ktrunc=None,
            sigma8_z=None,
            neff=None,
            poisson_par=None,
            fstar=None
        ):
    
        galaxy_linear_bias = [None] * self.nbins
        pk_1h = [None] * self.nbins
        pk_2h = [None] * self.nbins
        pk_tot = [None] * self.nbins
    
        for i in range(self.nbins):
            poisson = self.poisson_func(self.mass, **poisson_par)
            if self.bnl:
                I_NL = self.I_NL(self.central_galaxy_profile[i] + self.satellite_galaxy_profile[i], self.central_galaxy_profile[i] + self.satellite_galaxy_profile[i], self.halobias, self.halobias, self.dndlnm, self.dndlnm, self.A_term,self.mean_density0, self.beta_nl, self.I12, self.I21, self.I22)
                pk_cc_2h = self.matter_power_lin * self.Ic_term[i] * self.Ic_term[i]
                pk_ss_2h = self.matter_power_lin * self.Is_term[i] * self.Is_term[i]
                pk_cs_2h = self.matter_power_lin * self.Ic_term[i] * self.Is_term[i]
            else:
                pk_cc_2h = self.matter_power_lin * self.Ic_term[i] * self.Ic_term[i] * self.two_halo_truncation(two_halo_ktrunc)[np.newaxis, :]
                pk_ss_2h = self.matter_power_lin * self.Is_term[i] * self.Is_term[i] * self.two_halo_truncation(two_halo_ktrunc)[np.newaxis, :]
                pk_cs_2h = self.matter_power_lin * self.Ic_term[i] * self.Is_term[i] * self.two_halo_truncation(two_halo_ktrunc)[np.newaxis, :]
    
            pk_cs_1h = self.compute_1h_term(self.central_galaxy_profile[i], self.satellite_galaxy_profile[i], self.mass[np.newaxis, np.newaxis, :], self.dndlnm[:, np.newaxis, :]) * self.one_halo_truncation(one_halo_ktrunc)
            pk_ss_1h = self.compute_1h_term(self.satellite_galaxy_profile[i] * poisson, self.satellite_galaxy_profile[i], self.mass[np.newaxis, np.newaxis, :], self.dndlnm[:, np.newaxis, :]) * self.one_halo_truncation(one_halo_ktrunc)
    
            pk_1h[i] = 2.0 * pk_cs_1h + pk_ss_1h
            pk_2h[i] = pk_cc_2h + pk_ss_2h + 2.0 * pk_cs_2h
            if self.bnl:
                pk_2h[i] += self.matter_power_lin * I_NL
    
            pk_tot[i] = pk_1h[i] + pk_2h[i]
            galaxy_linear_bias[i] = np.sqrt(self.Ic_term[i] * self.Ic_term[i] + self.Is_term[i] * self.Is_term[i] + 2.0 * self.Ic_term[i] * self.Is_term[i])
            
        return pk_1h, pk_2h, pk_tot, galaxy_linear_bias
    
    def compute_power_spectrum_gm(
            self,
            one_halo_ktrunc=None,
            two_halo_ktrunc=None,
            sigma8_z=None,
            neff=None,
            poisson_par=None,
            fstar=None
        ):
    
        galaxy_linear_bias = [None] * self.nbins
        pk_1h = [None] * self.nbins
        pk_2h = [None] * self.nbins
        pk_tot = [None] * self.nbins
        
        for i in range(self.nbins):
            if self.mead_correction == 'feedback':
                self.matter_profile_1h = self.matter_profile_with_feedback
            elif self.mead_correction == 'fit' or self.pointmass:
                self.matter_profile_1h = self.matter_profile_with_feedback_stellar_fraction_from_obs(fstar[i])
            else:
                self.matter_profile_1h = self.matter_profile
                
            if self.bnl:
                I_NL = self.I_NL(self.central_galaxy_profile[i] + self.satellite_galaxy_profile[i], self.matter_profile, self.halobias, self.halobias, self.dndlnm, self.dndlnm, self.A_term, self.mean_density0, self.beta_nl, self.I12, self.I21,self.I22)
                pk_cm_2h = self.matter_power_lin * self.Ic_term[i] * self.Im_term
                pk_sm_2h = self.matter_power_lin * self.Is_term[i] * self.Im_term
            else:
                pk_cm_2h = self.matter_power_lin * self.Ic_term[i] * self.Im_term * self.two_halo_truncation(two_halo_ktrunc)[np.newaxis, :]
                pk_sm_2h = self.matter_power_lin * self.Is_term[i] * self.Im_term * self.two_halo_truncation(two_halo_ktrunc)[np.newaxis, :]
    
            pk_cm_1h = self.compute_1h_term(self.central_galaxy_profile[i], self.matter_profile_1h, self.mass[np.newaxis, np.newaxis, :], self.dndlnm[:, np.newaxis, :]) * self.one_halo_truncation(one_halo_ktrunc)
            pk_sm_1h = self.compute_1h_term(self.satellite_galaxy_profile[i], self.matter_profile_1h, self.mass[np.newaxis, np.newaxis, :], self.dndlnm[:, np.newaxis, :]) * self.one_halo_truncation(one_halo_ktrunc)
    
            pk_1h[i] = pk_cm_1h + pk_sm_1h
            pk_2h[i] = pk_cm_2h + pk_sm_2h
            if self.bnl:
                pk_2h[i] += self.matter_power_lin * I_NL
    
            pk_tot[i] = pk_1h[i] + pk_2h[i]
            galaxy_linear_bias[i] = np.sqrt(self.Ic_term[i] * self.Im_term + self.Is_term[i] * self.Im_term)
    
        return pk_1h, pk_2h, pk_tot, galaxy_linear_bias



class AlignmentSpectra(GalaxySpectra):
    def __init__(self,
            alignment_gi = None,
            wkm_sat = None,
            t_eff = None,
            beta_cen = None,
            beta_sat = None,
            mpivot_cen = None,
            mpivot_sat = None,
            growth_factor = None,
            scale_factor = None,
            mass_avg = None,
            matter_power_nl = None,
            fortuna = False,
            **galaxy_spectra_kwargs
        ):
        
        # Call super init MUST BE DONE FIRST.
        super().__init__(**galaxy_spectra_kwargs)
        
        self.beta_cen = beta_cen
        self.beta_sat = beta_sat
        self.mpivot_cen = mpivot_cen
        self.mpivot_sat = mpivot_sat
        self.mass_avg = mass_avg
        self.t_eff = t_eff
        self.alignment_gi = alignment_gi
        self.wkm_sat = wkm_sat
        self.growth_factor = growth_factor
        self.scale_factor = scale_factor
        self.fortuna = fortuna
        
        self.alignment_amplitude_2h, self.alignment_amplitude_2h_II, self.C1 = self.compute_two_halo_alignment
    
        if self.fortuna:
            self.matter_power_nl = matter_power_nl
            if not len(self.f_c) == self.nbins:
                raise ValueError('f_c needs to have same length as number of bins provided')
            self.peff = (1.0 - self.t_eff) * self.matter_power_nl + self.t_eff * self.matter_power_lin
        
    def compute_central_galaxy_alignment_profile(self, scale_factor, growth_factor, f_c, C1, mass, beta=None, mpivot=None, mass_avg=None):
        """
        Compute the central galaxy alignment profile.
        """
        if beta is not None and mpivot is not None and mass_avg is not None:
            additional_term = (mass_avg / mpivot) ** beta
        else:
            additional_term = 1.0
        return f_c * (C1 / growth_factor) * mass * additional_term# * scale_factor**2.0

    def compute_satellite_galaxy_alignment_profile(self, Nsat, numdenssat, f_s, wkm_sat, beta=None, mpivot=None, mass_avg=None):
        """
        Compute the satellite galaxy alignment profile.
        """
        if beta is not None and mpivot is not None and mass_avg is not None:
            additional_term = (mass_avg / mpivot) ** beta
        else:
            additional_term = 1.0
        return f_s * Nsat * wkm_sat / numdenssat * additional_term

    @property
    def central_alignment_profile(self):
        """
        Prepare the grid in z, k and mass for the central alignment
        f_cen/n_cen N_cen gamma_hat(k,M)
        where gamma_hat(k,M) is the Fourier transform of the density weighted shear, i.e. the radial dependent power law
        times the NFW profile, here computed by the module wkm, while gamma_1h is only the luminosity dependence factor.
        """
        profile = [
            self.compute_central_galaxy_alignment_profile(
                self.scale_factor[:, :, np.newaxis],
                self.growth_factor[:, :, np.newaxis],
                self.f_c[i][:, np.newaxis, np.newaxis],
                self.C1,
                self.mass[np.newaxis, np.newaxis, :],
                self.beta_cen,
                self.mpivot_cen,
                self.mass_avg[i][:, np.newaxis, np.newaxis]
            ) for i in range(self.nbins)
        ]
        return profile

    @property
    def satellite_alignment_profile(self):
        """
        Prepare the grid in z, k and mass for the satellite alignment
        f_sat/n_sat N_sat gamma_hat(k,M)
        where gamma_hat(k,M) is the Fourier transform of the density weighted shear, i.e. the radial dependent power law
        times the NFW profile, here computed by the module wkm, while gamma_1h is only the luminosity dependence factor.
        """
        profile = [
            self.compute_satellite_galaxy_alignment_profile(
                self.Nsat[i][:, np.newaxis, :],
                self.numdensat[i][:, np.newaxis, np.newaxis],
                self.f_s[i][:, np.newaxis, np.newaxis],
                self.wkm_sat.transpose(0, 2, 1),
                self.beta_sat,
                self.mpivot_sat,
                self.mass_avg[i][:, np.newaxis, np.newaxis]
            ) for i in range(self.nbins)
        ]
        return profile


    @property
    def Ic_align_term(self):
        I_g_align = [
            self.compute_Ig_term(
                self.central_alignment_profile[i],
                self.mass[np.newaxis, np.newaxis, :],
                self.dndlnm[:, np.newaxis, :],
                self.halobias[:, np.newaxis, :]
            ) for i in range(self.nbins)
        ]
        return [I_g_align[i] + self.A_term * self.central_alignment_profile[i][:, :, 0] * self.mean_density0[:, np.newaxis] / self.mass[0]
                for i in range(self.nbins)]
     
    @property
    def Is_align_term(self):
        I_g_align = [
            self.compute_Ig_term(
                self.satellite_alignment_profile[i],
                self.mass[np.newaxis, np.newaxis, :],
                self.dndlnm[:, np.newaxis, :],
                self.halobias[:, np.newaxis, :]
            ) for i in range(self.nbins)
        ]
        return [I_g_align[i] + self.A_term * self.satellite_alignment_profile[i][:, :, 0] * self.mean_density0[:, np.newaxis] / self.mass[0]
                for i in range(self.nbins)]
    
    @property
    def compute_two_halo_alignment(self):
        """
        The IA amplitude at large scales, including the IA prefactors.
    
        :param alignment_gi: double array 1d (nz), alignment coefficient for GI
        :param growth_factor: double array 2d (nz, nk), growth factor normalized to be 1 at z=0
        :param mean_density0: double, mean matter density of the Universe at redshift z=0
        :return: tuple of double array 2d (nz, nk), the large scale alignment amplitudes (GI and II)
        """
        # Linear alignment coefficient
        C1 = 5e-14
    
        # Calculate alignment amplitudes using broadcasting
        alignment_amplitude_2h = -self.alignment_gi[:, np.newaxis] * (C1 * self.mean_density0[:, np.newaxis] / self.growth_factor)
        alignment_amplitude_2h_II = (self.alignment_gi[:, np.newaxis] * C1 * self.mean_density0[:, np.newaxis] / self.growth_factor)**2.0
        self.C1 = C1 * self.alignment_gi[:, np.newaxis, np.newaxis]
    
        return alignment_amplitude_2h, alignment_amplitude_2h_II, C1 * self.alignment_gi[:, np.newaxis, np.newaxis]

    def compute_power_spectrum_mi(
            self,
            one_halo_ktrunc=None,
            two_halo_ktrunc=None,
            sigma8_z=None,
            neff=None,
            poisson_par=None,
            fstar=None
        ):
    
        galaxy_linear_bias = None
        pk_1h = [None] * self.nbins
        pk_2h = [None] * self.nbins
        pk_tot = [None] * self.nbins
    
        for i in range(self.nbins):
            if self.mead_correction == 'feedback':
                self.matter_profile_1h = self.matter_profile_with_feedback
            elif self.mead_correction == 'fit' or self.pointmass:
                self.matter_profile_1h = self.matter_profile_with_feedback_stellar_fraction_from_obs(fstar[i])
            else:
                self.matter_profile_1h = self.matter_profile
                
            if self.bnl:
                I_NL = self.I_NL(self.central_alignment_profile[i] + self.satellite_alignment_profile[i], self.matter_profile, self.halobias, self.halobias, self.dndlnm, self.dndlnm, self.A_term, self.mean_density0, self.beta_nl, self.I12, self.I21,self.I22)
                pk_sm_2h = (-1.0) * self.matter_power_lin * self.Is_align_term[i] * self.Im_term
                pk_cm_2h = (-1.0) * self.matter_power_lin * self.Ic_align_term[i] * self.Im_term
            elif self.fortuna:
                pk_cm_2h = self.f_c[i][:, np.newaxis] * self.peff * self.alignment_amplitude_2h * self.two_halo_truncation_ia(two_halo_ktrunc)[np.newaxis, :]
            else:
                pk_sm_2h = (-1.0) * self.matter_power_lin * self.Is_align_term[i] * self.Im_term * self.two_halo_truncation_ia(two_halo_ktrunc)[np.newaxis, :]
                pk_cm_2h = (-1.0) * self.matter_power_lin * self.Ic_align_term[i] * self.Im_term * self.two_halo_truncation_ia(two_halo_ktrunc)[np.newaxis, :]
    
            pk_sm_1h = (-1.0) * self.compute_1h_term(self.matter_profile_1h, self.satellite_alignment_profile[i], self.mass[np.newaxis, np.newaxis, :], self.dndlnm[:, np.newaxis, :]) * self.one_halo_truncation_ia(one_halo_ktrunc)
            #pk_cm_1h = (-1.0) * self.compute_1h_term(self.matter_profile_1h, self.central_alignment_profile[i], self.mass[np.newaxis, np.newaxis, :], self.dndlnm[:, np.newaxis, :]) * self.one_halo_truncation_ia(one_halo_ktrunc)
    
            if self.bnl:
                pk_1h[i] = pk_sm_1h
                pk_2h[i] = pk_cm_2h + pk_sm_2h - self.matter_power_lin * I_NL
                pk_tot[i] = pk_sm_1h + pk_cm_2h + pk_sm_2h - self.matter_power_lin * I_NL
            elif self.fortuna:
                pk_1h[i] = pk_sm_1h
                pk_2h[i] = pk_cm_2h
                pk_tot[i] = pk_sm_1h + pk_cm_2h
            else:
                pk_1h[i] = pk_sm_1h
                pk_2h[i] = pk_cm_2h + pk_sm_2h
                pk_tot[i] = pk_sm_1h + pk_cm_2h + pk_sm_2h
                
        return pk_1h, pk_2h, pk_tot, galaxy_linear_bias
    
    def compute_power_spectrum_ii(
            self,
            one_halo_ktrunc=None,
            two_halo_ktrunc=None,
            sigma8_z=None,
            neff=None,
            poisson_par=None,
            fstar=None
        ):
    
        galaxy_linear_bias = None
        pk_1h = [None] * self.nbins
        pk_2h = [None] * self.nbins
        pk_tot = [None] * self.nbins
        
        # Needs Poisson parameter as well!
        for i in range(self.nbins):
            if self.bnl:
                I_NL_ss = self.I_NL(self.satellite_alignment_profile[i], self.satellite_alignment_profile[i], self.halobias, self.halobias, self.dndlnm, self.dndlnm, self.A_term, self.mean_density0, self.beta_nl, self.I12, self.I21, self.I22)
                I_NL_cc = self.I_NL(self.central_alignment_profile[i], self.central_alignment_profile[i], self.halobias, self.halobias, self.dndlnm, self.dndlnm, self.A_term, self.mean_density0, self.beta_nl, self.I12, self.I21, self.I22)
                I_NL_cs = self.I_NL(self.central_alignment_profile[i], self.satellite_alignment_profile[i], self.halobias, self.halobias, self.dndlnm, self.dndlnm, self.A_term, self.mean_density0, self.beta_nl, self.I12, self.I21, self.I22)
                pk_ss_2h = self.matter_power_lin * self.Is_align_term[i] * self.Is_align_term[i] + self.matter_power_lin * I_NL_ss
                pk_cc_2h = self.matter_power_lin * self.Ic_align_term[i] * self.Ic_align_term[i] + self.matter_power_lin * I_NL_cc
                pk_cs_2h = self.matter_power_lin * self.Ic_align_term[i] * self.Is_align_term[i] + self.matter_power_lin * I_NL_cs
            elif self.fortuna:
                pk_cc_2h = self.f_c[i][i][:, np.newaxis]**2.0 * self.peff * self.alignment_amplitude_2h_II * self.two_halo_truncation_ia(two_halo_ktrunc)[np.newaxis, :]
            else:
                pk_ss_2h = self.matter_power_lin * self.Is_align_term[i] * self.Is_align_term[i] * self.two_halo_truncation_ia(two_halo_ktrunc)[np.newaxis, :]
                pk_cc_2h = self.matter_power_lin * self.Ic_align_term[i] * self.Ic_align_term[i] * self.two_halo_truncation_ia(two_halo_ktrunc)[np.newaxis, :]
                pk_cs_2h = self.matter_power_lin * self.Ic_align_term[i] * self.Is_align_term[i] * self.two_halo_truncation_ia(two_halo_ktrunc)[np.newaxis, :]
    
            pk_ss_1h = self.compute_1h_term(self.satellite_alignment_profile[i], self.satellite_alignment_profile[i], self.mass[np.newaxis, np.newaxis, :], self.dndlnm[:, np.newaxis, :]) * self.one_halo_truncation_ia(one_halo_ktrunc)
            #pk_cs_1h = self.compute_1h_term(self.central_alignment_profile[i], self.satellite_alignment_profile[i], self.mass[np.newaxis, np.newaxis, :], self.dndlnm[:, np.newaxis, :]) * self.one_halo_truncation_ia(one_halo_ktrunc)
    
            if self.fortuna:
                pk_1h[i] = pk_ss_1h
                pk_2h[i] = pk_cc_2h
                pk_tot[i] = pk_ss_1h + pk_cc_2h
            else:
                pk_1h[i] = pk_ss_1h
                pk_2h[i] = pk_cc_2h + pk_ss_2h + pk_cs_2h
                pk_tot[i] = pk_ss_1h + pk_cc_2h + pk_ss_2h + pk_cs_2h
                
        return pk_1h, pk_2h, pk_tot, galaxy_linear_bias
    
    def compute_power_spectrum_gi(
            self,
            one_halo_ktrunc=None,
            two_halo_ktrunc=None,
            sigma8_z=None,
            neff=None,
            poisson_par=None,
            fstar=None
        ):
    
        galaxy_linear_bias = None
        pk_1h = [None] * self.nbins
        pk_2h = [None] * self.nbins
        pk_tot = [None] * self.nbins
        
        for i in range(self.nbins):
            if self.bnl:
                I_NL_cc = self.I_NL(self.central_galaxy_profile[i], self.central_alignment_profile[i], self.halobias, self.halobias, self.dndlnm, self.dndlnm, self.A_term, self.mean_density0, self.beta_nl, self.I12, self.I21, self.I22)
                I_NL_cs = self.I_NL(self.central_galaxy_profile[i], self.satellite_alignment_profile[i], self.halobias, self.halobias, self.dndlnm, self.dndlnm, self.A_term, self.mean_density0, self.beta_nl, self.I12, self.I21, self.I22)
                pk_cc_2h = self.matter_power_lin * self.Ic_term[i] * self.Ic_align_term[i] + self.matter_power_lin * I_NL_cc
                pk_cs_2h = self.matter_power_lin * self.Ic_term[i] * self.Is_align_term[i] + self.matter_power_lin * I_NL_cs
            elif self.fortuna:
                pk_cc_2h = -1.0 * self.peff * self.Ic_term[i] * self.alignment_amplitude_2h[:,] * self.two_halo_truncation_ia(two_halo_ktrunc)[np.newaxis, :]
            else:
                pk_cc_2h = self.matter_power_lin * self.Ic_term[i] * self.Ic_align_term[i] * self.two_halo_truncation_ia(two_halo_ktrunc)[np.newaxis, :]
                pk_cs_2h = self.matter_power_lin * self.Ic_term[i] * self.Is_align_term[i] * self.two_halo_truncation_ia(two_halo_ktrunc)[np.newaxis, :]
    
            pk_cs_1h = self.compute_1h_term(self.central_galaxy_profile[i], self.satellite_alignment_profile[i], self.mass[np.newaxis, np.newaxis, :], self.dndlnm[:, np.newaxis, :]) * self.one_halo_truncation_ia(one_halo_ktrunc)
    
            if self.fortuna:
                pk_1h[i] = pk_cs_1h
                pk_2h[i] = pk_cc_2h
                pk_tot[i] = pk_cs_1h + pk_cc_2h
            else:
                pk_1h[i] = pk_cs_1h
                pk_2h[i] = pk_cs_2h + pk_cc_2h
                pk_tot[i] = pk_cs_1h + pk_cs_2h + pk_cc_2h
    
        return pk_1h, pk_2h, pk_tot, galaxy_linear_bias
