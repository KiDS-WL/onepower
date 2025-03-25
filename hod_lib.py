import numpy as np
from scipy.integrate import simps

# Conversion functions

def convert_to_luminosity(abs_mag, abs_mag_sun):
    """Convert absolute magnitude to luminosity."""
    logL = -0.4 * (abs_mag - abs_mag_sun)
    return 10.0**logL

def convert_to_magnitudes(L, abs_mag_sun):
    """Convert luminosity to absolute magnitude."""
    logL = np.log10(L)
    Mr = -2.5 * logL + abs_mag_sun
    return Mr

# ------------------------------------------#
#    Conditional Observable Function (COF)  #
# ------------------------------------------#
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

def COF_cen(obs, mass, hod_par):
    """
    COF for Central galaxies.
    eq 17 of D23: 2210.03110:
    Φc(O|M) = 1/[√(2π) ln(10) σ_c O] exp[ -log(O/O∗c)^2/ (2 σ_c^2) ]
    Note Φc(O|M) is unitless.
    """
    mean_obs_c = cal_mean_obs_c(mass, hod_par)  # O∗c
    COF_c = (1.0 / (np.sqrt(2.0 * np.pi) * np.log(10.0) * hod_par.sigma_log10_O_c * obs) *
             np.exp(-(np.log10(obs / mean_obs_c))**2 / (2.0 * hod_par.sigma_log10_O_c**2)))
    return COF_c

def COF_sat(obs, mass, hod_par):
    """
    COF for satellite galaxies.
    eq 18 of D23: 2210.03110:
    Φs(O|M) = ϕ∗s/O∗s (O/O∗s)^α_s exp [−(O/O∗s)^2], O*s is O∗s(M) = 0.56 O∗c(M)
    Note Φs(O|M) is unitless.
    """
    obs_s_star = hod_par.norm_s * cal_mean_obs_c(mass, hod_par)
    obs_tilde = obs / obs_s_star
    phi_star_val = phi_star_s(mass, hod_par)
    COF_s = (phi_star_val / obs_s_star) * (obs_tilde**hod_par.alpha_s) * np.exp(-obs_tilde**hod_par.beta_s)
    return COF_s

def obs_func(mass, phi_x, dn_dlnM_normalised, axis=-1):
    """
    The Observable function, this is Φs(O|M), Φc(O|M) integrated over the halo mass weighted
    with the Halo Mass Function (HMF) to give:  Φs(O),Φc(O)
    Φx(O) =int Φx(O|M) n(M) dM,
    dn_dlnM_normalised is basically n(M) x mass, it is the output of hmf
    The differential mass function in terms of natural log of m,
    len=len(m) [units \(h^3 Mpc^{-3}\)]
    dn(m)/ dln m eq1 of 1306.6721
    obs_func unit is h^3 Mpc^{-3} dex^-1
    """
    integrand = phi_x * dn_dlnM_normalised / mass
    obs_function = simps(integrand, mass, axis=axis)
    return obs_function

def cal_mean_obs_c(mass, hod_par):
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
    mean_obs_c = (hod_par.Obs_norm_c * (mass / hod_par.M_char)**hod_par.g_1 /
                  (1.0 + (mass / hod_par.M_char))**(hod_par.g_1 - hod_par.g_2))
    return mean_obs_c

def phi_star_s(mass, hod_par):
    """
    pivot COF used in eq 21 of D23: 2210.03110
    using a bias expantion around the pivot mass
    eq 22 of D23: 2210.03110
    log[ϕ∗s(M)] = b0 + b1(log m13) , m13 is logM_pivot, m13 = M/(hod_par.pivot M⊙ h−1)
    """
    logM_pivot = np.log10(mass) - hod_par.pivot
    log_phi_s = hod_par.b0 + hod_par.b1 * logM_pivot + hod_par.b2 * (logM_pivot**2.0)
    return 10.0**log_phi_s


# TODO: section 4 of https://arxiv.org/pdf/1512.03050 defines <N|M> differently to us?! eq 5 is the same.
# but then eq 8 is not what we expect!
def compute_hod(obs, COF_x):
    """
    The HOD is computed by integrating over the COFs
    eq 23 of D23: 2210.03110
    ⟨Nx|M⟩ =int_{O_low}^{O_high} Φx(O|M) dO
    """
    hod_x = simps(COF_x, obs)
    return hod_x

def compute_stellar_fraction(obs, phi_x):
    """
    The mean value of the observable for the given galaxy population for a given halo mass.
    O is weighted by the number of galaxies with the peroperty O for each halo mass: Φx(O|M)
    f_star = int_{O_low}^{O_high} Φx(O|M) O dO
    """
    integral = simps(phi_x * obs, obs)
    return integral

def compute_number_density(mass, hod_x, dn_dlnM_normalised):
    """
    Total number density of galaxies with the given HOD, e.g. central and satellite galaxies
    This is an integral over the HOD and the halo mass function to remove the halo mass dependence.
    Nx = int ⟨Nx|M⟩ n(M) dM
    """
    integrand = hod_x * dn_dlnM_normalised / mass
    n_density = simps(integrand, mass)
    return n_density

def compute_avg_halo_mass(mass, hod_x, dn_dlnM_normalised):
    """
    The mean halo mass for the given population of galaxies, e.g. central and satellite galaxies
    M_mean = int ⟨Nx|M⟩ M n(M) dM
    """
    integrand = hod_x * dn_dlnM_normalised
    mean_halo_mass = simps(integrand, mass)
    return mean_halo_mass

def compute_galaxy_linear_bias(mass, hod_x, halo_bias, dn_dlnM_normalised):
    """
    Mean linear halo bias for the given population of galaxies.
    b_lin_x = int ⟨Nx|M⟩ b_h(M) n(M) dM
    """
    bg_integrand = hod_x * halo_bias * dn_dlnM_normalised / mass
    bg_integral = simps(bg_integrand, mass)
    return bg_integral
