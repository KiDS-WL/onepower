# Library of the power spectrum module

import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator, UnivariateSpline
from scipy.integrate import simps
from scipy.special import erf
from scipy.optimize import curve_fit
import warnings


# from darkmatter_lib import compute_u_dm, radvir_from_mass


# TODO: check simps integration

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

and the integal over beta_nl is

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

# -------------------------------------------------------------------------------------------------------------------- #
# Read in from block
# -------------------------------------------------------------------------------------------------------------------- #

def interpolate_in_z(input_grid, z_in, z_out,axis=0):
    """
    Interpolation in redshift
    Default redshift axis is the first one. 
    """
    
    f_interp = interp1d(z_in, input_grid, axis=axis)
    interpolated_grid = f_interp(z_out)
    
    return interpolated_grid

# Reads in linear matter power spectrum 
def get_linear_power_spectrum(block, z_vec):
    """
    Reads in linear matter power spectrum and downsamples
    """
    
    k_vec = block['matter_power_lin', 'k_h']
    z_pl  = block['matter_power_lin', 'z']
    matter_power_lin   = block['matter_power_lin', 'p_k']
    
    # interpolate in redshift
    plin = interpolate_in_z(matter_power_lin, z_pl, z_vec)
    
    return k_vec, plin

# Reads in the growth factor
def get_growth_factor(block, z_vec, k_vec):
    """
    Loads and interpolates the growth factor
    and scale factor
    """
    z_in  = block['growth_parameters', 'z']
    
    # reads in the growth factor and turns it into a 2D array that has this dimensions: len(z) x len(k)
    # all columns are identical
    growth_factor_in  = block['growth_parameters', 'd_z']
    growth_factor = interpolate_in_z(growth_factor_in, z_in, z_vec)
    growth_factor = growth_factor.flatten()[:,np.newaxis] * np.ones(k_vec.size)
 
    scale_factor  = 1./(1.+z_vec)
    scale_factor  = scale_factor.flatten()[:,np.newaxis] * np.ones(k_vec.size)
    
    return growth_factor, scale_factor


def get_nonlinear_power_spectrum(block, z_vec):
    """
    Reads in the non-linear matter power specturm and downsamples
    """
    
    k_nl = block['matter_power_nl', 'k_h']
    z_nl = block['matter_power_nl', 'z']
    matter_power_nl = block['matter_power_nl', 'p_k']
    p_nl = interpolate_in_z(matter_power_nl, z_nl, z_vec)
    
    return k_nl, p_nl


def log_linear_interpolation_k(power_in, k_in, k_out, axis=1, kind='linear'):
    """
    log-linear interpolation for power spectra. This works well for extrapolating to higher k.
    Ideally we want to have a different routine for interpolation (spline) and extrapolation (log-linear)
    """
    power_interp = interp1d(np.log(k_in), np.log(power_in), axis=axis, kind=kind, fill_value='extrapolate')
    power_out = np.exp(power_interp(np.log(k_out)))
    return power_out
    
def get_halo_functions(block, mass, z_vec):
    """
    Loads the halo mass function and linear halo bias, checks that the redshift and mass bins are the same as the inputs
    """
    
    # load the halo mass function
    mass_hmf    = block['hmf', 'm_h']
    z_hmf       = block['hmf', 'z']
    dndlnmh_hmf = block['hmf', 'dndlnmh']
    
    # load the halo bias
    mass_hbf     = block['halobias', 'm_h']
    z_hbf        = block['halobias', 'z']
    halobias_hbf = block['halobias', 'b_hb']
    
    # interpolate all the quantities that enter in the integrals
    #dn_dlnm = interpolate2d_HM(dndlnmh_hmf, mass_hmf, z_hmf, mass, z_vec)
    #b_dm    = interpolate2d_HM(halobias_hbf, mass_hbf, z_hbf, mass, z_vec)
    
    if((mass_hmf!=mass).any() or (mass_hbf!=mass).any()):
        raise Exception('The mass values are different to the input mass values.')
    if((z_hmf!=z_vec).any() or (z_hbf!=z_vec).any()):
        raise Exception('The redshift values are different to the input redshift values.')
    
    return dndlnmh_hmf, halobias_hbf #dn_dlnm, b_dm


# TODO: Check if this interpolation works well
def interpolate2d_HM(input_grid, x_in, y_in, x_out, y_out, method = 'linear'):
    
    f_interp = RegularGridInterpolator((x_in.T, y_in.T), input_grid.T,method=method, bounds_error=False, fill_value=None)
    xx, yy = np.meshgrid(x_out, y_out, sparse=True)
    interpolated = f_interp((xx.T, yy.T)).T
    
    return interpolated

def get_satellite_alignment(block, k_vec, mass, z_vec, suffix):
    """
    Loads and interpolates the wkm profiles needed for calculating the IA power spectra
    """
    # here I am assuming that the redshifts used in wkm_module and the pk_module match!
    wkm = np.empty([z_vec.size, mass.size, k_vec.size])

    for jz in range(0,z_vec.size):

        wkm_tmp  = block['wkm',f'w_km_{jz}{suffix}']
        k_wkm    = block['wkm',f'k_h_{jz}{suffix}']
        mass_wkm = block['wkm',f'mass_{jz}{suffix}']

        #CH: I've tested different approaches here and find interpolating log(w(k|m)/k^2) provides the best accuracy
        #given a low-resolution grid in logk and logm.
        lg_w_interp2d = RegularGridInterpolator((np.log10(k_wkm).T, np.log10(mass_wkm).T), np.log10(wkm_tmp/k_wkm**2).T, bounds_error=False, fill_value=None)
        
        lgkk, lgmm = np.meshgrid(np.log10(k_vec), np.log10(mass), sparse=True)
        lg_wkm_interpolated = lg_w_interp2d((lgkk.T, lgmm.T)).T
        wkm[jz] = 10**(lg_wkm_interpolated)*k_vec**2

    return wkm

def load_hods(block, section_name, suffix, z_vec, mass):
    """
    Loads and interpolates the hod quantities to match the
    calculation of power spectra
    """
    
    m_hod    = block[section_name, f'mass{suffix}']
    z_hod    = block[section_name, f'z{suffix}']
    Ncen_hod = block[section_name, f'n_cen{suffix}']
    Nsat_hod = block[section_name, f'n_sat{suffix}']
    numdencen_hod = block[section_name, f'number_density_cen{suffix}']
    numdensat_hod = block[section_name, f'number_density_sat{suffix}']
    f_c_hod = block[section_name, f'central_fraction{suffix}']
    f_s_hod = block[section_name, f'satellite_fraction{suffix}']
    mass_avg_hod = block[section_name, f'average_halo_mass{suffix}']  
    
    #If we're using an unconditional HOD, we need to define the stellar fraction with zeros
    try:
        f_star = block[section_name, f'f_star{suffix}']
    except:
        f_star = np.zeros((len(z_hod), len(m_hod)))  
    
    
    interp_Ncen  = RegularGridInterpolator((m_hod.T, z_hod.T), Ncen_hod.T, bounds_error=False, fill_value=0.0)
    interp_Nsat  = RegularGridInterpolator((m_hod.T, z_hod.T), Nsat_hod.T, bounds_error=False, fill_value=0.0)
    interp_fstar = RegularGridInterpolator((m_hod.T, z_hod.T), f_star.T, bounds_error=False, fill_value=0.0)
    
    # AD: Is extrapolation warranted here? Maybe make whole calculation on same grid/spacing/thingy!?
    interp_numdencen = interp1d(z_hod, numdencen_hod, fill_value='extrapolate', bounds_error=False)
    interp_numdensat = interp1d(z_hod, numdensat_hod, fill_value='extrapolate', bounds_error=False)
    interp_f_c = interp1d(z_hod, f_c_hod, fill_value=0.0, bounds_error=False)
    interp_f_s = interp1d(z_hod, f_s_hod, fill_value=0.0, bounds_error=False)
    interp_mass_avg = interp1d(z_hod, mass_avg_hod, fill_value=0.0, bounds_error=False)
    
    mm, zz = np.meshgrid(mass, z_vec, sparse=True)
    Ncen  = interp_Ncen((mm.T, zz.T)).T
    Nsat  = interp_Nsat((mm.T, zz.T)).T
    fstar = interp_fstar((mm.T, zz.T)).T
    
    numdencen = interp_numdencen(z_vec)
    numdensat = interp_numdensat(z_vec)
    f_c = interp_f_c(z_vec)
    f_s = interp_f_s(z_vec)
    mass_avg = interp_mass_avg(z_vec)
    
    return Ncen, Nsat, numdencen, numdensat, f_c, f_s, mass_avg, fstar

# TODO: unused, remove
# def load_galaxy_fractions(filename, z_vec):
#     z_file, fraction_file = np.loadtxt(filename, unpack=True)
#     if np.allclose(z_file, z_vec, atol=1e-3):
#         return fraction_file
#     else:
#         print('The redshift of the input galaxy fractions do not match the ranges'
#             'set in the pipeline. Performing interpolation.')
#         gal_frac_interp = interp(z_vec, z_file, fraction_file)
#         print( gal_frac_interp)
#         return gal_frac_interp

# -------------------------------------------------------------------------------------------------------------------- #
# Profiles
# -------------------------------------------------------------------------------------------------------------------- #

"""
    Each profile is calculated using a compute_*_profile function. But these functions are called through another function
    with the name of the profile, e.g. matter_profile. These latter functions simply create extra axis for the numpy arrays to
    avoid loops and speed up the process
"""


# ----------------#
# Matter Profiles
# ----------------#

def compute_matter_profile(mass, mean_density0, u_dm, fnu):
    """
    This is the matter halo profile. Feedback can be included through u_dm
    We lower the amplitude of W(M, k,z) in the one-halo term by the factor 1− fν ,
    where fν = Ων /Ωm is the neutrino mass fraction, to account for the fact that
    we assume that hot neutrinos cannot cluster in haloes and therefore
    do not contribute power to the one-halo term. Therefore W(M, k → 0,z) = (1− fν )M/ρ¯ and has units of volume
    This is the same as Mead et al. 2021
    """

    Wm_0 = mass / mean_density0
    Wm   = Wm_0 * u_dm * (1.0 - fnu)
    
    return Wm


def matter_profile(mass, mean_density0, u_dm, fnu):
    """
    Compute the grid in z, k, and M of the quantities described above
    Simple matter profile
    """
    
    profile = compute_matter_profile(mass[np.newaxis, np.newaxis, :], mean_density0[:, np.newaxis, np.newaxis], u_dm, fnu[:,np.newaxis,np.newaxis])
    
    return profile


def compute_matter_profile_with_feedback(mass, mean_density0, u_dm, z, omega_c, omega_m, omega_b, log10T_AGN, fnu):
    """
    eq 25 of 2009.01858
    matter profile including feedback as modelled by hmcode2020
    W(M, k) = [Ω_c/Ω_m+ fg(M)]W(M, k) + f∗ M/ρ¯
    The parameter 0 < f∗ < Ω_b/Ω_m can be thought of as an effective halo stellar mass fraction.
    
    Total matter profile from Mead2020 for baryonic feedback model
    Table 4 and eq 26 of 2009.01858
    f*(z) = f*_0 10^(z f*_z)
    
    This profile does not have 1-fnu correction as that is already accounted for in  dm_to_matter_frac
    """
    fstar = fs(log10T_AGN, z)

    dm_to_matter_frac = omega_c/omega_m
    f_gas = fg(mass, fstar, log10T_AGN, z, omega_b, omega_m)
    Wm_0 = mass / mean_density0
    Wm = (dm_to_matter_frac + f_gas) * Wm_0 * u_dm * (1.0 - fnu) + fstar * Wm_0
    #Wm = (dm_to_matter_frac + f_gas) * Wm_0 * u_dm + fstar * Wm_0
    return Wm

def matter_profile_with_feedback(mass, mean_density0, u_dm, z, omega_c, omega_m, omega_b, log10T_AGN, fnu):
    profile = compute_matter_profile_with_feedback(mass[np.newaxis, np.newaxis, :], mean_density0[:, np.newaxis, np.newaxis],
                                                    u_dm, z[:, np.newaxis, np.newaxis], omega_c, omega_m, omega_b,
                                                    log10T_AGN, fnu[:,np.newaxis,np.newaxis])
    return profile


def compute_matter_profile_with_feedback_stellar_fraction_from_obs(mass, mean_density0, u_dm, z, fstar, omega_c, omega_m, omega_b, fnu):
    """
    Total matter profile for a general baryonic feedback model
    using f* from HOD/CSMF/CLF that also provides for point mass estimate when used in the
    GGL power spectra
    
    This profile does not have 1-fnu correction as that is already accounted for in  dm_to_matter_frac
    """
    
    #fstar = block['pk_parameters', 'fstar'] # For now specified by the point mass!
    #ratio = 2.01*0.01 / np.median(fstar, axis=2)
    #fstar = fstar * ratio[:,:,np.newaxis] * 10.0**(z*(0.409))
    #print(ratio)
    #Tagn = 2.01/0.3 - fstar/(0.01*0.3)
    #print(np.max(np.abs(Tagn)))
    dm_to_matter_frac = omega_c/omega_m
    Wm_0 = mass / mean_density0
    f_gas_fit = fg_fit(mass, fstar, z, omega_b, omega_m)
    #Wm = (dm_to_matter_frac + f_gas_fit) * Wm_0 * u_dm + fstar * Wm_0
    Wm = (dm_to_matter_frac + f_gas_fit) * Wm_0 * u_dm * (1.0 - fnu) + fstar * Wm_0
    return Wm

def matter_profile_with_feedback_stellar_fraction_from_obs(mass, mean_density0, u_dm, z, fstar, omega_c, omega_m, omega_b, fnu):
    profile = compute_matter_profile_with_feedback_stellar_fraction_from_obs(mass[np.newaxis, np.newaxis, :],
                                                        mean_density0[:, np.newaxis, np.newaxis], u_dm,
                                                        z[:, np.newaxis, np.newaxis], fstar[:,np.newaxis,:],
                                                        omega_c, omega_m, omega_b, fnu[:,np.newaxis,np.newaxis])
    return profile


# ----------------#
# Galaxy Profiles
# ----------------#

def compute_galaxy_profile(Ngal, mean_number_density, fraction, u_gal=None):
    if u_gal is None:
        # this is relevant for centrals
        return fraction * Ngal / mean_number_density
    else:
        # This one for satellites
        return fraction * Ngal * u_gal / mean_number_density

# clustering - centrals
def central_profile(Ncen, numdencen, f_cen):
    profile = compute_galaxy_profile(Ncen, numdencen[:,np.newaxis], f_cen[:,np.newaxis])
    return profile

# clustering - satellites
def satellite_profile(Nsat, numdensat, f_sat, u_gal):
    profile = compute_galaxy_profile(Nsat[:,np.newaxis,:], numdensat[:,np.newaxis,np.newaxis], f_sat[:,np.newaxis,np.newaxis], u_gal)
    return profile


# ----------------#
# Aignment Profiles
# ----------------#

def compute_central_galaxy_alignment_profile(scale_factor, growth_factor, f_c, C1, mass):
    return f_c * (C1  / growth_factor) * mass# * scale_factor**2.0


def compute_satellite_galaxy_alignment_profile(Nsat, numdenssat, f_s, wkm_sat):
    return f_s * Nsat * wkm_sat / numdenssat


def compute_central_galaxy_alignment_profile_halo(growth_factor, f_c, C1, mass, beta, mpivot, mass_avg):
    return f_c * (C1  / growth_factor) * mass * (mass_avg/mpivot)**beta


def compute_satellite_galaxy_alignment_profile_halo(Nsat, numdenssat, f_s, wkm_sat, beta_sat, mpivot, mass_avg):
    return f_s * Nsat * wkm_sat / numdenssat * (mass_avg/mpivot)**beta_sat


# alignment - satellites
def satellite_alignment_profile(Nsat, numdensat, f_sat, wkm):
    """
    Prepare the grid in z, k and mass for the satellite alignment
    f_sat/n_sat N_sat gamma_hat(k,M)
    where gamma_hat(k,M) is the Fourier transform of the density weighted shear, i.e. the radial dependent power law
    times the NFW profile, here computed by the module wkm, while gamma_1h is only the luminosity dependence factor.
    """
    
    s_align_factor = compute_satellite_galaxy_alignment_profile(Nsat[:,np.newaxis,:], numdensat[:,np.newaxis,np.newaxis], f_sat[:,np.newaxis,np.newaxis], wkm.transpose(0,2,1))
    
    return s_align_factor
    
# alignment - centrals
def central_alignment_profile(mass, scale_factor, growth_factor, f_cen, C1):
    """
    Prepare the grid in z, k and mass for the central alignment
    f_cen/n_cen N_cen gamma_hat(k,M)
    where gamma_hat(k,M) is the Fourier transform of the density weighted shear, i.e. the radial dependent power law
    times the NFW profile, here computed by the module wkm, while gamma_1h is only the luminosity dependence factor.
    """
    
    c_align_factor = compute_central_galaxy_alignment_profile(scale_factor[:,:,np.newaxis], growth_factor[:,:,np.newaxis], f_cen[:,np.newaxis,np.newaxis], C1, mass[np.newaxis, np.newaxis, :])
    
    return c_align_factor
    
# alignment - centrals 2h: halo mass dependence
def central_alignment_profile_grid_halo(mass, scale_factor, growth_factor, f_cen, C1, beta, mpivot, mass_avg):
    """
    Prepare the grid in z, k and mass for the central alignment
    f_cen/n_cen N_cen gamma_hat(k,M)
    where gamma_hat(k,M) is the Fourier transform of the density weighted shear, i.e. the radial dependent power law
    times the NFW profile, here computed by the module wkm, while gamma_1h is only the luminosity dependence factor.
    """
    
    c_align_factor = compute_central_galaxy_alignment_profile_halo(growth_factor[:,:,np.newaxis], f_cen[:,np.newaxis,np.newaxis], C1, mass[np.newaxis, np.newaxis, :], beta, mpivot, mass_avg[:,np.newaxis,np.newaxis])
    
    return c_align_factor

# alignment - satellites 1h: halo mass dependence
def satellite_alignment_profile_grid_halo(Nsat, numdensat, f_sat, wkm, beta_sat, mpivot, mass_avg):
    """
    Prepare the grid in z, k and mass for the satellite alignment
    f_sat/n_sat N_sat gamma_hat(k,M)
    where gamma_hat(k,M) is the Fourier transform of the density weighted shear, i.e. the radial dependent power law
    times the NFW profile, here computed by the module wkm, while gamma_1h is only the luminosity dependence factor.
    """

    s_align_factor = compute_satellite_galaxy_alignment_profile_halo(Nsat[:,np.newaxis,:], numdensat[:,np.newaxis,np.newaxis], f_sat[:,np.newaxis,np.newaxis], wkm.transpose(0,2,1), beta_sat, mpivot, mass_avg[:,np.newaxis,np.newaxis])
    
    return s_align_factor


# -------------------------------------------------------------------------------------------------------------------- #
#  1 and 2 halo functions
# -------------------------------------------------------------------------------------------------------------------- #


def one_halo_truncation(k_vec, k_trunc=0.1):
    """
    1-halo term truncation at large scales (small k)
    """
    if k_trunc == None:
        return np.ones_like(k_vec)
    k_frac = k_vec/k_trunc
    return (k_frac**4.0)/(1.0 + k_frac**4.0)


def two_halo_truncation(k_vec, k_trunc=2.0):
    """
    2-halo term truncation at larger k-values (large k)
    """
    #TO-DO: figure out what ad-hoc values for f and nd are!
    #k_frac = k_vec/k_trunc
    #return 1.0 - f * (k_frac**nd)/(1.0 + k_frac**nd)
    if k_trunc == None:
        return np.ones_like(k_vec)
    return 0.5*(1.0+(erf(-(k_vec-k_trunc))))


def one_halo_truncation_old(k_vec, k_trunc=0.1):
    """
    1-halo term truncation at large scales (small k), this uses the error function
    """
    #return 1.-np.exp(-(k_vec/k_star)**2.)
    if k_trunc == None:
        return np.ones_like(k_vec)
    return erf(k_vec/k_trunc)


def two_halo_truncation_old(k_vec, k_trunc=2.0):
    """
    2-halo term truncation at larger k-values
    """
    if k_trunc == None:
        return np.ones_like(k_vec)
    return 0.5*(1.0+(erf(-(k_vec-k_trunc))))


def one_halo_truncation_ia(k_vec, k_trunc=4.0):
    if k_trunc == None:
        return np.ones_like(k_vec)
    return 1.-np.exp(-(k_vec/k_trunc)**2.)


def two_halo_truncation_ia(k_vec, k_trunc=6.0):
    if k_trunc == None:
        return np.ones_like(k_vec)
    return np.exp(-(k_vec/k_trunc)**2.)


def one_halo_truncation_mead(k_vec, sigma8_in):
    """
    1-halo term truncation in 2009.01858
    eq 17 and table 2
    """
    
    sigma8_z = sigma8_in[:,np.newaxis]
    # One-halo term damping wavenumber
    k_star = 0.05618 * sigma8_z**(-1.013) # h/Mpc
    k_frac = k_vec/k_star
    
    return (k_frac**4.0)/(1.0 + k_frac**4.0)


def two_halo_truncation_mead(k_vec, sigma8_in):
    """
    TODO: We don't do eq 15 which dewiggles the linear power
    eq 16 of 2009.01858
    As long as nd > 0, the multiplicative term in square brackets is
    unity for k << kd and (1 − f) for k >> kd.
    This damping is used instead of the regular 2-halo term integrals
    """
    
    sigma8_z = sigma8_in[:,np.newaxis]
    f = 0.2696 * sigma8_z**(0.9403)
    k_d = 0.05699 * sigma8_z**(-1.089)
    nd = 2.853
    k_frac = k_vec/k_d
    
    return 1.0 - f * (k_frac**nd)/(1.0 + k_frac**nd)


def transition_smoothing(neff, k_vec, p_1h, p_2h):
    """
    eq 23 and table 2 of 2009.01858
    This smooths the transition between 1 and 2 halo terms.
    α = 1 would correspond to a standard transition.
    α < 1 smooths the transition while α > 1 sharpens it.
    Delta^2(k) = k^3/(2 pi^2) P(k)
    ∆^2_hmcode(k,z) = {[∆^2_2h(k,z)]^α +[∆^2_1h(k,z)]^α}^1/α
    """
    
    delta_prefac = (k_vec**3.0)/(2.0*np.pi**2.0)
    # alpha = (1.875 * (1.603**block['hmf', 'neff'][:,np.newaxis]))
    alpha = (1.875 * (1.603**neff[:,np.newaxis]))
    Delta_1h = delta_prefac * p_1h
    Delta_2h = delta_prefac * p_2h
    Delta_hmcode = (Delta_1h**alpha + Delta_2h**alpha)**(1.0/alpha)
    
    return Delta_hmcode/delta_prefac


def compute_1h_term(profile_u, profile_v, mass, dn_dlnm_z):
    '''
    The 1h term of the power spectrum. All of the possible power spectra have the same structure at small scales:
    int W_u(k,z|M) W_v(k,z|M) n(M) dM
    where u,v are our fields, e.g. matter, galaxy, intrinsic alignment
    for example matter halo profile is: W_matter = (M/rho_m) U_matter(k,z|M)
    :param profile_u: array 1d (nmass), the given W_u for fixed value of k and z
    :param profile_v: array 1d (nmass), the given W_v for fixed value of k and z
    :param mass: array 1d (nmass)
    :param dn_dlnm_z: array 1d (nmass), the halo mass function at the given redshift z
    :return: scalar, the integral along the mass axis
    '''
    
    integrand = profile_u * profile_v * dn_dlnm_z / mass
    integral_1h = simps(integrand, mass)
    
    return integral_1h


# -------------------------------------------------------------------------------------------------------------------- #
# Gas fraction
# -------------------------------------------------------------------------------------------------------------------- #

def fg(mass, fstar, log10T_AGN, z, omega_b, omega_m, beta=2):
    """
    Gas fraction
    Eq 24 of 2009.01858
    fg(M) = [Ωb/Ωm− f∗] (M/Mb)^β/ (1 + (M/Mb)^β)
    where fg is the halo gas fraction, the pre-factor in parenthesis is the
    available gas reservoir, while Mb > 0 and β > 0 are fitted parameters.
    Haloes of M >> Mb are unaffected while those of M < Mb have
    lost more than half of their gas
    
    Gas fraction from Mead2020 for baryonic feedback model
        theta_agn = log10_TAGN - 7.8
    table 4 of 2009.01858, units of M_sun/h
    """
    theta_agn = log10T_AGN - 7.8
    mb = np.power(10.0, 13.87 + 1.81*theta_agn) * np.power(10.0, z*(0.195*theta_agn - 0.108))
    baryon_to_matter_fraction = omega_b/omega_m
    fg = (baryon_to_matter_fraction - fstar) * (mass/mb)**beta / (1.0+(mass/mb)**beta)
    
    return fg


def fg_fit(mass, fstar, z, omega_b, omega_m):
    """
    Gas fraction
    Eq 24 of 2009.01858
    fg(M) = [Ωb/Ωm− f∗] (M/Mb)^β/ (1 + (M/Mb)^β)
    where fg is the halo gas fraction, the pre-factor in parenthesis is the
    available gas reservoir, while Mb > 0 and β > 0 are fitted parameters.
    Haloes of M >> Mb are unaffected while those of M < Mb have
    lost more than half of their gas
    
    Gas fraction for a general baryonic feedback model
    """
    
    mb = 10**13.87#block['pk_parameters', 'm_b'] # free parameter.
    baryon_to_matter_fraction = omega_b/omega_m
    f = (baryon_to_matter_fraction - fstar) * (mass/mb)**2.0 / (1.0+(mass/mb)**2.0)
    
    return f

# -------------------------------------------------------------------------------------------------------------------- #
# Stellar fraction
# -------------------------------------------------------------------------------------------------------------------- #


def fs(log10T_AGN, z):
    """
    Stellar fraction from table 4 and eq 26 of 2009.01858 (Mead et al. 2021)
    f*(z) = f*_0 10^(z f*_z)
    """
    
    theta_agn = log10T_AGN - 7.8
    fstar_0 = (2.01 - 0.3*theta_agn)*0.01
    fstar_z = 0.409 + 0.0224*theta_agn
    fstar = fstar_0 * np.power(10.0, z*fstar_z)
    
    return fstar


def load_fstar_mm(block, section_name, z_vec, mass):
    """
    load stellar fraction that is calculated with the Cacciato HOD
    """
    
    if block.has_value(section_name,'f_star_extended'):
        f_star = block[section_name, 'f_star_extended']
        m_hod  = block[section_name, 'mass_extended']
        z_hod  = block[section_name, 'z_extended']
    else:
        raise ValueError(f'f_star_extended does NOT exist in {section_name}. \
            You have asked for stellar_fraction_from_observable_feedback which needs this quantity.\
            This is calculated as part of the hod_interface, if use_mead2020_corrections is set to stellar_fraction_from_observable_feedback. \
            Note that if hod_interface_unconditional does not predict this quantity.' )
    
    interp_fstar = RegularGridInterpolator((m_hod.T, z_hod.T), f_star.T, bounds_error=False, fill_value=None)
    mm, zz = np.meshgrid(mass, z_vec, sparse=True)
    fstar = interp_fstar((mm.T, zz.T)).T
    
    return fstar



# -------------------------------------------------------------------------------------------------------------------- #
# Two halo Integrals 
# -------------------------------------------------------------------------------------------------------------------- #

# 2-halo term integrals have this general form:
# I_x = int_0^infty dM b(M) W_x(M,k) n(M) 
# See the second term of eq 18 of Asgari, Mead, Heymans 2023: 2303.08752
# Note that dn_dlnm = n(M) M

def compute_A_term(mass, b_dm, dn_dlnm, mean_density0):
    """
    Integral over the missing haloes.
    This term is used to compensate for low mass haloes that are missing from the integral in the matter 2-halo term.
    Equation A.5 of Mead and Verde 2021, 2011.08858
    A(M_min) = 1−[1/ρ¯ int_M_min^infty dM M b(M) n(M)]
    Here all missing mass is assumed to be in halos of minimum mass M_min = min(mass)
    """
    
    integrand_m1 = b_dm * dn_dlnm * (1. / mean_density0)
    A = 1. - simps(integrand_m1, mass)
    if (A < 0.).any():  
        warnings.warn('Warning: Mass function/bias correction is negative!', RuntimeWarning)
        
    return A

def missing_mass_integral(mass, b_dm, dn_dlnm, mean_density0):
    A_term = compute_A_term(mass[np.newaxis,np.newaxis,:], b_dm[:,np.newaxis,:], dn_dlnm[:,np.newaxis,:], mean_density0[:,np.newaxis,np.newaxis])
    return A_term


def Im_term(mass, u_dm, b_dm, dn_dlnm, mean_density0, A_term):
    """
    # eq 35 of Asgari, Mead, Heymans 2023: 2303.08752
    # 2-halo term integral for matter, I_m = int_0^infty dM b(M) W_m(M,k) n(M) = int_0^infty dM b(M) M/rho_bar U_m(M,k) n(M)
    """
    
    I_m_term = compute_Im_term(mass[np.newaxis,np.newaxis,:], u_dm, b_dm[:,np.newaxis,:], dn_dlnm[:,np.newaxis,:], mean_density0[:,np.newaxis,np.newaxis])
    return I_m_term + A_term

def compute_Im_term(mass, u_dm, b_dm, dn_dlnm, mean_density0):
    integrand_m = b_dm * dn_dlnm * u_dm * (1. / mean_density0)
    I_m = simps(integrand_m, mass)
    return I_m

# TODO: write one that extrapolates intead of using A_term
# TODO: compare results 
def Im_term_v2(mass, u_dm, b_dm, dn_dlnm, mean_density0, A_term):
    I_m_term = compute_Im_term(mass[np.newaxis,np.newaxis,:], u_dm, b_dm[:,np.newaxis,:], dn_dlnm[:,np.newaxis,:], mean_density0[:,np.newaxis,np.newaxis])
    return I_m_term + A_term


def compute_Ig_term(profile, mass, dn_dlnm_z, b_m):
    integrand = profile * b_m * dn_dlnm_z / mass
    I_g = simps(integrand, mass)
    return I_g


def Is_term(mass, profile_satellites, b_m, dn_dlnm):
    I_s = compute_Ig_term(profile_satellites, mass[np.newaxis,np.newaxis,:], dn_dlnm[:,np.newaxis,:], b_m[:,np.newaxis,:])
    return I_s


def Ic_term(mass, profile_centrals, b_m, dn_dlnm, nk):
    I_c = np.tile(np.array([compute_Ig_term(profile_centrals, mass[np.newaxis,:], dn_dlnm, b_m)]).T, [1,nk])
    return I_c


def Ig_align_term(mass, profile_align, b_m, dn_dlnm, mean_density0, A_term):
    I_g_align = compute_Ig_term(profile_align, mass[np.newaxis,np.newaxis,:], dn_dlnm[:,np.newaxis,:], b_m[:,np.newaxis,:])
    return I_g_align + A_term * profile_align[:,:,0] * mean_density0[:,np.newaxis] / mass[0]


def compute_I_NL_term(k, z, W_1, W_2, b_1, b_2, mass_1, mass_2, dn_dlnm_z_1, dn_dlnm_z_2, A, rho_mean, B_NL_k_z):
    """
    uses eqs A.7 to A.10 fo Mead and Verde 2021, 2011.08858 to calculate the integral over beta_nl
    """

    if len(W_1.shape) < 3:
        W_1 = W_1[:,np.newaxis,:]
    if len(W_2.shape) < 3:
        W_2 = W_2[:,np.newaxis,:]
    
    W_1 = np.transpose(W_1, [0,2,1])
    W_2 = np.transpose(W_2, [0,2,1])
    

    # Takes the integral over mass_1
    # TODO: check that these integrals do the correct thing, keep this TODO
    integrand_22 = B_NL_k_z * W_1[:,:,np.newaxis,:] * W_2[:,np.newaxis,:,:] * b_1[:,:,np.newaxis,np.newaxis] * b_2[:,np.newaxis,:,np.newaxis] * dn_dlnm_z_1[:,:,np.newaxis,np.newaxis] * dn_dlnm_z_2[:,np.newaxis,:,np.newaxis] / (mass_1[np.newaxis,:,np.newaxis,np.newaxis] * mass_1[np.newaxis,np.newaxis,:,np.newaxis])
    integral_M1 = simps(integrand_22, mass_1, axis=1)
    integral_M2 = simps(integral_M1, mass_2, axis=1)
    I_22 = integral_M2
    
    # TODO: Compare this with pyhalomodel, keep this TODO

    I_11 = B_NL_k_z[:,0,0,:] * ((A**2.0) * W_1[:,0,:] * W_2[:,0,:] * (rho_mean[:,np.newaxis]**2.0)) / (mass_1[0] * mass_2[0])
    
    integrand_12 = B_NL_k_z[:,:,0,:] * W_2[:,:,:] * b_2[:,:,np.newaxis] * dn_dlnm_z_2[:,:,np.newaxis] / mass_2[np.newaxis,:,np.newaxis]
    integral_12 = simps(integrand_12, mass_2, axis=1)
    I_12 = A * W_1[:,0,:] * integral_12 * rho_mean[:,np.newaxis] / mass_1[0]
    
    integrand_21 = B_NL_k_z[:,0,:,:] * W_1[:,:,:] * b_1[:,:,np.newaxis] * dn_dlnm_z_1[:,:,np.newaxis] / mass_1[np.newaxis,:,np.newaxis]
    integral_21 = simps(integrand_21, mass_1, axis=1)
    I_21 = A * W_2[:,0,:] * integral_21 * rho_mean[:,np.newaxis] / mass_2[0]
    
    I_NL = I_11 + I_12 + I_21 + I_22

    return I_NL


def I_NL(mass_1, mass_2, factor_1, factor_2, bias_1, bias_2, dn_dlnm_1, dn_dlnm_2, k_vec, z_vec, A, rho_mean, beta_interp=None):
    I_NL = compute_I_NL_term(k_vec, z_vec, factor_1, factor_2, bias_1, bias_2, mass_1, mass_2, dn_dlnm_1, dn_dlnm_2, A, rho_mean, beta_interp)
    return I_NL

    
def low_k_truncation(k_vec, k_trunc):
    """
    Beta_nl low-k truncation
    """
    return 1.0/(1.0+np.exp(-(10.0*(np.log10(k_vec/k_trunc)))))


def high_k_truncation(k_vec, k_trunc):
    """
    Beta_nl high-k truncation
    """
    return 0.5*(1.0+(erf(-0.1*(k_vec-k_trunc))))#0.5*(1.0+(erf(-(k_vec-k_trunc))))


def minimum_halo_mass(emu):
    """
    Minimum halo mass for the set of cosmological parameters [Msun/h]
    """

    np_min = 200.    # Minimum number of halo particles
    npart = 2048.    # Cube root of number of simulation particles
    Lbox_HR = 1000. # Box size for high-resolution simulations [Mpc/h]
    Lbox_LR = 2000. # Box size for low-resolution simulations [Mpc/h]
    
    Om_m = emu.cosmo.get_Omega0()
    rhom = 2.77536627e11 * Om_m
    
    Mbox_HR = rhom*Lbox_HR**3
    mmin = Mbox_HR*np_min/npart**3
    
    vmin = Lbox_HR**3 * np_min/npart**3
    rmin = ((3.0*vmin) / (4.0*np.pi))**(1.0/3.0)
    
    return mmin, 2.0*np.pi/rmin
    
    
def rvir(emu, mass):
    Om_m = emu.cosmo.get_Omega0()
    rhom = 2.77536627e11 * Om_m
    return ((3.0 * mass) / (4.0 * np.pi * 200 * rhom)) ** (1.0 / 3.0)
    

def hl_envelopes_idx(data, dmin=1, dmax=1):
    """
    Input :
    data: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min
    lmin = (np.diff(np.sign(np.diff(data))) > 0).nonzero()[0] + 1
    # locals max
    lmax = (np.diff(np.sign(np.diff(data))) < 0).nonzero()[0] + 1
    
    # global min of dmin-chunks of locals min
    lmin = lmin[[i+np.argmin(data[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global max of dmax-chunks of locals max
    lmax = lmax[[i+np.argmax(data[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    return lmin,lmax


def compute_bnl_darkquest(z, log10M1, log10M2, k, emulator, block, kmax):
    M1 = 10.0**log10M1
    M2 = 10.0**log10M2
    # Parameters
    # Large 'linear' scale for linear halo bias [h/Mpc]
    klin = np.array([0.05])
    
    # Calculate beta_NL by looping over mass arrays
    beta_func = np.zeros((len(M1), len(M2), len(k)))
    b01 = np.zeros(len(M1))
    b02 = np.zeros(len(M2))
    # Linear power
    Pk_lin = emulator.get_pklin_from_z(k, z)
    #klin = np.array([k[np.argmax(Pk_lin)]])
    Pk_klin = emulator.get_pklin_from_z(klin, z)
    
    for iM, M0 in enumerate(M1):
        b01[iM] = np.sqrt(emulator.get_phh_mass(klin, M0, M0, z)/Pk_klin)
    #for iM, M0 in enumerate(M2):
    #    b02[iM] = np.sqrt(emulator.get_phh_mass(klin, M0, M0, z)/Pk_klin)
    
    for iM1, M01 in enumerate(M1):
        for iM2, M02 in enumerate(M2):
            if iM2 < iM1:
                # Use symmetry to not double calculate
                beta_func[iM1, iM2, :] = beta_func[iM2, iM1, :]
            else:
                # Linear halo bias
                b1 = b01[iM1]
                b2 = b01[iM2]
                    
                # Halo-halo power spectrum
                Pk_hh = emulator.get_phh_mass(k, M01, M02, z)
                
                #rmax = max(rvir(emulator, M01), rvir(emulator, M02))
                #kmax = 2.0*np.pi/rmax
                    
                # Create beta_NL
                shot_noise = lambda x, a: a
                popt, popc = curve_fit(shot_noise, k[k>50], Pk_hh[k>50])
                Pk_hh = Pk_hh - np.ones_like(k)*shot_noise(k, *popt)
            
                beta_func[iM1, iM2, :] = (Pk_hh/(b1*b2*Pk_lin) - 1.0 ) * low_k_truncation(k, klin)
                
                lmin, lmax = hl_envelopes_idx(np.abs(beta_func[iM1, iM2, :]+1))
                beta_func_interp = interp1d(k[lmax], np.abs(beta_func[iM1, iM2, lmax]+1), kind='quadratic', bounds_error=False, fill_value=(1.0, 0.0))
                beta_func[iM1, iM2, :] = beta_func_interp(k)-1.0
                
                
                Pk_hh0 = emulator.get_phh_mass(klin, M01, M02, z)
                Pk_hh0 = Pk_hh0 - np.ones_like(klin)*shot_noise(klin, *popt)
                db = Pk_hh0/(b1*b2*Pk_klin) - 1.0
                
                beta_func[iM1, iM2, :] = (((beta_func[iM1, iM2, :] + 1.0) * high_k_truncation(k, 60.0))/(db + 1.0) - 1.0) * low_k_truncation(k, klin)# * high_k_truncation(k, 10.0)
                #beta_func[iM1, iM2, :] = (beta_func[iM1, iM2, :] - db) * low_k_truncation(k, klin) * high_k_truncation(k, kmax)
                
    return beta_func
    
    
def create_bnl_interpolation_function(emulator, interpolation, z, block):
    
    lenM = 5
    lenk = 1000
    zc = z.copy()
    
    Mmin, kmax = minimum_halo_mass(emulator)
    M_up = np.log10((10.0**15.1))
    M_lo = np.log10((10.0**12.2))
    #M_lo = np.log10(Mmin)
    M = np.logspace(M_lo, M_up, lenM)
    k = np.logspace(-3.0, 2.0, lenk)
        
    beta_nl_interp_i = np.empty(len(z), dtype=object)
    for i,zi in enumerate(zc):
        beta_func = compute_bnl_darkquest(zi+0.01, np.log10(M), np.log10(M), k, emulator, block, kmax)
        beta_nl_interp_i[i] = RegularGridInterpolator([np.log10(M), np.log10(M), np.log10(k)], beta_func, fill_value=None, bounds_error=False, method='nearest')
    
    return beta_nl_interp_i


# TODO: go through the alignment module before fixing this 
def compute_two_halo_alignment(block, suffix, growth_factor, mean_density0):
    """
    The IA amplitude at large scales, including the IA prefactors.

    :param block: the CosmoSIS datablock
    :param nz: int, number of redshift bins
    :param nk: int, number of wave vector bins
    :param growth_factor: double array 2d (nz, nk), growth factor normalised to be 1 at z=0
    :param mean_density0: double, mean matter density of the Universe at redshift z=0
    Set in the option section.
    :return: double array 2d (nz, nk), double array 2d (nz, nk) : the large scale alignment amplitudes (GI and II)
    """
    
    # linear alignment coefficients
    C1 = 5.e-14
    # load the 2h (effective) amplitude of the alignment signal from the data block. 
    # This already includes the luminosity dependence if set. Double array [nz].
    alignment_gi = block[f'ia_large_scale_alignment{suffix}', 'alignment_gi']
    #alignment_ii = block[f'ia_large_scale_alignment{suffix}', 'alignment_ii']
    
    # Removing the loops!
    alignment_amplitude_2h = -alignment_gi[:,np.newaxis] * (C1 * mean_density0[:,np.newaxis] / growth_factor)
    alignment_amplitude_2h_II = (alignment_gi[:,np.newaxis] * C1 * mean_density0[:,np.newaxis] / growth_factor) ** 2.
    
    return alignment_amplitude_2h, alignment_amplitude_2h_II, C1 * alignment_gi[:,np.newaxis,np.newaxis]


def poisson_func(block, type, mass_avg, k_vec, z_vec):
    
    if type == 'scalar':
        poisson_num = block['pk_parameters', 'poisson'] * np.ones_like(mass_avg)
    elif type == 'power_law':
        poisson_num = block['pk_parameters', 'poisson'] * (mass_avg/block['pk_parameters', 'M_0'])**block['pk_parameters', 'slope']
    else:
        poisson_num = np.ones_like(mass_avg)
        
    return poisson_num



# ---- POWER SPECTRA ----#


# matter-matter
def compute_p_mm(k_vec, plin, z_vec, mass, dn_dln_m, matter_profile, I_m_term, one_halo_ktrunc, two_halo_ktrunc):

    # 2-halo term:
    pk_mm_2h = plin * I_m_term * I_m_term * two_halo_truncation(k_vec, two_halo_ktrunc)[np.newaxis,:]
    # 1-halo term
    pk_mm_1h = compute_1h_term(matter_profile, matter_profile, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation(k_vec, one_halo_ktrunc)[np.newaxis,:]
    # Total
    pk_mm_tot = pk_mm_1h + pk_mm_2h
    
    return pk_mm_1h, pk_mm_2h, pk_mm_tot
    
    
def compute_p_mm_bnl(k_vec, plin, z_vec, mass, dn_dln_m, matter_profile, I_m_term, I_NL_mm, one_halo_ktrunc):

    # 2-halo term:
    pk_mm_2h = plin * I_m_term * I_m_term + plin*I_NL_mm
    # 1-halo term
    pk_mm_1h = compute_1h_term(matter_profile, matter_profile, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation(k_vec, one_halo_ktrunc)[np.newaxis,:]
    # Total
    pk_mm_tot = pk_mm_1h + pk_mm_2h
    
    return pk_mm_1h, pk_mm_2h, pk_mm_tot
    
    
def compute_p_mm_mead(k_vec, plin, z_vec, mass, dn_dln_m, matter_profile, I_m_term, sigma8_z, neff):

    # 2-halo term:
    pk_mm_2h = plin * two_halo_truncation_mead(k_vec, sigma8_z)
    # 1-halo term
    pk_mm_1h = compute_1h_term(matter_profile, matter_profile, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation_mead(k_vec, sigma8_z)
    # Total
    pk_mm_tot = transition_smoothing(neff, k_vec, pk_mm_1h, pk_mm_2h)
    
    return pk_mm_1h, pk_mm_2h, pk_mm_tot
   

# galaxy-galaxy power spectrum
# TODO: combine this with the next function, but add an option for beta_nl
def compute_p_gg(block, k_vec, pk_lin, z_vec, mass, dn_dln_m, central_profile, satellite_profile, I_c_term, I_s_term, mass_avg, poisson_type, one_halo_ktrunc, two_halo_ktrunc):
    """
    p_tot = p_cs_1h + p_ss_1h + 2p_cs_2h + p_cc_2h
    """
    
    # 2-halo term:
    pk_cc_2h = pk_lin * I_c_term * I_c_term * two_halo_truncation(k_vec, two_halo_ktrunc)[np.newaxis,:]
    pk_ss_2h = pk_lin * I_s_term * I_s_term * two_halo_truncation(k_vec, two_halo_ktrunc)[np.newaxis,:]
    pk_cs_2h = pk_lin * I_c_term * I_s_term * two_halo_truncation(k_vec, two_halo_ktrunc)[np.newaxis,:]

    # 1-halo term:
    # TODO: check that the satellite 1 halo term is correct (eq 22 of the review)
    pk_cs_1h = compute_1h_term(central_profile[:,np.newaxis,:], satellite_profile, mass[np.newaxis,np.newaxis,:], dn_dln_m[:,np.newaxis,:]) * one_halo_truncation(k_vec, one_halo_ktrunc)
    pk_ss_1h = compute_1h_term(satellite_profile, satellite_profile, mass[np.newaxis,np.newaxis,:], dn_dln_m[:,np.newaxis,:]) * one_halo_truncation(k_vec, one_halo_ktrunc)
    
    # Total
    # AD: adding Poisson parameter to ph_ss_1h!
    # poisson = block['pk_parameters', 'poisson']
    # TODO: check if we can remove block dependence here. Maybe read in the parameters separately.
    poisson = poisson_func(block, poisson_type, mass_avg, k_vec, z_vec)[:,np.newaxis]
    pk_1h = 2.0*pk_cs_1h + poisson*pk_ss_1h
    pk_2h = pk_cc_2h + pk_ss_2h + 2.0*pk_cs_2h
    pk_tot = pk_1h + pk_2h

    # galaxy linear bias
    galaxy_linear_bias = np.sqrt(I_c_term ** 2. + I_s_term ** 2. + 2. * I_s_term * I_c_term)
    # changed this to have the poisson parameter included
    # return 2. * pk_cs_1h + pk_ss_1h, pk_cc_2h + pk_ss_2h + 2. * pk_cs_2h, pk_tot, galaxy_linear_bias
    return pk_1h, pk_2h, pk_tot, galaxy_linear_bias


def compute_p_gg_bnl(block, k_vec, pk_lin, z_vec, mass, dn_dln_m, central_profile, satellite_profile, I_c_term, I_s_term, I_NL_cs, I_NL_cc, I_NL_ss, mass_avg, poisson_type, one_halo_ktrunc):
    """
    p_tot = p_cs_1h + p_ss_1h + p_cs_2h + p_cc_2h
    """
    
    # 2-halo term:
    pk_cs_2h = pk_lin * I_c_term * I_s_term + pk_lin*I_NL_cs
    pk_cc_2h = pk_lin * I_c_term * I_c_term + pk_lin*I_NL_cc
    pk_ss_2h = pk_lin * I_s_term * I_s_term + pk_lin*I_NL_ss

    # 1-halo term:
    pk_cs_1h = compute_1h_term(central_profile[:,np.newaxis,:], satellite_profile, mass[np.newaxis,np.newaxis,:], dn_dln_m[:,np.newaxis,:]) * one_halo_truncation(k_vec, one_halo_ktrunc)
    pk_ss_1h = compute_1h_term(satellite_profile, satellite_profile, mass[np.newaxis,np.newaxis,:], dn_dln_m[:,np.newaxis,:]) * one_halo_truncation(k_vec, one_halo_ktrunc)

    

    poisson = poisson_func(block, poisson_type, mass_avg, k_vec, z_vec)[:,np.newaxis]
    pk_1h = 2.0*pk_cs_1h + poisson*pk_ss_1h
    pk_2h = pk_cc_2h + pk_ss_2h + 2.0*pk_cs_2h
    pk_tot = pk_1h + pk_2h
    
    # pk_tot_nbnl = 2. * pk_cs_1h + poisson * pk_ss_1h + pk_cc_2h - (pk_lin*I_NL_cc) + pk_ss_2h - (pk_lin*I_NL_ss) + (2. * pk_cs_2h) - (2. * pk_lin*I_NL_cs)

    # galaxy linear bias
    galaxy_linear_bias = np.sqrt(I_c_term ** 2. + I_s_term ** 2. + 2. * I_s_term * I_c_term)
    #print('p_nn succesfully computed')
    # return 2. * pk_cs_1h + pk_ss_1h, pk_cc_2h + pk_ss_2h + 2. * pk_cs_2h, pk_tot, galaxy_linear_bias
    return pk_1h, pk_2h, pk_tot, galaxy_linear_bias


# galaxy-matter power spectrum
# TODO: combine these two together
# TODO: should'nt the poisson parameter come into here as well?
# AD: No, no poisson parameter here, it only enters the 1-halo s-s term.
def compute_p_gm(block, k_vec, pk_lin, z_vec, mass, dn_dln_m, central_profile, satellite_profile, matter_profile, I_c_term, I_s_term, I_m_term, one_halo_ktrunc, two_halo_ktrunc):
    """
    p_tot = p_cm_1h + p_sm_1h + p_cm_2h + p_cm_2h
    """
    
    # 2-halo term:
    pk_cm_2h = pk_lin * I_c_term * I_m_term * two_halo_truncation(k_vec, two_halo_ktrunc)[np.newaxis,:]
    pk_sm_2h = pk_lin * I_s_term * I_m_term * two_halo_truncation(k_vec, two_halo_ktrunc)[np.newaxis,:]
    # 1-halo term
    pk_cm_1h = compute_1h_term(central_profile[:,np.newaxis], matter_profile, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation(k_vec, one_halo_ktrunc)[np.newaxis,:]
    pk_sm_1h = compute_1h_term(satellite_profile, matter_profile, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation(k_vec, one_halo_ktrunc)[np.newaxis,:]
    pk_1h = pk_cm_1h + pk_sm_1h
    pk_2h = pk_cm_2h + pk_sm_2h
    pk_tot = pk_1h + pk_2h
    # galaxy-matter linear bias
    galaxy_matter_linear_bias = np.sqrt(I_c_term * I_m_term + I_s_term * I_m_term)
    
    return pk_1h, pk_2h, pk_tot, galaxy_matter_linear_bias


def compute_p_gm_bnl(block, k_vec, pk_lin, z_vec, mass, dn_dln_m, central_profile, satellite_profile, matter_profile, I_c_term, I_s_term, I_m_term, I_NL_cm, I_NL_sm, one_halo_ktrunc):
    """
    p_tot = p_cm_1h + p_sm_1h + p_cm_2h + p_cm_2h
    """
    
    # 2-halo term:
    pk_cm_2h = pk_lin * I_c_term * I_m_term + pk_lin*I_NL_cm
    pk_sm_2h = pk_lin * I_s_term * I_m_term  + pk_lin*I_NL_sm
        # 1-halo term
    pk_cm_1h = compute_1h_term(central_profile[:,np.newaxis], matter_profile, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation(k_vec, one_halo_ktrunc)[np.newaxis,:]
    pk_sm_1h = compute_1h_term(satellite_profile, matter_profile, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation(k_vec, one_halo_ktrunc)[np.newaxis,:]
    pk_1h = pk_cm_1h + pk_sm_1h
    pk_2h = pk_cm_2h + pk_sm_2h
    pk_tot = pk_1h + pk_2h
    # galaxy-matter linear bias
    galaxy_matter_linear_bias = np.sqrt(I_c_term * I_m_term + I_s_term * I_m_term)
    
    return pk_1h, pk_2h, pk_tot, galaxy_matter_linear_bias


#################################################
#                                               #
#      INTRINSIC ALIGNMENT POWER SPECTRA        #
#                                               #
#################################################

"""
TO-DO: For the IA powerspectra that are not the implementation of Fortuna et al., we need to reconsider the
1-halo and 2-halo truncations:
    - generally truncate the 1-halo terms at smaller k (can be done from the config)
    - do not truncate the 2-halo terms (can also be done from the config)
    - use a different functional form, for instance the Mead et al. 2020 as in other power spectra above!
"""

# galaxy-matter power spectrum
def compute_p_mI_fortuna(block, k_vec, p_eff, z_vec, mass, dn_dln_m, matter_profile, s_align_factor, alignment_amplitude_2h, f_gal, one_halo_ktrunc, two_halo_ktrunc):
    """
    p_tot = p_sm_mI_1h + f_cen*p_cm_mI_2h + O(any other combination)
    """
    
    # 2-halo term:
    # pk_eff = (1.-t_eff)*plin+t_eff*pnl
    pk_cm_2h = compute_p_mI_two_halo(k_vec, p_eff, z_vec, f_gal, alignment_amplitude_2h) * two_halo_truncation_ia(k_vec, two_halo_ktrunc)[np.newaxis,:]
    # 1-halo term
    pk_sm_1h = (-1.0) * compute_1h_term(matter_profile, s_align_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation_ia(k_vec, one_halo_ktrunc)[np.newaxis,:]
    # prepare the 1h term
    pk_tot = pk_sm_1h + pk_cm_2h
    
    return pk_sm_1h, pk_cm_2h, pk_tot


def compute_p_mI(block, k_vec, p_lin, z_vec, mass, dn_dln_m, matter_profile, c_align_factor, s_align_factor, I_m_term, I_c_align_term, I_s_align_term, one_halo_ktrunc, two_halo_ktrunc):
    """
    p_tot = p_sm_mI_1h + f_cen*p_cm_mI_2h + O(any other combination)
    """
    
    pk_sm_1h = (-1.0) * compute_1h_term(matter_profile, s_align_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation_ia(k_vec, one_halo_ktrunc)[np.newaxis,:]
    #pk_cm_1h = (-1.0) * compute_1h_term(matter_profile, c_align_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation_ia(k_vec, one_halo_ktrunc)[np.newaxis,:]
    pk_sm_2h = (-1.0) * p_lin * I_m_term * I_s_align_term * two_halo_truncation_ia(k_vec, two_halo_ktrunc)[np.newaxis,:]
    pk_cm_2h = (-1.0) * p_lin * I_m_term * I_c_align_term * two_halo_truncation_ia(k_vec, two_halo_ktrunc)[np.newaxis,:]
    pk_tot = pk_sm_1h + pk_cm_2h + pk_sm_2h
    
    return pk_sm_1h, pk_cm_2h+pk_sm_2h, pk_tot
    
    
def compute_p_mI_bnl(block, k_vec, p_lin, z_vec, mass, dn_dln_m, matter_profile, c_align_factor, s_align_factor, I_m_term, I_c_align_term, I_s_align_term, I_NL_ia_cm, I_NL_ia_sm, one_halo_ktrunc):
    """
    p_tot = p_sm_mI_1h + f_cen*p_cm_mI_2h + O(any other combination)
    """
    
    pk_sm_1h = (-1.0) * compute_1h_term(matter_profile, s_align_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation_ia(k_vec, one_halo_ktrunc)[np.newaxis,:]
    #pk_cm_1h = (-1.0) * compute_1h_term(matter_profile, c_align_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation_ia(k_vec, one_halo_ktrunc)[np.newaxis,:]
    pk_sm_2h = (-1.0) * p_lin * I_m_term * I_s_align_term - p_lin*(I_NL_ia_sm)
    pk_cm_2h = (-1.0) * p_lin * I_m_term * I_c_align_term - p_lin*(I_NL_ia_cm)
    pk_tot = pk_sm_1h + pk_cm_2h + pk_sm_2h
    
    return pk_sm_1h, pk_cm_2h+pk_sm_2h, pk_tot


# intrinsic-intrinsic power spectrum
def compute_p_II_fortuna(block, k_vec, p_eff, z_vec, mass, dn_dln_m, s_align_factor, alignment_amplitude_2h_II, f_gal, one_halo_ktrunc, two_halo_ktrunc):
    """
    p_tot = p_ss_II_1h + p_cc_II_2h + O(p_sc_II_1h) + O(p_cs_II_2h)
    """
    
    # 2-halo term: This is simply the Linear Alignment Model weighted by the central galaxy fraction
    pk_cc_2h = compute_p_II_two_halo(k_vec, p_eff, z_vec, f_gal, alignment_amplitude_2h_II) * two_halo_truncation_ia(k_vec, two_halo_ktrunc)[np.newaxis,:]
    # 1-halo term
    pk_ss_1h = compute_1h_term(s_align_factor, s_align_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation_ia(k_vec, one_halo_ktrunc)[np.newaxis,:]
    pk_tot = pk_ss_1h + pk_cc_2h
    
    return pk_ss_1h, pk_cc_2h, pk_tot


# Needs Poisson parameter as well!
def compute_p_II(block, k_vec, p_lin, z_vec, mass, dn_dln_m, c_align_factor, s_align_factor, I_c_align_term, I_s_align_term, one_halo_ktrunc, two_halo_ktrunc):
    """
    p_tot = p_ss_II_1h + p_cc_II_2h + O(p_sc_II_1h) + O(p_cs_II_2h)
    """
    
    pk_ss_1h = compute_1h_term(s_align_factor, s_align_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation_ia(k_vec, one_halo_ktrunc)[np.newaxis,:]
    #pk_cs_1h = compute_1h_term(c_align_factor, s_align_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation_ia(k_vec, one_halo_ktrunc)[np.newaxis,:]
    pk_ss_2h = p_lin * I_s_align_term * I_s_align_term * two_halo_truncation_ia(k_vec, two_halo_ktrunc)[np.newaxis,:]
    pk_cc_2h = p_lin * I_c_align_term * I_c_align_term * two_halo_truncation_ia(k_vec, two_halo_ktrunc)[np.newaxis,:]
    pk_cs_2h = p_lin * I_c_align_term * I_s_align_term * two_halo_truncation_ia(k_vec, two_halo_ktrunc)[np.newaxis,:]
    pk_tot = pk_ss_1h + pk_cc_2h + pk_ss_2h + pk_cs_2h
    
    return pk_ss_1h, pk_cc_2h+pk_cs_2h+pk_cs_2h, pk_tot
    
    
# Needs Poisson parameter as well!
def compute_p_II_bnl(block, k_vec, p_lin, z_vec, mass, dn_dln_m, c_align_factor, s_align_factor, I_c_align_term, I_s_align_term, I_NL_ia_cc, I_NL_ia_cs, I_NL_ia_ss, one_halo_ktrunc):
    """
    p_tot = p_ss_II_1h + p_cc_II_2h + O(p_sc_II_1h) + O(p_cs_II_2h)
    """
    
    pk_ss_1h = compute_1h_term(s_align_factor, s_align_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation_ia(k_vec, one_halo_ktrunc)[np.newaxis,:]
    #pk_cs_1h = compute_1h_term(c_align_factor, s_align_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation_ia(k_vec, one_halo_ktrunc)[np.newaxis,:]
    pk_ss_2h = p_lin * I_s_align_term * I_s_align_term + p_lin*I_NL_ia_ss
    pk_cc_2h = p_lin * I_c_align_term * I_c_align_term + p_lin*I_NL_ia_cc
    pk_cs_2h = p_lin * I_c_align_term * I_s_align_term + p_lin*I_NL_ia_cs
    pk_tot = pk_ss_1h + pk_ss_2h + pk_cs_2h + pk_cc_2h
    
    return pk_ss_1h, pk_cc_2h+pk_cs_2h+pk_cs_2h, pk_tot


# galaxy-intrinsic power spectrum
#IT redefinition as dn_dln_m
def compute_p_gI_fortuna(block, k_vec, p_eff, z_vec, mass, dn_dln_m, central_profile, s_align_factor, I_c_term, alignment_amplitude_2h, one_halo_ktrunc, two_halo_ktrunc):
    """
    p_tot = p_cs_gI_1h + (2?)*p_cc_gI_2h + O(p_ss_gI_1h) + O(p_cs_gI_2h)
    """
    
    # 2-halo term:
    #IT Removed new_axis from alignment_amplitude_2h[:,np.newaxis] in the following line
    pk_cc_2h = -1.0 * p_eff * I_c_term * alignment_amplitude_2h[:,] * two_halo_truncation_ia(k_vec, two_halo_ktrunc)[np.newaxis,:]
    # 1-halo term
    pk_cs_1h = compute_1h_term(central_profile[:,np.newaxis,:], s_align_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation_ia(k_vec, one_halo_ktrunc)[np.newaxis,:]
    
    pk_tot = pk_cs_1h + pk_cc_2h
    
    return pk_cs_1h, pk_cc_2h, pk_tot

def compute_p_gI(block, k_vec, p_lin, z_vec, mass, dn_dln_m, central_profile, c_align_factor, s_align_factor, I_c_term, I_c_align_term, I_s_align_term, one_halo_ktrunc, two_halo_ktrunc):
    """
    p_tot = p_cs_gI_1h + (2?)*p_cc_gI_2h + O(p_ss_gI_1h) + O(p_cs_gI_2h)
    """
    
    pk_cs_1h = compute_1h_term(central_profile[:,np.newaxis,:], s_align_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation_ia(k_vec, one_halo_ktrunc)[np.newaxis,:]
    pk_cc_2h = p_lin * I_c_term * I_c_align_term * two_halo_truncation_ia(k_vec, two_halo_ktrunc)[np.newaxis,:]
    pk_cs_2h = p_lin * I_c_term * I_s_align_term * two_halo_truncation_ia(k_vec, two_halo_ktrunc)[np.newaxis,:]

    pk_tot = pk_cs_1h + pk_cs_2h + pk_cc_2h
    
    return pk_cs_1h, pk_cc_2h+pk_cs_2h, pk_tot

def compute_p_gI_bnl(block, k_vec, p_lin, z_vec, mass, dn_dln_m, central_profile, c_align_factor, s_align_factor, I_c_term, I_c_align_term, I_s_align_term, I_NL_ia_gc, I_NL_ia_gs, one_halo_ktrunc):
    """
    p_tot = p_cs_gI_1h + (2?)*p_cc_gI_2h + O(p_ss_gI_1h) + O(p_cs_gI_2h)
    """
    
    pk_cs_1h = compute_1h_term(central_profile[:,np.newaxis,:], s_align_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation_ia(k_vec, one_halo_ktrunc)[np.newaxis,:]
    pk_cc_2h = p_lin * I_c_term * I_c_align_term + p_lin*(I_NL_ia_gc)
    pk_cs_2h = p_lin * I_c_term * I_s_align_term + p_lin*(I_NL_ia_gs)

    pk_tot = pk_cs_1h + pk_cs_2h + pk_cc_2h
    
    return pk_cs_1h, pk_cc_2h+pk_cs_2h, pk_tot




############### TWO HALO ONLY ###################

# AD: Leaving as it is!

# galaxy-galaxy power spectrum
def compute_p_gg_two_halo(k_vec, plin, z_vec, bg):
    """
    p_tot = b_g**2 * p_lin
    """
    
    pk_tot = bg[:,np.newaxis] ** 2. * plin
    return pk_tot


# galaxy-matter power spectrum
def compute_p_gm_two_halo(k_vec, plin, z_vec, bg):
    """
    p_tot = bg * plin
    """
    
    pk_tot = bg[:,np.newaxis] * plin
    return pk_tot


# galaxy-matter power spectrum
def compute_p_mI_two_halo(k_vec, p_eff, z_vec, f_gal, alignment_amplitude_2h):
    # this is simply the Linear (or Nonlinear) Alignment Model, weighted by the central galaxy fraction
    pk_tot = f_gal[:,np.newaxis] * p_eff * alignment_amplitude_2h
    return pk_tot


# galaxy-intrinsic power spectrum
def compute_p_gI_two_halo(k_vec, p_eff, z_vec, f_gal, alignment_amplitude_2h, bg):
    """
    p_tot = bg * p_NLA
    """
    
    pk_tot = f_gal[:,np.newaxis] * bg[:,np.newaxis] * alignment_amplitude_2h * p_eff
    return pk_tot


# intrinsic-intrinsic power spectrum
def compute_p_II_two_halo(k_vec, p_eff, z_vec, f_gal, alignment_amplitude_2h_II):
    pk_tot = (f_gal[:,np.newaxis] ** 2.) * p_eff * alignment_amplitude_2h_II
    return pk_tot

#################################################

def get_normalised_profile(block, k_vec, mass, z_vec):
    """
    Reads the Fourier transform of the normalised Dark matter halo profile U.
    Checks that mass, redshift and k match the input.
    """
    z_udm    = block['fourier_nfw_profile', 'z']
    mass_udm = block['fourier_nfw_profile', 'm_h']
    k_udm    = block['fourier_nfw_profile', 'k_h']
    u_dm     = block['fourier_nfw_profile', 'ukm']
    u_sat    = block['fourier_nfw_profile', 'uksat']

    if((k_vec!=k_udm).any()):
        raise Exception('The profile k values are different to the input k values.')
    if((mass_udm!=mass).any()):
        raise Exception('The profile mass values are different to the input mass values.')
    if((z_udm!=z_vec).any()):
        raise Exception('The profile z values are different to the input z values.')
    return u_dm,u_sat


    # print(u_udm.shape,u_usat.shape)
    # u_udm    = np.reshape(u_udm, (np.size(z_udm),np.size(k_udm),np.size(mass_udm)))
    # u_usat   = np.reshape(u_usat, (np.size(z_udm),np.size(k_udm),np.size(mass_udm)))
    # print(u_udm.shape,u_usat.shape)
    # exit(1)
    # interpolate
    """
    # AD: Leaving this in for now if we need to revert back!
    nz = np.size(z_vec)
    nk = np.size(k_vec)
    nmass = np.size(mass)
    u_dm = np.array([interp_udm(mass_udm, k_udm, udm_z, mass, k_vec) for udm_z in u_udm])
    u_sat = np.array([interp_udm(mass_udm, k_udm, usat_z, mass, k_vec) for usat_z in u_usat])
    #"""
    # u_dm  = u_udm
    # u_sat = u_usat
    # print((u_udm<0).any(),(u_usat<0).any())
    # exit(1)
    # return np.abs(u_dm), np.abs(u_sat)
