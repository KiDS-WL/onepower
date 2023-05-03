# Library of the power spectrum module

import sys
import numpy as np
from scipy.interpolate import interp1d, interp2d, interpn
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from scipy.integrate import quad, simps, trapz
from scipy.interpolate import RegularGridInterpolator
import math
from scipy.special import erf
from itertools import count
import time
#import timing
from darkmatter_lib import compute_u_dm, radvir_from_mass
# import dill as pickle
from dark_emulator import darkemu


def one_halo_truncation(k_vec):
    k_star = 0.01
    #return 1.-np.exp(-(k_vec/k_star)**2.)
    return erf(k_vec/k_star)
    
def one_halo_truncation_mead(k_vec, block):
    sigma_var = (block['hmf', 'sigma_var'][:,np.newaxis])
    k_star = 0.05618 * sigma_var**(-1.013)
    return ((k_vec/k_star)**4.0)/(1.0 + (k_vec/k_star)**4.0)

def two_halo_truncation(k_vec):
    k_trunc = 2.0
    return 0.5*(1.0+(erf(-(k_vec-k_trunc))))
    
def two_halo_truncation_mead(k_vec, block):
    sigma_var = (block['hmf', 'sigma_var'][:,np.newaxis])
    f = 0.2696 * sigma_var**(0.9403)
    k_d = 0.05699 * sigma_var**(-1.089)
    nd = 2.853
    return 1.0 - (f*((k_vec/k_d)**nd)/(1.0 + (k_vec/k_d)**nd))

def two_halo_truncation_ia(k_vec):
    k_trunc = 6.0
    return np.exp(-(k_vec/k_trunc)**2.)
 
def one_halo_truncation_ia(k_vec):
    k_star = 4.0
    return 1.-np.exp(-(k_vec/k_star)**2.) 

def compute_2h_term(plin, I1, I2):
    '''
    The 2h term of the power spectrum. All of the possible power spectra have the same structure at large scales:
    P_2h,XY (k,z) = P(k,z)_lin I_X(k,z) I_Y(k,z)
    where I_X and I_Y are integrals, specific for each quantity (with X, Y = {matter, galaxy} )
    :param plin: array 2d, linear power spectrum in input
    :param I1: array 2d
    :param I2: array 2d
    :return: array 2d
    '''
    return plin * I1 * I2

def compute_1h_term(factor_1, factor_2, mass, dn_dlnm_z):
    '''
    The 1h term of the power spectrum. All of the possible power spectra have the same structure at small scales:
    \int f_X(k,z|M) f_Y(k,z|M) n(M) dM
    where X,Y = {matter, galaxy, intrinsic alignment}
    for example: f_matter = (M/rho_m) u(k,z|M)
    :param factor_1: array 1d (nmass), the given f_X for fixed value of k and z
    :param factor_2: array 1d (nmass), the given f_Y for fixed value of k and z
    :param mass: array 1d (nmass)
    :param dn_dlnm_z: array 1d (nmass), the halo mass function at the given redshift z
    :return: scalar, the integral along the mass axis
    '''
    integrand = factor_1 * factor_2 * dn_dlnm_z / mass
    sum_1h = simps(integrand, mass)
    return sum_1h


# -------------------------------------------------------------------------------------------------------------------- #
# One halo functions
# -------------------------------------------------------------------------------------------------------------------- #

# The f_X,Y factors that enters into the power spectra (analytical expression)
# Args : scalars
# Return: scalar

def fg(mass, fstar, theta_agn, z, block):
    
    mb = (10.0**(13.87 - 1.81*theta_agn) * 10.0**(z*(0.195*theta_agn - 0.108)))
    
    f = ((block['cosmological_parameters', 'omega_b']/block['cosmological_parameters', 'omega_m']) - fstar) * (mass/mb)**2.0 / (1.0+(mass/mb)**2.0)
    return f

def compute_matter_factor_baryon(mass, mean_density0, u_dm, z, block):

    theta_agn = block['halo_model_parameters', 'logT_AGN'] - 7.8
    
    fstar = ((2.01 - 0.30*theta_agn)*0.01 * 10.0**(z*(0.409 + 0.0224*theta_agn))) / (0.75 * (1.0+z)**(1.0/6.0))# / block['cosmological_parameters', 'h0'] / (1.0+z)**(1.0/3.0)# / block['cosmological_parameters', 'h_z'][:, np.newaxis, np.newaxis]**(1.0/3.0)
    
    return ((mass / mean_density0) * u_dm * ((block['cosmological_parameters', 'omega_c']/block['cosmological_parameters', 'omega_m']) + fg(mass, fstar, theta_agn, z, block))) + (fstar * (mass / mean_density0))
    
def fg_fit(mass, fstar, z, block):
    
    mb = block['pk_parameters', 'm_b'] # free parameter.
    
    f = ((block['cosmological_parameters', 'omega_b']/block['cosmological_parameters', 'omega_m']) - fstar) * (mass/mb)**2.0 / (1.0+(mass/mb)**2.0)
    return f

def compute_matter_factor_baryon_fit(mass, mean_density0, u_dm, z, block):

    fstar = block['pk_parameters', 'fstar'] # free parameter. Could it be specified by the point mass?
    
    return ((mass / mean_density0) * u_dm * ((block['cosmological_parameters', 'omega_c']/block['cosmological_parameters', 'omega_m']) + fg_fit(mass, fstar, z, block))) + (fstar * (mass / mean_density0))

# matter
def compute_matter_factor(mass, mean_density0, u_dm, block):
    return (mass / mean_density0) * u_dm * (1.0 - block['cosmological_parameters', 'fnu'][:,np.newaxis,np.newaxis])
# central galaxy position
def compute_central_galaxy_factor(Ncen, numdenscen, f_c):
    return f_c * Ncen / numdenscen
# satellite galaxy position
def compute_satellite_galaxy_factor(Nsat, numdenssat, f_s, u_gal):
    return f_s * Nsat * u_gal / numdenssat
# central galaxy alignment
def compute_central_galaxy_alignment_factor(scale_factor, growth_factor, f_c, C1, mass):
    return f_c * (C1  / growth_factor) * mass# * scale_factor**2.0
# satellite galaxy alignment
def compute_satellite_galaxy_alignment_factor(Nsat, numdenssat, f_s, wkm_sat):
    return f_s * Nsat * wkm_sat / numdenssat
# central galaxy alignment: halo mass dependence
# CHECK IF THIS MAKES SENSE IN THE TERM OF WINDOW FUNCTION COMPARED TO LA FORMALISM!
#def compute_central_galaxy_alignment_factor_halo(scale_factor, growth_factor, f_c, C1, mass, beta, m0):
#    return f_c * (C1  / growth_factor) * mass * (mass/m0)**beta # * scale_factor**2.0

# Compute the grid in z, k, and M of the quantities described above
# Args:
# Return:

# matter
def prepare_matter_factor_grid(mass, mean_density0, u_dm, block):
    m_factor = compute_matter_factor(mass[np.newaxis, np.newaxis, :], mean_density0[:, np.newaxis, np.newaxis], u_dm, block)
    return m_factor
    
def prepare_matter_factor_grid_baryon(mass, mean_density0, u_dm, z, block):
    m_factor = compute_matter_factor_baryon(mass[np.newaxis, np.newaxis, :], mean_density0[:, np.newaxis, np.newaxis], u_dm, z[:, np.newaxis, np.newaxis], block)
    return m_factor
    
def prepare_matter_factor_grid_baryon_fit(mass, mean_density0, u_dm, z, block):
    m_factor = compute_matter_factor_baryon_fit(mass[np.newaxis, np.newaxis, :], mean_density0[:, np.newaxis, np.newaxis], u_dm, z[:, np.newaxis, np.newaxis], block)
    return m_factor

# clustering - satellites
def prepare_satellite_factor_grid(Nsat, numdensat, f_sat, u_gal, nz, nk, nmass):
    s_factor = compute_satellite_galaxy_factor(Nsat[:,np.newaxis,:], numdensat[:,np.newaxis,np.newaxis], f_sat[:,np.newaxis,np.newaxis], u_gal)
    return s_factor

# clustering - centrals
def prepare_central_factor_grid(Ncen, numdencen, f_cen):
    c_factor = compute_central_galaxy_factor(Ncen, numdencen[:,np.newaxis], f_cen[:,np.newaxis])
    return c_factor

# alignment - satellites
def prepare_satellite_alignment_factor_grid(mass, Nsat, numdensat, f_sat, wkm, gamma_1h, nz, nk, nmass):
    """
    Prepare the grid in z, k and mass for the satellite alignment
    f_sat/n_sat N_sat gamma_hat(k,M)
    where gamma_hat(k,M) is the Fourier transform of the density weighted shear, i.e. the radial dependent power law
    times the NFW profile, here computed by the module wkm, while gamma_1h is only the luminosity dependence factor.
    :param mass:
    :param Nsat:
    :param numdensat:
    :param f_sat:
    :param wkm:
    :param gamma_1h:
    :param nz:
    :param nk:
    :param nmass:
    :return:
    """
    s_align_factor = compute_satellite_galaxy_alignment_factor(Nsat[:,np.newaxis,:], numdensat[:,np.newaxis,np.newaxis], f_sat[:,np.newaxis,np.newaxis], wkm.transpose(0,2,1))
    #s_align_factor *= gamma_1h[:, np.newaxis, np.newaxis]
    #print('s_align_factor successfully computed!')
    return s_align_factor
    
# alignment - centrals
def prepare_central_alignment_factor_grid(mass, scale_factor, growth_factor, f_cen, C1 , nz, nk, nmass):
    """
    Prepare the grid in z, k and mass for the central alignment
    f_cen/n_cen N_cen gamma_hat(k,M)
    where gamma_hat(k,M) is the Fourier transform of the density weighted shear, i.e. the radial dependent power law
    times the NFW profile, here computed by the module wkm, while gamma_1h is only the luminosity dependence factor.
    :param mass:
    :param Ncen:
    :param numdencen:
    :param f_cen:
    :param wkm:
    :param gamma_1h:
    :param nz:
    :param nk:
    :param nmass:
    :return:
    """
    c_align_factor = compute_central_galaxy_alignment_factor(scale_factor[:,:,np.newaxis], growth_factor[:,:,np.newaxis], f_cen[:,np.newaxis,np.newaxis], C1, mass[np.newaxis, np.newaxis, :])
    return c_align_factor
    
# alignment - centrals: halo mass dependence
#def prepare_central_alignment_factor_grid_halo(mass, scale_factor, growth_factor, f_cen, C1 , nz, nk, nmass, beta, m0):
#    """
#    Prepare the grid in z, k and mass for the central alignment
#    f_cen/n_cen N_cen gamma_hat(k,M)
#    where gamma_hat(k,M) is the Fourier transform of the density weighted shear, i.e. the radial dependent power law
#    times the NFW profile, here computed by the module wkm, while gamma_1h is only the luminosity dependence factor.
#    """
#    c_align_factor = compute_central_galaxy_alignment_factor_halo(scale_factor[:,:,np.newaxis], growth_factor[:,:,np.newaxis], f_cen[:,np.newaxis,np.newaxis], C1, mass[np.newaxis, np.newaxis, :], beta, m0) # need to check dimensionality of beta, m0!
#    return c_align_factor
    

# -------------------------------------------------------------------------------------------------------------------- #
# Two halo functions
# -------------------------------------------------------------------------------------------------------------------- #

# The I_X,Y integrals that enters into the 2h - power spectra (analytical expression)
# Args : scalars
# Return: scalar

def compute_A_term(mass, u_dm, b_dm, dn_dlnm, mean_density0):
    integrand_m1 = b_dm * dn_dlnm * (1. / mean_density0)
    A = 1. - simps(integrand_m1, mass)
    return A

def compute_Im_term(mass, u_dm, b_dm, dn_dlnm, mean_density0):
    # AD: check what happens if u_dm is removed. For considered k-ranges removing it should be fine.
    #integrand_m1 = b_dm * dn_dlnm * (1. / mean_density0)
    integrand_m2 = b_dm * dn_dlnm * u_dm * (1. / mean_density0)
    #I_m1 = 1. - simps(integrand_m1, mass)
    I_m2 = simps(integrand_m2, mass)
    #I_m = I_m1 + I_m2
    return I_m2

def compute_Ig_term(factor_1, mass, dn_dlnm_z, b_m):
    integrand = factor_1 * b_m * dn_dlnm_z / mass
    I_g = simps(integrand, mass)
    return I_g


def compute_I_NL_term(k, z, factor_1, factor_2, b_1, b_2, mass_1, mass_2, dn_dlnm_z_1, dn_dlnm_z_2, A, rho_mean, interpolation, B_NL_k_z, emulator):
    
    #B_NL_k_z = np.zeros((z.size, mass_1.size, mass_2.size, k.size))
    #indices = np.vstack(np.meshgrid(np.arange(z.size),np.arange(mass_1.size),np.arange(mass_2.size),np.arange(k.size))).reshape(4,-1).T
    #values = np.vstack(np.meshgrid(z, np.log10(mass_1), np.log10(mass_2), k)).reshape(4,-1).T
    
    #print(factor_1.shape, factor_2.shape)
    if len(factor_1.shape) < 3:
        factor_1 = factor_1[:,np.newaxis,:]
    if len(factor_2.shape) < 3:
        factor_2 = factor_2[:,np.newaxis,:]
    
    factor_1 = np.transpose(factor_1, [0,2,1])
    factor_2 = np.transpose(factor_2, [0,2,1])
    
    #to = time.time()
    # AD: This is slow because there is millions of values to evaluate B_NL_interp on. Could we reduce this to be less calls to the interpolating function?
    #B_NL_k_z[indices[:,0], indices[:,1], indices[:,2], indices[:,3]] = B_NL_interp(values)
    
    #print(time.time()-to)
    
    integrand = B_NL_k_z * factor_1[:,:,np.newaxis,:] * b_1[:,:,np.newaxis,np.newaxis] * dn_dlnm_z_1[:,:,np.newaxis,np.newaxis] / mass_1[np.newaxis,:,np.newaxis,np.newaxis]
    integral = simps(integrand, mass_1, axis=1)
    integrand_2 = integral * factor_2 * b_2[:,:,np.newaxis] * dn_dlnm_z_2[:,:,np.newaxis] / mass_2[np.newaxis,:,np.newaxis]
    beta_22 = simps(integrand_2, mass_2, axis=1)
     
    beta_11 = B_NL_k_z[:,0,0,:] * ((A**2.0) * factor_1[:,0,:] * factor_2[:,0,:] * (rho_mean[:,np.newaxis]**2.0)) / (mass_1[0] * mass_2[0])
    
    integrand_12 = B_NL_k_z[:,:,0,:] * factor_2[:,:,:] * b_2[:,:,np.newaxis] * dn_dlnm_z_2[:,:,np.newaxis] / mass_2[np.newaxis,:,np.newaxis]
    integral_12 = simps(integrand_12, mass_2, axis=1)
    beta_12 = A * factor_1[:,0,:] * integral_12 * rho_mean[:,np.newaxis] / mass_1[0]
    
    integrand_21 = B_NL_k_z[:,0,:,:] * factor_1[:,:,:] * b_1[:,:,np.newaxis] * dn_dlnm_z_1[:,:,np.newaxis] / mass_1[np.newaxis,:,np.newaxis]
    integral_21 = simps(integrand_21, mass_1, axis=1)
    beta_21 = A * factor_2[:,0,:] * integral_21 * rho_mean[:,np.newaxis] / mass_2[0]
    
    I_NL = beta_11 + beta_12 + beta_21 + beta_22
    #print(time.time()-to)
    return I_NL
   
def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]
   
def compute_bnl_darkquest(z, log10M1, log10M2, k, emulator, block):
    M1 = 10.0**log10M1
    M2 = 10.0**log10M2
    # Parameters
    # Large 'linear' scale for linear halo bias [h/Mpc]
    #klin = np.array([k[0]])
    #klin = np.array([0.02])
    
    # Calculate beta_NL by looping over mass arrays
    beta_func = np.zeros((len(M1), len(M2), len(k)))
    b01 = np.zeros(len(M1))
    b02 = np.zeros(len(M2))
    # Linear power
    Pk_lin = emulator.get_pklin_from_z(k, z)
    klin = np.array([k[np.argmax(Pk_lin)]])
    Pk_klin = emulator.get_pklin_from_z(klin, z)

    
    for iM, M0 in enumerate(M1):
        #b01[iM] = np.sqrt(emulator.get_phh_mass(klin, M0, M0, z)/Pk_klin)
        b01[iM] = np.nan_to_num(emulator.get_bias_mass(M0, z), nan=1.0, posinf=1.0, neginf=1.0)
    #for iM, M0 in enumerate(M2):
    #    b02[iM] = np.sqrt(emulator.get_phh_mass(klin, M0, M0, z1)/Pk_klin)
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
                    
                # Create beta_NL
                beta_func[iM1, iM2, :] = Pk_hh/(b1*b2*Pk_lin) - 1.0
                    
                Pk_hh0 = emulator.get_phh_mass(klin, M01, M02, z)
                db = Pk_hh0/(b1*b2*Pk_klin) - 1.0
        
                beta_func[iM1, iM2, :] = (beta_func[iM1, iM2, :] + 1.0)/(db + 1.0) - 1.0
    
    return beta_func
    
    
def create_bnl_interpolation_function(emulator, interpolation, z, block):
    # AD: The mass range in Bnl needs to be optimised. Preferrentially set to the maximum mass limits in DarkEmulator, with the largest number of bins possible.
    #M = np.logspace(12.0, 16.0, 5)
    #M = np.logspace(12.0, 14.0, 5)
    
    lenM = 5
    lenk = 50
    M = np.empty_like(z, dtype=np.object)
    k = np.empty_like(z, dtype=np.object)
    zc = z.copy()
    zc[zc>=0.5] = 0.5
    for i,zi in enumerate(zc):
        # Fitting the upper mass limit to the box size constraints as a function of redshift. Not stable.
        #M_up = 14.7788 - 0.624468*zi
        #M_up = 0.581217*zi**2 - 1.47736*zi + 16.0
        #M_up = 0.581217*zi**2 - 1.47736*zi + 14.9418
        #M_up = 0.581217*zi**2 - 1.47736*zi + 15.9418
        M_up = 14.0
        M_lo = 12.0
        M[i] = np.logspace(M_lo, M_up, lenM)# * 0.7 / block['cosmological_parameters', 'h0']
        k[i] = np.logspace(-2.0, np.log10(0.35 * (0.7 / block['cosmological_parameters', 'h0'])), lenk) # Need to correct k for h parameter here.
    #k = np.logspace(-2.0, 0.2, 50) #50)
    #beta_func = np.zeros((len(z), lenM, lenM, lenk))
    beta_nl_interp_i = np.empty(len(z), dtype=object)
    for i,zi in enumerate(zc):
        #zi += 1e-3
        #beta_func = np.nan_to_num(compute_bnl_darkquest(zi, np.log10(M[i]), np.log10(M[i]), k[i], emulator, block), nan=0.0, posinf=0.0, neginf=0.0)
        beta_func = compute_bnl_darkquest(zi, np.log10(M[i]), np.log10(M[i]), k[i], emulator, block)
        beta_nl_interp_i[i] = RegularGridInterpolator([np.log10(M[i]), np.log10(M[i]), k[i]], beta_func, fill_value=None, bounds_error=False, method='linear')
    
    return beta_nl_interp_i


def prepare_A_term(mass, u_dm, b_dm, dn_dlnm, mean_density0, nz, nk):
    A_term = compute_A_term(mass[np.newaxis,np.newaxis,:], u_dm, b_dm[:,np.newaxis,:], dn_dlnm[:,np.newaxis,:], mean_density0[:,np.newaxis,np.newaxis])
    return A_term

def prepare_Im_term(mass, u_dm, b_dm, dn_dlnm, mean_density0, nz, nk, A_term):
    I_m_term = compute_Im_term(mass[np.newaxis,np.newaxis,:], u_dm, b_dm[:,np.newaxis,:], dn_dlnm[:,np.newaxis,:], mean_density0[:,np.newaxis,np.newaxis])
    return I_m_term + A_term

def prepare_Is_term(mass, s_factor, b_m, dn_dlnm, nz, nk):
    I_s_term = compute_Ig_term(s_factor, mass[np.newaxis,np.newaxis,:], dn_dlnm[:,np.newaxis,:], b_m[:,np.newaxis,:])
    return I_s_term

def prepare_Ic_term(mass, c_factor, b_m, dn_dlnm, nz, nk):
    I_c_term = np.tile(np.array([compute_Ig_term(c_factor, mass[np.newaxis,:], dn_dlnm, b_m)]).T, [1,nk])
    return I_c_term
    
def prepare_Is_align_term(mass, s_align_factor, b_m, dn_dlnm, nz, nk, mean_density0, A_term):
    I_s_align_term = compute_Ig_term(s_align_factor, mass[np.newaxis,np.newaxis,:], dn_dlnm[:,np.newaxis,:], b_m[:,np.newaxis,:])
    return I_s_align_term + A_term * s_align_factor[:,:,0] * mean_density0[:,np.newaxis] / mass[0]
    
def prepare_Ic_align_term(mass, c_align_factor, b_m, dn_dlnm, nz, nk, mean_density0, A_term):
    I_c_align_term = compute_Ig_term(c_align_factor, mass[np.newaxis,np.newaxis,:], dn_dlnm[:,np.newaxis,:], b_m[:,np.newaxis,:])
    return I_c_align_term + A_term * c_align_factor[:,:,0] * mean_density0[:,np.newaxis] / mass[0]
    
def prepare_I_NL(mass_1, mass_2, factor_1, factor_2, bias_1, bias_2, dn_dlnm_1, dn_dlnm_2, nz, nk, k_vec, z_vec, A, rho_mean, emulator, interpolation, beta_interp=None):
    I_NL = compute_I_NL_term(k_vec, z_vec, factor_1, factor_2, bias_1, bias_2, mass_1, mass_2, dn_dlnm_1, dn_dlnm_2, A, rho_mean, interpolation, beta_interp, emulator)
    return I_NL
    
def compute_two_halo_alignment(block, suffix, nz, nk, growth_factor, mean_density0):
    """
    The IA amplitude at large scales, including the IA prefactors.

    :param block: the CosmoSIS datablock
    :param suffix: str, name of the sample as in the option section
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
    alignment_gi = block['ia_large_scale_alignment' + suffix, 'alignment_gi']
    #alignment_ii = block['ia_large_scale_alignment' + suffix, 'alignment_ii']
    
    # Removing the loops!
    alignment_amplitude_2h = -alignment_gi[:,np.newaxis] * (C1 * mean_density0[:,np.newaxis] / growth_factor)
    alignment_amplitude_2h_II = (alignment_gi[:,np.newaxis] * C1 * mean_density0[:,np.newaxis] / growth_factor) ** 2.
    
    return alignment_amplitude_2h, alignment_amplitude_2h_II, C1 * alignment_gi[:,np.newaxis,np.newaxis]



# ---- POWER SPECTRA ----#

def transition_smoothing(block, k_vec, one_halo, two_halo):
    delta_prefac = (k_vec**3.0)/(2.0*np.pi**2.0)
    alpha = (1.875 * (1.603**block['hmf', 'neff'][:,np.newaxis]))
    p_tot = (((delta_prefac * one_halo)**alpha + (delta_prefac * two_halo)**alpha)**(1.0/alpha))/delta_prefac
    return p_tot

# matter-matter
def compute_p_mm(block, k_vec, plin, z_vec, mass, dn_dln_m, m_factor, I_m_term, nz, nk):
    # 2-halo term:
    pk_mm_2h = compute_2h_term(plin, I_m_term, I_m_term) * two_halo_truncation(k_vec)[np.newaxis,:]
    # 1-halo term
    pk_mm_1h = compute_1h_term(m_factor, m_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation(k_vec)[np.newaxis,:]
    # Total
    pk_mm_tot = pk_mm_1h + pk_mm_2h
    #print('p_mm succesfully computed')
    return pk_mm_1h, pk_mm_2h, pk_mm_tot
    
def compute_p_mm_bnl(block, k_vec, plin, z_vec, mass, dn_dln_m, m_factor, I_m_term, nz, nk, I_NL_mm):
    # 2-halo term:
    pk_mm_2h = compute_2h_term(plin, I_m_term, I_m_term) + plin*I_NL_mm
    # 1-halo term
    pk_mm_1h = compute_1h_term(m_factor, m_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation(k_vec)[np.newaxis,:]
    # Total
    pk_mm_tot = pk_mm_1h + pk_mm_2h
    #print('p_mm succesfully computed')
    return pk_mm_1h, pk_mm_2h, pk_mm_tot
    
def compute_p_mm_mead(block, k_vec, plin, z_vec, mass, dn_dln_m, m_factor, I_m_term, nz, nk):
    # 2-halo term:
    pk_mm_2h = plin * two_halo_truncation_mead(k_vec, block)
    # 1-halo term
    pk_mm_1h = compute_1h_term(m_factor, m_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation_mead(k_vec, block)
    # Total
    pk_mm_tot = transition_smoothing(block, k_vec, pk_mm_1h, pk_mm_2h)
    return pk_mm_1h, pk_mm_2h, pk_mm_tot

# galaxy-galaxy power spectrum
def compute_p_gg(block, k_vec, pk_lin, z_vec, mass, dn_dln_m, c_factor, s_factor, I_c_term, I_s_term, nz, nk):
    #
    # p_tot = p_cs_1h + p_ss_1h + p_cs_2h + p_cc_2h
    #
    # 2-halo term:
    pk_cs_2h = compute_2h_term(pk_lin, I_c_term, I_s_term) * two_halo_truncation(k_vec)[np.newaxis,:]
    pk_cc_2h = compute_2h_term(pk_lin, I_c_term, I_c_term) * two_halo_truncation(k_vec)[np.newaxis,:]
    pk_ss_2h = compute_2h_term(pk_lin, I_s_term, I_s_term) * two_halo_truncation(k_vec)[np.newaxis,:]
    # 1-halo term:
    pk_cs_1h = compute_1h_term(c_factor[:,np.newaxis,:], s_factor, mass[np.newaxis,np.newaxis,:], dn_dln_m[:,np.newaxis,:]) * one_halo_truncation(k_vec)
    pk_ss_1h = compute_1h_term(s_factor, s_factor, mass[np.newaxis,np.newaxis,:], dn_dln_m[:,np.newaxis,:]) * one_halo_truncation(k_vec)

    
    # Total
    # AD: adding Poisson parameter to ph_ss_1h!
    poisson = block['pk_parameters', 'poisson']
    pk_tot = 2. * pk_cs_1h + poisson * pk_ss_1h + pk_cc_2h + pk_ss_2h + 2. * pk_cs_2h

    # in case, save in the datablock
    #block.put_grid('galaxy_cs_power_1h', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_cs_1h)
    #block.put_grid('galaxy_ss_power_1h', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_ss_1h)

    # galaxy linear bias
    galaxy_linear_bias = np.sqrt(I_c_term ** 2. + I_s_term ** 2. + 2. * I_s_term * I_c_term)
    #print('p_nn succesfully computed')
    return 2. * pk_cs_1h + pk_ss_1h, pk_cc_2h + pk_ss_2h + 2. * pk_cs_2h, pk_tot, galaxy_linear_bias

def compute_p_gg_bnl(block, k_vec, pk_lin, z_vec, mass, dn_dln_m, c_factor, s_factor, I_c_term, I_s_term, nz, nk, I_NL_cs, I_NL_cc, I_NL_ss):
    #
    # p_tot = p_cs_1h + p_ss_1h + p_cs_2h + p_cc_2h
    #
    # 2-halo term:
    pk_cs_2h = compute_2h_term(pk_lin, I_c_term, I_s_term) + pk_lin*I_NL_cs
    print('pk_cs_2h 1st term: ', compute_2h_term(pk_lin, I_c_term, I_s_term))
    print('pk_cs_2h 2nd term: ', pk_lin*I_NL_cs)
    pk_cc_2h = compute_2h_term(pk_lin, I_c_term, I_c_term) + pk_lin*I_NL_cc
    pk_ss_2h = compute_2h_term(pk_lin, I_s_term, I_s_term) + pk_lin*I_NL_ss

    # 1-halo term:
    pk_cs_1h = compute_1h_term(c_factor[:,np.newaxis,:], s_factor, mass[np.newaxis,np.newaxis,:], dn_dln_m[:,np.newaxis,:]) * one_halo_truncation(k_vec)
    pk_ss_1h = compute_1h_term(s_factor, s_factor, mass[np.newaxis,np.newaxis,:], dn_dln_m[:,np.newaxis,:]) * one_halo_truncation(k_vec)

    
    # Total
    # AD: adding Poisson parameter to ph_ss_1h!
    poisson = block['pk_parameters', 'poisson']
    pk_tot = 2. * pk_cs_1h + poisson * pk_ss_1h + pk_cc_2h + pk_ss_2h + 2. * pk_cs_2h
    
    pk_tot_nbnl = 2. * pk_cs_1h + poisson * pk_ss_1h + pk_cc_2h - (pk_lin*I_NL_cc) + pk_ss_2h - (pk_lin*I_NL_ss) + (2. * pk_cs_2h) - (2. * pk_lin*I_NL_cs)

    # in case, save in the datablock
    #block.put_grid('galaxy_cs_power_1h', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_cs_1h)
    #block.put_grid('galaxy_ss_power_1h', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_ss_1h)

    # galaxy linear bias
    galaxy_linear_bias = np.sqrt(I_c_term ** 2. + I_s_term ** 2. + 2. * I_s_term * I_c_term)
    #print('p_nn succesfully computed')
    return 2. * pk_cs_1h + pk_ss_1h, pk_cc_2h + pk_ss_2h + 2. * pk_cs_2h, pk_tot, galaxy_linear_bias


# galaxy-matter power spectrum
def compute_p_gm(block, k_vec, pk_lin, z_vec, mass, dn_dln_m, c_factor, s_factor, m_factor, I_c_term, I_s_term, I_m_term):
    #
    # p_tot = p_cm_1h + p_sm_1h + p_cm_2h + p_cm_2h
    #
    # 2-halo term:
    pk_cm_2h = compute_2h_term(pk_lin, I_c_term, I_m_term) * two_halo_truncation(k_vec)[np.newaxis,:]
    pk_sm_2h = compute_2h_term(pk_lin, I_s_term, I_m_term) * two_halo_truncation(k_vec)[np.newaxis,:]
    # 1-halo term
    pk_cm_1h = compute_1h_term(c_factor[:,np.newaxis], m_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation(k_vec)[np.newaxis,:]
    pk_sm_1h = compute_1h_term(s_factor, m_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation(k_vec)[np.newaxis,:]
    pk_tot = pk_cm_1h + pk_sm_1h + pk_cm_2h + pk_sm_2h
    #print('p_xgG succesfully computed')
    return pk_cm_1h+pk_sm_1h, pk_cm_2h+pk_cm_2h, pk_tot

def compute_p_gm_bnl(block, k_vec, pk_lin, z_vec, mass, dn_dln_m, c_factor, s_factor, m_factor, I_c_term, I_s_term, I_m_term, I_NL_cm, I_NL_sm):
    #
    # p_tot = p_cm_1h + p_sm_1h + p_cm_2h + p_cm_2h
    #
    # 2-halo term:
    pk_cm_2h = compute_2h_term(pk_lin, I_c_term, I_m_term) + pk_lin*I_NL_cm
    pk_sm_2h = compute_2h_term(pk_lin, I_s_term, I_m_term) + pk_lin*I_NL_sm
        # 1-halo term
    pk_cm_1h = compute_1h_term(c_factor[:,np.newaxis], m_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation(k_vec)[np.newaxis,:]
    pk_sm_1h = compute_1h_term(s_factor, m_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation(k_vec)[np.newaxis,:]
    pk_tot = pk_cm_1h + pk_sm_1h + pk_cm_2h + pk_sm_2h
    #print('p_xgG succesfully computed')
    return pk_cm_1h+pk_sm_1h, pk_cm_2h+pk_cm_2h, pk_tot

#################################################
#                                               #
#      INTRINSIC ALIGNMENT POWER SPECTRA        #
#                                               #
#################################################


# galaxy-matter power spectrum
def compute_p_mI_mc(block, k_vec, p_eff, z_vec, mass, dn_dln_m, m_factor, s_align_factor, alignment_amplitude_2h, nz, nk, f_gal):
    #
    # p_tot = p_sm_mI_1h + f_cen*p_cm_mI_2h + O(any other combination)
    #
    # 2-halo term:
    pk_cm_2h = compute_p_mI_two_halo(block, k_vec, p_eff, z_vec, nz, f_gal, alignment_amplitude_2h) * two_halo_truncation_ia(k_vec)[np.newaxis,:]
    # 1-halo term
    pk_sm_1h = (-1.0) * compute_1h_term(m_factor, s_align_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation_ia(k_vec)[np.newaxis,:]
    # prepare the 1h term
    pk_tot = pk_sm_1h + pk_cm_2h
    return pk_sm_1h, pk_cm_2h, pk_tot

def compute_p_mI(block, k_vec, p_lin, z_vec, mass, dn_dln_m, m_factor, c_align_factor, s_align_factor, I_m_term, I_c_align_term, I_s_align_term, nz, nk):
    
    pk_sm_1h = (-1.0) * compute_1h_term(m_factor, s_align_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation_ia(k_vec)[np.newaxis,:]
    #pk_cm_1h = (-1.0) * compute_1h_term(m_factor, c_align_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation_ia(k_vec)[np.newaxis,:]
    pk_sm_2h = (-1.0) * compute_2h_term(p_lin, I_m_term, I_s_align_term)
    pk_cm_2h = (-1.0) * compute_2h_term(p_lin, I_m_term, I_c_align_term)
    pk_tot = pk_sm_1h + pk_cm_2h + pk_sm_2h
    
    return pk_sm_1h, pk_cm_2h+pk_sm_2h, pk_tot
    
def compute_p_mI_bnl(block, k_vec, p_lin, z_vec, mass, dn_dln_m, m_factor, c_align_factor, s_align_factor, I_m_term, I_c_align_term, I_s_align_term, nz, nk, I_NL_ia_cm, I_NL_ia_sm):
    
    pk_sm_1h = (-1.0) * compute_1h_term(m_factor, s_align_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation_ia(k_vec)[np.newaxis,:]
    #pk_cm_1h = (-1.0) * compute_1h_term(m_factor, c_align_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation_ia(k_vec)[np.newaxis,:]
    pk_sm_2h = (-1.0) * compute_2h_term(p_lin, I_m_term, I_s_align_term) - p_lin*(I_NL_ia_sm)
    pk_cm_2h = (-1.0) * compute_2h_term(p_lin, I_m_term, I_c_align_term) - p_lin*(I_NL_ia_cm)
    pk_tot = pk_sm_1h + pk_cm_2h + pk_sm_2h
    
    
    return pk_sm_1h, pk_cm_2h+pk_sm_2h, pk_tot


# intrinsic-intrinsic power spectrum
def compute_p_II_mc(block, k_vec, p_eff, z_vec, mass, dn_dln_m, s_align_factor, alignment_amplitude_2h_II, nz, nk, f_gal):
    #
    # p_tot = p_ss_II_1h + p_cc_II_2h + O(p_sc_II_1h) + O(p_cs_II_2h)
    #
    # 2-halo term: This is simply the Linear Alignment Model weighted by the central galaxy fraction
    pk_cc_2h = compute_p_II_two_halo(block, k_vec, p_eff, z_vec, nz, f_gal, alignment_amplitude_2h_II) * two_halo_truncation_ia(k_vec)[np.newaxis,:]
    # 1-halo term
    pk_ss_1h = compute_1h_term(s_align_factor, s_align_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation_ia(k_vec)[np.newaxis,:]
    pk_tot = pk_ss_1h + pk_cc_2h
    return pk_ss_1h, pk_cc_2h, pk_tot

# Needs Poisson parameter as well!
def compute_p_II(block, k_vec, p_lin, z_vec, mass, dn_dln_m, c_align_factor, s_align_factor, I_c_align_term, I_s_align_term, nz, nk):
    
    pk_ss_1h = compute_1h_term(s_align_factor, s_align_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation_ia(k_vec)[np.newaxis,:]
    #pk_cs_1h = compute_1h_term(c_align_factor, s_align_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation_ia(k_vec)[np.newaxis,:]
    pk_ss_2h = compute_2h_term(p_lin, I_s_align_term, I_s_align_term)
    pk_cc_2h = compute_2h_term(p_lin, I_c_align_term, I_c_align_term)
    pk_cs_2h = compute_2h_term(p_lin, I_c_align_term, I_s_align_term)
    pk_tot = pk_ss_1h + pk_ss_2h + pk_cs_2h + pk_cc_2h
    
    return pk_ss_1h, pk_cc_2h+pk_cs_2h+pk_cs_2h, pk_tot
    
# Needs Poisson parameter as well!
def compute_p_II_bnl(block, k_vec, p_lin, z_vec, mass, dn_dln_m, c_align_factor, s_align_factor, I_c_align_term, I_s_align_term, nz, nk, I_NL_ia_cc, I_NL_ia_cs, I_NL_ia_ss):
    
    pk_ss_1h = compute_1h_term(s_align_factor, s_align_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation_ia(k_vec)[np.newaxis,:]
    #pk_cs_1h = compute_1h_term(c_align_factor, s_align_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation_ia(k_vec)[np.newaxis,:]
    pk_ss_2h = compute_2h_term(p_lin, I_s_align_term, I_s_align_term) + p_lin*I_NL_ia_ss
    pk_cc_2h = compute_2h_term(p_lin, I_c_align_term, I_c_align_term) + p_lin*I_NL_ia_cc
    pk_cs_2h = compute_2h_term(p_lin, I_c_align_term, I_s_align_term) + p_lin*I_NL_ia_cs
    pk_tot = pk_ss_1h + pk_ss_2h + pk_cs_2h + pk_cc_2h
    
    return pk_ss_1h, pk_cc_2h+pk_cs_2h+pk_cs_2h, pk_tot


# galaxy-intrinsic power spectrum
#IT redefinition as dn_dln_m
def compute_p_gI_mc(block, k_vec, p_eff, z_vec, mass, dn_dln_m, c_factor, s_align_factor, I_c_term, alignment_amplitude_2h, nz, nk):
    #
    # p_tot = p_cs_gI_1h + (2?)*p_cc_gI_2h + O(p_ss_gI_1h) + O(p_cs_gI_2h)
    #
    # 2-halo term:
    #IT Removed new_axis from alignment_amplitude_2h[:,np.newaxis] in the following line
    pk_cc_2h = -compute_2h_term(p_eff, I_c_term, alignment_amplitude_2h[:,]) * two_halo_truncation_ia(k_vec)[np.newaxis,:]
    # 1-halo term
    pk_cs_1h = compute_1h_term(c_factor[:,np.newaxis,:], s_align_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation_ia(k_vec)[np.newaxis,:]
    
    pk_tot = pk_cs_1h + pk_cc_2h
    return pk_cs_1h, pk_cc_2h, pk_tot

def compute_p_gI(block, k_vec, p_lin, z_vec, mass, dn_dln_m, c_factor, c_align_factor, s_align_factor, I_c_term, I_c_align_term, I_s_align_term, nz, nk):
    
    pk_cs_1h = compute_1h_term(c_factor[:,np.newaxis,:], s_align_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation_ia(k_vec)[np.newaxis,:]
    pk_cc_2h = compute_2h_term(p_lin, I_c_term, I_c_align_term)
    pk_cs_2h = compute_2h_term(p_lin, I_c_term, I_s_align_term)

    pk_tot = pk_cs_1h + pk_cs_2h + pk_cc_2h
    
    return pk_cs_1h, pk_cc_2h+pk_cs_2h, pk_tot

def compute_p_gI_bnl(block, k_vec, p_lin, z_vec, mass, dn_dln_m, c_factor, c_align_factor, s_align_factor, I_c_term, I_c_align_term, I_s_align_term, nz, nk, I_NL_ia_gc, I_NL_ia_gs):
    
    pk_cs_1h = compute_1h_term(c_factor[:,np.newaxis,:], s_align_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation_ia(k_vec)[np.newaxis,:]
    pk_cc_2h = compute_2h_term(p_lin, I_c_term, I_c_align_term) + p_lin*(I_NL_ia_gc)
    pk_cs_2h = compute_2h_term(p_lin, I_c_term, I_s_align_term) + p_lin*(I_NL_ia_gs)

    pk_tot = pk_cs_1h + pk_cs_2h + pk_cc_2h
    
    return pk_cs_1h, pk_cc_2h+pk_cs_2h, pk_tot




############### TWO HALO ONLY ###################

# AD: Not fixing this, as it seems to be not used at all.

# galaxy-galaxy power spectrum
def compute_p_gg_two_halo(block, k_vec, plin, z_vec, bg):
    #
    # p_tot = b_g**2 * p_lin
    #
    pk_tot = bg[:,np.newaxis] ** 2. * plin
    return pk_tot


# galaxy-matter power spectrum
def compute_p_gm_two_halo(block, k_vec, plin, z_vec, bg):
    #
    # p_tot = bg * plin
    #
    pk_tot = bg[:,np.newaxis] * plin
    return pk_tot


# galaxy-matter power spectrum
def compute_p_mI_two_halo(block, k_vec, p_eff, z_vec, nz, f_gal, alignment_amplitude_2h):
    #
    # p_tot = p_NLA
    #
    # this is simply the Linear (or Nonlinear) Alignment Model, weighted by the central galaxy fraction
    #pk_tot = np.zeros([len(z_vec), len(k_vec)])
    #print('Test')
    #print(f_gal.shape, p_eff.shape, alignment_amplitude_2h.shape)
    pk_tot = f_gal[:,np.newaxis] * p_eff * alignment_amplitude_2h
    return pk_tot


# galaxy-intrinsic power spectrum
def compute_p_gI_two_halo(block, k_vec, p_eff, z_vec, nz, f_gal, alignment_amplitude_2h, bg):
    #
    # p_tot = bg * p_NLA
    #
    #pk_tot = np.zeros([len(z_vec), len(k_vec)])
    #print('Test')
    #print(f_gal.shape, bg.shape, alignment_amplitude_2h.shape, p_eff.shape)
    pk_tot = f_gal[:,np.newaxis] * bg[:,np.newaxis] * alignment_amplitude_2h * p_eff
    return pk_tot


# intrinsic-intrinsic power spectrum
def compute_p_II_two_halo(block, k_vec, p_eff, z_vec, nz, f_gal, alignment_amplitude_2h_II):
    #pk_tot = np.zeros([len(z_vec), len(k_vec)])
    #print('Test')
    #print(f_gal.shape, p_eff.shape, alignment_amplitude_2h_II.shape)
    pk_tot = (f_gal[:,np.newaxis] ** 2.) * p_eff * alignment_amplitude_2h_II
    return pk_tot

#################################################


def interp_udm(mass_udm, k_udm, udm_z, mass, k_vec):
    #interp_udm = interp2d(mass_udm, k_udm, udm_z, kind='linear', bounds_error=False)
    #u_dm = interp_udm(mass, k_vec)
    interp_udm = RegularGridInterpolator((mass_udm.T, k_udm.T), udm_z.T, bounds_error=False, fill_value=None)
    mm, kk = np.meshgrid(mass, k_vec, sparse=True)
    u_dm = interp_udm((mm.T, kk.T)).T
    return u_dm

def compute_u_dm_grid(block, k_vec, mass, z_vec):
    start_time_udm = time.time()
    z_udm = block['fourier_nfw_profile', 'z']
    mass_udm = block['fourier_nfw_profile', 'm_h']
    k_udm = block['fourier_nfw_profile', 'k_h']
    u_udm = block['fourier_nfw_profile', 'ukm']
    u_usat = block['fourier_nfw_profile', 'uksat']
    u_udm = np.reshape(u_udm, (np.size(z_udm),np.size(k_udm),np.size(mass_udm)))
    u_usat = np.reshape(u_usat, (np.size(z_udm),np.size(k_udm),np.size(mass_udm)))
    # interpolate
    """
    nz = np.size(z_vec)
    nk = np.size(k_vec)
    nmass = np.size(mass)
    u_dm = np.array([interp_udm(mass_udm, k_udm, udm_z, mass, k_vec) for udm_z in u_udm])
    u_sat = np.array([interp_udm(mass_udm, k_udm, usat_z, mass, k_vec) for usat_z in u_usat])
    #"""
    u_dm = u_udm
    u_sat = u_usat
    #print('--- u_dm: %s seconds ---' % (time.time() - start_time_udm))
    return np.abs(u_dm), np.abs(u_sat)
