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
    k_star = 0.1
    return 1.-np.exp(-(k_vec/k_star)**2.)

def two_halo_truncation(k_vec):
    k_trunc = 2.0
    return 0.5*(1.0+(erf(-(k_vec-k_trunc))))

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

# matter
def compute_matter_factor(mass, mean_density0, u_dm):
    return (mass / mean_density0) * u_dm
# central galaxy position
def compute_central_galaxy_factor(Ncen, numdenscen, f_c):
    return f_c * Ncen / numdenscen
# satellite galaxy position
def compute_satellite_galaxy_factor(Nsat, numdenssat, f_s, u_gal):
    return f_s * Nsat * u_gal / numdenssat
# central galaxy alignment
def compute_central_galaxy_alignment_factor(scale_factor, growth_factor, f_c, C1):
    return f_c * C1 * scale_factor**2.0 / growth_factor
# satellite galaxy alignment
def compute_satellite_galaxy_alignment_factor(Nsat, numdenssat, f_s, wkm_sat):
    return f_s * Nsat * wkm_sat / numdenssat

# Compute the grid in z, k, and M of the quantities described above
# Args:
# Return:

# matter
def prepare_matter_factor_grid(mass, mean_density0, u_dm):
    m_factor = compute_matter_factor(mass[np.newaxis, np.newaxis, :], mean_density0[:, np.newaxis, np.newaxis], u_dm)
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
    c_align_factor = compute_central_galaxy_alignment_factor(scale_factor[:,:,np.newaxis], growth_factor[:,:,np.newaxis], f_cen[:,np.newaxis,np.newaxis], C1)
    return c_align_factor
    

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
    
    beta_11 = A**2.0 * factor_1[:,0,:] * factor_2[:,0,:] * rho_mean[:,np.newaxis]**2.0 / (mass_1[0] * mass_2[0])
    
    integrand_12 = B_NL_k_z[:,:,0,:] * factor_2[:,:,:] * b_2[:,:,np.newaxis] * dn_dlnm_z_2[:,:,np.newaxis] / mass_2[np.newaxis,:,np.newaxis]
    integral_12 = simps(integrand_12, mass_2, axis=1)
    beta_12 = A * factor_1[:,0,:] * integral_12 * rho_mean[:,np.newaxis] / mass_1[0]
    
    integrand_21 = B_NL_k_z[:,0,:,:] * factor_1[:,:,:] * b_1[:,:,np.newaxis] * dn_dlnm_z_1[:,:,np.newaxis] / mass_1[np.newaxis,:,np.newaxis]
    integral_21 = simps(integrand_21, mass_1, axis=1)
    beta_21 = A * factor_2[:,0,:] * integral_21 * rho_mean[:,np.newaxis] / mass_2[0]
    
    I_NL = beta_11 + beta_12 + beta_21 + beta_22
    #print(time.time()-to)
    return I_NL


def compute_bnl_darkquest(z, log10M1, log10M2, k, emulator):
    M1 = 10.0**log10M1
    M2 = 10.0**log10M2
    P_hh = emulator.get_phh_mass(k, M1, M2, z)
    Pk_lin = emulator.get_pklin_from_z(k, z)
    klin = 0.02 #large k to calculate bias
    Pk_klin = emulator.get_pklin_from_z(np.array([klin]), z)
    bM1 = np.sqrt(emulator.get_phh_mass(klin, M1, M1, z)/Pk_klin)
    bM2 = np.sqrt(emulator.get_phh_mass(klin, M2, M2, z)/Pk_klin)

    Bnl = P_hh/(bM1*bM2*Pk_lin) - 1.0
    
    return Bnl
    

def compute_bnl_darkquest_2(z, log10M1, log10M2, k, emulator):
    # Much faster than above func, mostly because it uses symmetry.
    M1 = 10.0**log10M1
    M2 = 10.0**log10M2
    # Parameters
    klin = np.array([0.02])  # Large 'linear' scale for linear halo bias [h/Mpc]
    
    # Calculate beta_NL by looping over mass arrays
    beta_func = np.zeros((len(z), len(M1), len(M2), len(k)))
    b01 = np.zeros(len(M1))
    b02 = np.zeros(len(M2))
    for iz1, z1 in enumerate(z):
        # Linear power
        Pk_lin = emulator.get_pklin_from_z(k, z1)
        Pk_klin = emulator.get_pklin_from_z(klin, z1)
        for iM, M0 in enumerate(M1):
            b01[iM] = np.sqrt(emulator.get_phh_mass(klin, M0, M0, z1)/Pk_klin)
        #for iM, M0 in enumerate(M2):
        #    b02[iM] = np.sqrt(emulator.get_phh_mass(klin, M0, M0, z1)/Pk_klin)
        for iM1, M01 in enumerate(M1):
            for iM2, M02 in enumerate(M2):
                if iM2 < iM1:
                    # Use symmetry to not double calculate
                    beta_func[iz1, iM1, iM2, :] = beta_func[iz1, iM2, iM1, :]
                else:
                    # Halo-halo power spectrum
                    Pk_hh = emulator.get_phh_mass(k, M01, M02, z1)
            
                    # Linear halo bias
                    b1 = b01[iM1]
                    b2 = b01[iM2]
                    
                    # Create beta_NL
                    beta_func[iz1, iM1, iM2, :] = Pk_hh/(b1*b2*Pk_lin) - 1.0
    
    return beta_func
    

def create_bnl_interpolation_function(emulator, interpolation):
    # AD: The mass range in Bnl needs to be optimised. Preferrentially set to the maximum mass limits in DarkEmulator, with the largest number of bins possible.
    M = np.logspace(12.1, 15.9, 15)#12.0, 14.0, 5)
    k = np.logspace(-2.0, 1.5, 50) #50)
    z = np.linspace(0.0, 0.5, 5)
    
    beta_func = compute_bnl_darkquest_2(z, np.log10(M), np.log10(M), k, emulator)
    if interpolation == True:
        beta_nl_interp = RegularGridInterpolator([z, np.log10(M), np.log10(M), k], beta_func, fill_value=None, bounds_error=False)
    else:
        beta_nl_interp = RegularGridInterpolator([z, np.log10(M), np.log10(M), k], beta_func, fill_value=0.0, bounds_error=False)
    return beta_nl_interp    


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
    
def prepare_Is_align_term(mass, s_align_factor, b_m, dn_dlnm, nz, nk):
    I_s_align_term = compute_Ig_term(s_align_factor, mass[np.newaxis,np.newaxis,:], dn_dlnm[:,np.newaxis,:], b_m[:,np.newaxis,:])
    return I_s_align_term
    
def prepare_Ic_align_term(mass, c_align_factor, b_m, dn_dlnm, nz, nk):
    I_c_align_term = compute_Ig_term(c_align_factor, mass[np.newaxis,np.newaxis,:], dn_dlnm[:,np.newaxis,:], b_m[:,np.newaxis,:])
    return I_c_align_term
    
def prepare_I_NL_mm(mass, m_factor, b_m, dn_dlnm, nz, nk, k_vec, z_vec, A, rho_mean, emulator, interpolation, beta_interp=None):
    # For Constance, do check this!
    print('preparing I_NL_mm')
    I_NL_mm = compute_I_NL_term(k_vec, z_vec, m_factor, m_factor, b_m, b_m, mass, mass, dn_dlnm, dn_dlnm, A, rho_mean, interpolation, beta_interp, emulator)
    return I_NL_mm

def prepare_I_NL_cs(mass, c_factor, s_factor, b_m, dn_dlnm, nz, nk, k_vec, z_vec, A, rho_mean, emulator, interpolation, beta_interp=None): #B_NL):
    print('preparing I_NL_cs')
    I_NL_cs = compute_I_NL_term(k_vec, z_vec, c_factor, s_factor, b_m, b_m, mass, mass, dn_dlnm, dn_dlnm, A, rho_mean, interpolation, beta_interp, emulator)
    print('nz: ', nz)
    print('I_NL_cs: ', I_NL_cs[0])
    return I_NL_cs

def prepare_I_NL_cc(mass, c_factor, b_m, dn_dlnm, nz, nk, k_vec, z_vec, A, rho_mean, emulator, interpolation, beta_interp=None):
    print('preparing I_NL_cc')
    I_NL_cc = compute_I_NL_term(k_vec, z_vec, c_factor, c_factor, b_m, b_m, mass, mass, dn_dlnm, dn_dlnm, A, rho_mean, interpolation, beta_interp, emulator)
    print('I_NL_cc: ', I_NL_cc[0])
    return I_NL_cc

def prepare_I_NL_ss(mass, s_factor, b_m, dn_dlnm, nz, nk, k_vec, z_vec, A, rho_mean, emulator, interpolation, beta_interp=None):
    print('preparing I_NL_ss')
    I_NL_ss = compute_I_NL_term(k_vec, z_vec, s_factor, s_factor, b_m, b_m, mass, mass, dn_dlnm, dn_dlnm, A, rho_mean, interpolation, beta_interp, emulator)
    print('I_NL_ss: ', I_NL_ss[0])
    return I_NL_ss

def prepare_I_NL_cm(mass, c_factor, m_factor, b_m, dn_dlnm, nz, nk, k_vec, z_vec, A, rho_mean, emulator, interpolation, beta_interp=None):
    print('preparing I_NL_cm')
    I_NL_cm = compute_I_NL_term(k_vec, z_vec, c_factor, m_factor, b_m, b_m, mass, mass, dn_dlnm, dn_dlnm, A, rho_mean, interpolation, beta_interp, emulator)
    return I_NL_cm 

def prepare_I_NL_sm(mass, s_factor, m_factor, b_m, dn_dlnm, nz, nk, k_vec, z_vec, A, rho_mean, emulator, interpolation, beta_interp=None):
    print('preparing I_NL_sm')
    I_NL_sm = compute_I_NL_term(k_vec, z_vec, s_factor, m_factor, b_m, b_m, mass, mass, dn_dlnm, dn_dlnm, A, rho_mean, interpolation, beta_interp, emulator)
    return I_NL_sm

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
    
    return alignment_amplitude_2h, alignment_amplitude_2h_II, C1 * mean_density0[:,np.newaxis,np.newaxis] * alignment_gi[:,np.newaxis,np.newaxis]



# ---- POWER SPECTRA ----#

# matter-matter
def compute_p_mm(block, k_vec, plin, z_vec, mass, dn_dln_m, m_factor, I_m_term, nz, nk):
    # 2-halo term:
    pk_mm_2h = compute_2h_term(plin, I_m_term, I_m_term) * two_halo_truncation(k_vec)[np.newaxis,:]
    # 1-halo term
    pk_mm_1h = compute_1h_term(m_factor, m_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation(k_vec)[np.newaxis,:]
    # Total
    pk_mm_tot= pk_mm_1h + pk_mm_2h
    #print('p_mm succesfully computed')
    return pk_mm_1h, pk_mm_2h, pk_mm_tot
    
def compute_p_mm_bnl(block, k_vec, plin, z_vec, mass, dn_dln_m, m_factor, I_m_term, nz, nk, I_NL_mm):
    # 2-halo term:
    pk_mm_2h = compute_2h_term(plin, I_m_term, I_m_term) + plin*I_NL_mm
    # 1-halo term
    pk_mm_1h = compute_1h_term(m_factor, m_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation(k_vec)[np.newaxis,:]
    # Total
    pk_mm_tot= pk_mm_1h + pk_mm_2h
    #print('p_mm succesfully computed')
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
# AD: We need:
#
#       p_sm_mI_1h + p_cm_mI_2h + p_sm_mi_2h + 2xBnl of course
#
def compute_p_mI(block, k_vec, p_eff, p_lin, z_vec, mass, dn_dln_m, m_factor, s_align_factor, I_m_term, I_c_align_term, I_s_align_term, alignment_amplitude_2h, nz, nk,
                  f_gal):
    #
    # p_tot = p_sm_mI_1h + f_cen*p_cm_mI_2h + O(any other combination)
    #
    # 2-halo term:
    pk_cm_2h = compute_p_mI_two_halo(block, k_vec, p_eff, z_vec, nz, f_gal, alignment_amplitude_2h) * two_halo_truncation_ia(k_vec)[np.newaxis,:]
    # 1-halo term
    pk_sm_1h = (-1.0) * compute_1h_term(m_factor, s_align_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation_ia(k_vec)[np.newaxis,:]
    # prepare the 1h term
    pk_tot = pk_sm_1h + pk_cm_2h
    import matplotlib.pyplot as plt
    for i in range(nz):
        plt.loglog(k_vec, -1 * pk_tot[i])
    # Above from Maria Cristina, belowe the added missing parts from Schneider & Bridle (+Bnl eventually):
    # Why 1h negative?
    pk_sm_1h = (-1.0) * compute_1h_term(m_factor, s_align_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation_ia(k_vec)[np.newaxis,:]
    pk_sm_2h = (-1.0) * compute_2h_term(p_lin, I_m_term, I_s_align_term)# * two_halo_truncation_ia(k_vec)[np.newaxis,:]
    pk_cm_2h = (-1.0) * compute_2h_term(p_lin, I_m_term, I_c_align_term)# * two_halo_truncation_ia(k_vec)[np.newaxis,:]
    pk_tot = pk_sm_1h + pk_cm_2h + pk_sm_2h
    for i in range(nz):
        plt.loglog(k_vec, -1 * pk_tot[i])
        plt.loglog(k_vec, -1 * pk_sm_1h[i])
        plt.loglog(k_vec, -1 * pk_sm_2h[i])
        plt.loglog(k_vec, -1 * pk_cm_2h[i])
    plt.show()
    quit()
    return pk_sm_1h, pk_cm_2h, pk_tot


# intrinsic-intrinsic power spectrum
# AD: We need:
#
#       p_ss_II_1h + p_ss_II_2h + p_cc_II_2h + p_sc_II_2h + 3xBnl of course
#
# Needs Poisson parameter as well!
def compute_p_II(block, k_vec, p_eff, z_vec, mass, dn_dln_m, s_align_factor, alignment_amplitude_2h_II, nz, nk, f_gal):
    #
    # p_tot = p_ss_II_1h + p_cc_II_2h + O(p_sc_II_1h) + O(p_cs_II_2h)
    #
    # 2-halo term: This is simply the Linear Alignment Model weighted by the central galaxy fraction
    pk_cc_2h = compute_p_II_two_halo(block, k_vec, p_eff, z_vec, nz, f_gal, alignment_amplitude_2h_II) * two_halo_truncation_ia(k_vec)[np.newaxis,:]
    # 1-halo term
    pk_ss_1h = compute_1h_term(s_align_factor, s_align_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation_ia(k_vec)[np.newaxis,:]
    pk_tot = pk_ss_1h + pk_cc_2h
    #print('p_II succesfully computed')
    return pk_ss_1h, pk_cc_2h, pk_tot


# galaxy-intrinsic power spectrum
#IT redefinition as dn_dln_m
# AD: We need:
#
#       p_cs_gI_1h + p_cc_gI_1h + p_ss_gI_1h + p_cs_gI_2h + p_cc_gI_2h + p_ss_gI_2h ?? + Bnl of course
#
def compute_p_gI(block, k_vec, p_eff, z_vec, mass, dn_dln_m, c_factor, s_align_factor, I_c_term, alignment_amplitude_2h,
                 nz, nk):
    #
    # p_tot = p_cs_gI_1h + (2?)*p_cc_gI_2h + O(p_ss_gI_1h) + O(p_cs_gI_2h)
    #
    # 2-halo term:
    #IT Removed new_axis from alignment_amplitude_2h[:,np.newaxis] in the following line
    pk_cc_2h = compute_2h_term(p_eff, I_c_term, alignment_amplitude_2h[:,]) * two_halo_truncation_ia(k_vec)[np.newaxis,:]
    # 1-halo term
    pk_cs_1h = compute_1h_term(c_factor[:,np.newaxis], s_align_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation_ia(k_vec)[np.newaxis,:]
    
    pk_tot = pk_cs_1h + pk_cc_2h
    # save in the datablock
    #block.put_grid('galaxy_cc_intrinsic_2h', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_cc_2h)
    #block.put_grid('galaxy_cs_intrinsic_1h', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_cs_1h)
    #IT Removed next line to save the pk in the interface. This function now returns the spectra
    #block.put_grid('galaxy_intrinsic_power', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_tot)
    #print('p_gI succesfully computed')
    return pk_cs_1h, pk_cc_2h, pk_tot




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
    interp_udm = interp2d(mass_udm, k_udm, udm_z, kind='linear', bounds_error=False)
    u_dm = interp_udm(mass, k_vec)
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
    nz = np.size(z_vec)
    nk = np.size(k_vec)
    nmass = np.size(mass)
    u_dm = np.array([interp_udm(mass_udm, k_udm, udm_z, mass, k_vec) for udm_z in u_udm])
    u_sat = np.array([interp_udm(mass_udm, k_udm, usat_z, mass, k_vec) for usat_z in u_usat])
    #print('--- u_dm: %s seconds ---' % (time.time() - start_time_udm))
    return np.abs(u_dm), np.abs(u_sat)
