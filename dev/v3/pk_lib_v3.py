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
# central galaxy (position)
def compute_central_galaxy_factor(Ncen, numdenscen, f_c):
    return f_c * Ncen / numdenscen
# satellite galaxy (position)
def compute_satellite_galaxy_factor(Nsat, numdenssat, f_s, u_gal):
    return f_s * Nsat * u_gal / numdenssat
# satellite galaxy (alignment)
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
    s_factor = np.empty([nz, nk, nmass])
    for jz in range(0, nz):
        for ik in range(0, nk):
            s_factor[jz, ik, :] = compute_satellite_galaxy_factor(Nsat[jz, :], numdensat[jz], f_sat[jz],
                                                                  u_gal[jz, ik, :])
    return s_factor

# clustering - centrals
def prepare_central_factor_grid(Ncen, numdencen, f_cen):
    c_factor = np.array([compute_central_galaxy_factor(Ncen_z, numdencen_z, f_cen_z) for Ncen_z, numdencen_z, f_cen_z in
                         zip(Ncen, numdencen, f_cen)])
    return c_factor

# alignment - satellites
def prepare_satellite_alignment_factor_grid(mass, Nsat, numdensat, f_sat, wkm, gamma_1h, nz, nk, nmass):
    '''
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
    '''
    s_align_factor = np.array([[compute_satellite_galaxy_alignment_factor(Nsat[jz, :], numdensat[jz],
                                                                          f_sat[jz], wkm[jz, :, ik])
                                for ik in range(0,nk)] for jz in range(0,nz)])
    #s_align_factor *= gamma_1h[:, np.newaxis, np.newaxis]
    '''
    s_align_factor = np.empty([nz, nk, nmass])
    for jz in range(0, nz):
        for ik in range(0, nk):
            s_align_factor[jz, ik, :] = compute_satellite_galaxy_alignment_factor(Nsat[jz, :], numdensat[jz], f_sat[jz],
                                                                                  wkm[jz, :, ik])
        s_align_factor[jz] *= gamma_1h[jz]
    '''
    print('s_align_factor successfully computed!')
    return s_align_factor

# -------------------------------------------------------------------------------------------------------------------- #
# Two halo functions
# -------------------------------------------------------------------------------------------------------------------- #

# The I_X,Y integrals that enters into the 2h - power spectra (analytical expression)
# Args : scalars
# Return: scalar

def compute_Im_term(mass, u_dm, b_dm, dn_dlnm, mean_density0):
    # AD: check what happens if u_dm is removed. For considered k-ranges removing it should be fine.
    integrand_m1 = b_dm * dn_dlnm * (1. / mean_density0)
    integrand_m2 = b_dm * dn_dlnm * u_dm * (1. / mean_density0)
    I_m1 = 1. - simps(integrand_m1, mass)
    I_m2 = simps(integrand_m2, mass)
    I_m = I_m1 + I_m2
    return I_m

def compute_Ig_term(factor_1, mass, dn_dlnm_z, b_m):
    integrand = factor_1 * b_m * dn_dlnm_z / mass
    I_g = simps(integrand, mass)
    return I_g

# Compute the grid in z and M (and eventually k) of the quantities described above

def prepare_Im_term(mass, u_dm, b_dm, dn_dlnm, mean_density0, nz, nk):
    I_m_term = np.array([[compute_Im_term(mass, u_dm[jz, ik, :], b_dm[jz], dn_dlnm[jz], mean_density0[jz])
                          for ik in range(0,nk)] for jz in range(0,nz)])
    return I_m_term

def prepare_Is_term(mass, s_factor, b_m, dn_dlnm, nz, nk):
    I_s_term = np.array([[compute_Ig_term(s_factor[jz, ik], mass, dn_dlnm[jz], b_m[jz]) for ik in range(0,nk)] for jz in range(0,nz)])
    return I_s_term

def prepare_Ic_term(mass, c_factor, b_m, dn_dlnm, nz, nk):
    #I_c_term = compute_Ig_term(c_factor, mass[np.newaxis,:], dn_dlnm, b_m)
    I_c_term = np.empty([nz, nk])
    for jz in range(0, nz):
        I_c_term[jz] = compute_Ig_term(c_factor[jz], mass, dn_dlnm[jz], b_m[jz])
    return I_c_term


def compute_two_halo_alignment(block, suffix, nz, nk, growth_factor, mean_density0):
    '''
    The IA amplitude at large scales, including the IA prefactors.

    :param block: the CosmoSIS datablock
    :param suffix: str, name of the sample as in the option section
    :param nz: int, number of redshift bins
    :param nk: int, number of wave vector bins
    :param growth_factor: double array 2d (nz, nk), growth factor normalised to be 1 at z=0
    :param mean_density0: double, mean matter density of the Universe at redshift z=0
    Set in the option section.
    :return: double array 2d (nz, nk), double array 2d (nz, nk) : the large scale alignment amplitudes (GI and II)
    '''
    # linear alignment coefficients
    C1 = 5.e-14
    # load the 2h (effective) amplitude of the alignment signal from the data block. 
    # This already includes the luminosity dependence if set. Double array [nz].
    alignment_gi = block["ia_large_scale_alignment" + suffix, "alignment_gi"]
    #alignment_ii = block["ia_large_scale_alignment" + suffix, "alignment_ii"]
    alignment_amplitude_2h = np.empty([nz, nk])
    alignment_amplitude_2h_II = np.empty([nz, nk])
    for jz in range(0, nz):
        alignment_amplitude_2h[jz] = -alignment_gi[jz] * (C1 * mean_density0[jz] / growth_factor[jz])
        # since the luminosity dependence is squared outside, the II amplitude is just GI squared
        alignment_amplitude_2h_II[jz] = (alignment_gi[jz] * C1 * mean_density0[jz] / growth_factor[jz]) ** 2.
        #alignment_amplitude_2h_II[jz] = alignment_ii[jz] * (C1 * mean_density0[jz] / growth_factor[jz]) ** 2.
    print (alignment_amplitude_2h, alignment_amplitude_2h_II)
    return alignment_amplitude_2h, alignment_amplitude_2h_II



# ---- POWER SPECTRA ----#

# matter-matter
def compute_p_mm_new(block, k_vec, plin, z_vec, mass, dn_dln_m, m_factor, I_m_term, nz, nk):
    # 2-halo term:
    pk_mm_2h = compute_2h_term(plin, I_m_term, I_m_term) * two_halo_truncation(k_vec)[np.newaxis,:]
    # 1-halo term
    pk_mm_1h = compute_1h_term(m_factor, m_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation(k_vec)[np.newaxis,:]
    # Total
    pk_mm_tot= pk_mm_1h + pk_mm_2h
    #print('p_mm succesfully computed')
    return pk_mm_1h, pk_mm_2h, pk_mm_tot


# galaxy-galaxy power spectrum
def compute_p_nn(block, k_vec, pk_lin, z_vec, mass, dn_dln_m, c_factor, s_factor, I_c_term, I_s_term, nz, nk):
    #
    # p_tot = p_cs_1h + p_ss_1h + p_cs_2h + p_cc_2h
    #
    # 2-halo term:
    pk_cs_2h = compute_2h_term(pk_lin, I_c_term, I_s_term) * two_halo_truncation(k_vec)[np.newaxis,:]
    pk_cc_2h = compute_2h_term(pk_lin, I_c_term, I_c_term) * two_halo_truncation(k_vec)[np.newaxis,:]
    pk_ss_2h = compute_2h_term(pk_lin, I_s_term, I_s_term) * two_halo_truncation(k_vec)[np.newaxis,:]
    # 1-halo term:
    pk_cs_1h = np.empty([nz, nk])
    pk_ss_1h = np.empty([nz, nk])
    for jz in range(0, nz):
        for ik in range(0, nk):
            pk_cs_1h[jz, ik] = compute_1h_term(c_factor[jz], s_factor[jz, ik], mass, dn_dln_m[jz])
            pk_ss_1h[jz, ik] = compute_1h_term(s_factor[jz, ik], s_factor[jz, ik], mass, dn_dln_m[jz])
        #mask_large_scales = k_vec < 0.01
        #pk_cs_1h[jz][mask_large_scales] = 0.0
        #pk_ss_1h[jz][mask_large_scales] = 0.0
        pk_cs_1h[jz] *= one_halo_truncation(k_vec)
        pk_ss_1h[jz] *= one_halo_truncation(k_vec)
    # Total
    # AD: add Poisson parameter to ph_ss_1h!
    pk_tot = 2. * pk_cs_1h + pk_ss_1h + pk_cc_2h + pk_ss_2h + 2. * pk_cs_2h

    # in case, save in the datablock
    #block.put_grid("galaxy_cs_power_1h", "z", z_vec, "k_h", k_vec, "p_k", pk_cs_1h)
    #block.put_grid("galaxy_ss_power_1h", "z", z_vec, "k_h", k_vec, "p_k", pk_ss_1h)

    # galaxy linear bias
    galaxy_linear_bias = np.sqrt(I_c_term ** 2. + I_s_term ** 2. + 2. * I_s_term * I_c_term)
    #print('p_nn succesfully computed')
    return 2. * pk_cs_1h + pk_ss_1h, pk_cc_2h + pk_ss_2h + 2. * pk_cs_2h, pk_tot, galaxy_linear_bias


# galaxy-matter power spectrum
def compute_p_xgG(block, k_vec, pk_lin, z_vec, mass, dn_dln_m, c_factor, s_factor, m_factor, I_c_term, I_s_term, I_m_term):
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


#################################################
#                                                                                                #
#                INTRINSIC ALIGNMENT POWER SPECTRA                #
#                                                                                                #
#################################################


# galaxy-matter power spectrum
def compute_p_xGI(block, k_vec, p_eff, z_vec, mass, dn_dln_m, m_factor, s_align_factor, alignment_amplitude_2h, nz, nk,
                  f_gal):
    #
    # p_tot = p_sm_GI_1h + f_cen*p_cm_GI_2h + O(any other combination)
    #
    # 2-halo term:
    pk_cm_2h = compute_p_xGI_two_halo(block, k_vec, p_eff, z_vec, nz, f_gal, alignment_amplitude_2h) * two_halo_truncation_ia(k_vec)[np.newaxis,:]
    # 1-halo term
    pk_sm_1h = - compute_1h_term(m_factor, s_align_factor, mass, dn_dln_m[:,np.newaxis]) * one_halo_truncation_ia(k_vec)[np.newaxis,:]
    # prepare the 1h term
    pk_tot = pk_sm_1h + pk_cm_2h
    # save in the datablock
    # block.put_grid("matter_intrinsic_2h", "z", z_vec, "k_h", k_vec, "p_k", pk_cm_2h)
    # block.put_grid("matter_intrinsic_1h", "z", z_vec, "k_h", k_vec, "p_k", pk_sm_1h)
    # block.put_grid("matter_intrinsic_power", "z", z_vec, "k_h", k_vec, "p_k", pk_tot)
    #print('p_xGI succesfully computed')
    return pk_sm_1h, pk_cm_2h, pk_tot


# intrinsic-intrinsic power spectrum
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
    '''
    # prepare the 1h term
    pk_cs_1h = np.empty([nz, nk])
    for jz in range(0, nz):
        for ik in range(0, nk):
            pk_cs_1h[jz, ik] = compute_1h_term(c_factor[jz], s_align_factor[jz, ik, :], mass, dn_dlnm[jz])
    # this is simply the Linear Alignment Model
    # NOTE: that here we are assuming the linear bias to be caused by the central galaxies only
    # if we want to use the bias of the entire sample, we can obtain it as:
    # bias = np.sqrt(Ic**2 + Is**2 + (2.*Ic*Is) [see the notes on the spiral notebook]
    align_2h = np.empty(I_c_term.shape)
    for jz in range(0, nz):
        align_2h[jz] = alignment_amplitude_2h[jz]
    # b_g = np.sqrt(I_c_term**2.+I_s_term**2.+2.*I_s_term*I_c_term)
    # pk_tot_2h =  b_g*(p_eff*align_2h)
    pk_cc_2h = compute_2h_term(p_eff, I_c_term, align_2h)
    '''
    pk_tot = pk_cs_1h + pk_cc_2h
    # save in the datablock
    #block.put_grid("galaxy_cc_intrinsic_2h", "z", z_vec, "k_h", k_vec, "p_k", pk_cc_2h)
    #block.put_grid("galaxy_cs_intrinsic_1h", "z", z_vec, "k_h", k_vec, "p_k", pk_cs_1h)
    #IT Removed next line to save the pk in the interface. This function now returns the spectra
    #block.put_grid("galaxy_intrinsic_power", "z", z_vec, "k_h", k_vec, "p_k", pk_tot)
    print('p_gI succesfully computed')
    return pk_cs_1h, pk_cc_2h, pk_tot




############### TWO HALO ONLY ###################

# galaxy-galaxy power spectrum
def compute_p_nn_two_halo(block, k_vec, plin, z_vec, bg):
    #
    # p_tot = b_g**2 * p_lin
    #
    pk_tot = np.zeros([len(z_vec), len(k_vec)])
    for jz in range(len(z_vec)):
        pk_tot[jz] = bg[jz] ** 2. * plin[jz]
    return pk_tot


# galaxy-matter power spectrum
def compute_p_xgG_two_halo(block, k_vec, plin, z_vec, bg):
    #
    # p_tot = bg * plin
    #
    pk_tot = np.zeros([len(z_vec), len(k_vec)])
    for jz in range(len(z_vec)):
        pk_tot[jz] = bg[jz] * plin[jz]
    return pk_tot


# galaxy-matter power spectrum
def compute_p_xGI_two_halo(block, k_vec, p_eff, z_vec, nz, f_gal, alignment_amplitude_2h):
    #
    # p_tot = p_NLA
    #
    # this is simply the Linear (or Nonlinear) Alignment Model, weighted by the central galaxy fraction
    pk_tot = np.zeros([len(z_vec), len(k_vec)])
    for jz in range(0, nz):
        pk_tot[jz] = f_gal[jz] * alignment_amplitude_2h[jz] * p_eff[jz]
    return pk_tot


# galaxy-intrinsic power spectrum
def compute_p_gI_two_halo(block, k_vec, p_eff, z_vec, nz, f_gal, alignment_amplitude_2h, bg):
    #
    # p_tot = bg * p_NLA
    #
    pk_tot = np.zeros([len(z_vec), len(k_vec)])
    for jz in range(0, nz):
        pk_tot[jz] = f_gal[jz] * bg[jz] * alignment_amplitude_2h[jz] * p_eff[jz]
    return pk_tot


# galaxy-intrinsic power spectrum
def compute_p_II_two_halo(block, k_vec, p_eff, z_vec, nz, f_gal, alignment_amplitude_2h_II):
    pk_tot = np.zeros([len(z_vec), len(k_vec)])
    for jz in range(0, nz):
        pk_tot[jz] = (f_gal[jz] ** 2.) * p_eff[jz] * alignment_amplitude_2h_II[jz]
    return pk_tot

#################################################


def interp_udm(mass_udm, k_udm, udm_z, mass, k_vec):
    interp_udm = interp2d(mass_udm, k_udm, udm_z, kind='linear', bounds_error=False)
    u_dm = interp_udm(mass, k_vec)
    return u_dm

def compute_u_dm_grid(block, k_vec, mass, z_vec):
    start_time_udm = time.time()
    z_udm = block["fourier_nfw_profile", "z"]
    mass_udm = block["fourier_nfw_profile", "m_h"]
    k_udm = block["fourier_nfw_profile", "k_h"]
    u_udm = block["fourier_nfw_profile", "ukm"]
    u_usat = block["fourier_nfw_profile", "uksat"]
    u_udm = np.reshape(u_udm, (np.size(z_udm),np.size(k_udm),np.size(mass_udm)))
    u_usat = np.reshape(u_usat, (np.size(z_udm),np.size(k_udm),np.size(mass_udm)))
    # interpolate
    nz = np.size(z_vec)
    nk = np.size(k_vec)
    nmass = np.size(mass)
    u_dm = np.array([interp_udm(mass_udm, k_udm, udm_z, mass, k_vec) for udm_z in u_udm])
    u_sat = np.array([interp_udm(mass_udm, k_udm, usat_z, mass, k_vec) for usat_z in u_usat])
    print("--- u_dm: %s seconds ---" % (time.time() - start_time_udm))
    return np.abs(u_dm), np.abs(u_sat)
