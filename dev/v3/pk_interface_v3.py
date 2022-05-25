# A new power spectrum module

# NOTE: no truncation (halo exclusion problem) applied!

from cosmosis.datablock import names, option_section
import sys
import numpy as np
from scipy.interpolate import interp1d, interp2d
from astropy.cosmology import FlatLambdaCDM, Flatw0waCDM
import astropy.units as u

# import hankel
from scipy.integrate import quad, simps, trapz
# from scipy.misc import factorial
# from scipy.special import legendre, sici, binom
import math

from load_utilities import get_linear_power_spectrum, get_halo_functions, \
    get_nonlinear_power_spectrum, compute_effective_power_spectrum, \
    get_satellite_alignment, load_growth_factor, load_hods, load_galaxy_fractions
from pk_lib_v3 import compute_p_mm_new, compute_u_dm_grid, \
    prepare_matter_factor_grid, prepare_Im_term, prepare_central_factor_grid, \
    prepare_satellite_factor_grid, prepare_Ic_term, prepare_Is_term, compute_p_nn, \
    prepare_satellite_alignment_factor_grid, compute_p_xgG, compute_p_gI, compute_p_xGI, compute_p_II, \
    compute_p_nn_two_halo, compute_p_xgG_two_halo, compute_p_xGI_two_halo, compute_p_gI_two_halo, \
    compute_p_II_two_halo, compute_two_halo_alignment

import time
#IT commented next line because it's not available in Python 2.7
#import timing
# from guppy import hpy

import os, errno

cosmo = names.cosmological_parameters


# --------- COSMOSIS MODULE ----------- #

def setup(options):
    # This function is called once per processor per chain.
    # It is a chance to read any fixed options from the configuration file,
    # load any data, or do any calculations that are fixed once.

    log_mass_min = options[option_section, "log_mass_min"]
    log_mass_max = options[option_section, "log_mass_max"]
    nmass = options[option_section, "nmass"]
    # log-spaced mass in units of M_sun/h
    dlog10m = (log_mass_max-log_mass_min)/nmass
    mass = 10.0 ** np.arange(log_mass_min, log_mass_max, dlog10m)

    zmin = options[option_section, "zmin"]
    zmax = options[option_section, "zmax"]
    nz = options[option_section, "nz"]
    z_vec = np.linspace(zmin, zmax, nz)

    pipeline = options[option_section, "pipeline"]

    # pk_obs = options[option_section, "pk_to_compute"]
    p_GG = options[option_section, "p_GG"]
    p_nn = options[option_section, "p_nn"]
    p_xgG = options[option_section, "p_xgG"]
    p_gI = options[option_section, "p_gI"]
    p_xGI = options[option_section, "p_xGI"]
    p_II = options[option_section, "p_II"]

    # initiate pipeline parameters
    ia_lum_dep_centrals = False
    ia_lum_dep_satellites = False
    gravitational = False
    galaxy = False
    alignment = False
    hod_section_name = ''
    two_halo_only = options[option_section, "two_halo_only"]
    f_red_cen_option = False

    if (two_halo_only == True) and (p_GG == True):
        gravitational = True
    elif (two_halo_only == False) and ((p_GG == True) or (p_xgG == True) or (p_xGI == True)):
        gravitational = True
    if (p_nn == True) or (p_xgG == True) or (p_gI == True) or (p_xGI == True) or (p_II == True) :
        galaxy = True
        hod_section_name = options[option_section, "hod_section_name"]
    if (p_gI == True) or (p_xGI == True) or (p_II == True):
        alignment = True
        #IT commented. No longer used
        #ia_lum_dep_centrals = options[option_section, "ia_luminosity_dependence_centrals"]
        #ia_lum_dep_satellites = options[option_section, "ia_luminosity_dependence_satellites"]
        #f_red_cen_option = options[option_section, "f_red_cen"] # TODO: consider to remove it (not particularly relevant anymore.. )
        #print('alignment is true')

    name = options.get_string(option_section, "name", default="").lower()
    if name:
        suffix = "_" + name
    else:
        suffix = ""

    # ============================================================================== #
    # this only makes sense in the context of the two halo only
    # f_red_cen = np.ones(nz)
    # if two_halo_only == True:
    #if f_red_cen_option:
    #    f_red_cen_file = options[option_section, "f_red_cen_file"]
    #    f_red_cen = load_galaxy_fractions(f_red_cen_file, z_vec)
    # if len(f_red_cen)!= nz:
    #    raise ValueError("The length of f_red_cen(z) does not match nz")
    #else:
    #    f_red_cen = np.ones(nz)
    #print(f_red_cen)
    # ============================================================================== #

    return mass, nmass, z_vec, nz, p_GG, p_nn, p_xgG, p_gI, p_xGI, p_II, gravitational, galaxy, alignment, \
           ia_lum_dep_centrals, ia_lum_dep_satellites, two_halo_only, pipeline, hod_section_name, suffix


# @profile
def execute(block, config):
    # This function is called every time you have a new sample of cosmological and other parameters.
    # It is the main workhorse of the code. The block contains the parameters and results of any
    # earlier modules, and the config is what we loaded earlier.

    mass, nmass, z_vec, nz, p_GG, p_nn, p_xgG, p_gI, p_xGI, p_II, gravitational, galaxy, alignment, \
    ia_lum_dep_centrals, ia_lum_dep_satellites, two_halo_only, pipeline, hod_section_name, suffix = config

    start_time = time.time()

    mean_density0 = block["density", "mean_density0"]
    

    # load linear power spectrum
    k_vec, plin, growth_factor = get_linear_power_spectrum(block, z_vec)
    nk = len(k_vec)

    # load nonlinear power spectrum (halofit)
    k_nl, p_nl = get_nonlinear_power_spectrum(block, z_vec)
    
    # AD: avoid this! (Maybe needed for IA part ...)
    # compute the effective power spectrum, mixing the linear and nonlinear one:
    #
    # (1.-t_eff)*plin + t_eff*p_nl
    #
    t_eff = block["pk_parameters", "trans_1hto2h"]
    pk_eff = compute_effective_power_spectrum(k_vec, plin, k_nl, p_nl, z_vec, t_eff)

    # initialise the galaxy bias
    bg = 1.0 # AD: ???

    # If the two_halo_only option is set True, then only the linear regime is computed and the linear bias is used (either computed by the
    # hod module or passed in the value	file (same structure as for the constant bias module)
    # Otherwise, compute the full power spectra (including the small scales)
    if two_halo_only == True:
        # preparing the integrals:
        if gravitational == True:
            # load the halo mass and bias functions from the datablock
            dn_dlnm, b_dm = get_halo_functions(block, pipeline, mass, z_vec)
            # prepare a grid for the navarro-frenk-white profile
            u_dm, u_sat = compute_u_dm_grid(block, k_vec, mass, z_vec)
            I_m_term = prepare_Im_term(mass, u_dm, b_dm, dn_dlnm, mean_density0, nz, nk)
            m_factor = prepare_matter_factor_grid(mass, mean_density0, u_dm)
        if galaxy == True:
            # load linear bias:
            bg = block["galaxy_bias" + suffix, "b"]
            if np.isscalar(bg): bg *= np.ones(nz)
        if alignment == True:
	    #IT commented ia_lum_dep_centrals
            alignment_amplitude_2h, alignment_amplitude_2h_II = compute_two_halo_alignment(block, suffix, nz, nk,
                                                                                           growth_factor, mean_density0
                                                                                           )#ia_lum_dep_centrals)
        # compute the power spectra
        if p_GG:
            # this is not very useful as for the lensing power spectrum it is usually used halofit
            raise ValueError("pGG not implemented for the two-halo only option\n")
            # AD: Implement proper matter-matter term using the halo model here. We do now want to use halofit.
            # compute_p_mm_new(block, k_vec, pk_eff, z_vec, mass, dn_dlnm, m_factor, I_m_term, nz, nk)
        if p_nn:
            pk_nn = compute_p_nn_two_halo(block, k_vec, pk_eff, z_vec, bg)
            block.put_grid("galaxy_power" + suffix, "z", z_vec, "k_h", k_vec, "p_k", pk_nn)
        if p_xgG:
            pk_xgG = compute_p_xgG_two_halo(block, k_vec, pk_eff, z_vec, bg)
            block.put_grid("matter_galaxy_power" + suffix, "z", z_vec, "k_h", k_vec, "p_k", pk_xgG)
        if p_xGI:
            #print('pGI...')
            pk_xGI = compute_p_xGI_two_halo(block, k_vec, pk_eff, z_vec, nz, f_red_cen, alignment_amplitude_2h)
            block.put_grid("matter_intrinsic_power" + suffix, "z", z_vec, "k_h", k_vec, "p_k", pk_xGI)
        if p_gI:
            pk_gI = compute_p_gI_two_halo(block, k_vec, pk_eff, z_vec, nz, f_red_cen, alignment_amplitude_2h, bg)
            block.put_grid("galaxy_intrinsic_power" + suffix, "z", z_vec, "k_h", k_vec, "p_k", pk_gI)
        if p_II:
            #print('pII...')
            pk_II = compute_p_II_two_halo(block, k_vec, pk_eff, z_vec, nz, f_red_cen, alignment_amplitude_2h_II)
            block.put_grid("intrinsic_power" + suffix, "z", z_vec, "k_h", k_vec, "p_k", pk_II)

    else:
        # load the halo mass and bias functions from the datablock and prepare a grid for the navarro-frenk-white profile
        dn_dlnm, b_dm = get_halo_functions(block, pipeline, mass, z_vec)
        u_dm, u_sat = compute_u_dm_grid(block, k_vec, mass, z_vec)

        # prepare the integrals
        if gravitational == True:
            # the matter integral and factor
            I_m_term = prepare_Im_term(mass, u_dm, b_dm, dn_dlnm, mean_density0, nz, nk)
            m_factor = prepare_matter_factor_grid(mass, mean_density0, u_dm)
        if (galaxy == True) or (alignment == True):
            # here we assume satellites perfectly follow nfw profile (i.e. they also have the same concentration):
            # u_sat(k*rs,c_gal) = u_dm(k*rs,c_dm)
            #print(hod_section_name)
            Ncen, Nsat, numdencen, numdensat, f_cen, f_sat = load_hods(block, hod_section_name, pipeline, z_vec, mass)
            if galaxy == True:
                # preparing the 1h term
                c_factor = prepare_central_factor_grid(Ncen, numdencen, f_cen)
                s_factor = prepare_satellite_factor_grid(Nsat, numdensat, f_sat, u_sat, nz, nk, nmass)
                # preparing the 2h term
                I_c_term = prepare_Ic_term(mass, c_factor, b_dm, dn_dlnm, nz, nk)
                I_s_term = prepare_Is_term(mass, s_factor, b_dm, dn_dlnm, nz, nk)
            if alignment == True:
		#IT commenting ia_lum_dep_centrals
                alignment_amplitude_2h, alignment_amplitude_2h_II = compute_two_halo_alignment(block, suffix, nz, nk,
                                                                                               growth_factor, mean_density0)#,
                                                                                               #ia_lum_dep_centrals)
                # ============================================================================== #
                # One halo alignment
                # ============================================================================== #
                ###########
                # gamma_1h am_amplitude is now computed inside the wkm module, so this is only the luminosity dependence of it
                gamma_1h = np.ones(nz)
                ###########
                #if ia_lum_dep_satellites == True:
                #    sat_l_term = block["ia_small_scale_alignment" + suffix, "alignment_1h"]
                #    gamma_1h *= sat_l_term
                # load the satellite density run w(k|m) for a perfect 3d radial alignment projected along the line of sight
                # it can either be constant or radial dependent -> this is computed in the wkm module, including the amplitude of the
                # signal (but not its luminosity dependence, which is a separate factor, see above)
                wkm = get_satellite_alignment(block, k_vec, mass, z_vec, suffix)
                # preparing the 1h term
                s_align_factor = prepare_satellite_alignment_factor_grid(mass, Nsat, numdensat, f_sat, wkm, gamma_1h, nz,
                                                                         nk, nmass)

        # compute the power spectra
        if p_GG == True:
            pk_mm_1h, pk_mm_2h, pk_mm_tot = compute_p_mm_new(block, k_vec, pk_eff, z_vec, mass, dn_dlnm, m_factor,
                                                             I_m_term, nz, nk)
            # save in the datablock
            #block.put_grid("matter_1h_power", "z", z_vec, "k_h", k_vec, "p_k", pk_mm_1h)
            #block.put_grid("matter_2h_power", "z", z_vec, "k_h", k_vec, "p_k", pk_mm_2h)
            block.put_grid("matter_power", "z", z_vec, "k_h", k_vec, "p_k", pk_mm_tot)
            # If you want to use this power spectrum to do cosmic shear, replace the lines above with the following:
            # block.replace("matter_power_nl", "z", z_vec, "k_h", k_vec, "p_k", pk_mm_tot)

        if p_nn == True:
            pk_nn_1h, pk_nn_2h, pk_nn, bg_halo_model = compute_p_nn(block, k_vec, plin, z_vec, mass, dn_dlnm, c_factor,
                                                                    s_factor, I_c_term, I_s_term, nz, nk)
            #block.put_grid("galaxy_power_1h" + suffix, "z", z_vec, "k_h", k_vec, "p_k", pk_nn_1h)
            #block.put_grid("galaxy_power_2h" + suffix, "z", z_vec, "k_h", k_vec, "p_k", pk_nn_2h)
            block.put_grid("galaxy_power" + suffix, "z", z_vec, "k_h", k_vec, "p_k", pk_nn)
            #block.put_grid("galaxy_linear_bias" + suffix, "z", z_vec, "k_h", k_vec, "galaxybiastotal", bg_halo_model)

        if p_xgG == True:
            #print("computing p_xgG...")
	    #IT Replacing pk_eff by plin
            pk_1h, pk_2h, pk_tot = compute_p_xgG(block, k_vec, plin, z_vec, mass, dn_dlnm, c_factor, s_factor, m_factor, I_c_term, I_s_term,
                          I_m_term)
            #block.put_grid("matter_galaxy_power_1h", "z", z_vec, "k_h", k_vec, "p_k", pk_1h)
            #block.put_grid("matter_galaxy_power_2h", "z", z_vec, "k_h", k_vec, "p_k", pk_2h)
	    #IT Adding suffix to matter_galaxy_power
            block.put_grid("matter_galaxy_power" + suffix, "z", z_vec, "k_h", k_vec, "p_k", pk_tot)
        if p_II == True:
            pk_II_1h, pk_II_2h, pk_II = compute_p_II(block, k_vec, pk_eff, z_vec, mass, dn_dlnm, s_align_factor,
                                                     alignment_amplitude_2h_II, nz, nk, f_cen)
            #block.put_grid("intrinsic_power_1h" + suffix, "z", z_vec, "k_h", k_vec, "p_k", pk_II_1h)
            #block.put_grid("intrinsic_power_2h" + suffix, "z", z_vec, "k_h", k_vec, "p_k", pk_II_2h)
            block.put_grid("intrinsic_power" + suffix, "z", z_vec, "k_h", k_vec, "p_k", pk_II)
        if p_gI == True:
            #print("computing p_gI...")
            pk_gI_1h, pk_gI_2h, pk_gI = compute_p_gI(block, k_vec, pk_eff, z_vec, mass, dn_dlnm, c_factor, s_align_factor, I_c_term, alignment_amplitude_2h, nz, nk)
	    #IT Added galaxy_intrinsic_power to datablock
            block.put_grid("galaxy_intrinsic_power" + suffix, "z", z_vec, "k_h", k_vec, "p_k", pk_gI)
        if p_xGI == True:
            #print("computing p_xGI...")
            # compute_p_xGI(block, k_vec, pk_eff, z_vec, mass, dn_dlnm, m_factor, s_align_factor, alignment_amplitude_2h, nz, nk, f_red_cen)
            pk_xGI_1h, pk_xGI_2h, pk_xGI = compute_p_xGI(block, k_vec, pk_eff, z_vec, mass, dn_dlnm, m_factor,
                                                         s_align_factor, alignment_amplitude_2h, nz, nk, f_cen)
            #block.put_grid("matter_intrinsic_power_1h" + suffix, "z", z_vec, "k_h", k_vec, "p_k", pk_xGI_1h)
            #block.put_grid("matter_intrinsic_power_2h" + suffix, "z", z_vec, "k_h", k_vec, "p_k", pk_xGI_2h)
            block.put_grid("matter_intrinsic_power" + suffix, "z", z_vec, "k_h", k_vec, "p_k", pk_xGI)

    # after = hp.heap()
    # leftover= after - before
    # print leftover

    #print("--- pk: %s seconds ---" % (time.time() - start_time))

    return 0


def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass



