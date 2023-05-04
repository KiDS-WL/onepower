# A new power spectrum module

# NOTE: no truncation (halo exclusion problem) applied!

from cosmosis.datablock import names, option_section
import sys
import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy import interp
from astropy.cosmology import FlatLambdaCDM, Flatw0waCDM
import astropy.units as u
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import quad, simps, trapz
import timeit
from collections import OrderedDict

# import hankel
from scipy.integrate import quad, simps, trapz
# from scipy.misc import factorial
# from scipy.special import legendre, sici, binom
import math
#import load_utilities
import pk_lib

import time

import os, errno

cosmo = names.cosmological_parameters

def get_linear_power_spectrum(block, z_vec):
    # AD: growth factor should be computed from camb/hmf directly, this way we can load Plin directly without this functions!
    k_vec = block['matter_power_lin', 'k_h']
    z_pl = block['matter_power_lin', 'z']
    matter_power_lin = block['matter_power_lin', 'p_k']
    growth_factor_zlin = block['growth_parameters', 'd_z'].flatten()[:,np.newaxis] * np.ones(k_vec.size)
    scale_factor_zlin = block['growth_parameters', 'a'].flatten()[:,np.newaxis] * np.ones(k_vec.size)
    gf_interp = interp1d(z_pl, growth_factor_zlin, axis=0)
    growth_factor = gf_interp(z_vec)
    a_interp = interp1d(z_pl, scale_factor_zlin, axis=0)
    scale_factor = a_interp(z_vec)
    # interpolate in redshift
    plin = interpolate1d_matter_power_lin(matter_power_lin, z_pl, z_vec)
    return k_vec, plin, growth_factor, scale_factor
    
def get_nonlinear_power_spectrum(block, z_vec):
    k_nl = block['matter_power_nl', 'k_h']
    z_nl = block['matter_power_nl', 'z']
    matter_power_nl = block['matter_power_nl', 'p_k']
    # this seems redundant
    p_nl = interpolate1d_matter_power_lin(matter_power_nl, z_nl, z_vec)
    return k_nl, p_nl
    
def compute_effective_power_spectrum(k_vec, plin, k_nl, p_nl, z_vec, t_eff):
    # interpolate
    #p_nl_interp = interp2d(k_nl, z_vec, p_nl)
    #pnl_int = p_nl_interp(k_vec, z_vec)
    p_nl_interp = RegularGridInterpolator((k_nl.T, z_vec.T), p_nl.T, bounds_error=False, fill_value=None)
    kk, zz = np.meshgrid(k_vec, z_vec, sparse=True)
    pnl_int = p_nl_interp((kk.T, zz.T)).T
    return (1.-t_eff)*plin+t_eff*pnl_int
    
    
def get_halo_functions(block, pipeline, mass, z_vec):
    if pipeline == True:
        # load the halo mass function
        dn_dlnm = block['hmf', 'dndlnmh']
        # load the halobias
        b_dm = block['halobias', 'b_hb']
    else:
        # load the halo mass function
        mass_hmf = block['hmf', 'm_h']
        z_hmf = block['hmf', 'z']
        dndlnmh_hmf = block['hmf', 'dndlnmh']
        # load the halobias
        mass_hbf = block['halobias', 'm_h']
        z_hbf = block['halobias', 'z']
        halobias_hbf = block['halobias', 'b_hb']
        # interpolate all the quantities that enter in the integrals
        dn_dlnm = interpolate2d_dndlnm(dndlnmh_hmf, mass_hmf, z_hmf, mass, z_vec)
        b_dm = interpolate2d_halobias(halobias_hbf, mass_hbf, z_hbf, mass, z_vec)
    return dn_dlnm, b_dm
    
# --------------------- #
#  satellite alignment  #
# --------------------- #

def load_growth_factor(block, z_vec):
    z_gwf = block['growth_parameters', 'z']
    D_gwf = block['growth_parameters', 'd_z']
    f_interp = interp1d(z_gwf, D_gwf)
    D_z = f_interp(z_vec)
    return D_z

def get_satellite_alignment(block, k_vec, mass, z_vec, suffix):
    # here I am assuming that the redshifts used in wkm_module and the pk_module match!
    #print( 'entering get_satellite_alignment..')
    wkm = np.empty([z_vec.size, mass.size, k_vec.size])
    for jz in range(0,z_vec.size):
        wkm_tmp = block['wkm_z%d'%jz + suffix,'w_km']
        k_wkm = block['wkm_z%d'%jz + suffix,'k_h']
        mass_wkm = block['wkm_z%d'%jz + suffix,'mass']
        #w_interp2d = interp2d(k_wkm, mass_wkm, wkm_tmp, bounds_error=False)#, fill_value=0)
        w_interp2d = RegularGridInterpolator((k_wkm.T, mass_wkm.T), wkm_tmp.T, bounds_error=False, fill_value=None)#, fill_value=0)
        #wkm_interpolated = w_interp2d((k_vec, mass))
        kk, mm = np.meshgrid(k_vec, mass, sparse=True)
        wkm_interpolated = w_interp2d((kk.T, mm.T)).T
        #print 'wkm_interp.shape = ', wkm_interpolated.shape
        wkm[jz] = wkm_interpolated
    #print( 'wkm.shape = ', wkm.shape)
    return wkm


# interpolation routines
def interpolate2d_dndlnm(dndlnmh_hmf, mass_hmf, z_hmf, mass, z_vec):
    #f_interp = interp2d(mass_hmf, z_hmf, dndlnmh_hmf)
    #hmf_interpolated = f_interp(mass, z_vec)
    f_interp = RegularGridInterpolator((mass_hmf.T, z_hmf.T), dndlnmh_hmf.T, bounds_error=False, fill_value=None)
    mm, zz = np.meshgrid(mass, z_vec, sparse=True)
    hmf_interpolated = f_interp((mm.T, zz.T)).T
    return hmf_interpolated
    
def interpolate2d_halobias(halobias_hbf, mass_hbf, z_hbf, mass, z_vec):
    #f_interp = interp2d(mass_hbf, z_hbf, halobias_hbf)
    #hbf_interpolated = f_interp(mass, z_vec)
    f_interp = RegularGridInterpolator((mass_hbf.T, z_hbf.T), halobias_hbf.T, bounds_error=False, fill_value=None)
    mm, zz = np.meshgrid(mass, z_vec, sparse=True)
    hbf_interpolated = f_interp((mm.T, zz.T)).T
    return hbf_interpolated
    
def interpolate1d_matter_power_lin(matter_power_lin, z_pl, z_vec):
    f_interp = interp1d(z_pl, matter_power_lin, axis=0)
    pk_interpolated = f_interp(z_vec)
    return pk_interpolated
    
    
# load the hod
def load_hods(block, section_name, pipeline, z_vec, mass):
    #section_name = 'hod' + suffix
    print (section_name)
    m_hod = block[section_name, 'mass']
    z_hod = block[section_name,  'z']
    Ncen_hod = block[section_name, 'n_cen']
    Nsat_hod = block[section_name, 'n_sat']
    numdencen_hod = block[section_name, 'number_density_cen']
    numdensat_hod = block[section_name, 'number_density_sat']
    f_c_hod = block[section_name, 'central_fraction']
    f_s_hod = block[section_name, 'satellite_fraction']
    #if pipeline == True:
    #    Ncen = Ncen_hod
    #    Nsat = Nsat_hod
    #    numdencen = numdencen_hod
    #    numdensat = numdensat_hod
    #    f_c = f_c_hod
    #    f_s = f_s_hod
    #else:
    #interp_Ncen = interp2d(m_hod, z_hod, Ncen_hod)
    #interp_Nsat = interp2d(m_hod, z_hod, Nsat_hod)
    interp_Ncen = RegularGridInterpolator((m_hod.T, z_hod.T), Ncen_hod.T, bounds_error=False, fill_value=None)
    interp_Nsat = RegularGridInterpolator((m_hod.T, z_hod.T), Nsat_hod.T, bounds_error=False, fill_value=None)
    # AD: Is extrapolation warranted here? Maybe make whole calculation on same grid/spacing/thingy!?
    interp_numdencen = interp1d(z_hod, numdencen_hod, fill_value='extrapolate')
    interp_numdensat = interp1d(z_hod, numdensat_hod, fill_value='extrapolate')
    interp_f_c = interp1d(z_hod, f_c_hod, fill_value='extrapolate')
    interp_f_s = interp1d(z_hod, f_s_hod, fill_value='extrapolate')
    #Ncen = interp_Ncen(mass, z_vec)
    #Nsat = interp_Nsat(mass, z_vec)
    mm, zz = np.meshgrid(mass, z_vec, sparse=True)
    Ncen = interp_Ncen((mm.T, zz.T)).T
    Nsat = interp_Nsat((mm.T, zz.T)).T
    #print ('z_hod', z_hod)
    #print ('z_vec', z_vec)
    numdencen = interp_numdencen(z_vec)
    numdensat = interp_numdensat(z_vec)
    f_c = interp_f_c(z_vec)
    f_s = interp_f_s(z_vec)
    
    return Ncen, Nsat, numdencen, numdensat, f_c, f_s
        
def load_galaxy_fractions(filename, z_vec):
    z_file, fraction_file = np.loadtxt(filename, unpack=True)
    if np.allclose(z_file, z_vec, atol=1e-3):
        return fraction_file
    else:
        print('The redshift of the input galaxy fractions do not match the ranges'
            'set in the pipeline. Performing interpolation.')
        gal_frac_interp = interp(z_vec, z_file, fraction_file)
        print( gal_frac_interp)
        return gal_frac_interp
    
    
# --------- COSMOSIS MODULE ----------- #

def setup(options):
    # This function is called once per processor per chain.
    # It is a chance to read any fixed options from the configuration file,
    # load any data, or do any calculations that are fixed once.
    

    log_mass_min = options[option_section, 'log_mass_min']
    log_mass_max = options[option_section, 'log_mass_max']
    nmass = options[option_section, 'nmass']
    # log-spaced mass in units of M_sun/h
    dlog10m = (log_mass_max-log_mass_min)/nmass
    mass = 10.0 ** np.arange(log_mass_min, log_mass_max, dlog10m)

    #nmass_bnl = options[option_section, 'nmass_bnl']
    #mass_bnl = np.logspace(log_mass_min, log_mass_max, nmass_bnl) 

    zmin = options[option_section, 'zmin']
    zmax = options[option_section, 'zmax']
    nz = options[option_section, 'nz']
    z_vec = np.linspace(zmin, zmax, nz)

    nk = options[option_section, 'nk']

    pipeline = options[option_section, 'pipeline']

    # pk_obs = options[option_section, 'pk_to_compute']
    p_mm = options.get_bool(option_section, 'p_mm', default=False)
    p_mm_bnl = options.get_bool(option_section, 'p_mm_bnl',default=False)
    p_gg = options.get_bool(option_section, 'p_gg',default=False)
    p_gg_bnl = options.get_bool(option_section, 'p_gg_bnl',default=False)
    p_gm = options.get_bool(option_section, 'p_gm',default=False)
    p_gm_bnl = options.get_bool(option_section, 'p_gm_bnl',default=False)
    p_gI = options.get_bool(option_section, 'p_gI',default=False)
    p_mI = options.get_bool(option_section, 'p_mI',default=False)
    p_II = options.get_bool(option_section, 'p_II',default=False)
    p_gI_bnl = options.get_bool(option_section, 'p_gI_bnl',default=False)
    p_mI_bnl = options.get_bool(option_section, 'p_mI_bnl',default=False)
    p_II_bnl = options.get_bool(option_section, 'p_II_bnl',default=False)
    p_gI_mc = options.get_bool(option_section, 'p_gI_mc',default=False)
    p_mI_mc = options.get_bool(option_section, 'p_mI_mc',default=False)
    p_II_mc = options.get_bool(option_section, 'p_II_mc',default=False)
    #interpolate_bnl = options.get_bool(option_section, 'interpolate_bnl',default=False)
    check_mead = options.has_value('hmf_and_halo_bias', 'use_mead2020_corrections')
    #use_mead = options['hmf_and_halo_bias', 'use_mead2020_corrections']
    #mead_version = options['camb', 'halofit_version']

    # initiate pipeline parameters
    ia_lum_dep_centrals = False
    ia_lum_dep_satellites = False
    gravitational = False
    galaxy = False
    bnl_gg = False
    bnl_gm = False
    bnl_mm = False
    bnl_ia = False
    alignment = False
    hod_section_name = ''
    two_halo_only = options[option_section, 'two_halo_only']
    f_red_cen_option = False

    if (p_gg == True) and (p_gg_bnl == True):
        print('Select either p_gg = True or p_gg_bnl = True, both compute the galaxy-galaxy power spectrum. p_gg_bnl includes beyond-linear halo bias in the galaxy power spectrum, p_gg does not.')
        sys.exit()

    if (p_gm == True) and (p_gm_bnl == True):
        print('Select either p_gm = True or p_gm_bnl = True, both compute the galaxy-matter power spectrum. p_gm_bnl includes beyond-linear halo bias in the galaxy power spectrum, p_gm does not.')
        sys.exit()
        
    if (p_mm == True) and (p_mm_bnl == True):
        print('Select either p_mm = True or p_mm_bnl = True, both compute the matter-matter power spectrum. p_mm_bnl includes beyond-linear halo bias in the matter power spectrum, p_mm does not.')
        sys.exit()
        
    if (p_mI == True) and (p_mI_bnl == True) and (p_mI_mc == True):
        print('Select either p_mI = True or p_mI_bnl = True or p_mI_mc = True, all compute the matter-intrinsic power spectrum. p_mI_bnl includes beyond-linear halo bias in the matter power spectrum, p_mI does not.')
        sys.exit()
        
    if (p_II == True) and (p_II_bnl == True) and (p_II_mc == True):
        print('Select either p_II = True or p_II_bnl = True or p_II_mc = True, all compute the matter-intrinsic power spectrum. p_II_bnl includes beyond-linear halo bias in the matter power spectrum, p_II does not.')
        sys.exit()
        
    if (p_gI == True) and (p_gI_bnl == True) and (p_gI_mc == True):
        print('Select either p_gI = True or p_gI_bnl = True or p_gI_mc = True, all compute the matter-intrinsic power spectrum. p_gI_bnl includes beyond-linear halo bias in the matter power spectrum, p_gI does not.')
        sys.exit()


    # Marika: what does the two halo only do? Do we need this?
    if (two_halo_only == True) and (p_mm == True) or (p_mm_bnl == True):
        gravitational = True
    elif (two_halo_only == False) and ((p_mm == True) or (p_mm_bnl == True) or (p_gm == True) or (p_mI == True) or (p_mI_bnl == True) or (p_gm_bnl == True)):
        gravitational = True
    if (p_gg == True) or (p_gm == True) or (p_gI == True) or (p_mI == True) or (p_II == True) or (p_gg_bnl == True) or (p_gm_bnl == True) or (p_gI_bnl == True) or (p_mI_bnl == True):
        galaxy = True
        hod_section_name = options[option_section, 'hod_section_name']
    if (p_gg_bnl == True):
        bnl_gg = True
    if (p_gm_bnl == True):
        bnl_gm = True
    if (p_mm_bnl == True):
        bnl_mm = True
    if (p_gI == True) or (p_mI == True) or (p_II == True) or (p_gI_bnl == True) or (p_mI_bnl == True) or (p_II_bnl == True) or (p_gI_mc == True) or (p_mI_mc == True) or (p_II_mc == True):
        alignment = True
    if (p_gI_bnl == True) or (p_mI_bnl == True) or (p_II_bnl == True):
        bnl_ia = True
        #IT commented. No longer used
        #ia_lum_dep_centrals = options[option_section, 'ia_luminosity_dependence_centrals']
        #ia_lum_dep_satellites = options[option_section, 'ia_luminosity_dependence_satellites']
        #f_red_cen_option = options[option_section, 'f_red_cen'] # TODO: consider to remove it (not particularly relevant anymore.. )
        #print('alignment is true')

    name = options.get_string(option_section, 'name', default='').lower()
    if name != '':
        suffix = '_' + name
    else:
        suffix = ''
        
    if check_mead:
        use_mead = options['hmf_and_halo_bias', 'use_mead2020_corrections']
        if use_mead == 'mead2020':
            mead_correction = 'nofeedback'
        elif use_mead == 'mead2020_feedback':
            mead_correction = 'feedback'
        elif use_mead == 'fit':
            mead_correction = 'fit'
    else:
        mead_correction = None

    # ============================================================================== #
    # this only makes sense in the context of the two halo only
    # f_red_cen = np.ones(nz)
    # if two_halo_only == True:
    #if f_red_cen_option:
    #    f_red_cen_file = options[option_section, 'f_red_cen_file']
    #    f_red_cen = load_galaxy_fractions(f_red_cen_file, z_vec)
    # if len(f_red_cen)!= nz:
    #    raise ValueError('The length of f_red_cen(z) does not match nz')
    #else:
    #    f_red_cen = np.ones(nz)
    #print(f_red_cen)
    # ============================================================================== #

    return mass, nmass, z_vec, nz, nk, p_mm, p_mm_bnl, p_gg, p_gg_bnl, p_gm, p_gm_bnl, p_gI, p_mI, p_II, p_gI_bnl, p_mI_bnl, p_II_bnl, p_gI_mc, p_mI_mc, p_II_mc, gravitational, galaxy, bnl_gg, bnl_gm, bnl_mm, bnl_ia, alignment, \
           ia_lum_dep_centrals, ia_lum_dep_satellites, two_halo_only, pipeline, hod_section_name, suffix, mead_correction


def execute(block, config):
    # This function is called every time you have a new sample of cosmological and other parameters.
    # It is the main workhorse of the code. The block contains the parameters and results of any
    # earlier modules, and the config is what we loaded earlier.

    mass, nmass, z_vec, nz, nk, p_mm, p_mm_bnl, p_gg, p_gg_bnl, p_gm, p_gm_bnl, p_gI, p_mI, p_II, p_gI_bnl, p_mI_bnl, p_II_bnl, p_gI_mc, p_mI_mc, p_II_mc, gravitational, galaxy, bnl_gg, bnl_gm, bnl_mm, bnl_ia, alignment, \
    ia_lum_dep_centrals, ia_lum_dep_satellites, two_halo_only, pipeline, hod_section_name, suffix, mead_correction = config

    start_time = time.time()

    mean_density0 = block['density', 'mean_density0']
    

    # Marika: Change this bit to read in k_vec and pk from the block directly. Get growth from camb
    # AD: If we can avoid interpolation, then yes. Looking at load_modules.py, we could leave them there to have more utility code separated. Could call them utilities. Dunno

    # load linear power spectrum
    k_vec_original, plin_original, growth_factor_original, scale_factor_original = get_linear_power_spectrum(block, z_vec)
    k_vec = np.logspace(np.log10(k_vec_original[0]), np.log10(k_vec_original[-1]), num=nk)
    
    # Marika: change this to avoid interpolation error.
    plin_k_interp = interp1d(k_vec_original, plin_original, axis=1, fill_value='extrapolate')
    plin = plin_k_interp(k_vec)
    growth_factor_interp = interp1d(k_vec_original, growth_factor_original, axis=1, fill_value='extrapolate')
    growth_factor = growth_factor_interp(k_vec)
    scale_factor_interp = interp1d(k_vec_original, scale_factor_original, axis=1, fill_value='extrapolate')
    scale_factor = scale_factor_interp(k_vec)

    # Marika: to here for now.
    #k_interp = interp1d(k_vec, plin, axis=1)
    #k_vec = np.logspace(np.log10(k_vec.min()), np.log10(k_vec.max()-1), nk)
    #plin = k_interp(k_vec)

    # load nonlinear power spectrum (halofit)
    k_nl, p_nl = get_nonlinear_power_spectrum(block, z_vec)
    plin_k_interp = interp1d(k_nl, p_nl, axis=1, fill_value='extrapolate')
    pnl = plin_k_interp(k_vec)
    block.replace_grid('matter_power_nl_mead', 'z', z_vec, 'k_h', k_vec, 'p_k', pnl)
    #block.replace_grid('matter_power_nl', 'z', z_vec, 'k_h', k_vec, 'p_k', pnl)
    
    # AD: avoid this! (Maybe needed for IA part ...)
    # compute the effective power spectrum, mixing the linear and nonlinear one:
    #
    # (1.-t_eff)*plin + t_eff*p_nl
    #
    t_eff = block['pk_parameters', 'trans_1hto2h']

    pk_eff = compute_effective_power_spectrum(k_vec, plin, k_nl, p_nl, z_vec, t_eff)

    # initialise the galaxy bias
    # bg = 1.0 # AD: ???
    
    # If the two_halo_only option is set True, then only the linear regime is computed and the linear bias is used (either computed by the
    # hod module or passed in the value	file (same structure as for the constant bias module)
    # Otherwise, compute the full power spectra (including the small scales)
    
    # load the halo mass and bias functions from the datablock
    dn_dlnm, b_dm = get_halo_functions(block, pipeline, mass, z_vec)
    # prepare a grid for the navarro-frenk-white profile
    u_dm, u_sat = pk_lib.compute_u_dm_grid(block, k_vec, mass, z_vec)
    
    #print(mean_density0)
    #mean_density0 = pk_lib.compute_mean_density(mass, dn_dlnm, block)
    #print(mean_density0)
    #quit()
    
    if two_halo_only == True:
        # preparing the integrals:
        if gravitational == True:
            A_term = pk_lib.prepare_A_term(mass, u_dm, b_dm, dn_dlnm, mean_density0, nz, nk)
            I_m_term = pk_lib.prepare_Im_term(mass, u_dm, b_dm, dn_dlnm, mean_density0, nz, nk, A_term)
            m_factor = pk_lib.prepare_matter_factor_grid(mass, mean_density0, u_dm)
        if galaxy == True:
            # load linear bias:
            bg = block['galaxy_bias' + suffix, 'b']
            if np.isscalar(bg): bg *= np.ones(nz)
        if alignment == True:
	    #IT commented ia_lum_dep_centrals
            alignment_amplitude_2h, alignment_amplitude_2h_II = pk_lib.compute_two_halo_alignment(block, suffix, nz, nk,
                                                                                           growth_factor, mean_density0
                                                                                           )#ia_lum_dep_centrals)
        # compute the power spectra
        if p_mm:
            # this is not very useful as for the lensing power spectrum it is usually used halofit
            raise ValueError('pmm not implemented for the two-halo only option\n')
            # AD: Implement proper matter-matter term using the halo model here. We do not want to use halofit.
            # compute_p_mm_new(block, k_vec, pk_eff, z_vec, mass, dn_dlnm, m_factor, I_m_term, nz, nk)
        if p_gg:
            pk_gg = pk_lib.compute_p_gg_two_halo(block, k_vec, pk_eff, z_vec, bg)
            block.put_grid('galaxy_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gg)
        if p_gm:
            pk_gm = pk_lib.compute_p_gm_two_halo(block, k_vec, pk_eff, z_vec, bg)
            block.put_grid('matter_galaxy_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gm)
        if p_mI:
            #print('pGI...')
            pk_mI = pk_lib.compute_p_mI_two_halo(block, k_vec, pk_eff, z_vec, nz, f_red_cen, alignment_amplitude_2h)
            block.put_grid('matter_intrinsic_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mI)
        if p_gI:
            pk_gI = pk_lib.compute_p_gI_two_halo(block, k_vec, pk_eff, z_vec, nz, f_red_cen, alignment_amplitude_2h, bg)
            block.put_grid('galaxy_intrinsic_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gI)
        if p_II:
            #print('pII...')
            pk_II = pk_lib.compute_p_II_two_halo(block, k_vec, pk_eff, z_vec, nz, f_red_cen, alignment_amplitude_2h_II)
            block.put_grid('intrinsic_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II)

    else:
        A_term = pk_lib.prepare_A_term(mass, u_dm, b_dm, dn_dlnm, mean_density0, nz, nk)

        if (bnl_gg == True) or (bnl_gm == True) or (bnl_mm == True) or (bnl_ia == True):
            beta_interp = block.get_double_array_nd('bnl', 'beta_interp')
            if beta_interp is False:
                raise ValueError('Non-linear halo bias module bnl not initialised!\n')

            
        # prepare the integrals
        if gravitational == True:
            # the matter integral and factor
            I_m_term = pk_lib.prepare_Im_term(mass, u_dm, b_dm, dn_dlnm, mean_density0, nz, nk, A_term)
            m_factor = pk_lib.prepare_matter_factor_grid(mass, mean_density0, u_dm, block)
            if mead_correction == 'feedback':
                m_factor_1h = pk_lib.prepare_matter_factor_grid_baryon(mass, mean_density0, u_dm, z_vec, block)
            elif mead_correction == 'fit':
                m_factor_1h = m_factor.copy() # for now does nothing!
                #m_factor_1h = pk_lib.prepare_matter_factor_grid_baryon_fit(mass, mean_density0, u_dm, z_vec, block)
            else:
                m_factor_1h = m_factor.copy()
                
            if bnl_mm == True:
                I_NL_mm = pk_lib.prepare_I_NL(mass, mass, m_factor, m_factor, b_dm, b_dm, dn_dlnm, dn_dlnm, nz, nk, k_vec, z_vec, A_term, mean_density0, beta_interp)
                
        if (galaxy == True) or (alignment == True):
            #print(hod_section_name)
            Ncen, Nsat, numdencen, numdensat, f_cen, f_sat = load_hods(block, hod_section_name, pipeline, z_vec, mass)
            if galaxy == True:
                # preparing the 1h term
                c_factor = pk_lib.prepare_central_factor_grid(Ncen, numdencen, f_cen)
                s_factor = pk_lib.prepare_satellite_factor_grid(Nsat, numdensat, f_sat, u_sat, nz, nk, nmass)
                # preparing the 2h term
                I_c_term = pk_lib.prepare_Ic_term(mass, c_factor, b_dm, dn_dlnm, nz, nk)
                I_s_term = pk_lib.prepare_Is_term(mass, s_factor, b_dm, dn_dlnm, nz, nk)
            
                if bnl_gg == True:
                    start = time.time()
                    I_NL_cs = pk_lib.prepare_I_NL(mass, mass, c_factor, s_factor, b_dm, b_dm, dn_dlnm, dn_dlnm, nz, nk, k_vec, z_vec, A_term, mean_density0, beta_interp)
                    end = time.time()
                    print('time I_NL_cs: ', end - start)
                    I_NL_ss = pk_lib.prepare_I_NL(mass, mass, s_factor, s_factor, b_dm, b_dm, dn_dlnm, dn_dlnm, nz, nk, k_vec, z_vec, A_term, mean_density0, beta_interp)
                    I_NL_cc = pk_lib.prepare_I_NL(mass, mass, c_factor, c_factor, b_dm, b_dm, dn_dlnm, dn_dlnm, nz, nk, k_vec, z_vec, A_term, mean_density0, beta_interp)

                if bnl_gm == True:
                    I_NL_cm = pk_lib.prepare_I_NL(mass, mass, c_factor, m_factor, b_dm, b_dm, dn_dlnm, dn_dlnm, nz, nk, k_vec, z_vec, A_term, mean_density0, beta_interp)
                    I_NL_sm = pk_lib.prepare_I_NL(mass, mass, s_factor, m_factor, b_dm, b_dm, dn_dlnm, dn_dlnm, nz, nk, k_vec, z_vec, A_term, mean_density0, beta_interp)
                    
            if alignment == True:
                #IT commenting ia_lum_dep_centrals
                # AD: Will probably be removed after some point when we get all the Bnl terms for IA added!
                alignment_amplitude_2h, alignment_amplitude_2h_II, C1 = pk_lib.compute_two_halo_alignment(block, suffix, nz, nk,
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
                #    sat_l_term = block['ia_small_scale_alignment' + suffix, 'alignment_1h']
                #    gamma_1h *= sat_l_term
                # load the satellite density run w(k|m) for a perfect 3d radial alignment projected along the line of sight
                # it can either be constant or radial dependent -> this is computed in the wkm module, including the amplitude of the
                # signal (but not its luminosity dependence, which is a separate factor, see above)
                wkm = get_satellite_alignment(block, k_vec, mass, z_vec, suffix)
                # preparing the central and satellite terms
                s_align_factor = pk_lib.prepare_satellite_alignment_factor_grid(mass, Nsat, numdensat, f_sat, wkm, gamma_1h, nz,
                                                                         nk, nmass)
                #if block['ia_large_scale_alignment' + suffix, 'instance'] == 'halo_mass':
                #    c_align_factor = pk_lib.prepare_central_alignment_factor_grid_halo(mass, scale_factor, growth_factor, f_cen, C1, nz,
                #                                                         nk, nmass, block['ia_large_scale_alignment' + suffix, 'beta_mh'], block['ia_large_scale_alignment' + suffix, 'M_0'])
                #else:
                c_align_factor = pk_lib.prepare_central_alignment_factor_grid(mass, scale_factor, growth_factor, f_cen, C1, nz,
                                                                         nk, nmass)

                I_c_align_term = pk_lib.prepare_Ic_align_term(mass, c_align_factor, b_dm, dn_dlnm, nz, nk, mean_density0, A_term)
                I_s_align_term = pk_lib.prepare_Is_align_term(mass, s_align_factor, b_dm, dn_dlnm, nz, nk, mean_density0, A_term)
                if bnl_ia == True:
                    if p_mI_bnl == True:
                        I_NL_ia_cm = pk_lib.prepare_I_NL(mass, mass, c_align_factor, m_factor, b_dm, b_dm, dn_dlnm, dn_dlnm, nz, nk, k_vec, z_vec, A_term, mean_density0, beta_interp)
                        I_NL_ia_sm = pk_lib.prepare_I_NL(mass, mass, s_align_factor, m_factor, b_dm, b_dm, dn_dlnm, dn_dlnm, nz, nk, k_vec, z_vec, A_term, mean_density0, beta_interp)
                    
                    if p_II_bnl == True:
                        I_NL_ia_cc = pk_lib.prepare_I_NL(mass, mass, c_align_factor, c_align_factor, b_dm, b_dm, dn_dlnm, dn_dlnm, nz, nk, k_vec, z_vec, A_term, mean_density0, beta_interp)
                        I_NL_ia_cs = pk_lib.prepare_I_NL(mass, mass, c_align_factor, s_align_factor, b_dm, b_dm, dn_dlnm, dn_dlnm, nz, nk, k_vec, z_vec, A_term, mean_density0, beta_interp)
                        I_NL_ia_ss = pk_lib.prepare_I_NL(mass, mass, s_align_factor, s_align_factor, b_dm, b_dm, dn_dlnm, dn_dlnm, nz, nk, k_vec, z_vec, A_term, mean_density0, beta_interp)
                        
                    if p_gI_bnl == True:
                        I_NL_ia_gc = pk_lib.prepare_I_NL(mass, mass, c_align_factor, c_factor, b_dm, b_dm, dn_dlnm, dn_dlnm, nz, nk, k_vec, z_vec, A_term, mean_density0, beta_interp)
                        I_NL_ia_gs = pk_lib.prepare_I_NL(mass, mass, s_align_factor, s_factor, b_dm, b_dm, dn_dlnm, dn_dlnm, nz, nk, k_vec, z_vec, A_term, mean_density0, beta_interp)
                
        # compute the power spectra
        if p_mm == True:
            if mead_correction == 'nofeedback':
                pk_mm_1h, pk_mm_2h, pk_mm_tot = pk_lib.compute_p_mm_mead(block, k_vec, plin, z_vec, mass, dn_dlnm, m_factor_1h,
                                                             I_m_term, nz, nk)
            else:
                pk_mm_1h, pk_mm_2h, pk_mm_tot = pk_lib.compute_p_mm(block, k_vec, plin, z_vec, mass, dn_dlnm, m_factor_1h,
                                                             I_m_term, nz, nk)
            # save in the datablock
            #block.put_grid('matter_1h_power', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm_1h)
            #block.put_grid('matter_2h_power', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm_2h)
            #block.put_grid('matter_power', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm_tot)
            # If you want to use this power spectrum to do cosmic shear, replace the lines above with the following:
            #block.replace_grid('matter_power_nl', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm_tot)
            
        if p_mm_bnl == True:
            pk_mm_1h_bnl, pk_mm_2h_bnl, pk_mm_tot_bnl = pk_lib.compute_p_mm_bnl(block, k_vec, plin, z_vec, mass, dn_dlnm, m_factor_1h,
                                                             I_m_term, nz, nk, I_NL_mm)
            # save in the datablock
            #block.put_grid('matter_1h_power', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm_1h)
            #block.put_grid('matter_2h_power', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm_2h)
            #block.put_grid('matter_power', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm_tot)
            # If you want to use this power spectrum to do cosmic shear, replace the lines above with the following:
            block.replace_grid('matter_power_nl', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm_tot_bnl)

        if p_gg == True:
            pk_gg_1h, pk_gg_2h, pk_gg, bg_halo_model = pk_lib.compute_p_gg(block, k_vec, plin, z_vec, mass, dn_dlnm, c_factor,
                                                                    s_factor, I_c_term, I_s_term, nz, nk)
            #block.put_grid('galaxy_power_1h' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gg_1h)
            #block.put_grid('galaxy_power_2h' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gg_2h)
            block.put_grid('galaxy_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gg)
            #block.put_grid('galaxy_linear_bias' + suffix, 'z', z_vec, 'k_h', k_vec, 'galaxybiastotal', bg_halo_model)

        if p_gg_bnl == True:
            print('beyond-linear halo bias selected')
            print('plin_shape: ', plin.shape)
            pk_gg_1h_bnl, pk_gg_2h_bnl, pk_gg_bnl, bg_halo_model_bnl = pk_lib.compute_p_gg_bnl(block, k_vec, plin, z_vec, mass, dn_dlnm, c_factor,
                                                                    s_factor, I_c_term, I_s_term, nz, nk, I_NL_cs, I_NL_cc, I_NL_ss)
            block.put_grid('galaxy_power_1h' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gg_1h_bnl)
            block.put_grid('galaxy_power_2h' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gg_2h_bnl)
            block.put_grid('galaxy_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gg_bnl)
            #block.put_grid('galaxy_linear_bias' + suffix, 'z', z_vec, 'k_h', k_vec, 'galaxybiastotal', bg_halo_model)

        if p_gm == True:
            #print('computing p_gm...')
            #IT Replacing pk_eff by plin
            pk_1h, pk_2h, pk_tot = pk_lib.compute_p_gm(block, k_vec, plin, z_vec, mass, dn_dlnm, c_factor, s_factor, m_factor_1h, I_c_term, I_s_term,
                          I_m_term)
            #block.put_grid('matter_galaxy_power_1h', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_1h)
            #block.put_grid('matter_galaxy_power_2h', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_2h)
            #IT Adding suffix to matter_galaxy_power
            block.put_grid('matter_galaxy_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_tot)
        
        if p_gm_bnl == True:
            print('computing p_gm with beyond-linear bias...')
            #IT Replacing pk_eff by plin
            pk_1h_bnl, pk_2h_bnl, pk_tot_bnl = pk_lib.compute_p_gm_bnl(block, k_vec, plin, z_vec, mass, dn_dlnm, c_factor, s_factor, m_factor_1h, I_c_term, I_s_term,
                          I_m_term, I_NL_cm, I_NL_sm)
            #block.put_grid('matter_galaxy_power_1h', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_1h)
            #block.put_grid('matter_galaxy_power_2h', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_2h)
            #IT Adding suffix to matter_galaxy_power
            block.put_grid('matter_galaxy_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_tot_bnl)

    
        # Intrinsic aligment power spectra (full halo model calculation)
        if p_II == True:
            pk_II_1h, pk_II_2h, pk_II = pk_lib.compute_p_II(block, k_vec, plin, z_vec, mass, dn_dlnm, c_align_factor, s_align_factor, I_c_align_term, I_s_align_term, nz, nk)
            #block.put_grid('intrinsic_power_1h' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II_1h)
            #block.put_grid('intrinsic_power_2h' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II_2h)
            block.put_grid('intrinsic_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II)
        if p_gI == True:
            #print('computing p_gI...')
            pk_gI_1h, pk_gI_2h, pk_gI = pk_lib.compute_p_gI(block, k_vec, plin, z_vec, mass, dn_dlnm, c_factor, c_align_factor, s_align_factor, I_c_term, I_c_align_term, I_s_align_term, nz, nk)
            #IT Added galaxy_intrinsic_power to datablock
            block.put_grid('galaxy_intrinsic_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gI)
        if p_mI == True:
            #print('computing p_GI...')
            # compute_p_GI(block, k_vec, pk_eff, z_vec, mass, dn_dlnm, m_factor, s_align_factor, alignment_amplitude_2h, nz, nk, f_red_cen)
            pk_mI_1h, pk_mI_2h, pk_mI = pk_lib.compute_p_mI(block, k_vec, plin, z_vec, mass, dn_dlnm, m_factor_1h,
                                                         c_align_factor, s_align_factor, I_m_term, I_c_align_term, I_s_align_term, nz, nk)
            #block.put_grid('matter_intrinsic_power_1h' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_GI_1h)
            #block.put_grid('matter_intrinsic_power_2h' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_GI_2h)
            block.put_grid('matter_intrinsic_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mI)
            
            
        # Intrinsic aligment power spectra (full halo model calculation)
        if p_II_bnl == True:
            pk_II_1h_bnl, pk_II_2h_bnl, pk_II_bnl = pk_lib.compute_p_II_bnl(block, k_vec, plin, z_vec, mass, dn_dlnm, c_align_factor, s_align_factor, I_c_align_term, I_s_align_term, nz, nk, I_NL_ia_cc, I_NL_ia_cs, I_NL_ia_ss)
            #block.put_grid('intrinsic_power_1h' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II_1h)
            #block.put_grid('intrinsic_power_2h' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II_2h)
            block.put_grid('intrinsic_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II_bnl)
        if p_gI_bnl == True:
            #print('computing p_gI...')
            pk_gI_1h_bnl, pk_gI_2h_bnl, pk_gI_bnl = pk_lib.compute_p_gI_bnl(block, k_vec, plin, z_vec, mass, dn_dlnm, c_factor, c_align_factor, s_align_factor, I_c_term, I_c_align_term, I_s_align_term, nz, nk, I_NL_ia_gc, I_NL_ia_gs)
            #IT Added galaxy_intrinsic_power to datablock
            block.put_grid('galaxy_intrinsic_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gI_bnl)
        if p_mI_bnl == True:
            #print('computing p_GI...')
            # compute_p_GI(block, k_vec, pk_eff, z_vec, mass, dn_dlnm, m_factor, s_align_factor, alignment_amplitude_2h, nz, nk, f_red_cen)
            pk_mI_1h_bnl, pk_mI_2h_bnl, pk_mI_bnl = pk_lib.compute_p_mI_bnl(block, k_vec, plin, z_vec, mass, dn_dlnm, m_factor_1h,
                                                         c_align_factor, s_align_factor, I_m_term, I_c_align_term, I_s_align_term, nz, nk, I_NL_ia_cm, I_NL_ia_sm)
            #block.put_grid('matter_intrinsic_power_1h' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_GI_1h)
            #block.put_grid('matter_intrinsic_power_2h' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_GI_2h)
            block.put_grid('matter_intrinsic_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mI_bnl)
            
            
        # Intrinsic aligment power spectra (implementation from Maria Cristina - 2h = LA/NLA mixture)
        if p_II_mc == True:
            pk_II_1h, pk_II_2h, pk_II = pk_lib.compute_p_II_mc(block, k_vec, pk_eff, z_vec, mass, dn_dlnm, s_align_factor, alignment_amplitude_2h_II, nz, nk, f_cen)
            #block.put_grid('intrinsic_power_1h' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II_1h)
            #block.put_grid('intrinsic_power_2h' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II_2h)
            block.put_grid('intrinsic_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II)
        if p_gI_mc == True:
            #print('computing p_gI...')
            pk_gI_1h, pk_gI_2h, pk_gI = pk_lib.compute_p_gI_mc(block, k_vec, pk_eff, z_vec, mass, dn_dlnm, c_factor, s_align_factor, I_c_term, alignment_amplitude_2h, nz, nk)
            #IT Added galaxy_intrinsic_power to datablock
            block.put_grid('galaxy_intrinsic_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gI)
        if p_mI_mc == True:
            #print('computing p_GI...')
            # compute_p_GI(block, k_vec, pk_eff, z_vec, mass, dn_dlnm, m_factor, s_align_factor, alignment_amplitude_2h, nz, nk, f_red_cen)
            pk_mI_1h, pk_mI_2h, pk_mI = pk_lib.compute_p_mI_mc(block, k_vec, pk_eff, z_vec, mass, dn_dlnm, m_factor_1h, s_align_factor, alignment_amplitude_2h, nz, nk, f_cen)
            #block.put_grid('matter_intrinsic_power_1h' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_GI_1h)
            #block.put_grid('matter_intrinsic_power_2h' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_GI_2h)
            block.put_grid('matter_intrinsic_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mI)

    return 0


def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass


