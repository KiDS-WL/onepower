# A new power spectrum module

# NOTE: no truncation (halo exclusion problem) applied!

from cosmosis.datablock import names, option_section
import sys
import numpy as np
from scipy.interpolate import interp1d, interp2d, RegularGridInterpolator
from collections import OrderedDict

import math
import pk_lib

import time

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
    
    
def get_halo_functions(block, mass, z_vec):
    
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
        wkm_tmp = block['wkm','w_km_%d'%jz + suffix]
        k_wkm = block['wkm','k_h_%d'%jz+suffix]
        mass_wkm = block['wkm','mass_%d'%jz+suffix]
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
    
def load_fstar_mm(block, section_name, z_vec, mass):
    m_hod = block[section_name, 'mass']
    z_hod = block[section_name,  'z']
    f_star = block[section_name, 'f_star']
    interp_fstar = RegularGridInterpolator((m_hod.T, z_hod.T), f_star.T, bounds_error=False, fill_value=None)
    mm, zz = np.meshgrid(mass, z_vec, sparse=True)
    fstar = interp_fstar((mm.T, zz.T)).T
    
    return fstar

# load the hod
def load_hods(block, section_name, z_vec, mass):
    
    m_hod = block[section_name, 'mass']
    z_hod = block[section_name,  'z']
    Ncen_hod = block[section_name, 'n_cen']
    Nsat_hod = block[section_name, 'n_sat']
    numdencen_hod = block[section_name, 'number_density_cen']
    numdensat_hod = block[section_name, 'number_density_sat']
    f_c_hod = block[section_name, 'central_fraction']
    f_s_hod = block[section_name, 'satellite_fraction']
    mass_avg_hod = block[section_name, 'average_halo_mass']
    f_star = block[section_name, 'f_star']
    
    #interp_Ncen = interp2d(m_hod, z_hod, Ncen_hod)
    #interp_Nsat = interp2d(m_hod, z_hod, Nsat_hod)
    interp_Ncen = RegularGridInterpolator((m_hod.T, z_hod.T), Ncen_hod.T, bounds_error=False, fill_value=0.0)
    interp_Nsat = RegularGridInterpolator((m_hod.T, z_hod.T), Nsat_hod.T, bounds_error=False, fill_value=0.0)
    interp_fstar = RegularGridInterpolator((m_hod.T, z_hod.T), f_star.T, bounds_error=False, fill_value=0.0)
    # AD: Is extrapolation warranted here? Maybe make whole calculation on same grid/spacing/thingy!?
    interp_numdencen = interp1d(z_hod, numdencen_hod, fill_value='extrapolate', bounds_error=False)
    interp_numdensat = interp1d(z_hod, numdensat_hod, fill_value='extrapolate', bounds_error=False)
    interp_f_c = interp1d(z_hod, f_c_hod, fill_value=0.0, bounds_error=False)
    interp_f_s = interp1d(z_hod, f_s_hod, fill_value=0.0, bounds_error=False)
    interp_mass_avg = interp1d(z_hod, mass_avg_hod, fill_value=0.0, bounds_error=False)
    #Ncen = interp_Ncen(mass, z_vec)
    #Nsat = interp_Nsat(mass, z_vec)
    mm, zz = np.meshgrid(mass, z_vec, sparse=True)
    Ncen = interp_Ncen((mm.T, zz.T)).T
    Nsat = interp_Nsat((mm.T, zz.T)).T
    fstar = interp_fstar((mm.T, zz.T)).T
    #print ('z_hod', z_hod)
    #print ('z_vec', z_vec)
    numdencen = interp_numdencen(z_vec)
    numdensat = interp_numdensat(z_vec)
    f_c = interp_f_c(z_vec)
    f_s = interp_f_s(z_vec)
    mass_avg = interp_mass_avg(z_vec)
    
    return Ncen, Nsat, numdencen, numdensat, f_c, f_s, mass_avg, fstar
        
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

    p_mm = options.get_bool(option_section, 'p_mm', default=False)
    p_gg = options.get_bool(option_section, 'p_gg',default=False)
    p_gm = options.get_bool(option_section, 'p_gm',default=False)
    p_gI = options.get_bool(option_section, 'p_gI',default=False)
    p_mI = options.get_bool(option_section, 'p_mI',default=False)
    p_II = options.get_bool(option_section, 'p_II',default=False)
    p_gI_mc = options.get_bool(option_section, 'p_gI_mc',default=False)
    p_mI_mc = options.get_bool(option_section, 'p_mI_mc',default=False)
    p_II_mc = options.get_bool(option_section, 'p_II_mc',default=False)
    bnl = options.get_bool(option_section, 'bnl', default=False)
    #interpolate_bnl = options.get_bool(option_section, 'interpolate_bnl',default=False)
    check_mead = options.has_value('hmf_and_halo_bias', 'use_mead2020_corrections')
    poisson_type = options.get_string(option_section, 'poisson_type',default='')
    point_mass = options.get_bool(option_section, 'point_mass',default=False)

    # initiate pipeline parameters
    ia_lum_dep_centrals = False
    ia_lum_dep_satellites = False
    gravitational = False
    galaxy = False
    alignment = False
    hod_section_name = ''
    two_halo_only = options[option_section, 'two_halo_only']
    f_red_cen_option = False

    
    if (p_mI == True) and (p_mI_mc == True):
        print('Select either p_mI = True or p_mI_mc = True, both compute the matter-intrinsic power spectrum. p_mI_mc is the implementation used in Fortuna et al. 2020 paper.')
        sys.exit()
        
    if (p_II == True) and (p_II_mc == True):
        print('Select either p_II = True or p_II_mc = True, all compute the matter-intrinsic power spectrum. p_II_mc is the implementation used in Fortuna et al. 2020 paper.')
        sys.exit()
        
    if (p_gI == True) and (p_gI_mc == True):
        print('Select either p_gI = True or p_gI_mc = True, all compute the matter-intrinsic power spectrum. p_gI_mc i is the implementation used in Fortuna et al. 2020 paper.')
        sys.exit()


    # Marika: what does the two halo only do? Do we need this?
    if (two_halo_only == True) and (p_mm == True):
        gravitational = True
    elif (two_halo_only == False) and ((p_mm == True) or (p_gm == True) or (p_mI == True)):
        gravitational = True
    if (p_gg == True) or (p_gm == True) or (p_gI == True) or (p_mI == True) or (p_II == True) or (p_gI_mc == True) or (p_mI_mc == True) or (p_II_mc == True):
        galaxy = True
        hod_section_name = options[option_section, 'hod_section_name']
    if (p_gI == True) or (p_mI == True) or (p_II == True) or (p_gI_mc == True) or (p_mI_mc == True) or (p_II_mc == True):
        alignment = True

    name = options.get_string(option_section, 'output_suffix', default='').lower()
    if name != '':
        suffix0 = '_' + name
    else:
        suffix0 = ''
        
    if check_mead:
        use_mead = options['hmf_and_halo_bias', 'use_mead2020_corrections']
        if use_mead == 'mead2020':
            mead_correction = 'nofeedback'
        elif use_mead == 'mead2020_feedback':
            mead_correction = 'feedback'
        elif use_mead == 'fit_feedback':
            mead_correction = 'fit'
            if not options.has_value(option_section, 'hod_section_name'):
                print('To use the fit option for feedback that links HOD derived stellar mass fraction to the baryon feedback one needs to provide the hod section name of used hod!')
                sys.exit()
            hod_section_name = options[option_section, 'hod_section_name']
    else:
        mead_correction = None


    return mass, nmass, z_vec, nz, nk, p_mm, p_gg, p_gm, p_gI, p_mI, p_II, p_gI_mc, p_mI_mc, p_II_mc, gravitational, galaxy, bnl, alignment, \
           ia_lum_dep_centrals, ia_lum_dep_satellites, two_halo_only, hod_section_name, suffix0, mead_correction, point_mass, poisson_type


def execute(block, config):
    # This function is called every time you have a new sample of cosmological and other parameters.
    # It is the main workhorse of the code. The block contains the parameters and results of any
    # earlier modules, and the config is what we loaded earlier.

    mass, nmass, z_vec, nz, nk, p_mm, p_gg, p_gm, p_gI, p_mI, p_II, p_gI_mc, p_mI_mc, p_II_mc, gravitational, galaxy, bnl, alignment, \
    ia_lum_dep_centrals, ia_lum_dep_satellites, two_halo_only, hod_section_name0, suffix0, mead_correction, point_mass, poisson_type = config


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
    dn_dlnm, b_dm = get_halo_functions(block, mass, z_vec)
    # prepare a grid for the navarro-frenk-white profile
    u_dm, u_sat = pk_lib.compute_u_dm_grid(block, k_vec, mass, z_vec)
    

    if two_halo_only == True:
        # preparing the integrals:
        if gravitational == True:
            A_term = pk_lib.prepare_A_term(mass, u_dm, b_dm, dn_dlnm, mean_density0)
            I_m_term = pk_lib.prepare_Im_term(mass, u_dm, b_dm, dn_dlnm, mean_density0, A_term)
            m_factor = pk_lib.prepare_matter_factor_grid(mass, mean_density0, u_dm)
        if (galaxy == True) or (alignment == True):
            hod_bins = block[hod_section_name0 + '_metadata', 'nbins']
            for nb in range(0,hod_bins):
                if hod_bins != 1:
                    hod_section_name = hod_section_name0 + '_{}'.format(nb+1)
                    suffix = suffix0 + '_{}'.format(nb+1)
                else:
                    hod_section_name = hod_section_name0
                    suffix = suffix0
                if galaxy == True:
                    # load linear bias:
                    bg = block['galaxy_bias' + suffix, 'b']
                    if np.isscalar(bg): bg *= np.ones(nz)
                if alignment == True:
                #IT commented ia_lum_dep_centrals
                    alignment_amplitude_2h, alignment_amplitude_2h_II = pk_lib.compute_two_halo_alignment(block, suffix, growth_factor, mean_density0)
                # compute the power spectra
                if p_gg:
                    pk_gg = pk_lib.compute_p_gg_two_halo(block, k_vec, pk_eff, z_vec, bg)
                    block.put_grid('galaxy_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gg)
                if p_gm:
                    pk_gm = pk_lib.compute_p_gm_two_halo(block, k_vec, pk_eff, z_vec, bg)
                    block.put_grid('matter_galaxy_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gm)
                if p_mI:
                    #print('pGI...')
                    pk_mI = pk_lib.compute_p_mI_two_halo(block, k_vec, pk_eff, z_vec, f_red_cen, alignment_amplitude_2h)
                    block.put_grid('matter_intrinsic_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mI)
                if p_gI:
                    pk_gI = pk_lib.compute_p_gI_two_halo(block, k_vec, pk_eff, z_vec, f_red_cen, alignment_amplitude_2h, bg)
                    block.put_grid('galaxy_intrinsic_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gI)
                if p_II:
                    #print('pII...')
                    pk_II = pk_lib.compute_p_II_two_halo(block, k_vec, pk_eff, z_vec, f_red_cen, alignment_amplitude_2h_II)
                    block.put_grid('intrinsic_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II)
        if p_mm:
            # this is not very useful as for the lensing power spectrum it is usually used halofit
            raise ValueError('pmm not implemented for the two-halo only option\n')
            # AD: Implement proper matter-matter term using the halo model here. We do not want to use halofit.
            # compute_p_mm_new(block, k_vec, pk_eff, z_vec, mass, dn_dlnm, m_factor, I_m_term, nz, nk)
    else:
        A_term = pk_lib.prepare_A_term(mass, u_dm, b_dm, dn_dlnm, mean_density0)

        if bnl == True:
            beta_interp = block.get_double_array_nd('bnl', 'beta_interp')
            if beta_interp.shape == np.array([0.0]).shape:
                raise ValueError('Non-linear halo bias module bnl not initialised!\n')

            
        # prepare the integrals
        if gravitational == True:
            # the matter integral and factor
            I_m_term = pk_lib.prepare_Im_term(mass, u_dm, b_dm, dn_dlnm, mean_density0, A_term)
            m_factor = pk_lib.prepare_matter_factor_grid(mass, mean_density0, u_dm, block)
            if mead_correction == 'feedback':
                m_factor_1h_mm = pk_lib.prepare_matter_factor_grid_baryon(mass, mean_density0, u_dm, z_vec, block)
            elif mead_correction == 'fit':
                fstar = load_fstar_mm(block, hod_section_name0 + '_metadata', z_vec, mass)
                m_factor_1h_mm = pk_lib.prepare_matter_factor_grid_baryon_fit(mass, mean_density0, u_dm, z_vec, fstar, block)
            else:
                m_factor_1h_mm = m_factor.copy()
                
            if bnl == True:
                I_NL_mm = pk_lib.prepare_I_NL(mass, mass, m_factor, m_factor, b_dm, b_dm, dn_dlnm, dn_dlnm, k_vec, z_vec, A_term, mean_density0, beta_interp)
                
        if (galaxy == True) or (alignment == True):
            hod_bins = block[hod_section_name0 + '_metadata', 'nbins']
            
            for nb in range(0,hod_bins):
                if hod_bins != 1:
                    hod_section_name = hod_section_name0 + '_{}'.format(nb+1)
                    suffix = suffix0 + '_{}'.format(nb+1)
                else:
                    hod_section_name = hod_section_name0
                    suffix = suffix0
                    
                Ncen, Nsat, numdencen, numdensat, f_cen, f_sat, mass_avg, fstar = load_hods(block, hod_section_name, z_vec, mass)
            
                if galaxy == True:
                    # preparing the 1h term
                    c_factor = pk_lib.prepare_central_factor_grid(Ncen, numdencen, f_cen)
                    s_factor = pk_lib.prepare_satellite_factor_grid(Nsat, numdensat, f_sat, u_sat)
                    # preparing the 2h term
                    I_c_term = pk_lib.prepare_Ic_term(mass, c_factor, b_dm, dn_dlnm, nz, nk)
                    I_s_term = pk_lib.prepare_Is_term(mass, s_factor, b_dm, dn_dlnm)
                    
                    if mead_correction == 'fit' or point_mass == True:
                        # Include point mass and gas contribution to the GGL power spectrum, defined from HOD
                        # Maybe extend to input the mass per bin!
                        m_factor_1h = pk_lib.prepare_matter_factor_grid_baryon_fit(mass, mean_density0, u_dm, z_vec, fstar, block)
                    else:
                        m_factor_1h = m_factor_1h_mm.copy()
                        
                    if bnl == True:
                        if p_gg == True:
                            I_NL_cs = pk_lib.prepare_I_NL(mass, mass, c_factor, s_factor, b_dm, b_dm, dn_dlnm, dn_dlnm, k_vec, z_vec, A_term, mean_density0, beta_interp)
                            I_NL_ss = pk_lib.prepare_I_NL(mass, mass, s_factor, s_factor, b_dm, b_dm, dn_dlnm, dn_dlnm, k_vec, z_vec, A_term, mean_density0, beta_interp)
                            I_NL_cc = pk_lib.prepare_I_NL(mass, mass, c_factor, c_factor, b_dm, b_dm, dn_dlnm, dn_dlnm, k_vec, z_vec, A_term, mean_density0, beta_interp)
    
                        if p_gm == True:
                            I_NL_cm = pk_lib.prepare_I_NL(mass, mass, c_factor, m_factor, b_dm, b_dm, dn_dlnm, dn_dlnm, k_vec, z_vec, A_term, mean_density0, beta_interp)
                            I_NL_sm = pk_lib.prepare_I_NL(mass, mass, s_factor, m_factor, b_dm, b_dm, dn_dlnm, dn_dlnm, k_vec, z_vec, A_term, mean_density0, beta_interp)
                    
                if alignment == True:
                    #IT commenting ia_lum_dep_centrals
                    # AD: Will probably be removed after some point when we get all the Bnl terms for IA added!
                    alignment_amplitude_2h, alignment_amplitude_2h_II, C1 = pk_lib.compute_two_halo_alignment(block, suffix0, growth_factor, mean_density0)
                    # ============================================================================== #
                    # One halo alignment
                    # ============================================================================== #
                    # load the satellite density run w(k|m) for a perfect 3d radial alignment projected along the line of sight
                    # it can either be constant or radial dependent -> this is computed in the wkm module, including the amplitude of the
                    # signal (but not its luminosity dependence, which is a separate factor, see above)
                    wkm = get_satellite_alignment(block, k_vec, mass, z_vec, suffix0)
                    # preparing the central and satellite terms
                    if block['ia_small_scale_alignment' + suffix0, 'instance'] == 'halo_mass':
                        s_align_factor = pk_lib.prepare_satellite_alignment_factor_grid_halo(Nsat, numdensat, f_sat, wkm, block['ia_small_scale_alignment' + suffix0, 'beta_sat'],   block['ia_small_scale_alignment' + suffix0, 'M_pivot'], mass_avg)
                    else:
                        s_align_factor = pk_lib.prepare_satellite_alignment_factor_grid(Nsat, numdensat, f_sat, wkm)
                    if block['ia_large_scale_alignment' + suffix0, 'instance'] == 'halo_mass':
                        c_align_factor = pk_lib.prepare_central_alignment_factor_grid_halo(mass, scale_factor, growth_factor, f_cen, C1, block['ia_large_scale_alignment' + suffix0, 'beta'],  block['ia_large_scale_alignment' + suffix0, 'M_pivot'], mass_avg)
                    else:
                        c_align_factor = pk_lib.prepare_central_alignment_factor_grid(mass, scale_factor, growth_factor, f_cen, C1)
    
                    I_c_align_term = pk_lib.prepare_Ic_align_term(mass, c_align_factor, b_dm, dn_dlnm, mean_density0, A_term)
                    I_s_align_term = pk_lib.prepare_Is_align_term(mass, s_align_factor, b_dm, dn_dlnm, mean_density0, A_term)
                    if bnl == True:
                        if p_mI == True:
                            I_NL_ia_cm = pk_lib.prepare_I_NL(mass, mass, c_align_factor, m_factor, b_dm, b_dm, dn_dlnm, dn_dlnm, k_vec, z_vec, A_term, mean_density0, beta_interp)
                            I_NL_ia_sm = pk_lib.prepare_I_NL(mass, mass, s_align_factor, m_factor, b_dm, b_dm, dn_dlnm, dn_dlnm, k_vec, z_vec, A_term, mean_density0, beta_interp)
                
                        if p_II == True:
                            I_NL_ia_cc = pk_lib.prepare_I_NL(mass, mass, c_align_factor, c_align_factor, b_dm, b_dm, dn_dlnm, dn_dlnm, k_vec, z_vec, A_term, mean_density0, beta_interp)
                            I_NL_ia_cs = pk_lib.prepare_I_NL(mass, mass, c_align_factor, s_align_factor, b_dm, b_dm, dn_dlnm, dn_dlnm, k_vec, z_vec, A_term, mean_density0, beta_interp)
                            I_NL_ia_ss = pk_lib.prepare_I_NL(mass, mass, s_align_factor, s_align_factor, b_dm, b_dm, dn_dlnm, dn_dlnm, k_vec, z_vec, A_term, mean_density0, beta_interp)
                
                        if p_gI == True:
                            I_NL_ia_gc = pk_lib.prepare_I_NL(mass, mass, c_align_factor, c_factor, b_dm, b_dm, dn_dlnm, dn_dlnm, k_vec, z_vec, A_term, mean_density0, beta_interp)
                            I_NL_ia_gs = pk_lib.prepare_I_NL(mass, mass, s_align_factor, s_factor, b_dm, b_dm, dn_dlnm, dn_dlnm, k_vec, z_vec, A_term, mean_density0, beta_interp)
                            
                # compute the power spectra
                if p_gg == True and bnl == False:
                    pk_gg_1h, pk_gg_2h, pk_gg, bg_halo_model = pk_lib.compute_p_gg(block, k_vec, plin, z_vec, mass, dn_dlnm, c_factor, s_factor, I_c_term, I_s_term, mass_avg, poisson_type)
                    #block.put_grid('galaxy_power_1h' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gg_1h)
                    #block.put_grid('galaxy_power_2h' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gg_2h)
                    block.put_grid('galaxy_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gg)
                    #block.put_grid('galaxy_linear_bias' + suffix, 'z', z_vec, 'k_h', k_vec, 'galaxybiastotal', bg_halo_model)
    
                if p_gg == True and bnl == True:
                    pk_gg_1h_bnl, pk_gg_2h_bnl, pk_gg_bnl, bg_halo_model_bnl = pk_lib.compute_p_gg_bnl(block, k_vec, plin, z_vec, mass, dn_dlnm, c_factor, s_factor, I_c_term, I_s_term, I_NL_cs, I_NL_cc, I_NL_ss, mass_avg, poisson_type)
                    #block.put_grid('galaxy_power_1h' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gg_1h_bnl)
                    #block.put_grid('galaxy_power_2h' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gg_2h_bnl)
                    block.put_grid('galaxy_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gg_bnl)
                    #block.put_grid('galaxy_linear_bias' + suffix, 'z', z_vec, 'k_h', k_vec, 'galaxybiastotal', bg_halo_model)
        
                if p_gm == True and bnl == False:
                    pk_1h, pk_2h, pk_tot = pk_lib.compute_p_gm(block, k_vec, plin, z_vec, mass, dn_dlnm, c_factor, s_factor, m_factor_1h, I_c_term, I_s_term, I_m_term)
                    #block.put_grid('matter_galaxy_power_1h', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_1h)
                    #block.put_grid('matter_galaxy_power_2h', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_2h)
                    block.put_grid('matter_galaxy_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_tot)
                
                if p_gm == True and bnl == True:
                    pk_1h_bnl, pk_2h_bnl, pk_tot_bnl = pk_lib.compute_p_gm_bnl(block, k_vec, plin, z_vec, mass, dn_dlnm, c_factor, s_factor, m_factor_1h, I_c_term, I_s_term, I_m_term, I_NL_cm, I_NL_sm)
                    #block.put_grid('matter_galaxy_power_1h', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_1h)
                    #block.put_grid('matter_galaxy_power_2h', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_2h)
                    block.put_grid('matter_galaxy_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_tot_bnl)
        
            
                # Intrinsic aligment power spectra (full halo model calculation)
                if p_II == True and bnl == False:
                    pk_II_1h, pk_II_2h, pk_II = pk_lib.compute_p_II(block, k_vec, plin, z_vec, mass, dn_dlnm, c_align_factor, s_align_factor, I_c_align_term, I_s_align_term)
                    #block.put_grid('intrinsic_power_1h' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II_1h)
                    #block.put_grid('intrinsic_power_2h' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II_2h)
                    block.put_grid('intrinsic_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II)
                if p_gI == True and bnl == False:
                    pk_gI_1h, pk_gI_2h, pk_gI = pk_lib.compute_p_gI(block, k_vec, plin, z_vec, mass, dn_dlnm, c_factor, c_align_factor, s_align_factor, I_c_term, I_c_align_term, I_s_align_term)
                    block.put_grid('galaxy_intrinsic_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gI)
                if p_mI == True and bnl == False:
                    pk_mI_1h, pk_mI_2h, pk_mI = pk_lib.compute_p_mI(block, k_vec, plin, z_vec, mass, dn_dlnm, m_factor_1h, c_align_factor, s_align_factor, I_m_term, I_c_align_term, I_s_align_term)
                    #block.put_grid('matter_intrinsic_power_1h' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_GI_1h)
                    #block.put_grid('matter_intrinsic_power_2h' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_GI_2h)
                    block.put_grid('matter_intrinsic_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mI)
                
                
                # Intrinsic aligment power spectra (full halo model calculation)
                if p_II == True and bnl == True:
                    pk_II_1h_bnl, pk_II_2h_bnl, pk_II_bnl = pk_lib.compute_p_II_bnl(block, k_vec, plin, z_vec, mass, dn_dlnm, c_align_factor, s_align_factor, I_c_align_term, I_s_align_term, I_NL_ia_cc, I_NL_ia_cs, I_NL_ia_ss)
                    #block.put_grid('intrinsic_power_1h' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II_1h)
                    #block.put_grid('intrinsic_power_2h' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II_2h)
                    block.put_grid('intrinsic_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II_bnl)
                if p_gI == True and bnl == True:
                    pk_gI_1h_bnl, pk_gI_2h_bnl, pk_gI_bnl = pk_lib.compute_p_gI_bnl(block, k_vec, plin, z_vec, mass, dn_dlnm, c_factor, c_align_factor, s_align_factor, I_c_term, I_c_align_term, I_s_align_term, I_NL_ia_gc, I_NL_ia_gs)
                    block.put_grid('galaxy_intrinsic_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gI_bnl)
                if p_mI == True and bnl == True:
                    pk_mI_1h_bnl, pk_mI_2h_bnl, pk_mI_bnl = pk_lib.compute_p_mI_bnl(block, k_vec, plin, z_vec, mass, dn_dlnm, m_factor_1h, c_align_factor, s_align_factor, I_m_term, I_c_align_term, I_s_align_term, I_NL_ia_cm, I_NL_ia_sm)
                    #block.put_grid('matter_intrinsic_power_1h' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_GI_1h)
                    #block.put_grid('matter_intrinsic_power_2h' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_GI_2h)
                    block.put_grid('matter_intrinsic_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mI_bnl)
                
                
                # Intrinsic aligment power spectra (implementation from Maria Cristina - 2h = LA/NLA mixture)
                if p_II_mc == True:
                    pk_II_1h, pk_II_2h, pk_II = pk_lib.compute_p_II_mc(block, k_vec, pk_eff, z_vec, mass, dn_dlnm, s_align_factor, alignment_amplitude_2h_II, f_cen)
                    #block.put_grid('intrinsic_power_1h' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II_1h)
                    #block.put_grid('intrinsic_power_2h' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II_2h)
                    block.put_grid('intrinsic_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II)
                if p_gI_mc == True:
                    pk_gI_1h, pk_gI_2h, pk_gI = pk_lib.compute_p_gI_mc(block, k_vec, pk_eff, z_vec, mass, dn_dlnm, c_factor, s_align_factor, I_c_term, alignment_amplitude_2h)
                    block.put_grid('galaxy_intrinsic_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gI)
                if p_mI_mc == True:
                    pk_mI_1h, pk_mI_2h, pk_mI = pk_lib.compute_p_mI_mc(block, k_vec, pk_eff, z_vec, mass, dn_dlnm, m_factor_1h, s_align_factor, alignment_amplitude_2h, f_cen)
                    #block.put_grid('matter_intrinsic_power_1h' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_GI_1h)
                    #block.put_grid('matter_intrinsic_power_2h' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_GI_2h)
                    block.put_grid('matter_intrinsic_power' + suffix, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mI)
                
        if p_mm == True and bnl == False:
            if mead_correction == 'nofeedback':
                pk_mm_1h, pk_mm_2h, pk_mm_tot = pk_lib.compute_p_mm_mead(block, k_vec, plin, z_vec, mass, dn_dlnm, m_factor_1h_mm, I_m_term)
            else:
                pk_mm_1h, pk_mm_2h, pk_mm_tot = pk_lib.compute_p_mm(block, k_vec, plin, z_vec, mass, dn_dlnm, m_factor_1h_mm, I_m_term)
            # save in the datablock
            #block.put_grid('matter_1h_power', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm_1h)
            #block.put_grid('matter_2h_power', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm_2h)
            #block.put_grid('matter_power', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm_tot)
            block.replace_grid('matter_power_nl', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm_tot)
            
        if p_mm == True and bnl == True:
            pk_mm_1h_bnl, pk_mm_2h_bnl, pk_mm_tot_bnl = pk_lib.compute_p_mm_bnl(block, k_vec, plin, z_vec, mass, dn_dlnm, m_factor_1h_mm, I_m_term, I_NL_mm)
            # save in the datablock
            #block.put_grid('matter_1h_power', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm_1h)
            #block.put_grid('matter_2h_power', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm_2h)
            #block.put_grid('matter_power', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm_tot)
            block.replace_grid('matter_power_nl', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm_tot_bnl)

    return 0


def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass


