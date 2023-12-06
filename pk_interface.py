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

We truncate the 1-halo term so that it doesn't dominate at large scales.

Linear matter power spectrum needs to be provided as well. The halo_model_ingredients and hod modules (for everything but mm) 
need to be run before this. 

Current power spectra that we predict are 
mm: matter-matter
gg: galaxy-galaxy
gm: galaxy-matter

II: intrinsic-intrinsic alignments
gI: galaxy-intrinsic alignment
mI: matter-intrinsic alignment
"""

# TODO: extend this so that it can accept any profile.

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


# TODO: change the name of this file and pk_lib file to make it clear what these do.

# [pk_bright]
# file= %(HM_PATH)s/pk_interface.py
# log_mass_min = %(logmassmin_def)s
# log_mass_max = %(logmassmax_def)s
# nmass = %(nmass_def)s
# zmin = %(zmin_def)s
# zmax = %(zmax_def)s
# nz = %(nz_def)s
# nk = %(nk_def)s
# pipeline = False
# p_mm = False
# p_mm_bnl = True
# p_gg = False
# p_gg_bnl = True
# p_gm = False
# p_gm_bnl = True
# p_gI = False
# p_mI = False
# p_II = False
# p_gI_bnl = False
# p_mI_bnl = False
# p_II_bnl = False
# two_halo_only = False
# hod_section_name = hod_KiDS_bright
# name = red
# poisson_type = scalar
# point_mass = True

def setup(options):

    # Read in the minimum and maximum halo mass
    # These are the same as the values that go into the halo model ingredients and the HOD sections, but they don't have to be.
    # Interpolation is done if the mass binning and range doesn't match
    log_mass_min = options[option_section, 'log_mass_min']
    log_mass_max = options[option_section, 'log_mass_max']
    nmass = options[option_section, 'nmass']
    # log-spaced mass in units of M_sun/h
    dlog10m = (log_mass_max-log_mass_min)/nmass
    mass    = 10.0 ** np.arange(log_mass_min, log_mass_max, dlog10m)

    #nmass_bnl = options[option_section, 'nmass_bnl']
    #mass_bnl = np.logspace(log_mass_min, log_mass_max, nmass_bnl) 

    zmin  = options[option_section, 'zmin']
    zmax  = options[option_section, 'zmax']
    nz    = options[option_section, 'nz']
    z_vec = np.linspace(zmin, zmax, nz)

    nk = options[option_section, 'nk']

    p_mm = options.get_bool(option_section, 'p_mm',default=False)
    p_gg = options.get_bool(option_section, 'p_gg',default=False)
    p_gm = options.get_bool(option_section, 'p_gm',default=False)
    p_gI = options.get_bool(option_section, 'p_gI',default=False)
    p_mI = options.get_bool(option_section, 'p_mI',default=False)
    p_II = options.get_bool(option_section, 'p_II',default=False)
    # TODO: these are the IA power as in Fortuna et al. 2021. Change the name.
    p_gI_mc = options.get_bool(option_section, 'p_gI_mc',default=False)
    p_mI_mc = options.get_bool(option_section, 'p_mI_mc',default=False)
    p_II_mc = options.get_bool(option_section, 'p_II_mc',default=False)
    # If True uses beta_nl
    bnl     = options.get_bool(option_section, 'bnl',default=False)
    #interpolate_bnl = options.get_bool(option_section, 'interpolate_bnl',default=False)

    # TODO: change this: generally not good practice to look into a difference section other than option_section.
    # As names can change in the ini file.
    check_mead    = options.has_value('hmf_and_halo_bias', 'use_mead2020_corrections')

    poisson_type  = options.get_string(option_section, 'poisson_type',default='')
    point_mass    = options.get_bool(option_section, 'point_mass',default=False)
    two_halo_only = options[option_section, 'two_halo_only']

    # initiate pipeline parameters
    # TODO: Check what each of these does
    ia_lum_dep_centrals = False
    ia_lum_dep_satellites = False
    matter = False
    galaxy = False
    alignment = False
    hod_section_name = ''
    f_red_cen_option = False

    # change to raise
    #  TODO: change the name of *_mc to something more descriptive
    if (p_mI == True) and (p_mI_mc == True):
        raise Exception('Select either p_mI = True or p_mI_mc = True, both compute the matter-intrinsic power spectrum. p_mI_mc is the implementation used in Fortuna et al. 2021 paper.')
        # print('Select either p_mI = True or p_mI_mc = True, both compute the matter-intrinsic power spectrum. p_mI_mc is the implementation used in Fortuna et al. 2021 paper.')
        # sys.exit()
        
    if (p_II == True) and (p_II_mc == True):
        raise Exception('Select either p_II = True or p_II_mc = True, all compute the matter-intrinsic power spectrum. p_II_mc is the implementation used in Fortuna et al. 2021 paper.')

        
    if (p_gI == True) and (p_gI_mc == True):
        raise Exception('Select either p_gI = True or p_gI_mc = True, all compute the matter-intrinsic power spectrum. p_gI_mc i is the implementation used in Fortuna et al. 2020 paper.')



    # TODO: what does the two halo only do? Do we need this?
    if ((p_mm == True) or (p_gm == True) or (p_mI == True)):
        matter = True
    if (p_gg == True) or (p_gm == True) or (p_gI == True) or (p_mI == True) or (p_II == True) or (p_gI_mc == True) or (p_mI_mc == True) or (p_II_mc == True):
        galaxy = True
        hod_section_name = options[option_section, 'hod_section_name']
    if (p_gI == True) or (p_mI == True) or (p_II == True) or (p_gI_mc == True) or (p_mI_mc == True) or (p_II_mc == True):
        alignment = True

    population_name = options.get_string(option_section, 'output_suffix', default='').lower()
    if population_name != '':
        pop_name = f'_{population_name}'
    else:
        pop_name = ''
    
    # TODO: this has to be changed see comment above about check_mead
    # if check_mead:
    #     use_mead = options['hmf_and_halo_bias', 'use_mead2020_corrections']
    #     if use_mead == 'mead2020':
    #         mead_correction = 'nofeedback'
    #     elif use_mead == 'mead2020_feedback':
    #         mead_correction = 'hmcode2020_feedback'
    #     elif use_mead == 'stellar_fraction_from_observable_feedback':
    #         mead_correction = 'stellar_fraction_from_observable_feedback'
    #         if not options.has_value(option_section, 'hod_section_name'):
    #             raise ValueError('To use the fit option for feedback that links HOD derived stellar mass fraction to the baryon \
    #                              feedback one needs to provide the hod section name of used hod!')
    #         hod_section_name = options[option_section, 'hod_section_name']
    # else:
    #     mead_correction = None


    use_mead = options.get_string(option_section, 'use_mead2020_corrections', default='None')
    if use_mead == 'mead2020':
        mead_correction = 'nofeedback'
    elif use_mead == 'mead2020_feedback':
        mead_correction = 'feedback'
    elif use_mead == 'fit_feedback':
        mead_correction = 'fit'
    else:
        mead_correction = None

    return mass, nmass, z_vec, nz, nk, \
           p_mm, p_gg, p_gm, p_gI, p_mI, p_II, p_gI_mc, p_mI_mc, p_II_mc, \
           matter, galaxy, bnl, alignment, \
           ia_lum_dep_centrals, ia_lum_dep_satellites, two_halo_only, hod_section_name, \
           mead_correction, point_mass, poisson_type, \
           pop_name


def execute(block, config):

    mass, nmass, z_vec, nz, nk, \
    p_mm, p_gg, p_gm, p_gI, p_mI, p_II, p_gI_mc, p_mI_mc, p_II_mc, \
    matter, galaxy, bnl, alignment, \
    ia_lum_dep_centrals, ia_lum_dep_satellites, two_halo_only, hod_section_name, \
    mead_correction, point_mass, poisson_type, \
    pop_name = config

    # TODO: This has the same length as nz but the same value in each element
    mean_density0 = block['density', 'mean_density0']

    # Marika: Change this bit to read in k_vec and pk from the block directly. Get growth from camb
    # AD: If we can avoid interpolation, then yes. Looking at load_modules.py, we could leave them there to have more utility code separated. 
    # Could call them utilities. Dunno

    # TODO: move all interpolations into this function
    k_vec_original, plin_original = pk_lib.get_linear_power_spectrum(block, z_vec)
    # load growth factor
    k_vec_original,  growth_factor_original, scale_factor_original = pk_lib.get_growth_factor(block, z_vec)
    # load nonlinear power spectrum
    k_nl, p_nl = pk_lib.get_nonlinear_power_spectrum(block, z_vec)
    
    # TODO: Why is k_vec defined? Why not just use k_vec_original?
    k_vec = np.logspace(np.log10(k_vec_original[0]), np.log10(k_vec_original[-1]), num=nk)

    plin_k_interp = interp1d(k_vec_original, plin_original, axis=1, fill_value='extrapolate')
    plin = plin_k_interp(k_vec)
    growth_factor_interp = interp1d(k_vec_original, growth_factor_original, axis=1, fill_value='extrapolate')
    growth_factor = growth_factor_interp(k_vec)
    scale_factor_interp = interp1d(k_vec_original, scale_factor_original, axis=1, fill_value='extrapolate')
    scale_factor = scale_factor_interp(k_vec)

    # TODO: Why is k_vec defined? Why not just use k_vec_original?
    plin_k_interp = interp1d(k_nl, p_nl, axis=1, fill_value='extrapolate')
    pnl = plin_k_interp(k_vec)
    # TODO: this shouldn't be replaced
    block.replace_grid('matter_power_nl_mead', 'z', z_vec, 'k_h', k_vec, 'p_k', pnl)
    #block.replace_grid('matter_power_nl', 'z', z_vec, 'k_h', k_vec, 'p_k', pnl)

    # AD: avoid this! (Maybe needed for IA part ...)
    # compute the effective power spectrum, mixing the linear and nonlinear one:
    #
    # (1.-t_eff)*plin + t_eff*p_nl
    #
    t_eff = block['pk_parameters', 'trans_1hto2h']

    # TODO: do we need this? Does this need to do interpolation? both p_nl and plin are already interpolated
    # pk_eff = pk_lib.compute_effective_power_spectrum(k_vec, plin, k_nl, p_nl, z_vec, t_eff)
    pk_eff = (1.-t_eff)*plin+t_eff*pnl

    # initialise the galaxy bias
    # bg = 1.0 # AD: ???
    
    # If the two_halo_only option is set True, then only the linear regime is computed and the linear bias is used (either computed by the
    # hod module or passed in the value	file (same structure as for the constant bias module)
    # Otherwise, compute the full power spectra (including the small scales)

    # load the halo mass and bias functions from the datablock
    dn_dlnm, b_dm = pk_lib.get_halo_functions(block, mass, z_vec)
    # prepare a grid for the navarro-frenk-white profile
    #TODO: see the comments for this function in pk_lib
    u_dm, u_sat  = pk_lib.compute_u_dm_grid(block, k_vec, mass, z_vec)
    
    # TODO: check that mean_density for A should be mean_density at redshift zero.
    # A_term       = pk_lib.missing_mass_integral(mass, b_dm, dn_dlnm, mean_density0)
    
    # bg = block[f'galaxy_bias{suffix}', 'b']
    # print(bg)
    # exit()
    # TODO: CHECK THESE later
    if two_halo_only == True:
        if matter == True:
            A_term   = pk_lib.missing_mass_integral(mass, b_dm, dn_dlnm, mean_density0)
            I_m = pk_lib.Im_term(mass, u_dm, b_dm, dn_dlnm, mean_density0, A_term)
            matter_profile = pk_lib.matter_profile(mass, mean_density0, u_dm)
            
        if (galaxy == True) or (alignment == True):
            # TODO: Change where this is looking for things, as I have removed metadata
            hod_bins = block[hod_section_name, 'nbins']
            
            for nb in range(0,hod_bins):
                if hod_bins != 1:
                    suffix = f'{pop_name}_{nb+1}'
                else:
                    suffix = f'{pop_name}'
                    
                if galaxy == True:
                    # load linear bias:
                    # TODO: change this to galaxy_bias_section_name
                    bg = block[hod_section_name, f'b{suffix}']
                    if np.isscalar(bg): bg *= np.ones(nz)
                    
                if alignment == True:
                #IT commented ia_lum_dep_centrals
                    alignment_amplitude_2h, alignment_amplitude_2h_II = pk_lib.compute_two_halo_alignment(block, pop_name, growth_factor, mean_density0)
                    
                # compute the power spectra
                if p_gg:
                    pk_gg = pk_lib.compute_p_gg_two_halo(block, k_vec, pk_eff, z_vec, bg)
                    block.put_grid(f'galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gg)
                if p_gm:
                    pk_gm = pk_lib.compute_p_gm_two_halo(block, k_vec, pk_eff, z_vec, bg)
                    block.put_grid(f'matter_galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gm)
                if p_mI:
                    #print('pGI...')
                    pk_mI = pk_lib.compute_p_mI_two_halo(block, k_vec, pk_eff, z_vec, f_red_cen, alignment_amplitude_2h)
                    block.put_grid(f'matter_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mI)
                if p_gI:
                    pk_gI = pk_lib.compute_p_gI_two_halo(block, k_vec, pk_eff, z_vec, f_red_cen, alignment_amplitude_2h, bg)
                    block.put_grid(f'galaxy_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gI)
                if p_II:
                    #print('pII...')
                    pk_II = pk_lib.compute_p_II_two_halo(block, k_vec, pk_eff, z_vec, f_red_cen, alignment_amplitude_2h_II)
                    block.put_grid(f'intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II)
                    
        if p_mm:
            # this is not very useful as for the lensing power spectrum it is usually used halofit
            raise ValueError('pmm not implemented for the two-halo only option\n')
            # AD: Implement proper matter-matter term using the halo model here. We do not want to use halofit.
            # compute_p_mm_new(block, k_vec, pk_eff, z_vec, mass, dn_dlnm, matter_profile, I_m, nz, nk)
    
    # TODO: starting checks from here 
    else:
        # TODO: Check beta_interp
        # Add the non-linear P_hh to the 2h term
        if bnl == True:
            # Reads beta_nl from the block
            if block.has_value('bnl', 'beta_interp'):
                beta_interp = block.get_double_array_nd('bnl', 'beta_interp')
            else:
                raise Exception("You've set bnl = True. Looked for beta_intep in bnl, but didn't find it. Run bnl_interface.py to set this up.\n")
            if beta_interp.shape == np.array([0.0]).shape:
                raise ValueError('Non-linear halo bias module bnl is not initialised, or you have deleted it too early! \
                    This might be because you ran bnl_interface_delete.py before this module. \n')
                    
        # Accounts for the missing low mass haloes in the integrals for the 2h term.
        # Assumes all missing mass is in haloes of mass M_min.
        # This is calculated separately for each redshift
        # TODO: check if this is needed for the IA section
        A_term = pk_lib.missing_mass_integral(mass, b_dm, dn_dlnm, mean_density0)
        
        # If matter auto or cross power spectra are set to True
        if matter == True:
            # 2h term integral for matter
            I_m = pk_lib.Im_term(mass, u_dm, b_dm, dn_dlnm, mean_density0, A_term)
            # f_nu = omega_nu/omega_m with the same length as redshift
            fnu = block['cosmological_parameters', 'fnu']
            # Matter halo profile
            matter_profile = pk_lib.matter_profile(mass, mean_density0, u_dm, fnu)
            # TODO: Why is there a matter profile and a matter_profile_1h_mm?
            
            if mead_correction == 'hmcode2020_feedback':
                omega_c    = block['cosmological_parameters', 'omega_c']
                omega_m    = block['cosmological_parameters', 'omega_m']
                log10T_AGN = block['halo_model_parameters', 'logT_AGN']
                matter_profile_1h_mm = pk_lib.matter_profile_with_feedback(mass, mean_density0, u_dm, z_vec, omega_c, omega_m, omega_b, log10T_AGN)
            elif mead_correction == 'fit':
                # Reads f_star_extended form the HOD section. Either need to use a conditional HOD to get this value or to put it in block some other way.
                fstar_mm = load_fstar_mm(block, hod_section_name, z_vec, mass)
                matter_profile_1h_mm = pk_lib.matter_profile_with_feedback_stellar_fraction_from_observable(mass, mean_density0, u_dm, z_vec, fstar_mm, omega_c, omega_m, omega_b)
            else:
                matter_profile_1h_mm = matter_profile.copy()
                
            if bnl == True:
                # TODO: This one uses matter_profile not matter_profile_1h_mm. Shouldn't we use the same profile everywhere?
                # Compare these
                I_NL_mm = pk_lib.I_NL(mass, mass, matter_profile, matter_profile, 
                                    b_dm, b_dm, dn_dlnm, dn_dlnm, k_vec, 
                                    z_vec, A_term, mean_density0, beta_interp)
                I_NL_mm_1h = pk_lib.I_NL(mass, mass, matter_profile_1h_mm, matter_profile_1h_mm, 
                                    b_dm, b_dm, dn_dlnm, dn_dlnm, k_vec, 
                                    z_vec, A_term, mean_density0, beta_interp)
        
            if p_mm == True and bnl == False:
                if mead_correction == 'nofeedback':
                    sigma8_z = block['hmf', 'sigma8_z']
                    neff     = block['hmf', 'neff']
                    pk_mm_1h, pk_mm_2h, pk_mm_tot = pk_lib.compute_p_mm_mead(k_vec, plin, z_vec, mass, 
                                                                            dn_dlnm, matter_profile_1h_mm, 
                                                                            I_m, sigma8_z,neff)
                else:
                    pk_mm_1h, pk_mm_2h, pk_mm_tot = pk_lib.compute_p_mm(k_vec, plin, z_vec, mass, 
                                                                        dn_dlnm, matter_profile_1h_mm, I_m)
                # save in the datablock
                #block.put_grid('matter_1h_power', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm_1h)
                #block.put_grid('matter_2h_power', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm_2h)
                #block.put_grid('matter_power', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm_tot)
                block.replace_grid('matter_power_nl', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm_tot)
                
            if p_mm == True and bnl == True:
                pk_mm_1h_bnl, pk_mm_2h_bnl, pk_mm_tot_bnl = pk_lib.compute_p_mm_bnl(k_vec, plin, z_vec, mass, dn_dlnm, 
                                                                                    matter_profile_1h_mm, I_m, I_NL_mm)
                # save in the datablock
                #block.put_grid('matter_1h_power', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm_1h)
                #block.put_grid('matter_2h_power', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm_2h)
                #block.put_grid('matter_power', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm_tot)
                block.replace_grid('matter_power_nl', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm_tot_bnl)

        if (galaxy == True) or (alignment == True):
            # TODO: metadata does not exist change this
            hod_bins = block[hod_section_name, 'nbins']
            
            for nb in range(0,hod_bins):
                if hod_bins != 1:
                    suffix = f'{pop_name}_{nb+1}'
                    suffix_hod = f'_{nb+1}'
                else:
                    suffix = f'{pop_name}'
                    suffix_hod = ''
                    
                Ncen, Nsat, numdencen, numdensat, f_cen, f_sat, mass_avg, fstar = pk_lib.load_hods(block, hod_section_name, suffix_hod, z_vec, mass)
            
                if galaxy == True:
                    # preparing the 1h term
                    # TODO: check if Nsat and Ncen need to be in a grid with respect to z
                    # Computes the profiles for centrals and satellites. 
                    # These are the W_u(M,k) functions in Asgari, Mead, Heymans 2023: 2303.08752
                    # Centrals are assumed to be in the centre of the halo, therefore no need for the normalised profile, U.
                    profile_c = pk_lib.central_profile(Ncen, numdencen, f_cen)
                    profile_s = pk_lib.satellite_profile(Nsat, numdensat, f_sat, u_sat)
                    # calculate the 2-halo integrals for centrals and satelites
                    I_c = pk_lib.Ic_term(mass, profile_c, b_dm, dn_dlnm, nk)
                    I_s = pk_lib.Is_term(mass, profile_s, b_dm, dn_dlnm)
                    
                    if mead_correction == 'fit' or point_mass == True:
                        # Include point mass and gas contribution to the GGL power spectrum, defined from HOD
                        # Maybe extend to input the mass per bin!
                        omega_c    = block['cosmological_parameters', 'omega_c']
                        omega_m    = block['cosmological_parameters', 'omega_m']
                        omega_b    = block['cosmological_parameters', 'omega_b']
                        matter_profile_1h = pk_lib.matter_profile_with_feedback_stellar_fraction_from_obs(mass, mean_density0, 
                                                                                                        u_dm, z_vec, fstar, 
                                                                                                        omega_c, omega_m, omega_b)
                    else:
                        matter_profile_1h = matter_profile_1h_mm.copy()
                        
                    if bnl == True:
                        if p_gg == True:
                            I_NL_cs = pk_lib.I_NL(mass, mass, profile_c, profile_s, b_dm, b_dm,
                                dn_dlnm, dn_dlnm, k_vec, z_vec, A_term, mean_density0, beta_interp)
                            I_NL_ss = pk_lib.I_NL(mass, mass, profile_s, profile_s, b_dm, b_dm, 
                                dn_dlnm, dn_dlnm, k_vec, z_vec, A_term, mean_density0, beta_interp)
                            I_NL_cc = pk_lib.I_NL(mass, mass, profile_c, profile_c, b_dm, b_dm, 
                                dn_dlnm, dn_dlnm, k_vec, z_vec, A_term, mean_density0, beta_interp)
    
                        if p_gm == True:
                            I_NL_cm = pk_lib.I_NL(mass, mass, profile_c, matter_profile, b_dm, b_dm, 
                                dn_dlnm, dn_dlnm, k_vec, z_vec, A_term, mean_density0, beta_interp)
                            I_NL_sm = pk_lib.I_NL(mass, mass, profile_s, matter_profile, b_dm, b_dm, 
                                dn_dlnm, dn_dlnm, k_vec, z_vec, A_term, mean_density0, beta_interp)
                    
                if alignment == True:
                    #IT commenting ia_lum_dep_centrals
                    # AD: Will probably be removed after some point when we get all the Bnl terms for IA added!
                    alignment_amplitude_2h, alignment_amplitude_2h_II, C1 = pk_lib.compute_two_halo_alignment(block, pop_name,
                                                                                                growth_factor, mean_density0)
                    # ============================================================================== #
                    # One halo alignment
                    # ============================================================================== #
                    # load the satellite density run w(k|m) for a perfect 3d radial alignment projected along the line of sight
                    # it can either be constant or radial dependent -> this is computed in the wkm module, 
                    # including the amplitude of the
                    # signal (but not its luminosity dependence, which is a separate factor, see above)
                    wkm = pk_lib.get_satellite_alignment(block, k_vec, mass, z_vec, pop_name)
                    # preparing the central and satellite terms
                    if block[f'ia_small_scale_alignment{pop_name}', 'instance'] == 'halo_mass':
                        beta_sat = block[f'ia_small_scale_alignment{pop_name}', 'beta_sat']
                        M_pivot  = block[f'ia_small_scale_alignment{pop_name}', 'M_pivot']
                        s_align_profile = pk_lib.satellite_alignment_profile_grid_halo(Nsat, numdensat, f_sat, wkm,
                                                                                    beta_sat,  M_pivot , mass_avg)
                    else:
                        s_align_profile = pk_lib.satellite_alignment_profile(Nsat, numdensat, f_sat, wkm)
                    if block[f'ia_large_scale_alignment{pop_name}', 'instance'] == 'halo_mass':
                        beta    = block[f'ia_large_scale_alignment{pop_name}', 'beta']
                        M_pivot = block[f'ia_large_scale_alignment{pop_name}', 'M_pivot']
                        c_align_profile = pk_lib.central_alignment_profile_grid_halo(mass, scale_factor, growth_factor,
                                                                                    f_cen, C1,beta,  M_pivot, mass_avg)
                    else:
                        c_align_profile = pk_lib.central_alignment_profile(mass, scale_factor, growth_factor, f_cen, C1)
                    # TODO: does this need the A_term?
                    I_c_align_term = pk_lib.Ig_align_term(mass, c_align_profile, b_dm, dn_dlnm, mean_density0, A_term)
                    I_s_align_term = pk_lib.Ig_align_term(mass, s_align_profile, b_dm, dn_dlnm, mean_density0, A_term)
                    if bnl == True:
                        if p_mI == True:
                            I_NL_ia_cm = pk_lib.I_NL(mass, mass, c_align_profile, matter_profile, b_dm, b_dm, 
                                dn_dlnm, dn_dlnm, k_vec, z_vec, A_term, mean_density0, beta_interp)
                            I_NL_ia_sm = pk_lib.I_NL(mass, mass, s_align_profile, matter_profile, b_dm, b_dm, 
                                dn_dlnm, dn_dlnm, k_vec, z_vec, A_term, mean_density0, beta_interp)
                
                        if p_II == True:
                            I_NL_ia_cc = pk_lib.I_NL(mass, mass, c_align_profile, c_align_profile, b_dm, b_dm, 
                                dn_dlnm, dn_dlnm, k_vec, z_vec, A_term, mean_density0, beta_interp)
                            I_NL_ia_cs = pk_lib.I_NL(mass, mass, c_align_profile, s_align_profile, b_dm, b_dm, 
                                dn_dlnm, dn_dlnm, k_vec, z_vec, A_term, mean_density0, beta_interp)
                            I_NL_ia_ss = pk_lib.I_NL(mass, mass, s_align_profile, s_align_profile, b_dm, b_dm, 
                                dn_dlnm, dn_dlnm, k_vec, z_vec, A_term, mean_density0, beta_interp)
                
                        if p_gI == True:
                            I_NL_ia_gc = pk_lib.I_NL(mass, mass, c_align_profile, profile_c, b_dm, b_dm, 
                                dn_dlnm, dn_dlnm, k_vec, z_vec, A_term, mean_density0, beta_interp)
                            I_NL_ia_gs = pk_lib.I_NL(mass, mass, s_align_profile, profile_s, b_dm, b_dm, 
                                dn_dlnm, dn_dlnm, k_vec, z_vec, A_term, mean_density0, beta_interp)
                            
                # compute the power spectra
                if p_gg == True and bnl == False:
                    pk_gg_1h, pk_gg_2h, pk_gg, bg_halo_model = pk_lib.compute_p_gg(block, k_vec, plin, z_vec, 
                                mass, dn_dlnm, profile_c, profile_s, I_c, I_s, mass_avg, poisson_type)
                    #block.put_grid(f'galaxy_power_1h{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gg_1h)
                    #block.put_grid(f'galaxy_power_2h{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gg_2h)
                    block.put_grid(f'galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gg)
                    #block.put_grid(f'galaxy_linear_bias{suffix}', 'z', z_vec, 'k_h', k_vec, 'galaxybiastotal', bg_halo_model)
    
                if p_gg == True and bnl == True:
                    pk_gg_1h_bnl, pk_gg_2h_bnl, pk_gg_bnl, bg_halo_model_bnl = pk_lib.compute_p_gg_bnl(block, k_vec, plin, z_vec, 
                        mass, dn_dlnm, profile_c, profile_s, I_c, I_s, I_NL_cs, I_NL_cc, I_NL_ss, mass_avg, poisson_type)
                    #block.put_grid(f'galaxy_power_1h{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gg_1h_bnl)
                    #block.put_grid(f'galaxy_power_2h{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gg_2h_bnl)
                    block.put_grid(f'galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gg_bnl)
                    #block.put_grid(f'galaxy_linear_bias{suffix}', 'z', z_vec, 'k_h', k_vec, 'galaxybiastotal', bg_halo_model)
        
                if p_gm == True and bnl == False:
                    pk_1h, pk_2h, pk_tot, galaxy_matter_linear_bias = pk_lib.compute_p_gm(block, k_vec, plin, z_vec, 
                        mass, dn_dlnm, profile_c, profile_s, matter_profile_1h, I_c, I_s, I_m)
                    #block.put_grid(f'matter_galaxy_power_1h{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_1h)
                    #block.put_grid(f'matter_galaxy_power_2h{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_2h)
                    block.put_grid(f'matter_galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_tot)
                
                if p_gm == True and bnl == True:
                    pk_1h_bnl, pk_2h_bnl, pk_tot_bnl, galaxy_matter_linear_bias = pk_lib.compute_p_gm_bnl(block, k_vec, plin, z_vec, 
                        mass, dn_dlnm, profile_c, profile_s, matter_profile_1h, I_c, I_s, I_m, I_NL_cm, I_NL_sm)
                    #block.put_grid(f'matter_galaxy_power_1h{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_1h)
                    #block.put_grid(f'matter_galaxy_power_2h{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_2h)
                    block.put_grid(f'matter_galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_tot_bnl)
        
            
                # Intrinsic aligment power spectra (full halo model calculation)
                if p_II == True and bnl == False:
                    pk_II_1h, pk_II_2h, pk_II = pk_lib.compute_p_II(block, k_vec, plin, z_vec, 
                        mass, dn_dlnm, c_align_profile, s_align_profile, I_c_align_term, I_s_align_term)
                    #block.put_grid(f'intrinsic_power_1h{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II_1h)
                    #block.put_grid(f'intrinsic_power_2h{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II_2h)
                    block.put_grid(f'intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II)
                if p_gI == True and bnl == False:
                    pk_gI_1h, pk_gI_2h, pk_gI = pk_lib.compute_p_gI(block, k_vec, plin, z_vec, 
                        mass, dn_dlnm, profile_c, c_align_profile, s_align_profile, I_c, I_c_align_term, I_s_align_term)
                    block.put_grid(f'galaxy_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gI)
                if p_mI == True and bnl == False:
                    pk_mI_1h, pk_mI_2h, pk_mI = pk_lib.compute_p_mI(block, k_vec, plin, z_vec, 
                        mass, dn_dlnm, matter_profile_1h, c_align_profile, s_align_profile, I_m, I_c_align_term, I_s_align_term)
                    #block.put_grid(f'matter_intrinsic_power_1h{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_GI_1h)
                    #block.put_grid(f'matter_intrinsic_power_2h{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_GI_2h)
                    block.put_grid(f'matter_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mI)
                
                
                # Intrinsic aligment power spectra (full halo model calculation)
                if p_II == True and bnl == True:
                    pk_II_1h_bnl, pk_II_2h_bnl, pk_II_bnl = pk_lib.compute_p_II_bnl(block, k_vec, plin, z_vec, 
                        mass, dn_dlnm, c_align_profile, s_align_profile, I_c_align_term, I_s_align_term, 
                        I_NL_ia_cc, I_NL_ia_cs, I_NL_ia_ss)
                    #block.put_grid(f'intrinsic_power_1h{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II_1h)
                    #block.put_grid(f'intrinsic_power_2h{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II_2h)
                    block.put_grid(f'intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II_bnl)
                if p_gI == True and bnl == True:
                    pk_gI_1h_bnl, pk_gI_2h_bnl, pk_gI_bnl = pk_lib.compute_p_gI_bnl(block, k_vec, plin, z_vec, 
                        mass, dn_dlnm, profile_c, c_align_profile, s_align_profile, I_c, I_c_align_term, I_s_align_term, 
                        I_NL_ia_gc, I_NL_ia_gs)
                    block.put_grid(f'galaxy_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gI_bnl)
                if p_mI == True and bnl == True:
                    pk_mI_1h_bnl, pk_mI_2h_bnl, pk_mI_bnl = pk_lib.compute_p_mI_bnl(block, k_vec, plin, z_vec, 
                        mass, dn_dlnm, matter_profile_1h, c_align_profile, s_align_profile, I_m, I_c_align_term, I_s_align_term, 
                        I_NL_ia_cm, I_NL_ia_sm)
                    #block.put_grid(f'matter_intrinsic_power_1h{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_GI_1h)
                    #block.put_grid(f'matter_intrinsic_power_2h{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_GI_2h)
                    block.put_grid(f'matter_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mI_bnl)
                
                
                # Intrinsic aligment power spectra (implementation from Maria Cristina - 2h = LA/NLA mixture)
                if p_II_mc == True:
                    pk_II_1h, pk_II_2h, pk_II = pk_lib.compute_p_II_mc(block, k_vec, pk_eff, z_vec, 
                        mass, dn_dlnm, s_align_profile, alignment_amplitude_2h_II, f_cen)
                    #block.put_grid(f'intrinsic_power_1h{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II_1h)
                    #block.put_grid(f'intrinsic_power_2h{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II_2h)
                    block.put_grid(f'intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II)
                if p_gI_mc == True:
                    pk_gI_1h, pk_gI_2h, pk_gI = pk_lib.compute_p_gI_mc(block, k_vec, pk_eff, z_vec, 
                        mass, dn_dlnm, profile_c, s_align_profile, I_c, alignment_amplitude_2h)
                    block.put_grid(f'galaxy_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gI)
                if p_mI_mc == True:
                    pk_mI_1h, pk_mI_2h, pk_mI = pk_lib.compute_p_mI_mc(k_vec, pk_eff, z_vec, 
                        mass, dn_dlnm, matter_profile_1h, s_align_profile, alignment_amplitude_2h, f_cen)
                    #block.put_grid(f'matter_intrinsic_power_1h{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_GI_1h)
                    #block.put_grid(f'matter_intrinsic_power_2h{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_GI_2h)
                    block.put_grid(f'matter_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mI)
                
    return 0


def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass


