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
import numpy as np
import numbers
import pk_lib

# from scipy.interpolate import interp1d
# from scipy.interpolate import interp2d, RegularGridInterpolator
# from collections import OrderedDict
# import sys
# import time

# cosmological parameters section name in block
cosmo_params = names.cosmological_parameters

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


def get_string_or_none(options, name, default):
    """
    A helper function to return a number or None explicitly from config files
    """
    if options.has_value(option_section, name):
        test_param = options.get(option_section, name)
        if isinstance(test_param, numbers.Number):
            param = options.get_double(option_section, name, default)
        if isinstance(test_param, str):
            str_in = options.get_string(option_section, name)
            if str_in == 'None':
                param = None
    else:
        param = options.get_double(option_section, name, default)

    if not isinstance(param, (numbers.Number, type(None))):
        raise ValueError(f'Parameter {name} is not an instance of a number or NoneType!')
    
    return param

##############################################################################################################################

def setup(options):

    # Read in the minimum and maximum halo mass
    # These are the same as the values that go into the halo model ingredients and the HOD sections, but they don't have to be.

    p_mm = options.get_bool(option_section, 'p_mm', default=False)
    p_gg = options.get_bool(option_section, 'p_gg', default=False)
    p_gm = options.get_bool(option_section, 'p_gm', default=False)
    p_gI = options.get_bool(option_section, 'p_gI', default=False)
    p_mI = options.get_bool(option_section, 'p_mI', default=False)
    p_II = options.get_bool(option_section, 'p_II', default=False)

    # If true use the IA formalism of Fortuna et al. 2021: Truncated NLA at high k + 1-halo term 
    fortuna = options.get_bool(option_section, 'fortuna', default=False)
    # If True uses beta_nl
    bnl     = options.get_bool(option_section, 'bnl', default=False)

    poisson_type  = options.get_string(option_section, 'poisson_type', default='')
    point_mass    = options.get_bool(option_section, 'point_mass', default=False)
    
    dewiggle      = options.get_bool(option_section, 'dewiggle', default=False)

    # Fortuna introduces a truncation of the 1-halo term at large scales to avoid the halo exclusion problem
    # and a truncation of the NLA 2-halo term at small scales to avoid double-counting of the 1-halo term
    # The user can change these values.
    one_halo_ktrunc_ia = get_string_or_none(options, 'one_halo_ktrunc_ia', default=4.0) # h/Mpc or None
    two_halo_ktrunc_ia = get_string_or_none(options, 'two_halo_ktrunc_ia', default=6.0) # h/Mpc or None
    # General truncation of non-IA terms:
    one_halo_ktrunc = get_string_or_none(options, 'one_halo_ktrunc', default=0.1) # h/Mpc or None
    two_halo_ktrunc = get_string_or_none(options, 'two_halo_ktrunc', default=2.0) # h/Mpc or None
    
    # initiate pipeline parameters
    matter = False
    galaxy = False
    alignment = False
    
    hod_section_name = options.get_string(option_section, 'hod_section_name')

    # if (p_mI == True) and (p_mI_fortuna == True):
    #     raise Exception('Select either p_mI = True or p_mI_fortuna = True, \
    #                     both compute the matter-intrinsic power spectrum. \
    #                     p_mI_fortuna is the implementation used in Fortuna et al. 2021 paper.')
        
    # if (p_II == True) and (p_II_fortuna == True):
    #     raise Exception('Select either p_II = True or p_II_fortuna = True, \
    #                     all compute the matter-intrinsic power spectrum. \
    #                     p_II_fortuna is the implementation used in Fortuna et al. 2021 paper.')
        
    # if (p_gI == True) and (p_gI_fortuna == True):
    #     raise Exception('Select either p_gI = True or p_gI_fortuna = True, \
    #                     all compute the matter-intrinsic power spectrum. \
    #                     p_gI_fortuna i is the implementation used in Fortuna et al. 2021 paper.')

    if ((p_mm == True) or (p_gm == True) or (p_mI == True)):
        matter = True
    if (p_gg == True) or (p_gm == True) or (p_gI == True) or (p_mI == True) or (p_II == True):
        galaxy = True
    if (p_gI == True) or (p_mI == True) or (p_II == True):
        alignment = True

    population_name = options.get_string(option_section, 'output_suffix', default='').lower()
    if population_name != '':
        pop_name = f'_{population_name}'
    else:
        pop_name = ''

    check_mead    = options.has_value(option_section, 'use_mead2020_corrections')
    if check_mead:
        use_mead = options[option_section, 'use_mead2020_corrections']
        if use_mead == 'mead2020':
            mead_correction = 'nofeedback'
        elif use_mead == 'mead2020_feedback':
            mead_correction = 'feedback'
        elif use_mead == 'fit_feedback':
            mead_correction = 'fit'
            if not options.has_value(option_section, 'hod_section_name'):
                raise ValueError('To use the fit option for feedback that links HOD derived stellar mass fraction to the baryon \
                                  feedback one needs to provide the hod section name of used hod!')
        else:
            mead_correction = None
    else:
        mead_correction = None

    return p_mm, p_gg, p_gm, p_gI, p_mI, p_II, fortuna, \
           matter, galaxy, bnl, alignment, \
           one_halo_ktrunc, two_halo_ktrunc, one_halo_ktrunc_ia, two_halo_ktrunc_ia,\
           hod_section_name, mead_correction, dewiggle, point_mass, poisson_type, pop_name

def execute(block, config):

    p_mm, p_gg, p_gm, p_gI, p_mI, p_II, fortuna, \
    matter, galaxy, bnl, alignment,\
    one_halo_ktrunc, two_halo_ktrunc, one_halo_ktrunc_ia, two_halo_ktrunc_ia,\
    hod_section_name, mead_correction, dewiggle, point_mass, poisson_type, pop_name = config

    # load the halo mass, halo bias, mass and redshifts from the datablock
    dn_dlnm, b_dm, mass, z_vec = pk_lib.get_halo_functions(block)

    # Reads in the Fourier transform of the normalised dark matter halo profile 
    u_dm, u_sat, k_vec  = pk_lib.get_normalised_profile(block, mass, z_vec)

    nk = len(k_vec)

    # Interpolates in z only
    k_vec_original, plin_original = pk_lib.get_linear_power_spectrum(block, z_vec)
    # Using log-linear extrapolation which works better with power spectra, not so impotant when interpolating. 
    plin = pk_lib.log_linear_interpolation_k(plin_original, k_vec_original, k_vec)
    # load growth factor and scale factor
    growth_factor, scale_factor = pk_lib.get_growth_factor(block, z_vec, k_vec)

    # Optionally de-wiggle linear power spectrum as in Mead 2020:
    if mead_correction in ['feedback', 'nofeedback'] or dewiggle == True:
        plin = pk_lib.dewiggle(plin, k_vec, block)

    # AD: The following two lines only used for testing, need to be removed later on!
    # k_nl, p_nl = pk_lib.get_nonlinear_power_spectrum(block, z_vec)
    # pnl = pk_lib.log_linear_interpolation_k(p_nl, k_nl, k_vec)
    # block.replace_grid('matter_power_nl_mead', 'z', z_vec, 'k_h', k_vec, 'p_k', pnl)
    #block.replace_grid('matter_power_nl', 'z', z_vec, 'k_h', k_vec, 'p_k', pnl)
    
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
    mean_density0 = block['density', 'mean_density0'] * np.ones(len(z_vec))
    A_term = pk_lib.missing_mass_integral(mass, b_dm, dn_dlnm, mean_density0)
    
    # f_nu = omega_nu/omega_m with the same length as redshift
    fnu     = block[cosmo_params, 'fnu'] * np.ones(len(z_vec))
    omega_c = block[cosmo_params, 'omega_c']
    omega_m = block[cosmo_params, 'omega_m']
    omega_b = block[cosmo_params, 'omega_b']
    
    # If matter auto or cross power spectra are set to True
    if matter == True:
        # 2h term integral for matter
        I_m = pk_lib.Im_term(mass, u_dm, b_dm, dn_dlnm, mean_density0, A_term)
        # Matter halo profile
        matter_profile = pk_lib.matter_profile(mass, mean_density0, u_dm, np.zeros_like(fnu))
        # TODO: Why is there a matter profile and a matter_profile_1h_mm?
        
        if mead_correction == 'feedback':
            log10T_AGN = block['halo_model_parameters', 'logT_AGN']
            matter_profile_1h_mm = pk_lib.matter_profile_with_feedback(mass, mean_density0, u_dm, z_vec, omega_c, omega_m, omega_b, log10T_AGN, fnu)
        elif mead_correction == 'fit':
            # Reads f_star_extended form the HOD section. Either need to use a conditional HOD to get this value or to put it in block some other way.
            fstar_mm = pk_lib.load_fstar_mm(block, hod_section_name, z_vec, mass)
            mb = 10.0**block['halo_model_parameters', 'm_b']
            matter_profile_1h_mm = pk_lib.matter_profile_with_feedback_stellar_fraction_from_obs(mass, mean_density0, u_dm, z_vec, mb, fstar_mm, omega_c, omega_m, omega_b, fnu)
        else:
            matter_profile_1h_mm = pk_lib.matter_profile(mass, mean_density0, u_dm, fnu)
            
        if bnl == True:
            # TODO: This one uses matter_profile not matter_profile_1h_mm. Shouldn't we use the same profile everywhere?
            # AD: No, I_NL and 2-halo functions should use the mater_profile, no 1h! 
            # The corrections applied do not hold true for 2h regime!
            I_NL_mm = pk_lib.I_NL(mass, mass, matter_profile, matter_profile,
                                b_dm, b_dm, dn_dlnm, dn_dlnm, k_vec,
                                z_vec, A_term, mean_density0, beta_interp)
    
        if p_mm == True and bnl == False:
            if mead_correction in ['feedback', 'nofeedback']:
                sigma8_z = block['hmf', 'sigma8_z']
                neff     = block['hmf', 'neff']
                pk_mm_1h, pk_mm_2h, pk_mm_tot = pk_lib.compute_p_mm_mead(k_vec, plin, z_vec, mass,
                                                                        dn_dlnm, matter_profile_1h_mm, 
                                                                        I_m, sigma8_z, neff)
            else:
                pk_mm_1h, pk_mm_2h, pk_mm_tot = pk_lib.compute_p_mm(k_vec, plin, z_vec, mass,
                                                                    dn_dlnm, matter_profile_1h_mm, I_m,
                                                                    one_halo_ktrunc, two_halo_ktrunc)
            # save in the datablock
            # TODO: change this after testing.
            # block.put_grid('matter_1h_power', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm_1h)
            # block.put_grid('matter_2h_power', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm_2h)
            # block.put_grid('matter_power', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm_tot)
            # block.put_grid('matter_power_nl', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm_tot)
            
        if p_mm == True and bnl == True:
            pk_mm_1h, pk_mm_2h, pk_mm_tot = pk_lib.compute_p_mm_bnl(k_vec, plin, z_vec, mass, dn_dlnm,
                                                                    matter_profile_1h_mm, I_m, I_NL_mm,
                                                                    one_halo_ktrunc)
            # save in the datablock
            # TODO: change this after testing.
        block.put_grid('matter_1h_power', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm_1h)
        block.put_grid('matter_2h_power', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm_2h)
        # block.put_grid('matter_power', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm_tot)
        block.put_grid('matter_power_nl', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm_tot)
    # end of matter
    ##############################################################################################################
    if (galaxy == True) or (alignment == True):
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
                    mb = 10.0**block['halo_model_parameters', 'm_b']
                    matter_profile_1h = pk_lib.matter_profile_with_feedback_stellar_fraction_from_obs(mass, mean_density0,
                                                                                                    u_dm, z_vec, mb, fstar,
                                                                                                    omega_c, omega_m, omega_b, fnu)
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
            # end of galaxy setup
            ##############################################################################################################
            if alignment == True:
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
                                                                                beta_sat,  M_pivot, mass_avg)
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
            # end of IA setup
            ##############################################################################################################         
            # compute the power spectra and galaxy bias
            # TODO: check bg_halo_model
            
            if p_gg:
                if bnl:
                    pk_gg_1h, pk_gg_2h, pk_gg, bg_halo_model = pk_lib.compute_p_gg_bnl(block, k_vec, plin, z_vec, 
                    mass, dn_dlnm, profile_c, profile_s, I_c, I_s, I_NL_cs, I_NL_cc, I_NL_ss, mass_avg, poisson_type, one_halo_ktrunc)
                else:
                    pk_gg_1h, pk_gg_2h, pk_gg, bg_halo_model = pk_lib.compute_p_gg(block, k_vec, plin, z_vec, 
                            mass, dn_dlnm, profile_c, profile_s, I_c, I_s, mass_avg, poisson_type, one_halo_ktrunc, two_halo_ktrunc)
                block.put_grid(f'galaxy_power_1h{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gg_1h)
                block.put_grid(f'galaxy_power_2h{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gg_2h)
                block.put_grid(f'galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gg)
                block.put_grid(f'galaxy_linear_bias{suffix}', 'z', z_vec, 'k_h', k_vec, 'galaxybiastotal', bg_halo_model)
    
            if p_gm:
                if bnl:
                    pk_1h, pk_2h, pk_tot, galaxy_matter_linear_bias = pk_lib.compute_p_gm_bnl(block, k_vec, plin, z_vec, 
                        mass, dn_dlnm, profile_c, profile_s, matter_profile_1h, I_c, I_s, I_m, I_NL_cm, I_NL_sm, one_halo_ktrunc)
                else:
                    pk_1h, pk_2h, pk_tot, galaxy_matter_linear_bias = pk_lib.compute_p_gm(block, k_vec, plin, z_vec, 
                        mass, dn_dlnm, profile_c, profile_s, matter_profile_1h, I_c, I_s, I_m, one_halo_ktrunc, two_halo_ktrunc)
                block.put_grid(f'matter_galaxy_power_1h{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_1h)
                block.put_grid(f'matter_galaxy_power_2h{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_2h)
                block.put_grid(f'matter_galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_tot)
    
            # Intrinsic aligment power spectra (full halo model calculation)
            if fortuna:
                # Only used in Fortuna et al. 2021 implementation of IA power spectra
                # computes the effective power spectrum, mixing the linear and nonlinear ones:
                # Defaullt in Fortuna et al. 2021 is the non-linear power spectrum, so t_eff defaults to 0
                #
                # (1.-t_eff)*pnl + t_eff*plin
                #
                # load nonlinear power spectrum
                k_nl, p_nl = pk_lib.get_nonlinear_power_spectrum(block, z_vec)
                pnl = pk_lib.log_linear_interpolation_k(p_nl, k_nl, k_vec)
                t_eff = block.get_double('pk_parameters', 'linear_fraction_fortuna', default=0.0)
                pk_eff = (1.-t_eff)*pnl + t_eff*plin
            
            if p_II:
                if fortuna:
                    pk_II_1h, pk_II_2h, pk_II = pk_lib.compute_p_II_fortuna(block, k_vec, pk_eff, z_vec,
                        mass, dn_dlnm, s_align_profile, alignment_amplitude_2h_II, f_cen, one_halo_ktrunc_ia, two_halo_ktrunc_ia)
                else:
                    if bnl:
                        pk_II_1h, pk_II_2h, pk_II = pk_lib.compute_p_II_bnl(block, k_vec, plin, z_vec, 
                            mass, dn_dlnm, c_align_profile, s_align_profile, I_c_align_term, I_s_align_term, 
                            I_NL_ia_cc, I_NL_ia_cs, I_NL_ia_ss, one_halo_ktrunc_ia)
                    else:
                        pk_II_1h, pk_II_2h, pk_II = pk_lib.compute_p_II(block, k_vec, plin, z_vec, 
                            mass, dn_dlnm, c_align_profile, s_align_profile, I_c_align_term, I_s_align_term, one_halo_ktrunc_ia, two_halo_ktrunc_ia)
                block.put_grid(f'intrinsic_power_1h{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II_1h)
                block.put_grid(f'intrinsic_power_2h{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II_2h)
                block.put_grid(f'intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II)
            
            if p_gI:
                if fortuna:
                    pk_gI_1h, pk_gI_2h, pk_gI = pk_lib.compute_p_gI_fortuna(block, k_vec, pk_eff, z_vec,
                        mass, dn_dlnm, profile_c, s_align_profile, I_c, alignment_amplitude_2h, one_halo_ktrunc_ia, two_halo_ktrunc_ia)
                else:
                    if bnl:
                        pk_gI_1h_bnl, pk_gI_2h_bnl, pk_gI_bnl = pk_lib.compute_p_gI_bnl(block, k_vec, plin, z_vec, 
                            mass, dn_dlnm, profile_c, c_align_profile, s_align_profile, I_c, I_c_align_term, I_s_align_term, 
                            I_NL_ia_gc, I_NL_ia_gs, one_halo_ktrunc_ia)
                    else:
                        pk_gI_1h, pk_gI_2h, pk_gI = pk_lib.compute_p_gI(block, k_vec, plin, z_vec, 
                            mass, dn_dlnm, profile_c, c_align_profile, s_align_profile, I_c, I_c_align_term, I_s_align_term, one_halo_ktrunc_ia, two_halo_ktrunc_ia)
                block.put_grid(f'galaxy_intrinsic_power_1h{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gI_1h)
                block.put_grid(f'galaxy_intrinsic_power_2h{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gI_2h)
                block.put_grid(f'galaxy_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gI)
            
            if p_mI:
                if fortuna:
                    pk_mI_1h, pk_mI_2h, pk_mI = pk_lib.compute_p_mI_fortuna(block, k_vec, pk_eff, z_vec,
                        mass, dn_dlnm, matter_profile_1h, s_align_profile, alignment_amplitude_2h, f_cen, one_halo_ktrunc_ia, two_halo_ktrunc_ia)
                else:
                    if bnl:
                        pk_mI_1h_bnl, pk_mI_2h_bnl, pk_mI_bnl = pk_lib.compute_p_mI_bnl(block, k_vec, plin, z_vec, 
                            mass, dn_dlnm, matter_profile_1h, c_align_profile, s_align_profile, I_m, I_c_align_term, I_s_align_term, 
                            I_NL_ia_cm, I_NL_ia_sm, one_halo_ktrunc_ia)
                    else:
                        pk_mI_1h, pk_mI_2h, pk_mI = pk_lib.compute_p_mI(block, k_vec, plin, z_vec, 
                            mass, dn_dlnm, matter_profile_1h, c_align_profile, s_align_profile, I_m, I_c_align_term, I_s_align_term, one_halo_ktrunc_ia, two_halo_ktrunc_ia)
                block.put_grid(f'matter_intrinsic_power_1h{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mI_1h)
                block.put_grid(f'matter_intrinsic_power_2h{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mI_2h)
                block.put_grid(f'matter_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mI)
    return 0


def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass


