"""
This module has two modes: response and direct.
If in direct mode it will directly calculate the power spectra.
If in response mode it will calculate the response of the halo model to different power spectra with respect to P_mm
and multiply that to an input P_mm to estimate the desired power, for example:
res_gg = P^hm_gg / P^hm_mm
res_gm = P^hm_gm / P^hm_mm
Then uses these in combination with input matter power spectra, P_mm, to create 3D power for P_gg and P_gm:
P_gg = res_gg * P_mm
P_gm = res_gm * P_mm

The following explains how P^hm_xy are calculated.
Calculates 3D power spectra using the halo model approach:
See section 2 of https://arxiv.org/pdf/2303.08752.pdf for details.

P_uv = P^2h_uv + P^1h_uv  (1)

P^1h_uv(k) = int_0^infty dM Wu(M, k) Wv(M, k) n(M)  (2)

P^2h_uv(k) = int_0^infty int_0^infty dM1 dM2 Phh(M1, M2, k) Wu(M1, k) Wv(M2, k) n(M1) n(M2)  (3)

Wx are the profile of the fields, u and v, showing how they fit into haloes.
n(M) is the halo mass function, quantifying the number of haloes of each mass, M.
Integrals are taken over halo mass.

The halo-halo power spectrum can be written as,

Phh(M1, M2, k) = b(M1) b(M2) P^lin_mm(k) (1 + beta_nl(M1, M2, k)) (4)

In the vanilla halo model, the 2-halo term is usually simplified by assuming that haloes are linearly biased with respect to matter.
This sets beta_nl to zero and effectively decouples the integrals in (3). Here we allow for both options to be calculated.
If you want the option with beta_nl, the beta_nl module has to be run before this module.

We truncate the 1-halo term so that it doesn't dominate at large scales.

Linear matter power spectrum needs to be provided as well. The halo_model_ingredients and hod modules (for everything but mm)
need to be run before this.

Current power spectra that we predict are:
mm: matter-matter
gg: galaxy-galaxy
gm: galaxy-matter

II: intrinsic-intrinsic alignments
gI: galaxy-intrinsic alignment
mI: matter-intrinsic alignment
"""

# TODO: IMPORTANT 1h term is too small compared to mead2020, see where this is coming from
# MA: It is smaller when mnu > 0 because of the fnu factor in matter profile. It might be a better approximation
# compared to simulations with neutrinos. We need to check this. I tried to compare with Euclid and Bacco emulator but
# had difficulties running them.

# NOTE: no truncation (halo exclusion problem) applied!

import numbers
import numpy as np
import pk_lib
from cosmosis.datablock import names, option_section

cosmo_params = names.cosmological_parameters



def get_string_or_none(cosmosis_block, section, name, default):
    """
    A helper function to return a number or None explicitly from config files
    or return None if no value is present.
    """
    if cosmosis_block.has_value(section, name):
        test_param = cosmosis_block.get(section, name)
        if isinstance(test_param, numbers.Number):
            param = cosmosis_block.get_double(section, name, default)
        if isinstance(test_param, str):
            str_in = cosmosis_block.get_string(section, name)
            if str_in == 'None':
                param = None
    else:
        try:
            param = cosmosis_block.get_double(section, name, default)
        except:
            param = None

    if not isinstance(param, (numbers.Number, type(None))):
        raise ValueError(f'Parameter {name} is not an instance of a number or NoneType!')

    return param

def setup(options):
    p_mm = options.get_bool(option_section, 'p_mm', default=False)
    p_gg = options.get_bool(option_section, 'p_gg', default=False)
    p_gm = options.get_bool(option_section, 'p_gm', default=False)
    p_gI = options.get_bool(option_section, 'p_gI', default=False)
    p_mI = options.get_bool(option_section, 'p_mI', default=False)
    p_II = options.get_bool(option_section, 'p_II', default=False)

    # If True, calculate the response of the halo model for the requested power spectra compared to matter power
    # multiplies this to input non-linear matter power spectra.
    response = options.get_bool(option_section, 'response', default=False)

    # If true, use the IA formalism of Fortuna et al. 2021: Truncated NLA at high k + 1-halo term
    fortuna = options.get_bool(option_section, 'fortuna', default=False)
    # If True, uses beta_nl
    bnl = options.get_bool(option_section, 'bnl', default=False)

    poisson_type = options.get_string(option_section, 'poisson_type', default='')
    point_mass = options.get_bool(option_section, 'point_mass', default=False)

    dewiggle = options.get_bool(option_section, 'dewiggle', default=False)

    # Fortuna introduces a truncation of the 1-halo term at large scales to avoid the halo exclusion problem
    # and a truncation of the NLA 2-halo term at small scales to avoid double-counting of the 1-halo term
    # The user can change these values.
    one_halo_ktrunc_ia = get_string_or_none(options, option_section, 'one_halo_ktrunc_ia', default=4.0)  # h/Mpc or None
    two_halo_ktrunc_ia = get_string_or_none(options, option_section, 'two_halo_ktrunc_ia', default=6.0)  # h/Mpc or None
    # General truncation of non-IA terms:
    one_halo_ktrunc = get_string_or_none(options, option_section, 'one_halo_ktrunc', default=0.1)  # h/Mpc or None
    two_halo_ktrunc = get_string_or_none(options, option_section, 'two_halo_ktrunc', default=2.0)  # h/Mpc or None

    # Initiate pipeline parameters
    matter = False
    galaxy = False
    alignment = False

    hod_section_name = options.get_string(option_section, 'hod_section_name')

    matter = p_mm or p_gm or p_mI
    galaxy = p_gg or p_gm or p_gI or p_mI or p_II
    alignment = p_gI or p_mI or p_II

    population_name = options.get_string(option_section, 'output_suffix', default='').lower()
    pop_name = f'_{population_name}' if population_name else ''

    # Option to set similar corrections to HMcode2020
    use_mead = options.get_string(option_section, 'use_mead2020_corrections', default='None')

    # Mapping of use_mead values to mead_correction values
    mead_correction_map = {
        'mead2020': 'nofeedback',
        'mead2020_feedback': 'feedback',
        'fit_feedback': 'fit',
        # Add more mappings here if needed
    }

    # Determine the mead_correction based on the mapping
    mead_correction = mead_correction_map.get(use_mead, None)
    if mead_correction == 'fit':
        if not options.has_value(option_section, 'hod_section_name'):
            raise ValueError('To use the fit option for feedback that links HOD derived stellar mass fraction to the baryon '
                             'feedback one needs to provide the hod section name of used hod!')

    return p_mm, p_gg, p_gm, p_gI, p_mI, p_II, response, fortuna, matter, galaxy, bnl, alignment, one_halo_ktrunc, two_halo_ktrunc, one_halo_ktrunc_ia, two_halo_ktrunc_ia, hod_section_name, mead_correction, dewiggle, point_mass, poisson_type, pop_name

def execute(block, config):
    p_mm, p_gg, p_gm, p_gI, p_mI, p_II, response, fortuna, matter, galaxy, bnl, alignment, one_halo_ktrunc, two_halo_ktrunc, one_halo_ktrunc_ia, two_halo_ktrunc_ia, hod_section_name, mead_correction, dewiggle, point_mass, poisson_type, pop_name = config

    # Load the halo mass, halo bias, mass, and redshifts from the datablock
    dn_dlnm, b_dm, mass, z_vec = pk_lib.get_halo_functions(block)

    # Reads in the Fourier transform of the normalized dark matter halo profile
    u_dm, u_sat, k_vec = pk_lib.get_normalised_profile(block, mass, z_vec)

    # Load the linear power spectrum and growth factor
    k_vec_original, plin_original = pk_lib.get_linear_power_spectrum(block, z_vec)
    plin = pk_lib.log_linear_interpolation_k(plin_original, k_vec_original, k_vec)
    growth_factor, scale_factor = pk_lib.get_growth_factor(block, z_vec, k_vec)

    if response:
        k_nl, p_nl = pk_lib.get_nonlinear_power_spectrum(block, z_vec)
        pk_mm_in = pk_lib.log_linear_interpolation_k(p_nl, k_nl, k_vec)

    # Optionally de-wiggle linear power spectrum as in Mead 2020:
    if mead_correction in ['feedback', 'nofeedback'] or dewiggle:
        plin = pk_lib.dewiggle(plin, k_vec, block)

    # AD: The following line is only used for testing, will be removed when we start running final chains!
    #k_nl, p_nl = pk_lib.get_nonlinear_power_spectrum(block, z_vec)
    #pk_mm_in = pk_lib.log_linear_interpolation_k(p_nl, k_nl, k_vec)
    #block.replace_grid('matter_power_nl_mead', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm_in)

    # Add the non-linear P_hh to the 2h term
    if bnl:
        # Reads beta_nl from the block
        if block.has_value('bnl', 'beta_interp'):
            beta_interp = block.get_double_array_nd('bnl', 'beta_interp')
        else:
            raise Exception("You've set bnl = True. Looked for beta_intep in bnl, but didn't find it. "
                            "Run bnl_interface.py to set this up.\n")
        if beta_interp.shape == np.array([0.0]).shape:
            raise ValueError('Non-linear halo bias module bnl is not initialized, or you have deleted it too early! '
                             'This might be because you ran bnl_interface_delete.py before this module. \n')

        integrand_12 = pk_lib.prepare_I12_integrand(b_dm, b_dm, mass, mass, dn_dlnm, dn_dlnm, beta_interp)
        integrand_21 = pk_lib.prepare_I21_integrand(b_dm, b_dm, mass, mass, dn_dlnm, dn_dlnm, beta_interp)
        integrand_22 = pk_lib.prepare_I22_integrand(b_dm, b_dm, mass, mass, dn_dlnm, dn_dlnm, beta_interp)

    # Accounts for the missing low mass haloes in the integrals for the 2h term.
    # Assumes all missing mass is in haloes of mass M_min.
    # This is calculated separately for each redshift
    # TODO: check if this is needed for the IA section
    mean_density0 = block['density', 'mean_density0'] * np.ones(len(z_vec))
    A_term = pk_lib.missing_mass_integral(mass, b_dm, dn_dlnm, mean_density0)

    # f_nu = omega_nu / omega_m with the same length as redshift
    fnu = block[cosmo_params, 'fnu'] * np.ones(len(z_vec))
    omega_c = block[cosmo_params, 'omega_c']
    omega_m = block[cosmo_params, 'omega_m']
    omega_b = block[cosmo_params, 'omega_b']

    # If matter auto or cross power spectra are set to True
    if matter:
        # 2h term integral for matter
        I_m = pk_lib.Im_term(mass, u_dm, b_dm, dn_dlnm, mean_density0, A_term)
        # Matter halo profile
        matter_profile = pk_lib.matter_profile(mass, mean_density0, u_dm, np.zeros_like(fnu))

        if mead_correction == 'feedback':
            log10T_AGN = block['halo_model_parameters', 'logT_AGN']
            matter_profile_1h_mm = pk_lib.matter_profile_with_feedback(mass, mean_density0, u_dm, z_vec, omega_c,
                                                                       omega_m, omega_b, log10T_AGN, fnu)
        elif mead_correction == 'fit':
            # Reads f_star_extended form the HOD section. Either need to use a conditional HOD to get this value or to put it in block some other way.
            fstar_mm = pk_lib.load_fstar_mm(block, hod_section_name, z_vec, mass)
            mb = 10.0**block['halo_model_parameters', 'm_b']
            matter_profile_1h_mm = pk_lib.matter_profile_with_feedback_stellar_fraction_from_obs(mass, mean_density0, u_dm, z_vec, mb,
                                                                                                 fstar_mm, omega_c, omega_m, omega_b, fnu)
        else:
            matter_profile_1h_mm = pk_lib.matter_profile(mass, mean_density0, u_dm, fnu)

        if bnl:
            I_NL_mm = pk_lib.I_NL(mass, mass, matter_profile, matter_profile, b_dm, b_dm,
                                  dn_dlnm, dn_dlnm, A_term, mean_density0, beta_interp, integrand_12, integrand_21, integrand_22)

        if p_mm or response:
            if bnl:
                pk_mm_1h, pk_mm_2h, pk_mm = pk_lib.compute_p_mm_bnl(k_vec, plin, mass, dn_dlnm,
                                                                   matter_profile_1h_mm, I_m, I_NL_mm,
                                                                   one_halo_ktrunc)
            else:
                if mead_correction in ['feedback', 'nofeedback']:
                    sigma8_z = block['hmf', 'sigma8_z']
                    neff = block['hmf', 'neff']
                    pk_mm_1h, pk_mm_2h, pk_mm = pk_lib.compute_p_mm_mead(k_vec, plin, mass,
                                                                         dn_dlnm, matter_profile_1h_mm,
                                                                         sigma8_z, neff)
                else:
                    pk_mm_1h, pk_mm_2h, pk_mm = pk_lib.compute_p_mm(k_vec, plin, mass,
                                                                    dn_dlnm, matter_profile_1h_mm, I_m,
                                                                    one_halo_ktrunc, two_halo_ktrunc)
            if response:
                # Here we save the computed Pmm to datablock as matter_power_hm,
                # but not replacing the Pnl with it, as in the response
                # method, the Pnl stays the same as one from CAMB
                block.put_grid('matter_power_hm', 'z', z_vec, 'k_h', k_vec, 'p_k_1h', pk_mm_1h)
                block.put_grid('matter_power_hm', 'z', z_vec, 'k_h', k_vec, 'p_k_2h', pk_mm_2h)
                block.put_grid('matter_power_hm', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm)
            else:
                block.put_grid('matter_power_nl', 'z', z_vec, 'k_h', k_vec, 'p_k_1h', pk_mm_1h)
                block.put_grid('matter_power_nl', 'z', z_vec, 'k_h', k_vec, 'p_k_2h', pk_mm_2h)
                block.put_grid('matter_power_nl', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm)

    if galaxy or alignment:
        hod_bins = block[hod_section_name, 'nbins']

        # Check number of observable-redshift bins and read in the input for the HOD of each bin
        for nb in range(hod_bins):
            suffix = f'{pop_name}_{nb+1}' if hod_bins != 1 else f'{pop_name}'
            suffix_hod = f'_{nb+1}' if hod_bins != 1 else ''

            N_cen, N_sat, numdencen, numdensat, f_cen, f_sat, mass_avg, fstar = pk_lib.load_hods(block,
                                                                                                 hod_section_name, suffix_hod, z_vec, mass)

            if galaxy:
                # Preparing the 1-halo term
                # Computes the profiles for centrals and satellites.
                # These are the W_u(M,k) functions in Asgari, Mead, Heymans 2023: 2303.08752
                # We assume that the centrals are in the centre of the halo,
                # and set their normalized profile to 1 everywhere.

                profile_c = pk_lib.galaxy_profile(N_cen, numdencen, f_cen, np.ones_like(u_sat))
                profile_s = pk_lib.galaxy_profile(N_sat, numdensat, f_sat, u_sat)

                # Calculate the 2-halo integrals for centrals and satellites
                I_c = pk_lib.Ig_term(mass, profile_c, b_dm, dn_dlnm)
                I_s = pk_lib.Ig_term(mass, profile_s, b_dm, dn_dlnm)

                if mead_correction == 'fit' or point_mass:
                    # Include point mass and gas contribution to the GGL power spectrum, defined from HOD
                    # Maybe extend to input the mass per bin!
                    mb = 10.0**block['halo_model_parameters', 'm_b']
                    matter_profile_1h = pk_lib.matter_profile_with_feedback_stellar_fraction_from_obs(mass, mean_density0,
                                                                                                    u_dm, z_vec, mb, fstar,
                                                                                                    omega_c, omega_m, omega_b, fnu)
                else:
                    matter_profile_1h = matter_profile_1h_mm.copy()

                if bnl:
                    if p_gg:
                        I_NL_gg = pk_lib.I_NL(mass, mass, profile_c + profile_s, profile_c + profile_s, b_dm, b_dm,
                                              dn_dlnm, dn_dlnm, A_term, mean_density0, beta_interp, integrand_12, integrand_21, integrand_22)
                    if p_gm:
                        I_NL_gm = pk_lib.I_NL(mass, mass, profile_c + profile_s, matter_profile, b_dm, b_dm,
                                              dn_dlnm, dn_dlnm, A_term, mean_density0, beta_interp, integrand_12, integrand_21, integrand_22)

            if alignment:
                # Load the 2h (effective) amplitude of the alignment signal from the data block.
                alignment_gi = block[f'ia_large_scale_alignment{pop_name}', 'alignment_gi']
                alignment_amplitude_2h, alignment_amplitude_2h_II, C1 = pk_lib.compute_two_halo_alignment(alignment_gi, growth_factor, mean_density0)
                wkm = pk_lib.get_satellite_alignment(block, k_vec, mass, z_vec, pop_name)
                # Preparing the central and satellite terms
                if block[f'ia_small_scale_alignment{pop_name}', 'instance'] == 'halo_mass':
                    beta_sat = block[f'ia_small_scale_alignment{pop_name}', 'beta_sat']
                    M_pivot = block[f'ia_small_scale_alignment{pop_name}', 'M_pivot']
                    s_align_profile = pk_lib.satellite_alignment_profile(N_sat, numdensat, f_sat, wkm,
                                                                         beta_sat, M_pivot, mass_avg)
                else:
                    s_align_profile = pk_lib.satellite_alignment_profile(N_sat, numdensat, f_sat, wkm)
                if block[f'ia_large_scale_alignment{pop_name}', 'instance'] == 'halo_mass':
                    beta = block[f'ia_large_scale_alignment{pop_name}', 'beta']
                    M_pivot = block[f'ia_large_scale_alignment{pop_name}', 'M_pivot']
                    c_align_profile = pk_lib.central_alignment_profile(mass, scale_factor, growth_factor,
                                                                      f_cen, C1, beta, M_pivot, mass_avg)
                else:
                    c_align_profile = pk_lib.central_alignment_profile(mass, scale_factor, growth_factor, f_cen, C1)
                # TODO: does this need the A_term?
                I_c_align_term = pk_lib.Ig_align_term(mass, c_align_profile, b_dm, dn_dlnm, mean_density0, A_term)
                I_s_align_term = pk_lib.Ig_align_term(mass, s_align_profile, b_dm, dn_dlnm, mean_density0, A_term)

                if bnl:
                    if p_mI:
                        I_NL_ia_gm = pk_lib.I_NL(mass, mass, c_align_profile + s_align_profile, matter_profile, b_dm, b_dm,
                                                 dn_dlnm, dn_dlnm, A_term, mean_density0, beta_interp, integrand_12, integrand_21, integrand_22)

                    if p_II:
                        I_NL_ia_cc = pk_lib.I_NL(mass, mass, c_align_profile, c_align_profile, b_dm, b_dm,
                                                 dn_dlnm, dn_dlnm, A_term, mean_density0, beta_interp, integrand_12, integrand_21, integrand_22)
                        I_NL_ia_cs = pk_lib.I_NL(mass, mass, c_align_profile, s_align_profile, b_dm, b_dm,
                                                 dn_dlnm, dn_dlnm, A_term, mean_density0, beta_interp, integrand_12, integrand_21, integrand_22)
                        I_NL_ia_ss = pk_lib.I_NL(mass, mass, s_align_profile, s_align_profile, b_dm, b_dm,
                                                 dn_dlnm, dn_dlnm, A_term, mean_density0, beta_interp, integrand_12, integrand_21, integrand_22)

                    if p_gI:
                        I_NL_ia_gc = pk_lib.I_NL(mass, mass, c_align_profile, profile_c, b_dm, b_dm,
                                                 dn_dlnm, dn_dlnm, A_term, mean_density0, beta_interp, integrand_12, integrand_21, integrand_22)
                        I_NL_ia_gs = pk_lib.I_NL(mass, mass, s_align_profile, profile_c, b_dm, b_dm,
                                                 dn_dlnm, dn_dlnm, A_term, mean_density0, beta_interp, integrand_12, integrand_21, integrand_22)

            if p_gg:
                # First check if the Poisson distribution for the satellites is disturbed.
                poisson_par = {
                    'poisson_type': poisson_type,
                    'poisson': get_string_or_none(block, 'pk_parameters', 'poisson', default=None),
                    'M_0': get_string_or_none(block, 'pk_parameters', 'M_0', default=None),
                    'slope': get_string_or_none(block, 'pk_parameters', 'slope', default=None)
                }

                if bnl:  # If bnl is True use the beyond linear halo bias formalism
                    pk_gg_1h, pk_gg_2h, pk_gg, bg_linear = pk_lib.compute_p_gg_bnl(k_vec, plin,
                                                                                  mass, dn_dlnm, profile_c, profile_s, I_c, I_s, I_NL_gg,
                                                                                  mass_avg, poisson_par, one_halo_ktrunc)
                else:  # If bnl is not true then just use linear halo bias
                    """
                    For Poisson distributed satellites, we have
                    < N_sat (N_sat-1) > = <N_sat>^2 = lambda^2 = <N_sat^2> - <N_sat>^2
                    We define the Poisson parameters as:
                    Poisson = <N_sat(N_sat-1)>/<N_sat>^2
                    A super-Poissonian distribution is a probability distribution
                    that has a larger variance than a Poisson distribution with the same mean.
                    Conversely, a sub-Poissonian distribution has a smaller variance.
                    Why do we assume Poisson distribution for satellites?
                    Poisson distribution assumes:
                    1- The satellites are independent. The existence of one satellite does not
                       affect another satellite.
                    2- Two satellites cannot exist in the same location.
                    3- The average number of satellites per halo is independent of any occurrence?
                    """
                    pk_gg_1h, pk_gg_2h, pk_gg, bg_linear = pk_lib.compute_p_gg(k_vec, plin,
                                                                               mass, dn_dlnm, profile_c, profile_s, I_c, I_s, mass_avg,
                                                                               poisson_par, one_halo_ktrunc, two_halo_ktrunc)
                block.put_grid(f'galaxy_linear_bias{suffix}', 'z', z_vec, 'k_h', k_vec, 'bg_linear', bg_linear)
                if response:
                    block.put_grid(f'galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_1h', pk_gg_1h / pk_mm * pk_mm_in)
                    block.put_grid(f'galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_2h', pk_gg_2h / pk_mm * pk_mm_in)
                    block.put_grid(f'galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gg / pk_mm * pk_mm_in)
                else:
                    block.put_grid(f'galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_1h', pk_gg_1h)
                    block.put_grid(f'galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_2h', pk_gg_2h)
                    block.put_grid(f'galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gg)

            if p_gm:
                if bnl:  # If bnl is True use the beyond linear halo bias formalism
                    pk_1h, pk_2h, pk_gm, bgm_linear = pk_lib.compute_p_gm_bnl(k_vec, plin,
                                                                             mass, dn_dlnm, profile_c, profile_s, matter_profile_1h, I_c, I_s, I_m,
                                                                             I_NL_gm, one_halo_ktrunc)
                else:  # If bnl is not true then just use linear halo bias
                    pk_1h, pk_2h, pk_gm, bgm_linear = pk_lib.compute_p_gm(k_vec, plin,
                                                                         mass, dn_dlnm, profile_c, profile_s, matter_profile_1h, I_c, I_s, I_m,
                                                                         one_halo_ktrunc, two_halo_ktrunc)
                block.put_grid(f'galaxy_matter_linear_bias{suffix}', 'z', z_vec, 'k_h', k_vec, 'bgm_linear', bgm_linear)
                if response:
                    block.put_grid(f'matter_galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_1h', pk_1h / pk_mm * pk_mm_in)
                    block.put_grid(f'matter_galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_2h', pk_2h / pk_mm * pk_mm_in)
                    block.put_grid(f'matter_galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gm / pk_mm * pk_mm_in)
                else:
                    block.put_grid(f'matter_galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_1h', pk_1h)
                    block.put_grid(f'matter_galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_2h', pk_2h)
                    block.put_grid(f'matter_galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gm)

            if fortuna:
                # Only used in Fortuna et al. 2021 implementation of IA power spectra
                # Computes the effective power spectrum, mixing the linear and nonlinear ones:
                # Default in Fortuna et al. 2021 is the non-linear power spectrum, so t_eff defaults to 0
                #
                # (1.-t_eff)*pnl + t_eff*plin
                #
                # Load nonlinear power spectrum
                k_nl, p_nl = pk_lib.get_nonlinear_power_spectrum(block, z_vec)
                pnl = pk_lib.log_linear_interpolation_k(p_nl, k_nl, k_vec)
                t_eff = block.get_double('pk_parameters', 'linear_fraction_fortuna', default=0.0)
                pk_eff = (1.0 - t_eff) * pnl + t_eff * plin

            if p_II:
                if fortuna:
                    pk_II_1h, pk_II_2h, pk_II = pk_lib.compute_p_II_fortuna(k_vec, pk_eff,
                                                                            mass, dn_dlnm, s_align_profile, alignment_amplitude_2h_II,
                                                                            f_cen, one_halo_ktrunc_ia, two_halo_ktrunc_ia)
                else:
                    if bnl:
                        pk_II_1h, pk_II_2h, pk_II = pk_lib.compute_p_II_bnl(k_vec, plin,
                                                                            mass, dn_dlnm, s_align_profile,
                                                                            I_c_align_term, I_s_align_term,
                                                                            I_NL_ia_cc, I_NL_ia_cs, I_NL_ia_ss,
                                                                            one_halo_ktrunc_ia)
                    else:
                        pk_II_1h, pk_II_2h, pk_II = pk_lib.compute_p_II(k_vec, plin,
                                                                        mass, dn_dlnm, s_align_profile,
                                                                        I_c_align_term, I_s_align_term,
                                                                        one_halo_ktrunc_ia, two_halo_ktrunc_ia)
                if response:
                    block.put_grid(f'intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_1h', pk_II_1h / pk_mm * pk_mm_in)
                    block.put_grid(f'intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_2h', pk_II_2h / pk_mm * pk_mm_in)
                    block.put_grid(f'intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II / pk_mm * pk_mm_in)
                else:
                    block.put_grid(f'intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_1h', pk_II_1h)
                    block.put_grid(f'intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_2h', pk_II_2h)
                    block.put_grid(f'intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II)

            if p_gI:
                if fortuna:
                    pk_gI_1h, pk_gI_2h, pk_gI = pk_lib.compute_p_gI_fortuna(k_vec, pk_eff,
                                                                            mass, dn_dlnm, profile_c, s_align_profile, I_c, alignment_amplitude_2h, one_halo_ktrunc_ia, two_halo_ktrunc_ia)
                else:
                    if bnl:
                        pk_gI_1h, pk_gI_2h, pk_gI = pk_lib.compute_p_gI_bnl(k_vec, plin,
                                                                            mass, dn_dlnm, profile_c, s_align_profile, I_c, I_c_align_term, I_s_align_term,
                                                                            I_NL_ia_gc, I_NL_ia_gs, one_halo_ktrunc_ia)
                    else:
                        pk_gI_1h, pk_gI_2h, pk_gI = pk_lib.compute_p_gI(k_vec, plin,
                                                                        mass, dn_dlnm, profile_c, s_align_profile, I_c,
                                                                        I_c_align_term, I_s_align_term, one_halo_ktrunc_ia, two_halo_ktrunc_ia)
                if response:
                    block.put_grid(f'galaxy_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_1h', pk_gI_1h / pk_mm * pk_mm_in)
                    block.put_grid(f'galaxy_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_2h', pk_gI_2h / pk_mm * pk_mm_in)
                    block.put_grid(f'galaxy_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gI / pk_mm * pk_mm_in)
                else:
                    block.put_grid(f'galaxy_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_1h', pk_gI_1h)
                    block.put_grid(f'galaxy_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_2h', pk_gI_2h)
                    block.put_grid(f'galaxy_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gI)

            if p_mI:
                if fortuna:
                    pk_mI_1h, pk_mI_2h, pk_mI = pk_lib.compute_p_mI_fortuna(k_vec, pk_eff,
                                                                            mass, dn_dlnm, matter_profile_1h, s_align_profile, alignment_amplitude_2h,
                                                                            f_cen, one_halo_ktrunc_ia, two_halo_ktrunc_ia)
                else:
                    if bnl:
                        pk_mI_1h, pk_mI_2h, pk_mI = pk_lib.compute_p_mI_bnl(k_vec, plin,
                                                                            mass, dn_dlnm, matter_profile_1h, s_align_profile, I_m, I_c_align_term, I_s_align_term,
                                                                            I_NL_ia_gm, one_halo_ktrunc_ia)
                    else:
                        pk_mI_1h, pk_mI_2h, pk_mI = pk_lib.compute_p_mI(k_vec, plin,
                                                                        mass, dn_dlnm, matter_profile_1h, s_align_profile, I_m, I_c_align_term, I_s_align_term,
                                                                        one_halo_ktrunc_ia, two_halo_ktrunc_ia)
                if response:
                    block.put_grid(f'matter_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_1h', pk_mI_1h / pk_mm * pk_mm_in)
                    block.put_grid(f'matter_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_2h', pk_mI_2h / pk_mm * pk_mm_in)
                    block.put_grid(f'matter_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mI / pk_mm * pk_mm_in)
                else:
                    block.put_grid(f'matter_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_1h', pk_mI_1h)
                    block.put_grid(f'matter_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_2h', pk_mI_2h)
                    block.put_grid(f'matter_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mI)

    return 0

def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass
