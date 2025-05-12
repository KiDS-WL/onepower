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

from cosmosis.datablock import names, option_section
import numpy as np
import numbers
import pk_util as pk_lib
from pk_lib_class import MatterSpectra, GalaxySpectra, AlignmentSpectra

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

    matter = p_mm
    galaxy = p_gg or p_gm
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

    matter_kwargs = {}
    galaxy_kwargs = {}
    align_kwargs = {}
    matter_kwargs['mead_correction'] = mead_correction
    matter_kwargs['dewiggle'] = dewiggle

    # Load the halo mass, halo bias, mass, and redshifts from the datablock
    dndlnm, halobias, mass, z_vec = pk_lib.get_halo_functions(block)
    
    # Reads in the Fourier transform of the normalized dark matter halo profile
    u_dm, u_sat, k_vec = pk_lib.get_normalised_profile(block, mass, z_vec)

    # Load the linear power spectrum and growth factor
    k_vec_original, plin_original = pk_lib.get_linear_power_spectrum(block, z_vec)
    plin = pk_lib.log_linear_interpolation_k(plin_original, k_vec_original, k_vec)
    growth_factor, scale_factor = pk_lib.get_growth_factor(block, z_vec, k_vec)

    matter_kwargs['dndlnm'] = dndlnm
    matter_kwargs['halobias'] = halobias
    matter_kwargs['mass'] = mass
    matter_kwargs['z_vec'] = z_vec
    matter_kwargs['u_dm'] = u_dm
    matter_kwargs['k_vec'] = k_vec
    matter_kwargs['matter_power_lin'] = plin

    if response or fortuna:
        k_nl, p_nl = pk_lib.get_nonlinear_power_spectrum(block, z_vec)
        pk_mm_in = pk_lib.log_linear_interpolation_k(p_nl, k_nl, k_vec)
        align_kwargs['matter_power_nl'] = pk_mm_in


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
        matter_kwargs['bnl'] = bnl
        matter_kwargs['beta_nl'] = beta_interp

    matter_kwargs['mean_density0'] = block['density', 'mean_density0'] * np.ones(len(z_vec))
    # f_nu = omega_nu / omega_m with the same length as redshift
    matter_kwargs['fnu'] = block[cosmo_params, 'fnu'] * np.ones(len(z_vec))
    matter_kwargs['omega_c'] = block[cosmo_params, 'omega_c']
    matter_kwargs['omega_m'] = block[cosmo_params, 'omega_m']
    matter_kwargs['omega_b'] = block[cosmo_params, 'omega_b']
    matter_kwargs['h0'] = block[cosmo_params, 'h0']
    matter_kwargs['n_s'] = block[cosmo_params, 'n_s']
    matter_kwargs['tcmb'] = block.get_double(cosmo_params, 'TCMB', default=2.7255)
    matter_kwargs['log10T_AGN'] = block['halo_model_parameters', 'logT_AGN']
    matter_kwargs['mb'] = 10.0**block['halo_model_parameters', 'm_b']
    sigma8_z = block['hmf', 'sigma8_z']
    neff = block['hmf', 'neff']
        
    if galaxy or alignment:
        hod_bins = block[hod_section_name, 'nbins']
        poisson_par = {
            'poisson_type': poisson_type,
            'poisson': get_string_or_none(block, 'pk_parameters', 'poisson', default=None),
            'M_0': get_string_or_none(block, 'pk_parameters', 'M_0', default=None),
            'slope': get_string_or_none(block, 'pk_parameters', 'slope', default=None)
        }

        N_cen = []
        N_sat = []
        numdencen = []
        numdensat = []
        f_cen = []
        f_sat = []
        mass_avg = []
        f_star = []
        # Check number of observable-redshift bins and read in the input for the HOD of each bin
        for nb in range(hod_bins):
            suffix = f'{pop_name}_{nb+1}' if hod_bins != 1 else f'{pop_name}'
            suffix_hod = f'_{nb+1}' if hod_bins != 1 else ''

            N_cen_in, N_sat_in, numdencen_in, numdensat_in, f_cen_in, f_sat_in, mass_avg_in, fstar_in = pk_lib.load_hods(block,
                                                                                                 hod_section_name, suffix_hod, z_vec, mass)
            N_cen.append(N_cen_in)
            N_sat.append(N_sat_in)
            numdencen.append(numdencen_in)
            numdensat.append(numdensat_in)
            f_cen.append(f_cen_in)
            f_sat.append(f_sat_in)
            mass_avg.append(mass_avg_in)
            f_star.append(fstar_in)
        
        galaxy_kwargs['u_sat'] = u_sat
        galaxy_kwargs['Ncen'] = N_cen
        galaxy_kwargs['Nsat'] = N_sat
        galaxy_kwargs['numdencen'] = numdencen
        galaxy_kwargs['numdensat'] = numdensat
        galaxy_kwargs['f_c'] = f_cen
        galaxy_kwargs['f_s'] = f_sat
        galaxy_kwargs['nbins'] = hod_bins
        galaxy_kwargs['pointmass'] = point_mass
    
    if alignment:
        align_kwargs['fortuna'] = fortuna
        align_kwargs['mass_avg'] = mass_avg
        align_kwargs['growth_factor'] = growth_factor
        align_kwargs['scale_factor'] = scale_factor
        # Load the 2h (effective) amplitude of the alignment signal from the data block.
        align_kwargs['alignment_gi'] = block[f'ia_large_scale_alignment{pop_name}', 'alignment_gi']
        align_kwargs['wkm_sat'] = pk_lib.get_satellite_alignment(block, k_vec, mass, z_vec, pop_name)
        align_kwargs['t_eff'] = block.get_double('pk_parameters', 'linear_fraction_fortuna', default=0.0)
        # Preparing the central and satellite terms
        if block[f'ia_small_scale_alignment{pop_name}', 'instance'] == 'halo_mass':
            align_kwargs['beta_sat'] = block[f'ia_small_scale_alignment{pop_name}', 'beta_sat']
            align_kwargs['mpivot_sat'] = block[f'ia_small_scale_alignment{pop_name}', 'M_pivot']
        else:
            align_kwargs['eta_sat'] = None
            align_kwargs['mpivot_sat'] = None
        if block[f'ia_large_scale_alignment{pop_name}', 'instance'] == 'halo_mass':
            align_kwargs['beta_cen'] = block[f'ia_large_scale_alignment{pop_name}', 'beta']
            align_kwargs['mpivot_cen'] = block[f'ia_large_scale_alignment{pop_name}', 'M_pivot']
        else:
            align_kwargs['beta_cen'] = None
            align_kwargs['mpivot_cen'] = None

    if matter:
        fstar_mm = pk_lib.load_fstar_mm(block, hod_section_name, z_vec, mass)
        matter_power = MatterSpectra(**matter_kwargs)
    if galaxy:
        comb_kwargs = {**matter_kwargs, **galaxy_kwargs}
        galaxy_power = GalaxySpectra(**comb_kwargs)
    if alignment:
        comb_kwargs = {**matter_kwargs, **galaxy_kwargs, **align_kwargs}
        alignment_power = AlignmentSpectra(**comb_kwargs)

    if p_mm:
        pk_mm_1h, pk_mm_2h, pk_mm, _ = matter_power.compute_power_spectrum_mm(
            one_halo_ktrunc = one_halo_ktrunc,
            two_halo_ktrunc = two_halo_ktrunc,
            sigma8_z = sigma8_z,
            neff = neff,
            fstar = fstar_mm
        )
        
        if response:
            # Here we save the computed Pmm to datablock as matter_power_hm,
            # but not replacing the Pnl with it, as in the response
            # method, the Pnl stays the same as one from CAMB
            block.put_grid('matter_power_hm', 'z', z_vec, 'k_h', k_vec, 'p_k_1h', pk_mm_1h[0])
            block.put_grid('matter_power_hm', 'z', z_vec, 'k_h', k_vec, 'p_k_2h', pk_mm_2h[0])
            block.put_grid('matter_power_hm', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm[0])
        else:
            block.put_grid('matter_power_nl', 'z', z_vec, 'k_h', k_vec, 'p_k_1h', pk_mm_1h[0])
            block.put_grid('matter_power_nl', 'z', z_vec, 'k_h', k_vec, 'p_k_2h', pk_mm_2h[0])
            block.put_grid('matter_power_nl', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mm[0])
        
    if p_gg:
        pk_gg_1h, pk_gg_2h, pk_gg, bg_linear = galaxy_power.compute_power_spectrum_gg(
            one_halo_ktrunc = one_halo_ktrunc,
            two_halo_ktrunc = two_halo_ktrunc,
            sigma8_z = sigma8_z,
            neff = neff,
            poisson_par = poisson_par,
            fstar = f_star
        )
        
        for nb in range(hod_bins):
            suffix = f'{pop_name}_{nb+1}' if hod_bins != 1 else f'{pop_name}'
            
            block.put_grid(f'galaxy_linear_bias{suffix}', 'z', z_vec, 'k_h', k_vec, 'bg_linear', bg_linear[nb])
            if response:
                block.put_grid(f'galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_1h', pk_gg_1h[nb] / pk_mm * pk_mm_in)
                block.put_grid(f'galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_2h', pk_gg_2h[nb] / pk_mm * pk_mm_in)
                block.put_grid(f'galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gg[nb] / pk_mm * pk_mm_in)
            else:
                block.put_grid(f'galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_1h', pk_gg_1h[nb])
                block.put_grid(f'galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_2h', pk_gg_2h[nb])
                block.put_grid(f'galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gg[nb])
    if p_gm:
    
        pk_gm_1h, pk_gm_2h, pk_gm, bgm_linear = galaxy_power.compute_power_spectrum_gm(
            one_halo_ktrunc = one_halo_ktrunc,
            two_halo_ktrunc = two_halo_ktrunc,
            sigma8_z = sigma8_z,
            neff = neff,
            poisson_par = poisson_par,
            fstar = f_star
        )
        for nb in range(hod_bins):
            suffix = f'{pop_name}_{nb+1}' if hod_bins != 1 else f'{pop_name}'
        
            block.put_grid(f'galaxy_matter_linear_bias{suffix}', 'z', z_vec, 'k_h', k_vec, 'bgm_linear', bgm_linear[nb])
            if response:
                block.put_grid(f'matter_galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_1h', pk_gm_1h[nb] / pk_mm * pk_mm_in)
                block.put_grid(f'matter_galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_2h', pk_gm_2h[nb] / pk_mm * pk_mm_in)
                block.put_grid(f'matter_galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gm[nb] / pk_mm * pk_mm_in)
            else:
                block.put_grid(f'matter_galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_1h', pk_gm_1h[nb])
                block.put_grid(f'matter_galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_2h', pk_gm_2h[nb])
                block.put_grid(f'matter_galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gm[nb])
    if p_II:
    
        pk_II_1h, pk_II_2h, pk_II, _ = alignment_power.compute_power_spectrum_ii(
            one_halo_ktrunc = one_halo_ktrunc_ia,
            two_halo_ktrunc = two_halo_ktrunc_ia,
            sigma8_z = sigma8_z,
            neff = neff,
            poisson_par = poisson_par,
            fstar = f_star
        )
        
        for nb in range(hod_bins):
            suffix = f'{pop_name}_{nb+1}' if hod_bins != 1 else f'{pop_name}'
        
            if response:
                block.put_grid(f'intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_1h', pk_II_1h[nb] / pk_mm * pk_mm_in)
                block.put_grid(f'intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_2h', pk_II_2h[nb] / pk_mm * pk_mm_in)
                block.put_grid(f'intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II[nb] / pk_mm * pk_mm_in)
            else:
                block.put_grid(f'intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_1h', pk_II_1h[nb])
                block.put_grid(f'intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_2h', pk_II_2h[nb])
                block.put_grid(f'intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II[nb])
    if p_gI:
    
        pk_gI_1h, pk_gI_2h, pk_gI, _ = alignment_power.compute_power_spectrum_gi(
            one_halo_ktrunc = one_halo_ktrunc_ia,
            two_halo_ktrunc = two_halo_ktrunc_ia,
            sigma8_z = sigma8_z,
            neff = neff,
            poisson_par = poisson_par,
            fstar = f_star
        )
        
        for nb in range(hod_bins):
            suffix = f'{pop_name}_{nb+1}' if hod_bins != 1 else f'{pop_name}'
            
            if response:
                block.put_grid(f'galaxy_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_1h', pk_gI_1h[nb] / pk_mm * pk_mm_in)
                block.put_grid(f'galaxy_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_2h', pk_gI_2h[nb] / pk_mm * pk_mm_in)
                block.put_grid(f'galaxy_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gI[nb] / pk_mm * pk_mm_in)
            else:
                block.put_grid(f'galaxy_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_1h', pk_gI_1h[nb])
                block.put_grid(f'galaxy_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_2h', pk_gI_2h[nb])
                block.put_grid(f'galaxy_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gI[nb])
    if p_mI:
        
        pk_mI_1h, pk_mI_2h, pk_mI, _ = alignment_power.compute_power_spectrum_mi(
            one_halo_ktrunc = one_halo_ktrunc_ia,
            two_halo_ktrunc = two_halo_ktrunc_ia,
            sigma8_z = sigma8_z,
            neff = neff,
            poisson_par = poisson_par,
            fstar = f_star
        )
        
        for nb in range(hod_bins):
            suffix = f'{pop_name}_{nb+1}' if hod_bins != 1 else f'{pop_name}'
            
            if response:
                block.put_grid(f'matter_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_1h', pk_mI_1h[nb] / pk_mm * pk_mm_in)
                block.put_grid(f'matter_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_2h', pk_mI_2h[nb] / pk_mm * pk_mm_in)
                block.put_grid(f'matter_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mI[nb] / pk_mm * pk_mm_in)
            else:
                block.put_grid(f'matter_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_1h', pk_mI_1h[nb])
                block.put_grid(f'matter_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_2h', pk_mI_2h[nb])
                block.put_grid(f'matter_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mI[nb])
                
    return 0

def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass
