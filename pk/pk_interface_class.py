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
import pk_util
from pk_lib_class import MatterSpectra, GalaxySpectra, AlignmentSpectra

cosmo_params = names.cosmological_parameters

parameters_models = {
    'Zheng': ['log10_Mmin', 'sigma', 'log10_M0', 'log10_M1', 'alpha'],
    'Zhai': ['log10_Mmin', 'sigma', 'log10_Msat', 'log10_Mcut', 'alpha'],
    'Cacciato': [
        'log10_obs_norm_c', 'log10_m_ch', 'g1', 'g2', 'sigma_log10_O_c',
        'norm_s', 'pivot', 'alpha_s', 'beta_s', 'b0', 'b1', 'b2'
    ]
}

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
    """Setup function to parse options and return configuration."""
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
    hod_values_name = options.get_string(option_section, 'hod_values_name', default='hod_parameters').lower()

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
    if mead_correction == 'fit' and not options.has_value(option_section, 'hod_section_name'):
        raise ValueError('To use the fit option for feedback that links HOD derived stellar mass fraction to the baryon feedback one needs to provide the hod section name of used hod!')

    hod_model = 'Cacciato'

    hod_params = {}
    hod_settings = {}
    if options.has_value(hod_section_name, 'observables_file'):
        hod_settings['observables_file'] = options.get_string(hod_section_name, 'observables_file')
        hod_settings['observable_z'] = True
    else:
        hod_settings['observables_file'] = None
        hod_settings['observable_z'] = False
        hod_settings['obs_min'] = np.asarray([options[hod_section_name, 'log10_obs_min']]).flatten()
        hod_settings['obs_max'] = np.asarray([options[hod_section_name, 'log10_obs_max']]).flatten()
        hod_settings['zmin'] = np.asarray([options[hod_section_name, 'zmin']]).flatten()
        hod_settings['zmax'] = np.asarray([options[hod_section_name, 'zmax']]).flatten()
        hod_settings['nz'] = options[hod_section_name, 'nz']
    hod_settings['nobs'] = options[hod_section_name, 'nobs']
    hod_settings['save_observable'] = options.get_bool(hod_section_name, 'save_observable', default=True)
    hod_settings['observable_mode'] = options.get_string(hod_section_name, 'observable_mode', default='obs_z')
    hod_settings['observable_h_unit'] = options.get_string(hod_section_name, 'observable_h_unit', default='1/h^2').lower()
    hod_settings['z_median'] = options.get_double(hod_section_name, 'z_median', default=0.1)
    
    hod_settings_mm = {}
    if options.has_value(hod_section_name, 'observables_file'):
        hod_settings_mm['observables_file'] = options.get_string(hod_section_name, 'observables_file')
    else:
        hod_settings_mm['observables_file'] = None
        hod_settings_mm['obs_min'] = np.array([hod_settings['obs_min'].min()])
        hod_settings_mm['obs_max'] = np.array([hod_settings['obs_max'].max()])
        hod_settings_mm['zmin'] = np.array([hod_settings['zmin'].min()])
        hod_settings_mm['zmax'] = np.array([hod_settings['zmax'].max()])
        hod_settings_mm['nz'] = 15
    hod_settings_mm['nobs'] = 100
    #hod_settings_mm['save_observable'] = options.get_bool(hod_section_name, 'save_observable', default=True)
    hod_settings_mm['observable_mode'] = options.get_string(hod_section_name, 'observable_mode', default='obs_z')
    hod_settings_mm['observable_h_unit'] = options.get_string(hod_section_name, 'observable_h_unit', default='1/h^2').lower()
    hod_settings_mm['z_median'] = options.get_double(hod_section_name, 'z_median', default=0.1)
    if options.has_value(hod_section_name, 'observable_section_name'):
        hod_settings['observable_section_name'] = options.get_string(
            hod_section_name, 'observable_section_name', default='stellar_mass_function'
        ).lower()
    

    return p_mm, p_gg, p_gm, p_gI, p_mI, p_II, response, fortuna, matter, galaxy, bnl, alignment, one_halo_ktrunc, two_halo_ktrunc, one_halo_ktrunc_ia, two_halo_ktrunc_ia, hod_section_name, mead_correction, dewiggle, point_mass, poisson_type, pop_name, hod_model, hod_params, hod_settings, hod_settings_mm, hod_values_name

def execute(block, config):
    """Execute function to compute power spectra based on configuration."""
    p_mm, p_gg, p_gm, p_gI, p_mI, p_II, response, fortuna, matter, galaxy, bnl, alignment, one_halo_ktrunc, two_halo_ktrunc, one_halo_ktrunc_ia, two_halo_ktrunc_ia, hod_section_name, mead_correction, dewiggle, point_mass, poisson_type, pop_name, hod_model, hod_params, hod_settings, hod_settings_mm, hod_values_name = config

    matter_kwargs = {
        'mead_correction': mead_correction,
        'dewiggle': dewiggle,
    }
    galaxy_kwargs = {}
    align_kwargs = {}

    # Load the halo mass, halo bias, mass, and redshifts from the datablock
    dndlnm, halobias, mass, z_vec = pk_util.get_halo_functions(block)
    
    # Reads in the Fourier transform of the normalized dark matter halo profile
    u_dm, u_sat, k_vec = pk_util.get_normalised_profile(block, mass, z_vec)

    # Load the linear power spectrum and growth factor
    k_vec_original, plin_original = pk_util.get_linear_power_spectrum(block, z_vec)
    plin = pk_util.log_linear_interpolation_k(plin_original, k_vec_original, k_vec)
    growth_factor, scale_factor = pk_util.get_growth_factor(block, z_vec, k_vec)

    matter_kwargs.update({
        'dndlnm': dndlnm,
        'halobias': halobias,
        'mass': mass,
        'z_vec': z_vec,
        'u_dm': u_dm,
        'k_vec': k_vec,
        'matter_power_lin': plin,
    })

    if response or fortuna:
        k_nl, p_nl = pk_util.get_nonlinear_power_spectrum(block, z_vec)
        pk_mm_in = pk_util.log_linear_interpolation_k(p_nl, k_nl, k_vec)
        align_kwargs['matter_power_nl'] = pk_mm_in


    # Add the non-linear P_hh to the 2h term
    if bnl:
        if block.has_value('bnl', 'beta_interp'):
            beta_interp = block.get_double_array_nd('bnl', 'beta_interp')
        else:
            raise Exception("You've set bnl = True. Looked for beta_intep in bnl, but didn't find it. Run bnl_interface.py to set this up.\n")
        if beta_interp.shape == np.array([0.0]).shape:
            raise ValueError('Non-linear halo bias module bnl is not initialized, or you have deleted it too early! This might be because you ran bnl_interface_delete.py before this module. \n')
        matter_kwargs.update({
            'bnl': bnl,
            'beta_nl': beta_interp,
        })

    matter_kwargs.update({
        'mean_density0': block['density', 'mean_density0'] * np.ones(len(z_vec)),
        'fnu': block[cosmo_params, 'fnu'] * np.ones(len(z_vec)),
        'omega_c': block[cosmo_params, 'omega_c'],
        'omega_m': block[cosmo_params, 'omega_m'],
        'omega_b': block[cosmo_params, 'omega_b'],
        'h0': block[cosmo_params, 'h0'],
        'n_s': block[cosmo_params, 'n_s'],
        'tcmb': block.get_double(cosmo_params, 'TCMB', default=2.7255),
        'log10T_AGN': block['halo_model_parameters', 'logT_AGN'],
        'mb': 10.0**block['halo_model_parameters', 'm_b'],
    })
    sigma8_z = block['hmf', 'sigma8_z']
    neff = block['hmf', 'neff']
        
    if galaxy or alignment:
        poisson_par = {
            'poisson_type': poisson_type,
            'poisson': get_string_or_none(block, 'pk_parameters', 'poisson', default=None),
            'M_0': get_string_or_none(block, 'pk_parameters', 'M_0', default=None),
            'slope': get_string_or_none(block, 'pk_parameters', 'slope', default=None)
        }
        #hod_bins = block[hod_section_name, 'nbins']
        #N_cen, N_sat, numdencen, numdensat, f_cen, f_sat, mass_avg, f_star = zip(*[
        #    pk_util.load_hods(block, hod_section_name, f'_{nb+1}' if hod_bins != 1 else '', z_vec, mass)
        #    for nb in range(hod_bins)
        #])

        galaxy_kwargs.update({
            'u_sat': u_sat,
            'pointmass': point_mass,
        })
        
        hod_params['A_cen'] = block[hod_values_name, 'A_cen'] if block.has_value(hod_values_name, 'A_cen') else None
        hod_params['A_sat'] = block[hod_values_name, 'A_sat'] if block.has_value(hod_values_name, 'A_sat') else None
        hod_parameters = parameters_models[hod_model]
        # Dinamically load required HOD parameters givent the model and number of bins!
        for param in hod_parameters:
            if hod_model == 'Cacciato':
                param_bin = param
                if not block.has_value(hod_values_name, param_bin):
                    raise Exception(f'Error: parameter {param} is needed for the requested hod model: {hod_model}')
                hod_params[param] = block[hod_values_name, param_bin]
            else:
                param_list = []
                for nb in range(nbins):
                    suffix = f'_{nb+1}' if nbins != 1 else ''
                    param_bin = f'{param}{suffix}'
                    if not block.has_value(hod_values_name, param_bin):
                        raise Exception(f'Error: parameter {param} is needed for the requested hod model: {hod_model}')
                    param_list.append(block[hod_values_name, param_bin])
                hod_params[param] = np.array(param_list)
        galaxy_kwargs.update({
            'hod_model': hod_model,
            'hod_params': hod_params,
            'hod_settings': hod_settings
        })
    matter_kwargs.update({
        'hod_model_mm': hod_model,
        'hod_params_mm': hod_params,
        'hod_settings_mm': hod_settings_mm
    })
    
    if alignment:
        align_kwargs.update({
            'fortuna': fortuna,
            'growth_factor': growth_factor,
            'scale_factor': scale_factor,
            'alignment_gi': block[f'ia_large_scale_alignment{pop_name}', 'alignment_gi'],
            't_eff': block.get_double('pk_parameters', 'linear_fraction_fortuna', default=0.0),
        })

        if block[f'ia_small_scale_alignment{pop_name}', 'instance'] == 'halo_mass':
            align_kwargs.update({
                'beta_sat': block[f'ia_small_scale_alignment{pop_name}', 'beta_sat'],
                'mpivot_sat': block[f'ia_small_scale_alignment{pop_name}', 'M_pivot'],
            })
        else:
            align_kwargs.update({
                'eta_sat': None,
                'mpivot_sat': None,
            })

        if block[f'ia_large_scale_alignment{pop_name}', 'instance'] == 'halo_mass':
            align_kwargs.update({
                'beta_cen': block[f'ia_large_scale_alignment{pop_name}', 'beta'],
                'mpivot_cen': block[f'ia_large_scale_alignment{pop_name}', 'M_pivot'],
            })
        else:
            align_kwargs.update({
                'beta_cen': None,
                'mpivot_cen': None,
            })
    
        align_params = {}
        align_params.update({
            'mass': mass,
            'z_vec': z_vec,
            'c': block['concentration_m', 'c'],
            'r_s': block['nfw_scale_radius_m', 'rs'],
            'rvir': block['virial_radius', 'rvir_m'],
            'nmass': 5,
            'n_hankel': 350,
            'nk': 10,
            'ell_max': 6,
            'gamma_1h_slope': block[f'intrinsic_alignment_parameters{pop_name}', 'gamma_1h_radial_slope'],
            'gamma_1h_amplitude': block[f'ia_small_scale_alignment{pop_name}', 'alignment_1h']
        })
        align_kwargs.update({
            'align_params': align_params,
        })

    if matter:
        matter_power = MatterSpectra(**matter_kwargs)
    if galaxy:
        comb_kwargs = {**matter_kwargs, **galaxy_kwargs}
        galaxy_power = GalaxySpectra(**comb_kwargs)
        hod = galaxy_power
    if alignment:
        comb_kwargs = {**matter_kwargs, **galaxy_kwargs, **align_kwargs}
        alignment_power = AlignmentSpectra(**comb_kwargs)
        hod = alignment_power

    if hod:
        N_cen = hod.hod.compute_hod_cen
        N_sat = hod.hod.compute_hod_sat
        N_tot = hod.hod.compute_hod
        numdens_cen = hod.hod.compute_number_density_cen
        numdens_sat = hod.hod.compute_number_density_sat
        numdens_tot = hod.hod.compute_number_density
        fraction_c = hod.hod.f_c
        fraction_s = hod.hod.f_s
        mass_avg = hod.mass_avg
        f_star = hod.fstar
        
        hod_bins = N_cen.shape[0]
        block.put_int(hod_section_name, 'nbins', hod_bins)
        block.put_bool(hod_section_name, 'observable_z', hod_settings['observable_z'])
        for nb in range(hod_bins):
            suffix = f'_{nb+1}' if hod_bins != 1 else ''
            block.put_grid(hod_section_name, f'z{suffix}', z_vec, f'mass{suffix}', mass, f'N_sat{suffix}', N_sat[nb])
            block.put_grid(hod_section_name, f'z{suffix}', z_vec, f'mass{suffix}', mass, f'N_cen{suffix}', N_cen[nb])
            block.put_grid(hod_section_name, f'z{suffix}', z_vec, f'mass{suffix}', mass, f'N_tot{suffix}', N_tot[nb])
            block.put_grid(hod_section_name, f'z{suffix}', z_vec, f'mass{suffix}', mass, f'f_star{suffix}', f_star[nb])
            block.put_double_array_1d(hod_section_name, f'number_density_cen{suffix}', numdens_cen[nb])
            block.put_double_array_1d(hod_section_name, f'number_density_sat{suffix}', numdens_sat[nb])
            block.put_double_array_1d(hod_section_name, f'number_density_tot{suffix}', numdens_tot[nb])
            block.put_double_array_1d(hod_section_name, f'central_fraction{suffix}', fraction_c[nb])
            block.put_double_array_1d(hod_section_name, f'satellite_fraction{suffix}', fraction_s[nb])
            block.put_double_array_1d(hod_section_name, f'average_halo_mass{suffix}', mass_avg[nb])
        
        if hod.obs_func is not None and hod_settings['save_observable']:
            obs_func = hod.obs_func
            obs_func_c = hod.obs_func_cen
            obs_func_s = hod.obs_func_sat
            obs_z = hod.obs_func_z
            obs_range = hod.obs_func_obs
            obs_bins = obs_range.shape[0]
            
            observable_section_name = hod_settings['observable_section_name']
            block.put(observable_section_name, 'obs_func_definition', 'obs_func * obs * ln(10)')
            block.put(observable_section_name, 'observable_mode', hod_settings['observable_mode'])

            for nb in range(obs_bins):
                if hod_settings['observable_mode'] == 'obs_zmed':
                    suffix_obs = '_med'
                    block.put_double_array_1d(observable_section_name, 'obs_val_med', np.squeeze(obs_range))
                    block.put_double_array_1d(observable_section_name, 'obs_func_med', np.squeeze(obs_func))
                    block.put_double_array_1d(observable_section_name, 'obs_func_med_c', np.squeeze(obs_func_c))
                    block.put_double_array_1d(observable_section_name, 'obs_func_med_s', np.squeeze(obs_func_s))
                else:
                    suffix_obs = f'_{nb+1}'
                    block.put_grid(observable_section_name, f'z_bin{suffix_obs}', obs_z[nb], f'obs_val{suffix_obs}', obs_range[nb, 0, :], f'obs_func{suffix_obs}', obs_func[nb])
                    block.put_grid(observable_section_name, f'z_bin{suffix_obs}', obs_z[nb], f'obs_val{suffix_obs}', obs_range[nb, 0, :], f'obs_func_c{suffix_obs}', obs_func_c[nb])
                    block.put_grid(observable_section_name, f'z_bin{suffix_obs}', obs_z[nb], f'obs_val{suffix_obs}', obs_range[nb, 0, :], f'obs_func_s{suffix_obs}', obs_func_s[nb])

    if p_mm or response:
        pk_mm_1h, pk_mm_2h, pk_mm, _ = matter_power.compute_power_spectrum_mm(
            one_halo_ktrunc = one_halo_ktrunc,
            two_halo_ktrunc = two_halo_ktrunc,
            sigma8_z = sigma8_z,
            neff = neff
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
            poisson_par = poisson_par
        )
        for nb in range(hod_bins):
            suffix = f'{pop_name}_{nb+1}' if hod_bins != 1 else f'{pop_name}'

            block.put_grid(f'galaxy_linear_bias{suffix}', 'z', z_vec, 'k_h', k_vec, 'bg_linear', bg_linear[nb])
            if response:
                block.put_grid(f'galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_1h', pk_gg_1h[nb] / pk_mm[0] * pk_mm_in)
                block.put_grid(f'galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_2h', pk_gg_2h[nb] / pk_mm[0] * pk_mm_in)
                block.put_grid(f'galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gg[nb] / pk_mm[0] * pk_mm_in)
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
            poisson_par = poisson_par
        )
        for nb in range(hod_bins):
            suffix = f'{pop_name}_{nb+1}' if hod_bins != 1 else f'{pop_name}'
        
            block.put_grid(f'galaxy_matter_linear_bias{suffix}', 'z', z_vec, 'k_h', k_vec, 'bgm_linear', bgm_linear[nb])
            if response:
                block.put_grid(f'matter_galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_1h', pk_gm_1h[nb] / pk_mm[0] * pk_mm_in)
                block.put_grid(f'matter_galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_2h', pk_gm_2h[nb] / pk_mm[0] * pk_mm_in)
                block.put_grid(f'matter_galaxy_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gm[nb] / pk_mm[0] * pk_mm_in)
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
            poisson_par = poisson_par
        )
        for nb in range(hod_bins):
            suffix = f'{pop_name}_{nb+1}' if hod_bins != 1 else f'{pop_name}'
        
            if response:
                block.put_grid(f'intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_1h', pk_II_1h[nb] / pk_mm[0] * pk_mm_in)
                block.put_grid(f'intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_2h', pk_II_2h[nb] / pk_mm[0] * pk_mm_in)
                block.put_grid(f'intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_II[nb] / pk_mm[0] * pk_mm_in)
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
            poisson_par = poisson_par
        )
        for nb in range(hod_bins):
            suffix = f'{pop_name}_{nb+1}' if hod_bins != 1 else f'{pop_name}'
            
            if response:
                block.put_grid(f'galaxy_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_1h', pk_gI_1h[nb] / pk_mm[0] * pk_mm_in)
                block.put_grid(f'galaxy_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_2h', pk_gI_2h[nb] / pk_mm[0] * pk_mm_in)
                block.put_grid(f'galaxy_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_gI[nb] / pk_mm[0] * pk_mm_in)
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
            poisson_par = poisson_par
        )
        for nb in range(hod_bins):
            suffix = f'{pop_name}_{nb+1}' if hod_bins != 1 else f'{pop_name}'
            
            if response:
                block.put_grid(f'matter_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_1h', pk_mI_1h[nb] / pk_mm[0] * pk_mm_in)
                block.put_grid(f'matter_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_2h', pk_mI_2h[nb] / pk_mm[0] * pk_mm_in)
                block.put_grid(f'matter_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mI[nb] / pk_mm[0] * pk_mm_in)
            else:
                block.put_grid(f'matter_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_1h', pk_mI_1h[nb])
                block.put_grid(f'matter_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k_2h', pk_mI_2h[nb])
                block.put_grid(f'matter_intrinsic_power{suffix}', 'z', z_vec, 'k_h', k_vec, 'p_k', pk_mI[nb])
                
    return 0

def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass
