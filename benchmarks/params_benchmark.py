import numpy as np

hod_params = {
    'log10_obs_norm_c': 10.521,
    'log10_m_ch': 11.145,
    'g1': 7.385,
    'g2': 0.201,
    'sigma_log10_O_c': 0.159,
    'norm_s': 0.562,
    'pivot': 13.0,
    'alpha_s': -0.847,
    'beta_s': 2,
    'b0': -0.120,
    'b1': 1.177,
    'b2': 0.0,      # Set to 0, not used in fit
    'A_cen': None,  # No assembly bias assumed
    'A_sat': None,  # No assembly bias assumed
}

hod_settings = {
    'observables_file': None,
    'obs_min': np.array([9.1, 9.6, 9.95, 10.25, 10.5, 10.7]),
    'obs_max': np.array([9.6, 9.95, 10.25, 10.5, 10.7, 11.3]),
    'zmin': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    'zmax': np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
    'nz': 15,
    'nobs': 300,
    'observable_h_unit': '1/h^2',
}

obs_settings = {
    'observables_file': None,
    'obs_min': np.array([9.1]),
    'obs_max': np.array([11.3]),
    'zmin': np.array([0.25]),
    'zmax': np.array([0.25]),
    'nz': 1,
    'nobs': 300,
    'observable_h_unit': '1/h^2',
}

# constant Poisson model only takes one parameter
poisson_params = {
    'poisson': 0.417,
    #'pivot': None,
    #'slope': None
}

kwargs = {
        'k_vec': np.logspace(-4, 1, 100),
        'z_vec': np.array([0.12, 0.15, 0.18, 0.22, 0.27, 0.32]),
        'lnk_min': np.log(10**(-4.0)),
        'lnk_max': np.log(10**(4.0)),
        'dlnk': (np.log(10**(4.0)) - np.log(10**(-4.0))) / 100,
        'Mmin': 9.0,
        'Mmax': 16.0,
        'dlog10m': 0.05,
        'mdef_model': 'SOMean',
        'hmf_model': 'Tinker10',
        'bias_model': 'Tinker10',
        'halo_profile_model_dm': 'NFW',
        'halo_concentration_model_dm': 'Duffy08',
        'halo_profile_model_sat': 'NFW',
        'halo_concentration_model_sat': 'Duffy08',
        'transfer_model': 'CAMB',
        'growth_model': 'CambGrowth',
        'norm_cen': 0.939,  # normalisation of c(M) relation for matter/centrals
        'norm_sat': 0.84,   # normalisation of c(M) relation for satellites
        'overdensity': 200,
        'delta_c': 1.686,
        'one_halo_ktrunc': 0.1,
        'two_halo_ktrunc': 2.0,
        'bnl': True,

        'omega_c': 0.25,    # Cold dark matter density
        'omega_b': 0.05,    # Baryonic matter density
        'h0': 0.7,          # Dimensionless Hubble parameter
        'n_s': 0.9,         # Spectral index (note different key name than CAMB)
        'sigma_8': 0.8,     # RMS linear density fluctuation in 8 Mpc/h spheres
        'm_nu': 0.06,       # Neutrino mass
        'tcmb': 2.7255,
        'w0': -1.0,
        'wa': 0.0,

        'poisson_model': 'constant',
        'poisson_params': poisson_params,
        'pointmass': True,

        'hod_model': 'Cacciato',
        'hod_params': hod_params,
        'hod_settings': hod_settings,
        'obs_settings': obs_settings,
        'compute_observable': True,
}
