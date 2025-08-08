# A new power spectrum module

# NOTE: no truncation (halo exclusion problem) applied, as it is included in BNL!

import numpy as np
from collections import OrderedDict
from cosmosis.datablock import names, option_section
from dark_emulator import darkemu
from scipy.interpolate import RegularGridInterpolator, interp1d

#import sys
#sys.path.insert(0, "/net/home/fohlen13/dvornik/halo_model_mc/halomodel_for_cosmosis/package/pk")
#from bnl import NonLinearBias
from onepower.bnl import NonLinearBias

cosmo_params = names.cosmological_parameters

def setup(options):
    # This function is called once per processor per chain.
    # It is a chance to read any fixed options from the configuration file,
    # load any data, or do any calculations that are fixed once.

    log_mass_min = options[option_section, 'log_mass_min']
    log_mass_max = options[option_section, 'log_mass_max']
    nmass = options[option_section, 'nmass']
    dlog10m = (log_mass_max - log_mass_min) / nmass
    mass = 10 ** np.arange(log_mass_min, log_mass_max, dlog10m)

    # TODO: We might need to specify the mass binning of bnl, but for now it is not user accessible!
    # nmass_bnl = options[option_section, 'nmass_bnl']
    # mass_bnl = np.logspace(log_mass_min, log_mass_max, nmass_bnl)

    zmin = options[option_section, 'zmin']
    zmax = options[option_section, 'zmax']
    nz = options[option_section, 'nz']
    z_vec = np.linspace(zmin, zmax, nz)

    nk = options[option_section, 'nk']
    bnl = options.get_bool(option_section, 'bnl', default=False)

    if bnl:
        cached_bnl = {
            'num_calls': 0,
            'cached_bnl': None,
            'update_bnl': options[option_section, 'update_bnl']
        }
    else:
        cached_bnl = None

    return mass, nmass, z_vec, nz, nk, bnl, cached_bnl

def execute(block, config):
    # This function is called every time you have a new sample of cosmological and other parameters.
    # It is the main workhorse of the code. The block contains the parameters and results of any
    # earlier modules, and the config is what we loaded earlier.

    mass, nmass, z_vec, nz, nk, bnl, cached_bnl = config

    # load linear power spectrum
    k_vec_original = block['matter_power_lin', 'k_h']
    k_vec = np.logspace(np.log10(k_vec_original[0]), np.log10(k_vec_original[-1]), num=nk)

    if bnl:
        num_calls = cached_bnl['num_calls']
        update_bnl = cached_bnl['update_bnl']

        if num_calls % update_bnl == 0:
            bnl = NonLinearBias(
                mass = mass,
                z_vec = z_vec,
                k_vec = k_vec,
                h0 = block[cosmo_params, 'h0'],
                A_s = block[cosmo_params, 'A_s'],
                omega_b = block[cosmo_params, 'omega_b'],
                omega_c = block[cosmo_params, 'omega_c'],
                omega_lambda = 1.0 - block[cosmo_params, 'omega_m'],
                n_s = block[cosmo_params, 'n_s'],
                w0 = block[cosmo_params, 'w'],
                z_dep = False
            )

            beta_interp = bnl.bnl
            cached_bnl['cached_bnl'] = beta_interp
        else:
            beta_interp = cached_bnl['cached_bnl']

        cached_bnl['num_calls'] = num_calls + 1
        block.put_double_array_nd('bnl', 'beta_interp', beta_interp)
    else:
        block.put_double_array_nd('bnl', 'beta_interp', np.array([0.0]))

    return 0

def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass
