# A new power spectrum module

# NOTE: no truncation (halo exclusion problem) applied, as it is included in BNL!

import bnl_util
import numpy as np
from collections import OrderedDict
from cosmosis.datablock import names, option_section
from dark_emulator import darkemu
from scipy.interpolate import RegularGridInterpolator, interp1d

cosmo = names.cosmological_parameters

def test_cosmo(cparam_in):
    """Returns the edge values for DarkQuest emulator if the values are outside the emulator range."""
    cparam_range = OrderedDict([
        ['omegab', [0.0211375, 0.0233625]],
        ['omegac', [0.10782, 0.13178]],
        ['Omagede', [0.54752, 0.82128]],
        ['ln(10^10As)', [2.4752, 3.7128]],
        ['ns', [0.916275, 1.012725]],
        ['w', [-1.2, -0.8]]
    ])

    cparam_in = cparam_in.reshape(1, 6)
    cparam_out = np.copy(cparam_in)

    for i, (key, edges) in enumerate(cparam_range.items()):
        if cparam_in[0, i] < edges[0]:
            cparam_out[0, i] = edges[0]
        if cparam_in[0, i] > edges[1]:
            cparam_out[0, i] = edges[1]

    return cparam_out

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
    interpolate_bnl = options.get_bool(option_section, 'interpolate_bnl', default=True)
    # TODO: Interpolation option currently not working, will need to implement in the future!

    if bnl:
        emulator = darkemu.base_class()
        cached_bnl = {
            'num_calls': 0,
            'cached_bnl': None,
            'update_bnl': options[option_section, 'update_bnl']
        }
    else:
        emulator = None
        cached_bnl = None

    return mass, nmass, z_vec, nz, nk, bnl, interpolate_bnl, emulator, cached_bnl

def execute(block, config):
    # This function is called every time you have a new sample of cosmological and other parameters.
    # It is the main workhorse of the code. The block contains the parameters and results of any
    # earlier modules, and the config is what we loaded earlier.

    mass, nmass, z_vec, nz, nk, bnl, interpolate_bnl, emulator, cached_bnl = config

    # load linear power spectrum
    k_vec_original = block['matter_power_lin', 'k_h']
    k_vec = np.logspace(np.log10(k_vec_original[0]), np.log10(k_vec_original[-1]), num=nk)

    if bnl:
        num_calls = cached_bnl['num_calls']
        update_bnl = cached_bnl['update_bnl']

        if num_calls % update_bnl == 0:
            ombh2 = block['cosmological_parameters', 'ombh2']
            omch2 = block['cosmological_parameters', 'omch2']
            omega_lambda = block['cosmological_parameters', 'omega_lambda']
            A_s = block['cosmological_parameters', 'A_s']
            n_s = block['cosmological_parameters', 'n_s']
            w = block['cosmological_parameters', 'w']
            #cparam = np.array([ombh2, omch2, omega_lambda, np.log(A_s*10.0**10.0), n_s, w])
            cparam = test_cosmo(np.array([ombh2, omch2, omega_lambda, np.log(A_s*10.0**10.0), n_s, w]))
            #print('cparam: ', cparam)
            emulator.set_cosmology(cparam)

            beta_interp_tmp = bnl_util.create_bnl_interpolation_function(emulator, interpolate_bnl, z_vec, block)

            beta_interp = np.zeros((z_vec.size, mass.size, mass.size, k_vec.size))
            indices = np.vstack(np.meshgrid(np.arange(mass.size), np.arange(mass.size), np.arange(k_vec.size), copy=False)).reshape(3, -1).T
            values = np.vstack(np.meshgrid(np.log10(mass), np.log10(mass), np.log10(k_vec), copy=False)).reshape(3, -1).T
            # for i,zi in enumerate(z_vec):
            #    beta_interp[i,indices[:,0], indices[:,1], indices[:,2]] = beta_interp_tmp[i](values)

            beta_interp2 = np.zeros((mass.size, mass.size, k_vec.size))
            beta_interp2[indices[:, 0], indices[:, 1], indices[:, 2]] = beta_interp_tmp(values)

            beta_interp = beta_interp2[np.newaxis, :, :, :]
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
