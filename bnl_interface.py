# A new power spectrum module

# NOTE: no truncation (halo exclusion problem) applied, as it is included in BNL!

from cosmosis.datablock import names, option_section
import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator
from dark_emulator import darkemu
from collections import OrderedDict

import pk_lib

cosmo = names.cosmological_parameters

def test_cosmo(cparam_in):
    # Returns the edge values for DarkQuest emulator if the values are outside the emulator range
    cparam_range = OrderedDict((["omegab", [0.0211375, 0.0233625]],
                          ["omegac", [0.10782, 0.13178]],
                          ["Omagede", [0.54752, 0.82128]],
                          ["ln(10^10As)", [2.4752, 3.7128]],
                          ["ns", [0.916275, 1.012725]],
                          ["w", [-1.2, -0.8]]))

    cparam_in = cparam_in.reshape(1, 6)
    cparam_out = np.copy(cparam_in)

    for i, (key, edges) in enumerate(cparam_range.items()):
        if cparam_in[0, i] < edges[0]:
            cparam_out[0, i] = edges[0]
        if cparam_in[0, i] > edges[1]:
            cparam_out[0, i] = edges[1]
    return cparam_out
 
 
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
     
     
def interpolate1d_matter_power_lin(matter_power_lin, z_pl, z_vec):
    f_interp = interp1d(z_pl, matter_power_lin, axis=0)
    pk_interpolated = f_interp(z_vec)
    return pk_interpolated
    

    
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

    # TODO: We might need to specify the mass bining of bnl, but for now it is not user accessible!
    #nmass_bnl = options[option_section, 'nmass_bnl']
    #mass_bnl = np.logspace(log_mass_min, log_mass_max, nmass_bnl) 

    zmin = options[option_section, 'zmin']
    zmax = options[option_section, 'zmax']
    nz = options[option_section, 'nz']
    z_vec = np.linspace(zmin, zmax, nz)

    nk = options[option_section, 'nk']

    bnl = options.get_bool(option_section, 'bnl', default=False)
    # TODO: Interpolatation option currently not working, will need to implement in the future!
    interpolate_bnl = options.get_bool(option_section, 'interpolate_bnl', default=True)
    
    if bnl == True:
        #initialise emulator
        emulator = darkemu.base_class()
        cached_bnl = {}
        cached_bnl['num_calls'] = 0
        cached_bnl['cached_bnl'] = None
        cached_bnl['update_bnl'] = options[option_section, 'update_bnl']
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
    k_vec_original, plin_original, growth_factor_original, scale_factor_original = get_linear_power_spectrum(block, z_vec)
    k_vec = np.logspace(np.log10(k_vec_original[0]), np.log10(k_vec_original[-1]), num=nk)
    

    if bnl == True:
        
        num_calls = cached_bnl['num_calls']
        update_bnl = cached_bnl['update_bnl']
            
        if num_calls % update_bnl == 0:
            ombh2 = block['cosmological_parameters', 'ombh2']
            omch2 = block['cosmological_parameters', 'omch2'] # - 0.00064 #need to subtract the neutrino density to get h right in DQ emulator!
            omega_lambda = block['cosmological_parameters', 'omega_lambda']
            A_s = block['cosmological_parameters', 'A_s']
            n_s = block['cosmological_parameters', 'n_s']
            w = block['cosmological_parameters', 'w']
            #cparam = np.array([ombh2, omch2, omega_lambda, np.log(A_s*10.0**10.0),n_s,w])
            cparam = test_cosmo(np.array([ombh2, omch2, omega_lambda, np.log(10**10*A_s),n_s,w]))
            #print('cparam: ', cparam)
            emulator.set_cosmology(cparam)
                
            beta_interp_tmp = pk_lib.create_bnl_interpolation_function(emulator, interpolate_bnl, z_vec, block)
            print('Created b_nl interpolator')
                
            beta_interp = np.zeros((z_vec.size, mass.size, mass.size, k_vec.size))
            indices = np.vstack(np.meshgrid(np.arange(mass.size),np.arange(mass.size),np.arange(k_vec.size), copy = False)).reshape(3,-1).T
            values = np.vstack(np.meshgrid(np.log10(mass), np.log10(mass), np.log10(k_vec), copy = False)).reshape(3,-1).T
            for i,zi in enumerate(z_vec):
                beta_interp[i,indices[:,0], indices[:,1], indices[:,2]] = beta_interp_tmp[i](values)
    
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


