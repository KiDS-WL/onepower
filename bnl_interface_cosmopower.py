# TO-DO: This is just an initial import of the CosmoPower beyond linear bias module from Pierre Burger
# It still needs to be verified and tested
# In the pipeline it can be replaced with the bnl_interface.py module, with only a couple of extra parameters
# that are specific to get the CosmoPower predictions

import os
from cosmosis.datablock import names, option_section
import sys
import traceback
from scipy.interpolate import CubicSpline, RectBivariateSpline, interp1d, RegularGridInterpolator
import cosmopower as cp
from collections import OrderedDict

import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # to use CPU if GPU is avalible otherwise the GPU memory runs out of memory. It also does not slower the predtiction.

# These are pre-defined strings we use as datablock
# section names
cosmo = names.cosmological_parameters
distances = names.distances

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
            print(f'{cparam_in[0, i]} is smaller than {edges[0]} of key: {key}')
            exit()
        if cparam_in[0, i] > edges[1]:
            print(f'{cparam_in[0, i]} is larger than {edges[0]} of key: {key}')
            exit()


def setup(options):

    config = {}

    config['kmax'] = options.get_double(option_section, 'kmax', default=10.0)
    config['kmin'] = options.get_double(option_section, 'kmin', default=1e-5)
    config['nk'] = options.get_int(option_section, 'nk', default=200)
    config['use_specific_k_modes'] = options.get_bool(option_section, 'use_specific_k_modes', default=False)

    config['zmin'] = options[option_section, 'zmin']
    config['zmax'] = options[option_section, 'zmax']
    config['len_zvec'] = options[option_section, 'nz']
    config['z_vec'] = np.linspace(config['zmin'], config['zmax'], config['len_zvec'])

    
    config['M_up'] = options[option_section, 'log_mass_max']
    config['M_low'] = options[option_section, 'log_mass_min']
    config['len_Mvec'] = options[option_section, 'nmass']
    config['M_vec'] = np.logspace(config['M_low'], config['M_up'], config['len_Mvec'])
    lower_limit = 10.0**12.0
    upper_limit = 10.0**14.0
    config['M_vec'][config['M_vec'] < lower_limit] = lower_limit
    config['M_vec'][config['M_vec'] > upper_limit] = upper_limit

    config['k_vec'] = np.logspace(np.log10(1e-3), np.log10(100.0), num=1000)
    config['len_kvec'] = len(config['k_vec'])

    bnl = options.get_bool(option_section, 'bnl', default=False)
    config['bnl'] = bnl
    
    if bnl == True:
        #initialise emulator
        path_2_trained_emulator = options.get_string(option_section, 'path_2_trained_emulator')
        config['bnl_emulator'] = cp.cosmopower_NN(restore=True, restore_filename = path_2_trained_emulator)
    else:
        config['bnl_emulator'] = None

    # Return all this config information
    return config


def get_cosmopower_inputs(block, z_vec, len_zvec, len_Mvec, M_vec):

    # Get parameters from block and give them the
    # names and form that class expects

    z_list = []
    log10M1_list = []
    log10M2_list = []
    for i in range(len_zvec):
        for j in range(len_Mvec):
            for k in range(len_Mvec):
                z_list.append(z_vec[i])
                log10M1_list.append(np.log10(M_vec[j]))
                log10M2_list.append(np.log10(M_vec[k]))
    # AD: Just thinking out loud, is this creating a set of all combinations?
    # This must be easier to do with itertools...

    ombh2 = block['cosmological_parameters', 'ombh2']
    omch2 = block['cosmological_parameters', 'omch2'] # - 0.00064 #need to subtract the neutrino density to get h right in DQ emulator!
    omega_lambda = block['cosmological_parameters', 'omega_lambda']
    A_s = block['cosmological_parameters', 'A_s']
    n_s = block['cosmological_parameters', 'n_s']
    w = block['cosmological_parameters', 'w']

    test_cosmo(np.array([ombh2, omch2, omega_lambda, np.log(10**10*A_s),n_s,w]))

    params_bnl = {'obh2':ombh2*np.ones_like(z_list),
                  'omch2': omch2*np.ones_like(z_list),
                  'omega_lambda': omega_lambda*np.ones_like(z_list),
                  'logAs': np.log(10**10*A_s)*np.ones_like(z_list),
                  'ns': n_s*np.ones_like(z_list),
                  'w': w*np.ones_like(z_list),
                  'z': z_list,
                  'logM1': log10M1_list,
                  'logM2': log10M2_list
                  }

    return params_bnl


def execute(block, config):

    bnl_emulator  = config['bnl_emulator']
    bnl  = config['bnl']
    use_specific_k_modes = config['use_specific_k_modes']
    
    if bnl == True:
        params_bnl = get_cosmopower_inputs(block, config['z_vec'], config['len_zvec'], config['len_Mvec'], config['M_vec'])
        bnl_functions = bnl_emulator.predictions_np(params_bnl).reshape(config['len_zvec'], config['len_Mvec'], config['len_Mvec'], config['len_kvec'])
        print('bnl read from emulator, creating 3D array')
        """
        # Test to see what exactly was emulated
        import matplotlib.pyplot as pl
        params_all = np.load('/net/home/fohlen13/dvornik/bnl_training/outputs/train_betanl_parameter.npz')
        center_params = {'obh2': [np.mean(params_all['obh2'])],
                          'omch2': [np.mean(params_all['omch2'])],
                          'omega_lambda': [np.mean(params_all['omega_lambda'])],
                          'logAs': [np.mean(params_all['logAs'])],
                          'ns': [np.mean(params_all['ns'])],
                          'w': [np.mean(params_all['w'])],
                          'z': [np.mean(params_all['z'])],
                          'logM1': [np.mean(params_all['logM1'])],
                          'logM2': [np.mean(params_all['logM2'])],
                            }
        print(center_params)
        beta_center = np.load('/net/home/fohlen13/dvornik/bnl_training/bnl_vectors/bnl_center_vector_'+str(0)+'.npz')#np.load('/net/home/fohlen13/dvornik/bnl_training/outputs/beta_nl_func_center.npz')['beta_func'][0]
        beta_center_emu = bnl_emulator.predictions_np(center_params)
        pl.plot(config['k_vec'], beta_center['beta_func'][0])
        pl.plot(config['k_vec'], beta_center_emu[0])
        pl.xscale('log')
        pl.show()
        quit()
        #"""
        
        if use_specific_k_modes:
        
            # load linear power spectrum
            k_vec_original, plin_original, growth_factor_original, scale_factor_original = get_linear_power_spectrum(block, config['z_vec'])
            k_new = np.logspace(np.log10(k_vec_original[0]), np.log10(k_vec_original[-1]), num=config['nk'])
            
            bnl_functions_new = np.zeros(shape=(config['len_zvec'], config['len_Mvec'], config['len_Mvec'], len(k_new)))
            std_bnl = np.std(bnl_functions, axis=(0,1,2))
            mean_bnl = np.mean(bnl_functions, axis=(0,1,2))
            
            # Can call axis in the interp1d function to avoid loops!
            bnl_spline = interp1d(np.log10(config['k_vec']), bnl_functions, axis=-1, fill_value=0.0, bounds_error=False)
            bnl_functions_new = bnl_spline(np.log10(k_new))
            beta_cosmopower = bnl_functions_new
        else:
            beta_cosmopower = bnl_functions
        block.put_double_array_nd('bnl', 'beta_interp', beta_cosmopower)
    
    else:
        block.put_double_array_nd('bnl', 'beta_interp', np.array([0.0]))

    return 0


