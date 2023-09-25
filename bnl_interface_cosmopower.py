# TO-DO: This is just an initial import of the CosmoPower beyond linear bias module from Pierre Burger
# It still needs to be verified and tested
# In the pipeline it can be replaced with the bnl_interface.py module, with only a couple of extra parameters
# that are specific to get the CosmoPower predictions

from builtins import str
import os
from cosmosis.datablock import names, option_section
import sys
import traceback
from scipy.interpolate import CubicSpline
from scipy.interpolate import RectBivariateSpline
import cosmopower as cp
from scipy.interpolate import interp1d, RegularGridInterpolator
from collections import OrderedDict

import cosmopower
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # to use CPU if GPU is avalible otherwise the GPU memory runs out of memory. It also does not slower the predtiction.

# These are pre-defined strings we use as datablock
# section names
cosmo = names.cosmological_parameters
distances = names.distances


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
            print(str(cparam_in[0, i])+' is smaller than '+str(edges[0])+' of key: '+str(key))
            exit()
        if cparam_in[0, i] > edges[1]:
            print(str(cparam_in[0, i])+' is larger than '+str(edges[0])+' of key: '+str(key))
            exit()


def setup(options):

    config = {
        'kmax': options.get_double(option_section, 'kmax', default=10.0),
        'kmin': options.get_double(option_section, 'kmin', default=1e-5),
        'nk': options.get_int(option_section, 'nk', default=200),
        'use_specific_k_modes': options.get_bool(option_section, 'use_specific_k_modes', default=False),
    }

    zmin = options[option_section, 'zmin']
    zmax = options[option_section, 'zmax']
    len_zvec = options[option_section, 'nz']
    z_vec = np.linspace(zmin, zmax, len_zvec)

    
    M_up = options[option_section, 'log_mass_max']
    M_low = options[option_section, 'log_mass_min']
    len_Mvec = options[option_section, 'nmass']
    M_vec = np.logspace(M_low, M_up, len_Mvec)

    k_vec = np.logspace(np.log10(1e-2), np.log10(1.0), num=1000)
    len_kvec = len(k_vec)

    bnl = options.get_bool(option_section, 'bnl', default=False)


    if bnl == True:
        #initialise emulator
        path_2_trained_emulator = options.get_string(option_section, 'path_2_trained_emulator')
        bnl_emulator = cp.cosmopower_NN(restore=True,
                            restore_filename=os.path.join(path_2_trained_emulator+'/bnl_emulator_modified'))
        cached_bnl = {}
        cached_bnl['num_calls'] = 0
        cached_bnl['cached_bnl'] = None
        cached_bnl['update_bnl'] = options[option_section, 'update_bnl']
    else:
        bnl_emulator = None
        cached_bnl = None

    # Return all this config information
    return bnl_emulator,z_vec,len_zvec,M_vec,len_Mvec,k_vec,len_kvec,bnl,cached_bnl


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
                log10M1_list.append(M_vec[j])
                log10M2_list.append(M_vec[k])

        
    len_z_list = len(z_list)

    ombh2 = block['cosmological_parameters', 'ombh2']
    omch2 = block['cosmological_parameters', 'omch2'] # - 0.00064 #need to subtract the neutrino density to get h right in DQ emulator!
    omega_lambda = block['cosmological_parameters', 'omega_lambda']
    A_s = block['cosmological_parameters', 'A_s']
    n_s = block['cosmological_parameters', 'n_s']
    w = block['cosmological_parameters', 'w']

    test_cosmo(np.array([ombh2, omch2, omega_lambda, np.log(10**10*A_s),n_s,w]))

    params_bnl = {'obh2':[ombh2]*len_z_list,
          'omch2': [omch2]*len_z_list,
          'omega_lambda': [omega_lambda]*len_z_list,
          'logAs': [np.log(10**10*A_s)]*len_z_list,
          'ns': [n_s]*len_z_list,
          'w': [w]*len_z_list,
          'z': z_list,
          'logM1': log10M1_list,
          'logM2': log10M2_list
           }

    return params_bnl


def execute(block, config):

    bnl_emulator,z_vec,len_zvec,M_vec,len_Mvec,k_vec,len_kvec,bnl,cached_bnl = config

    if bnl == True:
        params_bnl = get_cosmopower_inputs(block, z_vec, len_zvec, len_Mvec, M_vec)
        bnl_functions=bnl_emulator.predictions_np(params_bnl).reshape(len_zvec,len_Mvec,len_Mvec,len_kvec)

        if(config['use_specific_k_modes']):
            k_new = np.logspace(np.log10(config['kmin']), np.log10(config['kmax']),num=config['nk'])
            bnl_functions_new = np.zeros(shape=(len_zvec,len_Mvec,len_Mvec,len(k_new)))
            for i in range(len_zvec):
                for j in range(len_Mvec):
                    for k in range(len_Mvec):
                        bnl_spline = CubicSpline(k_vec,bnl_functions[i][j][k])
                        bnl_functions_new[i][j][k] = bnl_spline(k_new)
            
            cached_bnl['cached_bnl'] = bnl_functions_new
        else:

            cached_bnl['cached_bnl'] = bnl_functions
    
    else:
        block.put_bool('bnl', 'beta_interp', False)

    return 0


