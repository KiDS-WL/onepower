from cosmosis.datablock import names, option_section
import numpy as np


cosmo = names.cosmological_parameters


# --------- COSMOSIS MODULE ----------- #

def setup(options):
    # This function is called once per processor per chain.
    # It is a chance to read any fixed options from the configuration file,
    # load any data, or do any calculations that are fixed once.
    
    sampler_name = options['runtime', 'sampler']
    if sampler_name != 'test':
        delete_bnl = False
    else:
        delete_bnl = True
    
    return delete_bnl


def execute(block, config):
    # This function is called every time you have a new sample of cosmological and other parameters.
    # It is the main workhorse of the code. The block contains the parameters and results of any
    # earlier modules, and the config is what we loaded earlier.

    delete_bnl = config
    print(delete_bnl)
    if delete_bnl == True:
        block.replace_double_array_nd('bnl', 'beta_interp', np.array([0.0]))
    else:
        pass
        
        
    return 0


def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass


