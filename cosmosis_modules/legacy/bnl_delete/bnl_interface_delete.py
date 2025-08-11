import numpy as np
from cosmosis.datablock import names, option_section

cosmo = names.cosmological_parameters

def setup(options):
    # This function is called once per processor per chain.
    # It is a chance to read any fixed options from the configuration file,
    # load any data, or do any calculations that are fixed once.

    sampler_name = options['runtime', 'sampler']
    keep_bnl = options.get_bool(option_section, 'keep_bnl', default=False)

    # Determine whether to delete the bnl array
    delete_bnl = sampler_name != 'test' and not keep_bnl

    return delete_bnl


def execute(block, config):
    # This function is called every time you have a new sample of cosmological and other parameters.
    # It is the main workhorse of the code. The block contains the parameters and results of any
    # earlier modules, and the config is what we loaded earlier.

    delete_bnl = config

    if block.has_value('bnl', 'beta_interp') and delete_bnl:
        block.replace_double_array_nd('bnl', 'beta_interp', np.array([0.0]))
        print('Deleting the large beta_interp array. If you want to keep this set keep_bnl = True')

    return 0


def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass
