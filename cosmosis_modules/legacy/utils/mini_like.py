import numpy as np
from cosmosis.datablock import names, option_section
from cosmosis.datablock.cosmosis_py import errors


def setup(options):
    like_name = options.get_string(option_section, "like_name")
    input_section_name = options.get_string(option_section, "input_section_name", default="likelihood")

    return like_name, input_section_name

def execute(block, config):
    like_name, input_section_name = config

    d = block[input_section_name, "data"]
    mu = block[input_section_name, "theory"]
    #print('data:',np.round(d,decimals=4))
    #print('theory:',np.round(mu,decimals=4))
    # dir(block)
    # for key in block.keys('hod_parameters_bright'):
    #     print(key, '=', block[key])

    inv_cov = block[input_section_name, "inv_covariance"]
    r = d - mu

    chi2 = float(r @ inv_cov @ r)
    ln_like = -0.5*chi2

    block[names.data_vector, like_name+"_CHI2"] = chi2
    block[names.likelihoods, like_name+"_LIKE"] = ln_like

    return 0

def clean(config):
    pass
