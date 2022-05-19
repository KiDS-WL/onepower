from cosmosis.datablock import names, option_section
import sys
import numpy as np
from scipy.interpolate import interp1d, interp2d
import time

import math

from darkmatter_lib import concentration, radvir_from_mass, scale_radius, compute_u_dm

# We have a collection of commonly used pre-defined block section names.
# If none of the names here is relevant for your calculation you can use any
# string you want instead.

cosmo = names.cosmological_parameters


# --------------------------------------------------------------------------------#

def setup(options):
    # This function is called once per processor per chain.
    # It is a chance to read any fixed options from the configuration file,
    # load any data, or do any calculations that are fixed once.

    log_mass_min = options[option_section, "log_mass_min"]
    log_mass_max = options[option_section, "log_mass_max"]
    nmass = options[option_section, "nmass"]
    # log-spaced mass in units of M_sun/h
    dlog10m = (log_mass_max-log_mass_min)/nmass
    mass = 10.0 ** np.arange(log_mass_min, log_mass_max, dlog10m) #To be consistent with hmf!

    zmin = options[option_section, "zmin"]
    zmax = options[option_section, "zmax"]
    nz = options[option_section, "nz"]
    z_vec = np.linspace(zmin, zmax, nz)
    print(z_vec)
    model_cm = options[option_section, "model"]
    mdef = options[option_section, "mdef_model"]
    overdensity = options[option_section, "overdensity"]

    return z_vec, nz, mass, nmass, model_cm, mdef, overdensity


def execute(block, config):
    # This function is called every time you have a new sample of cosmological and other parameters.
    # It is the main workhorse of the code. The block contains the parameters and results of any
    # earlier modules, and the config is what we loaded earlier.

    z, nz, mass, nmass, model_cm, mdef, overdensity = config
    start_time = time.time()

    rho_m = block["density", "mean_density0"]
    print('rho_m = ', rho_m)

    # compute the virial radius and the scale radius associated with a halo of mass M
    rho_halo = overdensity * rho_m # array 1d (size of rhom)

    norm_cen = block["nfw_halo", "norm_cen"]
    norm_sat = block["nfw_halo", "norm_sat"]

    conc_cen = norm_cen * concentration(block, mass, z, model_cm, mdef, overdensity)
    conc_sat = norm_sat * concentration(block, mass, z, model_cm, mdef, overdensity)
    rvir_cen = radvir_from_mass(mass, rho_halo)
    rvir_sat = radvir_from_mass(mass, rho_halo)
    r_s_cen = scale_radius(rvir_cen, conc_cen)
    r_s_sat = scale_radius(rvir_sat, conc_sat)

    # compute the Fourier-transform of the NFW profile (normalised to the mass of the halo)
    k = np.logspace(-2,6,200) # AD: inherit the range from Plin? That would avoid intepolations ...
    u_dm_cen = compute_u_dm(k, r_s_cen, conc_cen)
    u_dm_sat = compute_u_dm(k, r_s_sat, conc_sat)

    block.put_grid("concentration_dm", "z", z, "m_h", mass, "c", conc_cen)
    block.put_grid("concentration_sat", "z", z, "m_h", mass, "c", conc_sat)
    block.put_grid("nfw_scale_radius_dm", "z", z, "m_h", mass, "rs", r_s_cen)
    block.put_grid("nfw_scale_radius_sat", "z", z, "m_h", mass, "rs", r_s_sat)
    #block.put_grid("virial_radius", "z", z, "m_h", mass, "rvir", rvir)
    #print(rvir[0].shape)
    block.put_double_array_1d("virial_radius", "m_h", mass)
    block.put_double_array_1d("virial_radius", "rvir_dm", rvir_cen[0])
    block.put_double_array_1d("virial_radius", "rvir_sat", rvir_sat[0])


    block.put_double_array_1d("fourier_nfw_profile", "z", z)
    block.put_double_array_1d("fourier_nfw_profile", "m_h", mass)
    block.put_double_array_1d("fourier_nfw_profile", "k_h", k)
    block.put_double_array_nd("fourier_nfw_profile", "ukm", u_dm_cen)
    block.put_double_array_nd("fourier_nfw_profile", "uksat", u_dm_sat)
    
    return 0


def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass
