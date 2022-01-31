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
    mass = np.logspace(log_mass_min, log_mass_max, nmass)

    zmin = options[option_section, "zmin"]
    zmax = options[option_section, "zmax"]
    nz = options[option_section, "nz"]
    z_vec = np.linspace(zmin, zmax, nz)
    print(z_vec)

    return z_vec, nz, mass, nmass


def execute(block, config):
    # This function is called every time you have a new sample of cosmological and other parameters.
    # It is the main workhorse of the code. The block contains the parameters and results of any
    # earlier modules, and the config is what we loaded earlier.

    z, nz, mass, nmass = config
    start_time = time.time()

    # ---- Comology ----#
    # Load cosmological parameters
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u

    this_cosmo = FlatLambdaCDM(H0=block[cosmo, "hubble"], Om0=block[cosmo, "omega_m"], Ob0=block[cosmo, "omega_b"],
                               Tcmb0=2.725)
    rho_crit0 = this_cosmo.critical_density0.to(u.M_sun * u.Mpc ** (-3.)) / (this_cosmo.h ** 2.)
    mean_density0 = rho_crit0.value * this_cosmo.Om0
    rho_m = np.array([mean_density0]) #* np.ones(nz)  #(1.+z)** 3.  # mean_density0 * (1.+np.zeros(nz))**3. # array 1d [nz]
    print('rho_m = ', rho_m)
    print('mean_density0 = ', mean_density0)

    # compute the virial radius and the scale radius associated with a halo of mass M
    rho_halo = 200. * rho_m # array 1d (size of rhom)
    #print ('rho_halo.size = ', rho_halo.shape)
    eta = 0.
    conc = (1.+eta)*concentration(block, mass, z, 'Duffy2008_mean')
    rvir = radvir_from_mass(mass, rho_halo)
    r_s = scale_radius(rvir, conc)

    # compute the Fourier-transform of the NFW profile (normalised to the mass of the halo)
    k = np.logspace(-2,6,200)
    u_dm = compute_u_dm(k, r_s, conc)

    block.put_grid("concentration", "z", z, "m_h", mass, "c", conc)
    block.put_grid("nfw_scale_radius", "z", z, "m_h", mass, "rs", r_s)
    #block.put_grid("virial_radius", "z", z, "m_h", mass, "rvir", rvir)
    #print(rvir[0].shape)
    block.put_double_array_1d("virial_radius", "m_h", mass)
    block.put_double_array_1d("virial_radius", "rvir", rvir[0])


    block.put_double_array_1d("fourier_nfw_profile", "z", z)
    block.put_double_array_1d("fourier_nfw_profile", "m_h", mass)
    block.put_double_array_1d("fourier_nfw_profile", "k_h", k)
    block.put_double_array_nd("fourier_nfw_profile", "ukm", u_dm)

    return 0


def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass
