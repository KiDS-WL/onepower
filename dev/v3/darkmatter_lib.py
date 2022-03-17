from cosmosis.datablock import names, option_section
import sys
import numpy as np
from scipy.interpolate import interp1d, interp2d
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

import hankel
from scipy.integrate import quad, simps, trapz
from scipy.special import legendre, sici, binom
import math


# Concentration-mass relations

def concentration(block, mass, z_vec, model):
    nz = len(z_vec)
    nmass = len(mass)
    c = np.empty([nz, nmass])

    if model == 'CooraySheth2002':
        nu_grid = block["hmf_nu", 'nu']
        mass_dn = block["hmf_nu", 'm_h']
        z_dn = block["hmf_nu", 'z']
        f_int_nu = interp2d(mass_dn, z_dn, nu_grid)
        nu = f_int_nu(mass, z_vec)
        # find mstar
        mstar = np.zeros(nz)

        # import matplotlib.pyplot as plt
        for jz in range(0, nz):
            mass_int = interp1d(nu[jz], mass)
            print(z_vec[jz], nu[jz].min(), nu[jz].max())
            mstar[jz] = mass_int(1.0)
        # plt.plot(mass, nu[jz])
        # plt.plot(mstar, np.ones(nz), 'ro')
        print(mstar)
        # plt.xscale('log')
        # plt.show()
        for jz in range(0, nz):
            mass_term = (mass / mstar[jz]) ** (-0.13)
            for im in range(0, nmass):
                c[jz, im] = (9. / (1. + z_vec[jz])) * mass_term[im]

    if model == 'Duffy2008_mean':
        mass_term = 10.14 / ((mass / (2.e+12)) ** 0.081)
        c = mass_term[np.newaxis,:] / ((1. + z_vec[:,np.newaxis]) ** 1.01)

    if model == 'Duffy2008_crit':
        mass_term = 5.71 / ((mass / (2.e+12)) ** 0.084)
        for jz in range(0, nz):
            for im in range(0, nmass):
                c[jz, im] = mass_term[im] / ((1. + z_vec[jz]) ** 0.47)

    #if model == "Mandelbaum2008":
        #mass_term = 1. / ((mass / (1.E+14)) ** 0.1)
        #c_dm = 5. * mass_term / (1. + z)
    return c

# At the moment, only Duffy is actually used in the pipeline (first function)
'''
def concentration(mass, z):
    mass_term = 1. / ((mass / (2.e12)) ** 0.081)
    c_dm = 10.14 * mass_term / ((1. + z) ** 1.01)
    return c_dm
'''


def scalar_rvir(mass, rho_halo):
    return ((3. * mass) / (4. * np.pi * rho_halo)) ** (1. / 3.)

# Virial radius associated with a given mass (depends on the definition of rho_halo)
def radvir_from_mass(mass, rho_halo):
    # mass : array1d or scalar. The mass of the halo in units of Msun/h
    # rho_halo : array1d or scalar. The matter density of the halo. It can either be Delta x rho_m(z) or
    #            Delta x rho_m(z=0) - where rho_m is the mean matter density of the Universe (evaluated at z or 0,
    #            depending on the convention used in the model) - or be Delta x rho_c, or even the Delta_vir
    rvir = np.array([scalar_rvir(mass,rhalo) for rhalo in rho_halo])
    return rvir


def scale_radius(rvir, conc):
    # rvir : array-1d or 2d, depending on the model definition ([nz,nmass] or simply [nmass]). The virial radius of the
    #        halo, in units of Mpc/h
    # conc : array-2d [nz,nmass]. The concentration of the halo, computed from one of the available fitting functions.
    #print('nz conc = ', nz)
    #if np.isscalar(rvir):
    #    r_s = rvir[np.newaxis,:]/conc
    #else:
    r_s = rvir/conc
    #nz = np.size(conc, axis=0)
    #r_s_test = np.empty(np.shape(conc))
    #for jz in range(0, nz):
    #    r_s_test[jz] = rvir[0] / conc[jz]  # array 1d [nmass]
    #    print (r_s_test[jz]/r_s[jz])
    return r_s

# AD: check normalisations! On k->0 the u_dm should be 1!
# Analytic Fourier transform of the Navarro-Frenk-White density profile
def norm_fourier(x, c):
    # Note: x = k*r_scale where r_scale= r200/c
    si, ci = sici((1. + c) * x)
    si2, ci2 = sici(x)
    sinterm = np.sin(x) * (si - si2) - np.sin(c * x) / ((1 + c) * x)
    costerm = np.cos(x) * (ci - ci2)
    # note that the factor 4 pi rho_s r_s^3 appears both in the numerator than in the mass, so it can be simplified
    rescaled_mass = np.log(1. + c) - c / (1. + c)
    u_fourier = (sinterm + costerm) / rescaled_mass
    return u_fourier

# compute the analytic fourier-transform of the nfw profile
def compute_u_dm(k_vec, rs, conc):
    # k : array-1d. The wave vector in units of h/Mpc
    # rs : array-2d [nz,nmass]. The scale radius.
    # c_dm : array-2d [nz,nmass]. The concentration of the halo as a function of redshift and mass.
    # return: array-3d
    nz = np.size(conc, axis=0)
    #nmass = np.size(conc, axis=1)
    nk = np.size(k_vec)
    u_dm = np.array([[norm_fourier(k * rs[jz], conc[jz]) for k in k_vec] for jz in range(0,nz)])
    '''
    u_dm = np.empty([nz, nk, nmass])
    for jz in range(nz):
        for  ik in range(0,nk):
                u_dm[jz, ik, :] = norm_fourier(k[ik] * rs[jz], conc[jz]) # array-1d [nmass]
    '''
    return u_dm

#######
