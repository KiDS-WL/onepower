import numpy as np
from scipy.special import sici

def scalar_rvir(mass, rho_halo):
    return ((3. * mass) / (4. * np.pi * rho_halo)) ** (1. / 3.)


# Virial radius associated with a given mass (depends on the definition of rho_halo)
def radvir_from_mass(mass, rho_halo):
    # mass : array1d or scalar. The mass of the halo in units of Msun/h
    # rho_halo : array1d or scalar. The matter density of the halo. It can either be Delta x rho_m(z) or
    #            Delta x rho_m(z=0) - where rho_m is the mean matter density of the Universe (evaluated at z or 0,
    #            depending on the convention used in the model) - or be Delta x rho_c, or even the Delta_vir
    
    rvir = scalar_rvir(mass[np.newaxis,:],rho_halo[:,np.newaxis])
    return rvir


def scale_radius(rvir, conc):
    # rvir : array-1d or 2d, depending on the model definition ([nz,nmass] or simply [nmass]). The virial radius of the
    #        halo, in units of Mpc/h
    # conc : array-2d [nz,nmass]. The concentration of the halo, computed from one of the available fitting functions.
    r_s = rvir/conc
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
def compute_u_dm(k_vec, rs, conc, mass):
    # k : array-1d. The wave vector in units of h/Mpc
    # rs : array-2d [nz,nmass]. The scale radius.
    # c_dm : array-2d [nz,nmass]. The concentration of the halo as a function of redshift and mass.
    # return: array-3d
    
    nz = np.size(conc, axis=0)
    nk = np.size(k_vec)
    
    u_dm = norm_fourier(k_vec[:,np.newaxis] * rs[:,np.newaxis,:], conc[:,np.newaxis,:])
    u_dm = u_dm/np.expand_dims(u_dm[:,0,:], 1) # Force normalisation to 1!
    """
    # This works but needs further refinement before we can completely replace the default NFW
    #print(u_dm.shape)
    from halomod.profiles import NFW, NFWInf #GeneralizedNFW as NFW
    profile = NFW('Bullock01')#, alpha=1.25)
    u_dm_halomod = np.zeros((conc.shape[0], k_vec.size, mass.size))
    #print(u_dm_halomod.shape)
    for i,ci in enumerate(conc):
        u_dm_halomod[i,:,:] = profile.u(k_vec, mass, norm='m', c=ci, coord='k')
    #print(u_dm_halomod.shape)
    print(u_dm/u_dm_halomod)
    #quit()
    #"""
    
    return u_dm

#######
