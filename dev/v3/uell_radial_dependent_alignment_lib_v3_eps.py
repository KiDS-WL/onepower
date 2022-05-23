'''
This module computes the radial dependent part of the satellite alignment term.
The satellite alignment is modelled following the revised version of Schneider&Bridle 2010
(SB10) by Fortuna et al. 2020. We compute the Fourier transform of the alignment, 
which we divide in two components as described in SB10 : an angular dependent part,
implemented in the wkm_angular_part_eps.py module and a radial dependent part, whose main
functions are described below. 

The Fourier tranform of the density weighted satellite shear is defined as (Fortuna et al. 2020)
NOTE: we omit here the HOD term as it is only mass and redshift dependent (does not depend on 
r, theta or phi). We reconstruct the full density weighted shear via the multiplication of the
HOD term in the pk_interface.py module. -> This might be changed in the future to have a 
full density weighted shear output.

\hat{\gamma}^I_s (k,M) = F(\gamma^I(r,M) u(r,M))                                       (1)

where u(r,M) is the normalised NFW profile, rho_NFW(r,M)/M with M the mass of the halo, and 
\gamma^I(r,M) is the projected (2D) radial dependent satellite alignment, here modelled as

\gamma^I(r, theta, M) = \bar{gamma}^I(r,M,z) sin(theta) =
                      = a_1h (L/L0)^zeta (r sin(theta) /r_vir)^b  .          (2)

In practise, we work with gamma_1h_amplitude(z) = a_1h (L(z)/L0)^zeta, which can potentially 
be a function of redshift (in a flux-limited survey this is inherited by the luminosity; if
the sample is volume complete, gamma_1h_amplitude(z) = gamma_1h_amplitude at all redshifts,
unless a specific redshift dependence is included by the user - currently not implemented).

a_1h(L/L0)^zeta do not dependent on r, and thus is pre-computed in the module 
ia_amplitudes_all_modes.py, which returns the effective amplitude after luminosity scaling.
Any effective amplitude can be passed to the module.
NOTE that we need to include the amplitude to asses the correct threshold for the alignment 
in the core, as discussed below.


We can divide the angular and radial part and define \bar{gamma}^I(r,M) as the 3D galaxy 
shear, which reads

\bar{gamma}^I(r, M, z) = gamma_1h_amplitude(z) (r/rvir)^b ,                             (3)

while the sin(theta)^b is treated in the wkm angular part module.

---

The Fourier transform of the projected satellite shear thus reads

F(\gamma^I(r,M) u(r,M)) = \int_0^{2pi} dphi \int_0^{pi} dtheta sin^(2+b)(theta) \
                            \int_0^{\infty} dr r^2 \gamma^I(r,M) (\rho_NFW(r,M) / M) e^{i kr}

where r and k are both 3D vectors and the product kr has to be read as a scalar product of 
the components. The square in the sin(theta) comes from the fact that we are considering 
a projected shear and the b comes from eq. (2). 

Following Appendix B2 in SB10 and Appendix C in Fortuna et al. (2020) we can use the wave
expansion to separate the angular and radial part of the integrals. The radial part becomes:

---------------------------------- radial component: ------------------------------------

u_ell(z,M,k) = \int dr r^2 gamma_1h_amplitude (r/rvir)**b \rho_NFW(r,M) j_l(kr) / M_NFW

-----------------------------------------------------------------------------------------

The NFW mass ca be expressed analytically as

M_NFW = 4 pi \rho_s r_s^3 (ln(1+c) - c/(1+c) ) .

The NFW profile at the numerator is instead

\rho_NFW(r,M) = \rho_s / ((r/r_s)(1+r/r_s))

and thus \rho_s can be simplified. We implement all of these functions dropping the dependence
on rho_s. 

The integral over j_l(kr) is a Hankel transform that we implement using the hankel.py module.

We use the transform j_nu(x) = \sqrt(\pi/(2x)) J_(nu + 0.5) (x)

with x = kr and thus the integral over dx becomes an integral over dr k, i.e.

\int dx f(x) J_nu(x) => \int dx/k x^2/k^2 u_l(x/k, M) \sqrt(\pi/(2x)) J_(l+0.5)

thus: f(x) = x^2 u_l(x/k,M) \sqrt(\pi/(2x))

and then we divide everything by k^3.

---

We also include a constant core to avoid the power law to explode at small r. The radial
part of the projected shear thus contains a piecewise function: for r<0.06 Mpc/h the shear
is constant with amplitude equal to the value of the shear at r= 0.06 Mpc/h. We also 
require the shear to not exceed the maximum value of 0.3, corresponding to a perfectly aligned
satellite.

'''



import numpy as np
import hankel
import math

import time
from itertools import count
from numpy import absolute, array, arange, cos, exp, linspace, log10, \
                  logspace, max as npmax, median, newaxis, ones, outer,\
                  pi, sin,  sum as npsum, zeros
from scipy.integrate import quad, simps, trapz
from scipy.interpolate import interp1d
from scipy.special import legendre, sici, binom
import math
import hankl

from wkm_angular_part_eps import *

import matplotlib.pyplot as plt


#-----------------------------------------------------------------------#
#								u_ell									#
#-----------------------------------------------------------------------#


# Since we are interested on the normalised nfw profile only, 
# here rho_s is removed both in the nfw profile and in the nfw mass 
def nfw_profile(r, rs):
    x = r/rs
    f_x = x*(1.+x)**2
    rho_nfw = 1./f_x
    return rho_nfw


def mass_nfw(r_s, c):
    mnfw =  4.*np.pi*(r_s**3.)*(np.log(1.+c)-c/(1.+c))
    return mnfw


def nfw_profile_trunc(r, rs, rvir):
    mask_above_rvir = r>=rvir
    nfw = nfw_profile(r, rs)
    nfw[mask_above_rvir] = 0.0
    return nfw


def gamma_r_nfw_profile(r, rs, rvir, A, b, rcore=0.06):
    mask_small_r = r<rcore
    gamma = A*(r/rvir)**b
    # note that in the case of non-radial dependent alignment, this cut is not recommendable since it introduces ringing
    gamma[mask_small_r] = A*(rcore/rvir)**b
    #mask_gamma = gamma > 0.3
    gamma[gamma > 0.3] = 0.3
    gamma_weighted = gamma * nfw_profile_trunc(r,rs,rvir)
    return gamma_weighted


# not used
def gamma_r_nfw_profile_notrunc(r, rs, rvir, b):
    gamma_nfw = (r/rvir)**b * nfw_profile(r,rs)
    return gamma_nfw


# not used
def vector_step_function(x, threshold):
    mask_x = x<threshold
    y = np.zeros(x.shape)
    y[mask_x] = 1.
    return y


# virial radius 	
def radvir(m, rho_halo):
    radvir_constants = 3./(4.*math.pi*rho_halo)
    r_vir = (m*radvir_constants)**(1./3.) #Mpc/h
    return r_vir


#ht = hankel.SphericalHankelTransform(nu= 0,		# The order of the bessel function
#                     N = 1000,          			# Number of steps in the integration
#                     h = 0.03)						# Proxy for "size" of steps in integration


def do_transform(h_transf, k, rs_z_m, rvir_m, mnfw_z_m, gamma_1h_amplitude, gamma_b):
    nfw_f = lambda x: gamma_r_nfw_profile(x / k, rs_z_m, rvir_m, gamma_1h_amplitude, gamma_b) * (
                x ** 2.) * np.sqrt(np.pi / (2. * x))
    uk_l_z_m_k = h_transf.integrate(nfw_f)[0] / (k ** 3. * mnfw_z_m)
    #print (np.size(uk_l_z_m_k))
    return uk_l_z_m_k
    

def nfw_f_test(x, k, r_s, rvir, gamma_1h_amplitude, gamma_b):

    func = gamma_r_nfw_profile(x/k, r_s, rvir, gamma_1h_amplitude, gamma_b)*(x**2.)*np.sqrt(np.pi/(2.*x))

    return func

def uell_gamma_r_nfw(gamma_r_nfw_profile, gamma_1h_amplitude, gamma_b, k_vec, z, r_s, rvir, c, mass, ell):
    to = time.time()
    msize = np.size(mass)
    zsize = np.size(z)
    mnfw = mass_nfw(r_s, c)

    h_transf = hankel.hankel.HankelTransform(ell+0.5,300,0.01)
    
    uk_l = np.zeros([z.size, msize, k_vec.size])
    #print(uk_l.shape)
    #print(k_vec.shape, r_s.shape, rvir.shape, gamma_1h_amplitude.shape)
    for jz in range(z.size):
        for im in range(msize):
            for kk in range(k_vec.size):
                nfw_f = lambda x: gamma_r_nfw_profile(x/k_vec[kk], r_s[jz,im], rvir[im], gamma_1h_amplitude[jz], gamma_b)*(x**2.)*np.sqrt(np.pi/(2.*x))
                uk_l[jz, im, kk] = h_transf.integrate(nfw_f)[0]/(k_vec[kk]**3. * mnfw[jz,im])
    print(time.time()-to)
    ukll = uk_l.copy()
    to = time.time()
    x = np.logspace(-15, 15, 80)
    for jz in range(z.size):
        for im in range(msize):
            #for kk in range(k_vec.size):
            uk_l[jz, im, :] = hankl.FFTLog(x, nfw_f_test(x, k_vec, r_s[jz,im], rvir[im], gamma_1h_amplitude[jz], gamma_b), q=1.0, mu=ell+0.5)[1] / (k_vec**3. * mnfw[jz,im])
    print('uell transform: ', time.time()-to)
    #print(uk_l/ukll)
    print(np.allclose(ukll, uk_l))
    plt.plot(ukll[0,0,:])
    plt.plot(uk_l[0,0,:])
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    
    quit()
    #uk_l = h_transf.integrate(nfw_f)[0]/(k_vec**3. * mnfw)
    return uk_l

#------------------------------ uell -----------------------------------#

# create a 4dim array containing u_ell as a function of l,z,m and k
# uell[l,z,m,k]
def IA_uell_gamma_r_hankel(gamma_1h_amplitude, gamma_b, k, c, z, r_s, rvir, mass, ell_max):
    to = time.time()
    uell_ia = [uell_gamma_r_nfw(gamma_r_nfw_profile, gamma_1h_amplitude, gamma_b, k, z, r_s, rvir, c, mass, il) for il in range(0,ell_max+1,2)]
    print('uell array: ', time.time()-to)
    '''
    msize = np.size(c, axis=1) #c.shape[1]
    print ('msize IAuell = ', msize)
    uell_ia = np.zeros((ell_max+1, z.size, msize, k.size)) 
    for i in range(0,int(ell_max/2+1)):
            uell_ia[2*i] = uell_gamma_r_nfw(gamma_r_nfw_profile, gamma_1h_amplitude, gamma_b, k, z, r_s, rvir, c, mass, 2*i)
    '''
    return np.array(uell_ia)

#-----------------------------------------------------------------------#

#------------------------------ w(k|m) ---------------------------------#

#integral of the angular part in eq B8 (SB10) using the Legendre polynomials
#assuming theta_e=theta, phi_e=phi (perfect radial alignment)

def wkm_my_fell(uell, theta_k, phi_k, ell_max, gamma_b):
    to = time.time()
    #nl = np.size(uell, axis=0)
    nz = np.size(uell, axis=1)
    nm = np.size(uell, axis=2)
    nk = np.size(uell, axis=3)
    #print 'nl, nz, nm, nk = ', nl, nz, nm, nk
    sum_ell = np.zeros([nz,nm,nk])
    for ell in range(0,ell_max+1,2):
        angular = another_fell(theta_k, phi_k, ell, gamma_b)
        c_ = np.real(angular)
        d_ = np.imag(angular)
        for jz in range(nz):
            for im in range(nm):
                radial = (1j)**(ell) * (2.*ell + 1.) * uell[int(ell/2),jz,im]
                a_ = np.real(radial)
                b_ = np.imag(radial)
                sum_ell[jz,im] = sum_ell[jz,im] + (a_*c_ - b_*d_)  + 1j*(a_*d_ + b_*c_)
    print('wkm: ', time.time()-to)
    return np.sqrt(np.real(sum_ell)**2.+np.imag(sum_ell)**2.)
