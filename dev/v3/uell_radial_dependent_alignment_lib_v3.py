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

from wkm_angular_part import *

#import matplotlib.pyplot as plt


NewtonGravConst = 4.2994E-9
rho_crit0= (3.*10**4)/(8.*math.pi*NewtonGravConst) 		#critical density at redshift 0


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

# I've checked the above implementation is equivalent!
def mass_nfw_test(r_s, c, zsize, msize):
    mnfw_test =  4.*np.pi*(r_s**3.)*(np.log(1.+c)-c/(1.+c))
    mnfw = np.zeros([zsize, msize])
    for jz in range(0,zsize):
        mnfw[jz] = 4.*np.pi*(r_s[jz]**3.)*(np.log(1.+c[jz])-c[jz]/(1.+c[jz]))
        print (mnfw_test[jz]/mnfw[jz])
    return mnfw


def nfw_profile_trunc(r, rs, rvir):
    mask_above_rvir = r>=rvir
    nfw = nfw_profile(r, rs)
    nfw[mask_above_rvir] = 0.0
    return nfw

def gamma_r_nfw_profile(r, rs, rvir, A, b):
    rcore = 0.06 #Mpc/h
    mask_small_r = r<rcore
    gamma = A*(r/rvir)**b
    # note that in the case of non-radial dependent alignment, this cut is not recommendable since it introduces ringing
    gamma[mask_small_r] = A*(rcore/rvir)**b
    mask_gamma = gamma > 0.3
    gamma[mask_gamma] = 0.3
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


#Navarro-Frenk-White density profile
# only for comparison -> note that the normalisation is different, due to the shear!
def norm_fourier(x, c):
    # Note: x = k*r_scale where r_scale= r200/c
    si, ci = sici((1+c)*x)
    si2, ci2 = sici(x)
    sinterm = np.sin(x)*( si-si2 ) -np.sin(c*x)/((1+c)*x)
    costerm = np.cos(x)*( ci-ci2 )
    f_c = np.log(1+c)-c/(1+c)
    u_fourier = ( sinterm + costerm )/f_c
    return u_fourier


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

def uell_gamma_r_nfw(gamma_r_nfw_profile, gamma_1h_amplitude, gamma_b, k_vec, z, r_s, rvir, c, mass, ell):
    msize = np.size(mass)
    zsize = np.size(z)
    mnfw = mass_nfw(r_s, c)

    h_transf = hankel.hankel.HankelTransform(ell+0.5,300,0.01)
    #h_transf = hankel.hankel.HankelTransform(ell+0.5,1000,0.00005)

    #uk_l = np.array([[[do_transform(h_transf, k_vec[ik], r_s[jz,im], rvir[im], mnfw[jz,im], gamma_1h_amplitude, gamma_b) for ik in range(0,k_vec.size)] for im in range(0,msize)] for jz in range(0,z.size)])
    #print (uk_l.shape)


    uk_l = np.zeros([z.size, msize, k_vec.size])
    for jz in range(z.size):
        for im in range(msize):
            for kk in range(k_vec.size):
                nfw_f = lambda x: gamma_r_nfw_profile(x/k_vec[kk], r_s[jz,im], rvir[im], gamma_1h_amplitude[jz], gamma_b)*(x**2.)*np.sqrt(np.pi/(2.*x))
                uk_l[jz, im, kk] = h_transf.integrate(nfw_f)[0]/(k_vec[kk]**3. * mnfw[jz,im])
    return uk_l

#------------------------------ uell -----------------------------------#

# create a 4dim array containing u_ell as a function of l,z,m and k
# uell[l,z,m,k]
def IA_uell_gamma_r_hankel(gamma_1h_amplitude, gamma_b, k, c, z, r_s, rvir, mass, ell_max):
    uell_ia = [uell_gamma_r_nfw(gamma_r_nfw_profile, gamma_1h_amplitude, gamma_b, k, z, r_s, rvir, c, mass, il) for il in range(0,ell_max+1,2)]
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

def wkm_my_fell(uell, theta_k, phi_k, ell_max):
    #nl = np.size(uell, axis=0)
    nz = np.size(uell, axis=1)
    nm = np.size(uell, axis=2)
    nk = np.size(uell, axis=3)
    #print 'nl, nz, nm, nk = ', nl, nz, nm, nk
    sum_ell = np.zeros([nz,nm,nk])
    for ell in range(0,ell_max+1,2):
        angular = another_fell(theta_k, phi_k, ell)
        c_ = np.real(angular)
        d_ = np.imag(angular)
        for jz in range(nz):
            for im in range(nm):
                radial = (1j)**(ell) * (2.*ell + 1.) * uell[int(ell/2),jz,im]
                a_ = np.real(radial)
                b_ = np.imag(radial)
                sum_ell[jz,im] = sum_ell[jz,im] + (a_*c_ - b_*d_)  + 1j*(a_*d_ + b_*c_)
    return np.sqrt(np.real(sum_ell)**2.+np.imag(sum_ell)**2.)
