"""
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

We use the relationship between spherical to normal bessel functions: j_nu(kr) = \sqrt(\pi/(2kr)) J_(nu + 0.5) (kr)

The Hankel transform is the implemented in the hankel.py module
ht = [HankelTransform(ell+0.5,N,h)  (where N and h are sensitivity settings)

F = ht.transform(f,k) calculates int f(r) J_(ell+0.5)(kr) r dr

f(r) =\sqrt(r \pi / 2k) gamma_1h_amplitude (r/rvir)**b \rho_NFW(r,M) / M_NFW

---

We also include a constant core to avoid the power law to explode at small r. The radial
part of the projected shear thus contains a piecewise function: for r<0.06 Mpc/h the shear
is constant with amplitude equal to the value of the shear at r= 0.06 Mpc/h. We also 
require the shear to not exceed the maximum value of 0.3, corresponding to a perfectly aligned
satellite.

"""



import numpy as np
from scipy.integrate import simps
from scipy.special import legendre, binom

#-----------------------------------------------------------------------#
#                           Angular parts                               #
#-----------------------------------------------------------------------#


def I_x(a,b):
    eps = 1e-10
    x = np.linspace(-1.+eps,1.-eps,500)
    int = simps((1.-x**2.)**(a/2.)*x**b,x)
    return int
    
def legendre_coefficients(l,m):
    # note that scipy.special.legendre returns an array with the coefficients of the legendre polynomials
    return legendre(l)[m]

# Computes the angular part of the satellite intrinsic shear field,
# Eq. (C8) in `Fortuna et al. 2021 <https://arxiv.org/abs/2003.02700>`
def calculate_f_ell(theta_k, phi_k, l, gamma_b):

    phase = np.cos(2.*phi_k) + 1j*np.sin(2.*phi_k)

    # Follow CCL by hard-coding for most common cases (b=0, b=-2) to gain speed 
    # (in CCL gain is ~1.3sec - gain here depends on how many times this is called).
    if theta_k==np.pi/2.:
        if gamma_b==0:
            pre_calc_f_ell = np.array([0,0,2.77582637, 0 , -0.19276603, 0,
                                   0.04743899, 0, -0.01779024, 0,
                                   0.00832446, 0, -0.00447308, 0])
            return pre_calc_f_ell[l]*phase
    
        if gamma_b==-2:
            pre_calc_f_ell = np.array([0,0,4.71238898, 0, -2.61799389, 0,
                                         2.06167032, 0, -1.76714666, 0,
                                        1.57488973, 0, -1.43581368, 0])
            return pre_calc_f_ell[l]*phase

    # If either of the above expressions are met the return statement is executed and the function ends.
    # Otherwise, the function continues to calculate the general case.
    gj = np.array([0, 0, np.pi / 2, 0, np.pi / 2, 0, 15 * np.pi / 32,
                       0, 7 * np.pi / 16, 0, 105 * np.pi / 256, 0])
    
    sum1 = 0.
    for m in range(0,l+1):
        sum2=0.
        for j in range(0,m+1):
            sum2 += binom(m,j) * gj[j] * np.sin(theta_k)**(j) * np.cos(theta_k)**(m-j) * I_x(j+gamma_b,m-j)
        sum1 += binom(l,m)*binom(0.5*(l+m-1.),l)*sum2
    return 2.**l * sum1*phase

            
def compare_leg_coeff(l,m):
    print(legendre_coefficients(l,m))
    print(2.**l * binom(l,m)*binom(0.5*(l+m-1.),l))
    return


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
    

def gamma_r_nfw_profile_trunc(r, rs, rvir, A, b, rcore=0.06):
    mask_small_r = r<rcore
    gamma = A*(r/rvir)**b
    # note that in the case of non-radial dependent alignment, this cut is not recommendable since it introduces ringing
    gamma[mask_small_r] = A*(rcore/rvir)**b
    gamma[gamma > 0.3] = 0.3
    gamma_weighted = gamma * nfw_profile_trunc(r,rs,rvir)
    return gamma_weighted


def gamma_r_nfw_profile_notrunc(r, rs, rvir, A, b, rcore=0.06):
    mask_small_r = r<rcore
    gamma = A*(r/rvir)**b
    # note that in the case of non-radial dependent alignment, this cut is not recommendable since it introduces ringing
    gamma[mask_small_r] = A*(rcore/rvir)**b
    gamma[gamma > 0.3] = 0.3
    gamma_weighted = gamma * nfw_profile(r,rs)
    return gamma_weighted


# not used
def vector_step_function(x, threshold):
    mask_x = x<threshold
    y = np.zeros(x.shape)
    y[mask_x] = 1.
    return y


# virial radius 	
def radvir(m, rho_halo):
    radvir_constants = 3./(4.*np.pi*rho_halo)
    r_vir = (m*radvir_constants)**(1./3.) #Mpc/h
    return r_vir
    

def uell_gamma_r_nfw(gamma_r_nfw_profile, gamma_1h_amplitude, gamma_b, k_vec, z, r_s, rvir, c, mass, ell, h_transf):
    msize = np.size(mass)
    zsize = np.size(z)
    mnfw = mass_nfw(r_s, c)

    #h_transf = HankelTransform(ell+0.5,N_hankel,pi/N_hankel)
    #Note even though ell is not used in this function, h_transf depends on ell
    #We initialise the class in setup as it only depends on predefined ell values
    
    uk_l = np.zeros([zsize, msize, k_vec.size])

    #Note: I experimented coding the use of Simpson integration for where the Bessel function is flat 
    #and then switching to the Hankel transform for where the Bessel function oscillates.
    #This is more accurate than using the Hankel transform for all k values with lower accuracy 
    #settings, but it's slower than using the Hankel transform for all k values.
    #It's also difficult to decide how to define the transition between the two methods.
    #Given that low-k accuracy is unimportant for IA, I've decided to use the Hankel transform for all k values.
    
    for jz in range(zsize):
        for im in range(msize):
            #The python lambda "anonymous function" is new to me so an explaination is in order:
            #this line connects nfw_f to the executable function gamma_r_nfw_profile varying x and keeping the other parameters fixed
            nfw_f = lambda x: gamma_r_nfw_profile(x, r_s[jz,im], rvir[im], gamma_1h_amplitude[jz], gamma_b) * np.sqrt((x*np.pi)/2.0)
            uk_l[jz, im, :] = h_transf.transform(nfw_f, k_vec)[0] / (k_vec**0.5 * mnfw[jz,im])
    return uk_l


#------------------------------ uell -----------------------------------#

# create a 4dim array containing u_ell as a function of l,z,m and k
# uell[l,z,m,k]
def IA_uell_gamma_r_hankel(gamma_1h_amplitude, gamma_b, k, c, z, r_s, rvir, mass, ell_max, h_transf):
    uell_ia = [uell_gamma_r_nfw(gamma_r_nfw_profile_notrunc, gamma_1h_amplitude, gamma_b, k, z, r_s, rvir, c, mass, il, h_transf[i]) for i,il in enumerate(range(0,ell_max+1,2))]
    return np.array(uell_ia)

#-----------------------------------------------------------------------#

#------------------------------ w(k|m) ---------------------------------#

#integral of the angular part in eq B8 (SB10) using the Legendre polynomials
#assuming theta_e=theta, phi_e=phi (perfect radial alignment)

def wkm_f_ell(uell, theta_k, phi_k, ell_max, gamma_b):
    nz = uell.shape[1]
    nm = uell.shape[2]
    nk = uell.shape[3]
    
    sum_ell = np.zeros([nz,nm,nk])
    for ell in range(0,ell_max+1,2):
        angular = calculate_f_ell(theta_k, phi_k, ell, gamma_b)
        c_ = np.real(angular)
        d_ = np.imag(angular)
        radial = (1j)**(ell) * (2.*ell + 1.) * uell[int(ell/2),:,:,:]
        a_ = np.real(radial)
        b_ = np.imag(radial)
        sum_ell = sum_ell + (a_*c_ - b_*d_)  + 1j*(a_*d_ + b_*c_)

    return np.sqrt(np.real(sum_ell)**2.+np.imag(sum_ell)**2.)

#Note CCL only calculates the real parts of w(k|m)f_ell and doesn't take the absolute value....
#which means you'll get negative values for wkm in CCL: they take the absolute value later.
