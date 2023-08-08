import numpy as np
from scipy.integrate import simps

# AD: make these general for stellar mass, add different options for flexibility (single power law, ...)
# AD: Superseeded by cf_lib.py but keeping this one here for now if we missed something ...

# Conversion functions

def convert_to_luminosity(abs_mag, abs_mag_sun):
	# Luminosities [L_sun h^2]
	logL = -0.4*(abs_mag - abs_mag_sun)	
	return 10.**logL
	
def convert_to_magnitudes(L, abs_mag_sun):
	logL = np.log10(L)
	# Mr -5 log(h)
	Mr = -2.5*logL+abs_mag_sun
	return Mr

# ------------------------------------------#
# 				HOD library					#
#-------------------------------------------#
def L_c (mass, hod) :
    lc = (hod.l_0*(mass/hod.m_1)**hod.g_1)/(1.+(mass/hod.m_1))**(hod.g_1-hod.g_2)
    return lc

def phi_star(mass, hod):
    logM_12 = np.log10(mass)-12.
    log_phi_s = hod.b0 + hod.b1*logM_12 + hod.b2*(logM_12**2.)
    return 10.**log_phi_s
    
def Ls_star(mass, hod):
    lc = L_c(mass, hod)
    return 0.562*lc

def clf_cen(L, mass, hod):
    # log10(e)/sqrt(2*pi) = 0.17325843097
    clf_c = np.log((0.17325843097/(hod.sigma_c))) +((-(np.log10(L)-np.log10(L_c(mass,hod)))**2.)/(2.*hod.sigma_c**2.)) - np.log(L)
    return np.exp(clf_c)

def clf_sat(L, mass, hod):
    Lstar = Ls_star(mass, hod)
    Ltilde = L/Lstar
    phistar = phi_star(mass, hod)
    clf_s = (phistar/Lstar)*(Ltilde**(hod.alpha_star))*np.exp(-Ltilde**2.)
    return clf_s

def LF(mass, phi_clf, dn_dlnM_normalised):
    lf_integrand = phi_clf*dn_dlnM_normalised/mass
    lf_integral = simps(lf_integrand, mass)
    return lf_integral

def compute_hod(L, phi_clf):
    hod_integral = simps(phi_clf, L)
    return hod_integral
    
def compute_number_density(mass, N_g, dn_dlnM_normalised):
    n_integrand = N_g*dn_dlnM_normalised/mass
    n_integral = simps(n_integrand, mass)
    return n_integral

def compute_galaxy_linear_bias(mass, N_g, halo_bias, dn_dlnM_normalised):
    bg_integrand = N_g*halo_bias*dn_dlnM_normalised/mass
    bg_integral = simps(bg_integrand, mass)
    return bg_integral
