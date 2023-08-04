import numpy as np
from scipy.integrate import simps

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
def mor(mass, hod, norm) :
    # mor = mass - observable relation
    # (observable can be galaxy luminosity or stellar mass)
    # returns the observable given halo mass. Assumed to be a double power law with characteristic
    # scale ml_1, normalisation ml_0 and slopes g_1 and g_2
    # We should generalise this (see GGL pipeline for ideas)
    mor = norm * (hod.ml_0*(mass/hod.ml_1)**hod.g_1)/(1.+(mass/hod.ml_1))**(hod.g_1-hod.g_2)
    return mor


def phi_star(mass, hod):
    logM_12 = np.log10(mass) - hod.pivot
    log_phi_s = hod.b0 + hod.b1*logM_12 + hod.b2*(logM_12**2.)
    return 10.**log_phi_s
    
    
def cf_cen(obs, mass, hod):
    # log10(e)/sqrt(2*pi) = 0.17325843097
    # AD: keeping this approximation in, but considering to replace with the on the fly calculation
    cf_c = np.log((0.17325843097/(hod.sigma_c))) +((-(np.log10(obs)-np.log10(mor(mass,hod,hod.norm_c)))**2.)/(2.*hod.sigma_c**2.)) - np.log(obs)
    return np.exp(cf_c)


def cf_sat(obs, mass, hod):
    obs_star = mor(mass, hod, hod.norm_s)
    obs_tilde = obs/obs_star
    phistar = phi_star(mass, hod)
    cf_s = (phistar/obs_star)*(obs_tilde**(hod.alpha_star))*np.exp(-obs_tilde**2.)
    return cf_s


def obs_func(mass, phi_clf, dn_dlnM_normalised, axis=-1):
    lf_integrand = phi_clf*dn_dlnM_normalised/mass
    lf_integral = simps(lf_integrand, mass, axis=axis)
    return lf_integral


def compute_hod(obs, phi_clf):
    hod_integral = simps(phi_clf, obs)
    return hod_integral
    
    
def compute_stellar_fraction(obs, phi_clf):
    integral = simps(phi_clf*obs, obs)
    return integral
   
   
def compute_number_density(mass, N_g, dn_dlnM_normalised):
    n_integrand = N_g*dn_dlnM_normalised/mass
    n_integral = simps(n_integrand, mass)
    return n_integral
    

def compute_avg_halo_mass(mass, N_g, dn_dlnM_normalised):
    n_integrand = N_g*dn_dlnM_normalised
    n_integral = simps(n_integrand, mass)
    return n_integral


def compute_galaxy_linear_bias(mass, N_g, halo_bias, dn_dlnM_normalised):
    bg_integrand = N_g*halo_bias*dn_dlnM_normalised/mass
    bg_integral = simps(bg_integrand, mass)
    return bg_integral
