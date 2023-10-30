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

# TODO change the names of some of these functions to make them more intuitive 
# eqs 19 and 20 of D23: 2210.03110, M∗c(M) = M0(M/M1)^γ1 / [1 + (M/M1)]^(γ1−γ2) and M∗s(M) = 0.56 M∗c(M)
# used to be mor
def obs_star(mass, hod_par, norm) :
    # (observable can be galaxy luminosity or stellar mass)
    # returns the observable given halo mass. Assumed to be a double power law with characteristic
    # scale ml_1, normalisation ml_0 and slopes g_1 and g_2
    # We should generalise this (see GGL pipeline for ideas)
    mor = norm * hod_par.ml_0 * (mass/hod_par.ml_1)**hod_par.g_1/(1.+(mass/hod_par.ml_1))**(hod_par.g_1-hod_par.g_2)
    return mor

# pivot mass used in eq 21 of D23: 2210.03110
# using a bias expantion around the pivot mass
# log[ϕ∗s(M)] = b0 + b1(log m13) ,m13 is logM_pivot, m13 = M/(hod_par.pivot M⊙ h−1), 
def phi_star(mass, hod_par):
    logM_pivot = np.log10(mass) - hod_par.pivot
    log_phi_s = hod_par.b0 + hod_par.b1*logM_pivot + hod_par.b2*(logM_pivot**2.)
    return 10.**log_phi_s


# eq 17 of D23: 2210.03110: Φc(M⋆|M) =1/[√(2π) ln(10) σ_c M⋆] exp[ -log(M⋆/M∗c)^2/ (2 σ_c^2) ]
# used to be cf_cen
def phi_cen(obs, mass, hod_par):
    obs_c_star = obs_star(mass,hod_par,hod_par.norm_c)
    phi_c = 1./(np.sqrt(2.*np.pi)* np.log(10)* hod_par.sigma_c*obs) * np.exp((-(np.log10(obs/obs_c_star))**2)/(2*hod_par.sigma_c**2.))
    # log10(e)/sqrt(2*pi) = 0.17325843097
    # cf_c = np.log((0.17325843097/(hod_par.sigma_c))) - np.log(obs) + ((-(np.log10(obs)-np.log10(obs_c_star))**2.)/(2.*hod_par.sigma_c**2.)) 
    # return np.exp(cf_c)
    return phi_c

# eq 18 of D23: 2210.03110:
# Φs(M⋆|M) =ϕ∗s/M∗s  (M⋆/M∗s)^α_s exp [−(M⋆/M∗s)^2], M*s is M∗s(M) = 0.56 M∗c(M) 
# used to be cf_sat
def phi_sat(obs, mass, hod_par):
    obs_s_star = obs_star(mass, hod_par, hod_par.norm_s)
    obs_tilde = obs/obs_s_star
    phi_star_val = phi_star(mass, hod_par)
    phi_s = (phi_star_val/obs_s_star)*(obs_tilde**(hod_par.alpha_star))*np.exp(-obs_tilde**2.)
    return phi_s


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
