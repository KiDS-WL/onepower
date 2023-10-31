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

# ------------------------------------------#
#    Conditional Observable Function        #
#-------------------------------------------#
# These are functions of halo mass, M, and the observable, O, which can be galaxy stellar mass or galaxy luminosity.
# The conditional observable Function is differnt for central and satellite galaxies. 
# Φ(O|M) = Φc(O|M) + Φs(O|M)  
# The halo mass dependence comes in through pivot observable values denoted by *, e.g. O∗c, O∗s

# Conditional Observable function for Central galaxies
# eq 17 of D23: 2210.03110: 
# Φc(O|M) =1/[√(2π) ln(10) σ_c O] exp[ -log(O/O∗c)^2/ (2 σ_c^2) ]
# used to be cf_cen
def phi_cen(obs, mass, hod_par):
    obs_star_c = obs_star(mass,hod_par,hod_par.norm_c) #O∗c
    phi_c = 1./(np.sqrt(2.*np.pi)* np.log(10)* hod_par.sigma_c*obs) * np.exp((-(np.log10(obs/obs_star_c))**2)/(2*hod_par.sigma_c**2.))
    # log10(e)/sqrt(2*pi) = 0.17325843097
    # cf_c = np.log((0.17325843097/(hod_par.sigma_c))) - np.log(obs) + ((-(np.log10(obs)-np.log10(obs_star_c))**2.)/(2.*hod_par.sigma_c**2.)) 
    # return np.exp(cf_c)
    return phi_c

# Conditional Observable function for satellite galaxies
# eq 18 of D23: 2210.03110:
# Φs(O|M) =ϕ∗s/O∗s  (O/O∗s)^α_s exp [−(O/O∗s)^2], O*s is O∗s(M) = 0.56 O∗c(M) 
# used to be cf_sat
def phi_sat(obs, mass, hod_par):
    obs_s_star = obs_star(mass, hod_par, hod_par.norm_s)
    obs_tilde = obs/obs_s_star
    phi_star_val = phi_star(mass, hod_par)
    phi_s = (phi_star_val/obs_s_star)*(obs_tilde**(hod_par.alpha_star))*np.exp(-obs_tilde**2.)
    return phi_s

# The Observable function, this is simply Φs(O|M), Φc(O|M) integrated over the halo mass to give:  Φs(O),Φc(O)
def obs_func(mass, phi, dn_dlnM_normalised, axis=-1):
    integrand = phi*dn_dlnM_normalised/mass
    obs_function = simps(integrand, mass, axis=axis)
    return obs_function


# eqs 19 and 20 of D23: 2210.03110 
# O∗c(M) = O_0 (M/M1)^γ1 / [1 + (M/M1)]^(γ1−γ2) 
# and 
# O∗s(M) = 0.56 O∗c(M)
# Here M1 is a characteristic mass scale, and O_0 is the normalization.
# used to be mor
def obs_star(mass, hod_par, norm) :
    # (observable can be galaxy luminosity or stellar mass)
    # returns the observable given halo mass. Assumed to be a double power law with characteristic
    # scale m_1, normalisation m_0 and slopes g_1 and g_2
    obs_star = norm * hod_par.O_0 * (mass/hod_par.m_1)**hod_par.g_1/(1.+(mass/hod_par.m_1))**(hod_par.g_1-hod_par.g_2)
    return obs_star

# pivot mass used in eq 21 of D23: 2210.03110
# using a bias expantion around the pivot mass
# log[ϕ∗s(M)] = b0 + b1(log m13) ,m13 is logM_pivot, m13 = M/(hod_par.pivot M⊙ h−1), 
def phi_star(mass, hod_par):
    logM_pivot = np.log10(mass) - hod_par.pivot
    log_phi_s = hod_par.b0 + hod_par.b1*logM_pivot + hod_par.b2*(logM_pivot**2.)
    return 10.**log_phi_s





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
