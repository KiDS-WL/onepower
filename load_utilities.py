from cosmosis.datablock import names, option_section
import sys
import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy import interp

from scipy.integrate import quad, simps, trapz


# Marika: remove this and move to the main code
def get_linear_power_spectrum(block, z_vec):
    # AD: growth factor should be computed from camb/hmf directly, this way we can load Plin directly without this functions!
    k_vec = block['matter_power_lin', 'k_h']
    print(k_vec.shape)
    z_pl = block['matter_power_lin', 'z']
    matter_power_lin = block['matter_power_lin', 'p_k']
    growth_factor_zlin = block['growth_parameters', 'd_z']
    gf_interp = interp1d(z_pl, growth_factor_zlin, axis=0)
    growth_factor = gf_interp(z_vec)
    # interpolate in redshift
    plin = interpolate1d_matter_power_lin(matter_power_lin, z_pl, z_vec)
    return k_vec, plin, growth_factor
	
def get_nonlinear_power_spectrum(block, z_vec):
    k_nl = block['matter_power_nl', 'k_h']
    z_nl = block['matter_power_nl', 'z']
    matter_power_nl = block['matter_power_nl', 'p_k']
    # this seems redundant
    p_nl = interpolate1d_matter_power_lin(matter_power_nl, z_nl, z_vec)
    return k_nl, p_nl
	
def compute_effective_power_spectrum(k_vec, plin, k_nl, p_nl, z_vec, t_eff):
    # interpolate
    p_nl_interp = interp2d(k_nl, z_vec, p_nl)
    pnl_int = p_nl_interp(k_vec, z_vec)
    return (1.-t_eff)*plin+t_eff*pnl_int
	
	
def get_halo_functions(block, pipeline, mass, z_vec):
    if pipeline == True:
        # load the halo mass function
        dn_dlnm = block['hmf', 'dndlnmh']
        # load the halobias
        b_dm = block['halobias', 'b_hb']
    else:
        # load the halo mass function
        mass_hmf = block['hmf', 'm_h']
        z_hmf = block['hmf', 'z']
        dndlnmh_hmf = block['hmf', 'dndlnmh']
        # load the halobias
        mass_hbf = block['halobias', 'm_h']
        z_hbf = block['halobias', 'z']
        halobias_hbf = block['halobias', 'b_hb']
        # interpolate all the quantities that enter in the integrals
        dn_dlnm = interpolate2d_dndlnm(dndlnmh_hmf, mass_hmf, z_hmf, mass, z_vec)
        b_dm = interpolate2d_halobias(halobias_hbf, mass_hbf, z_hbf, mass, z_vec)
    return dn_dlnm, b_dm
	
# --------------------- #	
#  satellite alignment  #
# --------------------- #	

def load_growth_factor(block, z_vec):
    z_gwf = block['growth_parameters', 'z']
    D_gwf = block['growth_parameters', 'd_z']
    f_interp = interp1d(z_gwf, D_gwf)
    D_z = f_interp(z_vec)
    return D_z

def get_satellite_alignment(block, k_vec, mass, z_vec, suffix):
    # here I am assuming that the redshifts used in wkm_module and the pk_module match!
    #print( 'entering get_satellite_alignment..')
    wkm = np.empty([z_vec.size, mass.size, k_vec.size])
    for jz in range(0,z_vec.size):
        wkm_tmp = block['wkm_z%d'%jz + suffix,'w_km']
        k_wkm = block['wkm_z%d'%jz + suffix,'k_h']
        mass_wkm = block['wkm_z%d'%jz + suffix,'mass']
        w_interp2d = interp2d(k_wkm, mass_wkm, wkm_tmp)
        wkm_interpolated = w_interp2d(k_vec, mass)
        #print 'wkm_interp.shape = ', wkm_interpolated.shape
        wkm[jz] = wkm_interpolated
    #print( 'wkm.shape = ', wkm.shape)
    return wkm


# interpolation routines
def interpolate2d_dndlnm(dndlnmh_hmf, mass_hmf, z_hmf, mass, z_vec):
    f_interp = interp2d(mass_hmf, z_hmf, dndlnmh_hmf)
    hmf_interpolated = f_interp(mass, z_vec)
    return hmf_interpolated
	
def interpolate2d_halobias(halobias_hbf, mass_hbf, z_hbf, mass, z_vec):
    f_interp = interp2d(mass_hbf, z_hbf, halobias_hbf)
    hbf_interpolated = f_interp(mass, z_vec)
    return hbf_interpolated
	
def interpolate1d_matter_power_lin(matter_power_lin, z_pl, z_vec):
    f_interp = interp1d(z_pl, matter_power_lin, axis=0)
    pk_interpolated = f_interp(z_vec)
    return pk_interpolated
	
	
# load the hod
def load_hods(block, section_name, pipeline, z_vec, mass):
    #section_name = 'hod' + suffix
    print (section_name)
    m_hod = block[section_name, 'mass']
    z_hod = block[section_name,  'z']
    Ncen_hod = block[section_name, 'n_cen']
    Nsat_hod = block[section_name, 'n_sat']
    numdencen_hod = block[section_name, 'number_density_cen']
    numdensat_hod = block[section_name, 'number_density_sat']
    f_c_hod = block[section_name, 'central_fraction']
    f_s_hod = block[section_name, 'satellite_fraction']
    #if pipeline == True:
    #	Ncen = Ncen_hod
    #	Nsat = Nsat_hod
    #	numdencen = numdencen_hod
    #	numdensat = numdensat_hod
    #	f_c = f_c_hod
    #	f_s = f_s_hod
    #else:
    interp_Ncen = interp2d(m_hod, z_hod, Ncen_hod)
    interp_Nsat = interp2d(m_hod, z_hod, Nsat_hod)
    # AD: Is extrapolation warranted here? Maybe make whole calculation on same grid/spacing/thingy!?
    interp_numdencen = interp1d(z_hod, numdencen_hod, fill_value='extrapolate')
    interp_numdensat = interp1d(z_hod, numdensat_hod, fill_value='extrapolate')
    interp_f_c = interp1d(z_hod, f_c_hod, fill_value='extrapolate')
    interp_f_s = interp1d(z_hod, f_s_hod, fill_value='extrapolate')
    Ncen = interp_Ncen(mass, z_vec)
    Nsat = interp_Nsat(mass, z_vec)
    #print ('z_hod', z_hod)
    #print ('z_vec', z_vec)
    numdencen = interp_numdencen(z_vec)
    numdensat = interp_numdensat(z_vec)
    f_c = interp_f_c(z_vec)
    f_s = interp_f_s(z_vec)
	
    return Ncen, Nsat, numdencen, numdensat, f_c, f_s
        
def load_galaxy_fractions(filename, z_vec):
    z_file, fraction_file = np.loadtxt(filename, unpack=True)
    if np.allclose(z_file, z_vec, atol=1e-3):
        return fraction_file
    else:
        print('The redshift of the input galaxy fractions do not match the ranges'
            'set in the pipeline. Performing interpolation.')
        gal_frac_interp = interp(z_vec, z_file, fraction_file)
        print( gal_frac_interp)
        return gal_frac_interp
	
