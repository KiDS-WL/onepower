# Library of the power spectrum module
import numexpr as ne
import numpy as np
import warnings
from scipy.integrate import quad, simpson, trapezoid
from scipy.interpolate import RegularGridInterpolator, UnivariateSpline, interp1d
from scipy.ndimage import gaussian_filter1d

# from scipy.special import erf
from scipy.optimize import curve_fit


def interpolate_in_z(input_grid, z_in, z_out, axis=0):
    """
    Interpolation in redshift
    Default redshift axis is the first one.
    """
    f_interp = interp1d(z_in, input_grid, axis=axis)
    return f_interp(z_out)

def log_linear_interpolation_k(power_in, k_in, k_out, axis=1, kind='linear'):
    """
    log-linear interpolation for power spectra. This works well for extrapolating to higher k.
    Ideally we want to have a different routine for interpolation (spline) and extrapolation (log-linear)
    """
    power_interp = interp1d(np.log(k_in), np.log(power_in), axis=axis, kind=kind, fill_value='extrapolate')
    return np.exp(power_interp(np.log(k_out)))

def get_linear_power_spectrum(block, z_vec):
    """
    Reads in linear matter power spectrum and downsamples
    """
    k_vec = block['matter_power_lin', 'k_h']
    z_pl = block['matter_power_lin', 'z']
    matter_power_lin = block['matter_power_lin', 'p_k']
    return k_vec, interpolate_in_z(matter_power_lin, z_pl, z_vec)

# Reads in the growth factor
def get_growth_factor(block, z_vec, k_vec):
    """
    Loads and interpolates the growth factor
    and scale factor
    Reads in the growth factor and turns it into a 2D array that has this dimensions: len(z) x len(k)
    all columns are identical
    """
    z_in = block['growth_parameters', 'z']
    growth_factor_in = block['growth_parameters', 'd_z']
    growth_factor = interpolate_in_z(growth_factor_in, z_in, z_vec)
    growth_factor = growth_factor.flatten()[:, np.newaxis] * np.ones(k_vec.size)
    scale_factor = 1.0 / (1.0 + z_vec)
    scale_factor = scale_factor.flatten()[:, np.newaxis] * np.ones(k_vec.size)
    return growth_factor, scale_factor

def get_nonlinear_power_spectrum(block, z_vec):
    """
    Reads in the non-linear matter power specturm and downsamples
    """
    k_nl = block['matter_power_nl', 'k_h']
    z_nl = block['matter_power_nl', 'z']
    matter_power_nl = block['matter_power_nl', 'p_k']
    return k_nl, interpolate_in_z(matter_power_nl, z_nl, z_vec)

def get_halo_functions(block):
    """
    Loads the halo mass function and linear halo bias
    """
    mass_hmf = block['hmf', 'm_h']
    z_hmf = block['hmf', 'z']
    dndlnmh = block['hmf', 'dndlnmh']
    mass_hbf = block['halobias', 'm_h']
    z_hbf = block['halobias', 'z']
    halobias = block['halobias', 'b_hb']
    return dndlnmh, halobias, mass_hmf, z_hmf

def get_normalised_profile(block, mass, z_vec):
    """
    Reads the Fourier transform of the normalised Dark matter halo profile U.
    Checks that mass, redshift and k match the input.
    """
    z_udm = block['fourier_nfw_profile', 'z']
    mass_udm = block['fourier_nfw_profile', 'm_h']
    k_udm = block['fourier_nfw_profile', 'k_h']
    u_dm = block['fourier_nfw_profile', 'ukm']
    u_sat = block['fourier_nfw_profile', 'uksat']
    # For now we assume that centrals are in the centre of the haloes so no need for
    # defnining their profile
    # u_cen    = block['fourier_nfw_profile', 'ukcen']

    #u_dm = interpolate_in_z(u_dm_in, z_udm, z_vec)
    #u_sat = interpolate_in_z(u_sat_in, z_udm, z_vec)
    if (mass_udm != mass).any():
        raise ValueError('The profile mass values are different to the input mass values.')
    return u_dm, u_sat, k_udm

def get_satellite_alignment(block, k_vec, mass, z_vec, suffix):
    """
    Loads and interpolates the wkm profiles needed for calculating the IA power spectra
    """
    wkm = np.empty([z_vec.size, mass.size, k_vec.size])
    for jz in range(z_vec.size):
        wkm_tmp = block['wkm', f'w_km_{jz}{suffix}']
        k_wkm = block['wkm', f'k_h_{jz}{suffix}']
        mass_wkm = block['wkm', f'mass_{jz}{suffix}']
        lg_w_interp2d = RegularGridInterpolator((np.log10(k_wkm).T, np.log10(mass_wkm).T),
                                                np.log10(wkm_tmp / k_wkm**2).T, bounds_error=False, fill_value=None)
        lgkk, lgmm = np.meshgrid(np.log10(k_vec), np.log10(mass), sparse=True)
        lg_wkm_interpolated = lg_w_interp2d((lgkk.T, lgmm.T)).T
        wkm[jz] = 10.0**(lg_wkm_interpolated) * k_vec**2.0
    return wkm

def get_satellite_alignment_new(block, k_vec, mass, z_vec, suffix):
    """
    Loads and interpolates the wkm profiles needed for calculating the IA power spectra
    """
    wkm = np.empty([z_vec.size, mass.size, k_vec.size])
    for jz in range(z_vec.size):
        wkm_tmp = block['wkm', f'w_km_{jz}{suffix}']
        wkm[jz, :, :] = wkm_tmp
    return wkm

def load_hods(block, section_name, suffix, z_vec, mass):
    """
    Loads and interpolates the hod quantities to match the
    calculation of power spectra
    """
    m_hod = block[section_name, f'mass{suffix}']
    z_hod = block[section_name, f'z{suffix}']
    Ncen_hod = block[section_name, f'n_cen{suffix}']
    Nsat_hod = block[section_name, f'n_sat{suffix}']
    numdencen_hod = block[section_name, f'number_density_cen{suffix}']
    numdensat_hod = block[section_name, f'number_density_sat{suffix}']
    f_c_hod = block[section_name, f'central_fraction{suffix}']
    f_s_hod = block[section_name, f'satellite_fraction{suffix}']
    mass_avg_hod = block[section_name, f'average_halo_mass{suffix}']

    if (m_hod != mass).any():
        raise ValueError('The HOD mass values are different to the input mass values.')

    #If we're using an unconditional HOD, we need to define the stellar fraction with zeros
    try:
        f_star = block[section_name, f'f_star{suffix}']
    except:
        f_star = np.zeros((len(z_hod), len(m_hod)))

    #interp_Ncen  = RegularGridInterpolator((m_hod.T, z_hod.T), Ncen_hod.T, bounds_error=False, fill_value=0.0)
    #interp_Nsat  = RegularGridInterpolator((m_hod.T, z_hod.T), Nsat_hod.T, bounds_error=False, fill_value=0.0)
    #interp_fstar = RegularGridInterpolator((m_hod.T, z_hod.T), f_star.T, bounds_error=False, fill_value=0.0)

    interp_Ncen = interp1d(z_hod, Ncen_hod, fill_value='extrapolate', bounds_error=False, axis=0)
    interp_Nsat = interp1d(z_hod, Nsat_hod, fill_value='extrapolate', bounds_error=False, axis=0)
    interp_fstar = interp1d(z_hod, f_star, fill_value='extrapolate', bounds_error=False, axis=0)
    interp_numdencen = interp1d(z_hod, numdencen_hod, fill_value='extrapolate', bounds_error=False)
    interp_numdensat = interp1d(z_hod, numdensat_hod, fill_value='extrapolate', bounds_error=False)
    interp_f_c = interp1d(z_hod, f_c_hod, fill_value=0.0, bounds_error=False)
    interp_f_s = interp1d(z_hod, f_s_hod, fill_value=0.0, bounds_error=False)
    interp_mass_avg = interp1d(z_hod, mass_avg_hod, fill_value=0.0, bounds_error=False)

    #mm, zz = np.meshgrid(mass, z_vec, sparse=True)
    #Ncen  = interp_Ncen((mm.T, zz.T)).T
    #Nsat  = interp_Nsat((mm.T, zz.T)).T
    #fstar = interp_fstar((mm.T, zz.T)).T
    Ncen = interp_Ncen(z_vec)
    Nsat = interp_Nsat(z_vec)
    fstar = interp_fstar(z_vec)
    numdencen = interp_numdencen(z_vec)
    numdensat = interp_numdensat(z_vec)
    f_c = interp_f_c(z_vec)
    f_s = interp_f_s(z_vec)
    mass_avg = interp_mass_avg(z_vec)

    return Ncen, Nsat, numdencen, numdensat, f_c, f_s, mass_avg, fstar

def load_hods_new(block, section_name, suffix, z_vec, mass):
    """
    Loads and interpolates the hod quantities to match the
    calculation of power spectra
    """
    m_hod = block[section_name, f'mass{suffix}']
    z_hod = block[section_name, f'z{suffix}']
    Ncen = block[section_name, f'n_cen{suffix}']
    Nsat = block[section_name, f'n_sat{suffix}']
    numdencen = block[section_name, f'number_density_cen{suffix}']
    numdensat = block[section_name, f'number_density_sat{suffix}']
    f_c = block[section_name, f'central_fraction{suffix}']
    f_s = block[section_name, f'satellite_fraction{suffix}']
    mass_avg = block[section_name, f'average_halo_mass{suffix}']

    if (m_hod != mass).any():
        raise ValueError('The HOD mass values are different to the input mass values.')

    #If we're using an unconditional HOD, we need to define the stellar fraction with zeros
    fstar = block[section_name, f'f_star{suffix}']

    return Ncen, Nsat, numdencen, numdensat, f_c, f_s, mass_avg, fstar

def load_fstar_mm(block, section_name, z_vec, mass):
    """
    Load stellar fraction that is calculated with the Cacciato HOD
    """
    if block.has_value(section_name, 'f_star_extended'):
        f_star = block[section_name, 'f_star_extended']
        m_hod = block[section_name, 'mass_extended']
        z_hod = block[section_name, 'z_extended']
    else:
        raise ValueError('f_star_extended does not exist in the provided section.')

    interp_fstar = RegularGridInterpolator((m_hod.T, z_hod.T), f_star.T, bounds_error=False, fill_value=None)
    mm, zz = np.meshgrid(mass, z_vec, sparse=True)
    return interp_fstar((mm.T, zz.T)).T
