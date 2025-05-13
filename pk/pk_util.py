# Library of the power spectrum module
import numpy as np
import numexpr as ne
from scipy.interpolate import interp1d, RegularGridInterpolator, UnivariateSpline
from scipy.integrate import simpson, quad, trapezoid
# from scipy.special import erf
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import warnings

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

def low_k_truncation(k_vec, k_trunc):
    """
    Beta_nl low-k truncation
    """
    return 1.0 / (1.0 + np.exp(-(10.0 * (np.log10(k_vec) - np.log10(k_trunc)))))

def high_k_truncation(k_vec, k_trunc):
    """
    Beta_nl high-k truncation
    """
    return 1.0 / (1.0 + np.exp((10.0 * (np.log10(k_vec) - np.log10(k_trunc)))))

def minimum_halo_mass(emu):
    """
    Minimum halo mass for the set of cosmological parameters [Msun/h]
    """
    np_min = 200.0 # Minimum number of halo particles
    npart = 2048.0 # Cube root of number of simulation particles
    Lbox_HR = 1000.0 # Box size for high-resolution simulations [Mpc/h]
    Lbox_LR = 2000.0 # Box size for low-resolution simulations [Mpc/h]

    Om_m = emu.cosmo.get_Omega0()
    rhom = 2.77536627e11 * Om_m

    Mbox_HR = rhom * Lbox_HR**3.0
    mmin = Mbox_HR * np_min / npart**3.0

    vmin = Lbox_HR**3.0 * np_min / npart**3.0
    rmin = ((3.0 * vmin) / (4.0 * np.pi))**(1.0 / 3.0)

    return mmin, 2.0 * np.pi / rmin

def rvir(emu, mass):
    Om_m = emu.cosmo.get_Omega0()
    rhom = 2.77536627e11 * Om_m
    return ((3.0 * mass) / (4.0 * np.pi * 200 * rhom))**(1.0 / 3.0)

def hl_envelopes_idx(data, dmin=1, dmax=1):
    """
    Extract high and low envelope indices from a 1D data signal.

    Parameters:
    data (1d-array): Data signal from which to extract high and low envelopes.
    dmin (int): Size of chunks for local minima, use this if the size of the input signal is too big.
    dmax (int): Size of chunks for local maxima, use this if the size of the input signal is too big.

    Returns:
    lmin, lmax (tuple of arrays): Indices of high and low envelopes of the input signal.
    """
    # Find local minima indices
    lmin = (np.diff(np.sign(np.diff(data))) > 0).nonzero()[0] + 1
    # Find local maxima indices
    lmax = (np.diff(np.sign(np.diff(data))) < 0).nonzero()[0] + 1

    # Global min of dmin-chunks of local minima
    lmin = lmin[[i + np.argmin(data[lmin[i:i + dmin]]) for i in range(0, len(lmin), dmin)]]
    # Global max of dmax-chunks of local maxima
    lmax = lmax[[i + np.argmax(data[lmax[i:i + dmax]]) for i in range(0, len(lmax), dmax)]]

    return lmin, lmax

def compute_bnl_darkquest(z, log10M1, log10M2, k, emulator, block, kmax):
    M1 = 10.0**log10M1
    M2 = 10.0**log10M2

    # Large 'linear' scale for linear halo bias [h/Mpc]
    klin = np.array([0.05])

    # Calculate beta_NL by looping over mass arrays
    beta_func = np.zeros((len(M1), len(M2), len(k)))

    # Linear power
    Pk_lin = emulator.get_pklin_from_z(k, z)
    Pk_klin = emulator.get_pklin_from_z(klin, z)

    # Calculate b01 for all M1
    b01 = np.zeros(len(M1))
    #b02 = np.zeros(len(M2))
    for iM, M0 in enumerate(M1):
        b01[iM] = np.sqrt(emulator.get_phh_mass(klin, M0, M0, z) / Pk_klin)

    for iM1, M01 in enumerate(M1):
        for iM2, M02 in enumerate(M2):
            if iM2 < iM1:
                # Use symmetry to not double calculate
                beta_func[iM1, iM2, :] = beta_func[iM2, iM1, :]
            else:
                # Linear halo bias
                b1 = b01[iM1]
                b2 = b01[iM2]
                    
                # Halo-halo power spectrum
                Pk_hh = emulator.get_phh_mass(k, M01, M02, z)
                
                #rmax = max(rvir(emulator, M01), rvir(emulator, M02))
                #kmax = 2.0*np.pi/rmax
                    
                # Create beta_NL
                shot_noise = lambda x, a: a
                popt, popc = curve_fit(shot_noise, k[(k > 100) & (k < 200)], Pk_hh[(k > 100) & (k < 200)])
                Pk_hh = Pk_hh - np.ones_like(k) * shot_noise(k, *popt)
            
                beta_func[iM1, iM2, :] = Pk_hh / (b1 * b2 * Pk_lin) - 1.0
                
                Pk_hh0 = emulator.get_phh_mass(klin, M01, M02, z)
                Pk_hh0 = Pk_hh0 - np.ones_like(klin)*shot_noise(klin, *popt)
                db = Pk_hh0 / (b1 * b2 * Pk_klin) - 1.0
                
                lmin, lmax = hl_envelopes_idx(np.abs(beta_func[iM1, iM2, :]+1.0))
                beta_func_interp = interp1d(k[lmax], np.abs(beta_func[iM1, iM2, lmax]+1.0), kind='quadratic', bounds_error=False, fill_value='extrapolate')
                beta_func[iM1, iM2, :] = (beta_func_interp(k) - 1.0)# * low_k_truncation(k, klin)
                db = (beta_func_interp(klin) - 1.0)
                
        
                #beta_func[iM1, iM2, :] = ((beta_func[iM1, iM2, :] + 1.0) * high_k_truncation(k, 30.0)/(db + 1.0) - 1.0) * low_k_truncation(k, klin)
                #beta_func[iM1, iM2, :] = ((beta_func[iM1, iM2, :] + 1.0)/(db + 1.0) - 1.0) #* low_k_truncation(k, klin) * high_k_truncation(k, 30.0)#/(1.0+z))
                beta_func[iM1, iM2, :] = (beta_func[iM1, iM2, :] - db) * low_k_truncation(k, klin) * high_k_truncation(k, 3.0*kmax)

    return beta_func

def create_bnl_interpolation_function(emulator, interpolation, z, block):
    lenM = 5
    lenk = 1000
    zc = z.copy()

    Mmin, kmax = minimum_halo_mass(emulator)
    M_up = np.log10((10.0**14.0))
    #M_lo = np.log10((10.0**12.0))
    M_lo = np.log10(Mmin)

    M = np.logspace(M_lo, M_up, lenM)
    k = np.logspace(-3.0, np.log10(200), lenk)
    beta_nl_interp_i = np.empty(len(z), dtype=object)
    beta_func = compute_bnl_darkquest(0.01, np.log10(M), np.log10(M), k, emulator, block, kmax)
    beta_nl_interp_i = RegularGridInterpolator([np.log10(M), np.log10(M), np.log10(k)], beta_func, fill_value=None, bounds_error=False, method='nearest')
    """
    for i,zi in enumerate(zc):
        #M = np.logspace(M_lo, M_up - 3.0*np.log10(1+zi), lenM)
        #beta_func = compute_bnl_darkquest(zi, np.log10(M), np.log10(M), k, emulator, block, kmax)
        beta_nl_interp_i[i] = RegularGridInterpolator([np.log10(M), np.log10(M), np.log10(k)],
                                                      beta_func, fill_value=None, bounds_error=False, method='nearest')
    """
    return beta_nl_interp_i

