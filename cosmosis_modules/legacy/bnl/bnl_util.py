# Library of the power spectrum module
import numexpr as ne
import numpy as np
import warnings
from scipy.integrate import quad, simpson, trapezoid
from scipy.interpolate import RegularGridInterpolator, UnivariateSpline, interp1d
from scipy.ndimage import gaussian_filter1d

# from scipy.special import erf
from scipy.optimize import curve_fit


def low_k_truncation(k_vec, k_trunc):
    """
    Beta_nl low-k truncation
    """
    return 1.0 / (1.0 + np.exp(-(10.0 * (np.log10(k_vec) - np.log10(k_trunc)))))

def high_k_truncation(k_vec, k_trunc):
    """
    Beta_nl high-k truncation
    """
    return 1.0 / (1.0 + np.exp(10.0 * (np.log10(k_vec) - np.log10(k_trunc))))

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
    M_up = np.log10(10.0**14.0)
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
