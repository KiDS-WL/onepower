from functools import cached_property
import numpy as np
from dark_emulator import darkemu
from collections import OrderedDict
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.optimize import curve_fit


class NonLinearBias:
    """
    A class to compute the 
    """
    def __init__(self,
            mass = None,
            z_vec = None,
            k_vec = None,
            h0 = 0.7,
            sigma_8 = 0.8,
            omega_b = 0.05,
            omega_c = 0.25,
            omega_lambda = 0.7,
            n_s = 1.0,
            w0 = -1.0
        ):
        self.mass = mass
        self.z_vec = z_vec
        self.k_vec = k_vec
        self.sigma_8 = sigma_8
        
        self.ombh2 = omega_b * h0**2.0
        self.omch2 = omega_c * h0**2.0
        self.omega_lambda = omega_lambda
        self.n_s = n_s
        self.w0 = w0
    
    @cached_property
    def emulator(self):
        A_s_init = 2.1e-9
        emu = darkemu.base_class()
        
        cparam = self.test_cosmo(np.array([self.ombh2, self.omch2, self.omega_lambda, np.log(A_s_init*10.0**10.0), self.n_s, self.w0]))
        emu.set_cosmology(cparam)

        sigma_8_init = emu.get_sigma8()
        scaling = (self.sigma_8/sigma_8_init)**2
        A_s = A_s_init * scaling
    
        cparam = self.test_cosmo(np.array([self.ombh2, self.omch2, self.omega_lambda, np.log(A_s*10.0**10.0), self.n_s, self.w0]))
        emu.set_cosmology(cparam)
        return emu
    
    @cached_property
    def bnl(self):
        beta_interp_tmp = self.create_bnl_interpolation_function
    
        indices = np.vstack(np.meshgrid(np.arange(self.mass.size), np.arange(self.mass.size), np.arange(self.k_vec.size), copy=False)).reshape(3, -1).T
        values = np.vstack(np.meshgrid(np.log10(self.mass), np.log10(self.mass), np.log10(self.k_vec), copy=False)).reshape(3, -1).T
    
        #beta_interp = np.zeros((self.z_vec.size, self.mass.size, self.mass.size, self.k_vec.size))
        # for i,zi in enumerate(self.z_vec):
        #    beta_interp[i,indices[:,0], indices[:,1], indices[:,2]] = beta_interp_tmp[i](values)
                
        beta_interp = np.zeros((self.mass.size, self.mass.size, self.k_vec.size))
        beta_interp[indices[:, 0], indices[:, 1], indices[:, 2]] = beta_interp_tmp(values)
    
        return beta_interp[np.newaxis, :, :, :]

    def low_k_truncation(self, k, k_trunc):
        """
        Beta_nl low-k truncation
        """
        return 1.0 / (1.0 + np.exp(-(10.0 * (np.log10(k) - np.log10(k_trunc)))))
    
    def high_k_truncation(self, k, k_trunc):
        """
        Beta_nl high-k truncation
        """
        return 1.0 / (1.0 + np.exp((10.0 * (np.log10(k) - np.log10(k_trunc)))))
    
    @property
    def minimum_halo_mass(self):
        """
        Minimum halo mass for the set of cosmological parameters [Msun/h]
        """
        np_min = 200.0 # Minimum number of halo particles
        npart = 2048.0 # Cube root of number of simulation particles
        Lbox_HR = 1000.0 # Box size for high-resolution simulations [Mpc/h]
        Lbox_LR = 2000.0 # Box size for low-resolution simulations [Mpc/h]
    
        Om_m = self.emulator.cosmo.get_Omega0()
        rhom = 2.77536627e11 * Om_m
    
        Mbox_HR = rhom * Lbox_HR**3.0
        mmin = Mbox_HR * np_min / npart**3.0
    
        vmin = Lbox_HR**3.0 * np_min / npart**3.0
        rmin = ((3.0 * vmin) / (4.0 * np.pi))**(1.0 / 3.0)
    
        return mmin, 2.0 * np.pi / rmin
    
    def rvir(self, mass):
        Om_m = self.emulator.cosmo.get_Omega0()
        rhom = 2.77536627e11 * Om_m
        return ((3.0 * mass) / (4.0 * np.pi * 200 * rhom))**(1.0 / 3.0)
    
    def hl_envelopes_idx(self, data, dmin=1, dmax=1):
        """
        Extract high and low envelope indices from a 1D data signal.
    
        Parameters:
        - data (1d-array): Data signal from which to extract high and low envelopes.
        - dmin (int): Size of chunks for local minima, use this if the size of the input signal is too big.
        - dmax (int): Size of chunks for local maxima, use this if the size of the input signal is too big.
    
        Returns:
        - lmin, lmax (tuple of arrays): Indices of high and low envelopes of the input signal.
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
    
    def compute_bnl_darkquest(self, z, log10M1, log10M2, k, kmax):
        M1 = 10.0**log10M1
        M2 = 10.0**log10M2
    
        # Large 'linear' scale for linear halo bias [h/Mpc]
        klin = np.array([0.05])
    
        # Calculate beta_NL by looping over mass arrays
        beta_func = np.zeros((len(M1), len(M2), len(k)))
    
        # Linear power
        Pk_lin = self.emulator.get_pklin_from_z(k, z)
        Pk_klin = self.emulator.get_pklin_from_z(klin, z)
    
        # Calculate b01 for all M1
        b01 = np.zeros(len(M1))
        #b02 = np.zeros(len(M2))
        for iM, M0 in enumerate(M1):
            b01[iM] = np.sqrt(self.emulator.get_phh_mass(klin, M0, M0, z) / Pk_klin)
    
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
                    Pk_hh = self.emulator.get_phh_mass(k, M01, M02, z)
                    
                    #rmax = max(self.rvir(M01), self.rvir(M02))
                    #kmax = 2.0*np.pi/rmax
                        
                    # Create beta_NL
                    shot_noise = lambda x, a: a
                    popt, popc = curve_fit(shot_noise, k[(k > 100) & (k < 200)], Pk_hh[(k > 100) & (k < 200)])
                    Pk_hh = Pk_hh - np.ones_like(k) * shot_noise(k, *popt)
                
                    beta_func[iM1, iM2, :] = Pk_hh / (b1 * b2 * Pk_lin) - 1.0
                    
                    Pk_hh0 = self.emulator.get_phh_mass(klin, M01, M02, z)
                    Pk_hh0 = Pk_hh0 - np.ones_like(klin)*shot_noise(klin, *popt)
                    db = Pk_hh0 / (b1 * b2 * Pk_klin) - 1.0
                    
                    lmin, lmax = self.hl_envelopes_idx(np.abs(beta_func[iM1, iM2, :]+1.0))
                    beta_func_interp = interp1d(k[lmax], np.abs(beta_func[iM1, iM2, lmax]+1.0), kind='quadratic', bounds_error=False, fill_value='extrapolate')
                    beta_func[iM1, iM2, :] = (beta_func_interp(k) - 1.0)# * low_k_truncation(k, klin)
                    db = (beta_func_interp(klin) - 1.0)
                    
            
                    #beta_func[iM1, iM2, :] = ((beta_func[iM1, iM2, :] + 1.0) * high_k_truncation(k, 30.0)/(db + 1.0) - 1.0) * low_k_truncation(k, klin)
                    #beta_func[iM1, iM2, :] = ((beta_func[iM1, iM2, :] + 1.0)/(db + 1.0) - 1.0) #* low_k_truncation(k, klin) * high_k_truncation(k, 30.0)#/(1.0+z))
                    beta_func[iM1, iM2, :] = (beta_func[iM1, iM2, :] - db) * self.low_k_truncation(k, klin) * self.high_k_truncation(k, 3.0*kmax)
    
        return beta_func
    
    @cached_property
    def create_bnl_interpolation_function(self):
        lenM = 5
        lenk = 1000
        zc = self.z_vec.copy()
    
        Mmin, kmax = self.minimum_halo_mass
        M_up = np.log10((10.0**14.0))
        #M_lo = np.log10((10.0**12.0))
        M_lo = np.log10(Mmin)
    
        M = np.logspace(M_lo, M_up, lenM)
        k = np.logspace(-3.0, np.log10(200), lenk)
        
        beta_func = self.compute_bnl_darkquest(0.01, np.log10(M), np.log10(M), k, kmax)
        beta_nl_interp_i = RegularGridInterpolator([np.log10(M), np.log10(M), np.log10(k)], beta_func, fill_value=None, bounds_error=False, method='nearest')
        """
        beta_nl_interp_i = np.empty(len(self.z_vec), dtype=object)
        for i,zi in enumerate(zc):
            #M = np.logspace(M_lo, M_up - 3.0*np.log10(1+zi), lenM)
            #beta_func = self.compute_bnl_darkquest(zi, np.log10(M), np.log10(M), k, kmax)
            beta_nl_interp_i[i] = RegularGridInterpolator([np.log10(M), np.log10(M), np.log10(k)],
                                                        beta_func, fill_value=None, bounds_error=False, method='nearest')
        """
        return beta_nl_interp_i

    def test_cosmo(self, cparam_in):
        """Returns the edge values for DarkQuest emulator if the values are outside the emulator range."""
        cparam_range = OrderedDict([
            ['omegab', [0.0211375, 0.0233625]],
            ['omegac', [0.10782, 0.13178]],
            ['Omagede', [0.54752, 0.82128]],
            ['ln(10^10As)', [2.4752, 3.7128]],
            ['ns', [0.916275, 1.012725]],
            ['w', [-1.2, -0.8]]
        ])
    
        cparam_in = cparam_in.reshape(1, 6)
        cparam_out = np.copy(cparam_in)
    
        for i, (key, edges) in enumerate(cparam_range.items()):
            if cparam_in[0, i] < edges[0]:
                cparam_out[0, i] = edges[0]
            if cparam_in[0, i] > edges[1]:
                cparam_out[0, i] = edges[1]
    
        return cparam_out
