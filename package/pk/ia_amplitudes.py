from functools import cached_property
import numpy as np
from scipy.integrate import simpson
from astropy.io import fits

"""
A module for computing aligment amplitudes.
This module provides classes and functions to calculate various properties 
related to the alignment amplitudes.
"""

class AlignmentAmplitudes:
    """
    A class to compute the 
    """
    def __init__(
            self,
            ...
        ):
        
        ...
        
        
    def mean_l_l0_to_beta(self, xlum, pdf, l0, beta):
        return simpson(pdf * (xlum / l0) ** beta, xlum)

    def broken_powerlaw(self, xlum, pdf, gamma_2h_lum, l0, beta, beta_low):
        alignment_ampl = np.where(xlum > l0, gamma_2h_lum * (xlum / l0) ** beta,
                                gamma_2h_lum * (xlum / l0) ** beta_low)
        return simpson(pdf * alignment_ampl, xlum)
    
    def compute_luminosity_pdf(self, z_loglum_file, zmin, zmax, nz, nlbins):
        galfile = fits.open(z_loglum_file)[1].data
        z_gal = np.array(galfile['z'])
        loglum_gal = np.array(galfile['loglum'])
    
        z_bins = np.linspace(zmin, zmax, nz)
        dz = 0.5 * (z_bins[1] - z_bins[0])
        z_edges = np.append(z_bins - dz, z_bins[-1] + dz)
    
        bincen = np.zeros([nz, nlbins])
        pdf = np.zeros([nz, nlbins])
    
        for i in range(nz):
            mask_z = (z_gal >= z_edges[i]) & (z_gal < z_edges[i + 1])
            loglum_bin = loglum_gal[mask_z]
            if loglum_bin.size:
                lum = 10.0 ** loglum_bin
                pdf_tmp, _lum_bins = np.histogram(lum, bins=nlbins, density=True)
                _dbin = (_lum_bins[-1] - _lum_bins[0]) / (1.0 * nlbins)
                bincen[i] = _lum_bins[:-1] + 0.5 * _dbin
                pdf[i] = pdf_tmp
    
        return bincen, pdf
