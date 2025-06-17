from functools import cached_property
import numpy as np
from scipy.integrate import simpson
from astropy.io import fits

"""
A module for computing alignment amplitudes.
This module provides classes and functions to calculate various properties
related to the alignment amplitudes.
"""


class AlignmentAmplitudes:
    """
    Parameters:
    -----------
    z_vec : array-like
        Redshift vector.
    central_ia_depends_on : str
        Dependency for central intrinsic alignment. Options: 'constant', 'luminosity', 'halo_mass'.
    satellite_ia_depends_on : str
        Dependency for satellite intrinsic alignment. Options: 'constant', 'luminosity', 'halo_mass'.
    gamma_2h_amplitude : float
        Amplitude for 2-halo term.
    beta_cen : float
        Beta parameter for central galaxies.
    beta_two : float, optional
        Secondary beta parameter.
    gamma_1h_amplitude : float
        Amplitude for 1-halo term.
    gamma_1h_slope : float
        Slope for 1-halo term.
    beta_sat : float
        Beta parameter for satellite galaxies.
    mpivot_cen : float, optional
        Pivot mass for central galaxies.
    mpivot_sat : float, optional
        Pivot mass for satellite galaxies.
    lpivot_cen : float, optional
        Pivot luminosity for central galaxies.
    lpivot_sat : float, optional
        Pivot luminosity for satellite galaxies.
    z_loglum_file_centrals : str, optional
        File path for central galaxy luminosity data.
    z_loglum_file_satellites : str, optional
        File path for satellite galaxy luminosity data.
    """
    def __init__(self,
            z_vec = None,
            central_ia_depends_on = 'halo_mass',
            satellite_ia_depends_on = 'halo_mass',
            gamma_2h_amplitude = 5.33,
            beta_cen = 0.44,
            beta_two = None,
            gamma_1h_amplitude = 0.0015,
            gamma_1h_slope = -2.0,
            beta_sat = 0.44,
            mpivot_cen = 13.5,
            mpivot_sat = 13.5,
            lpivot_cen = None,
            lpivot_sat = None,
            z_loglum_file_centrals = None,
            z_loglum_file_satellites = None,
        ):
        self.z_vec = z_vec
        self.central_ia_depends_on = central_ia_depends_on
        self.satellite_ia_depends_on = satellite_ia_depends_on
        
        self.gamma_2h_amp = gamma_2h_amplitude
        self.beta_cen = beta_cen
        self.beta_two = beta_two
        self.gamma_1h_slope = gamma_1h_slope
        self.gamma_1h_amp = gamma_1h_amplitude
        self.beta_sat = beta_sat
        
        self.mpivot_cen = 10.0 ** mpivot_cen if mpivot_cen is not None else None
        self.mpivot_sat = 10.0 ** mpivot_sat if mpivot_sat is not None else None
        self.lpivot_cen = 10.0 ** lpivot_cen if lpivot_cen is not None else None
        self.lpivot_sat = 10.0 ** lpivot_sat if lpivot_sat is not None else None
        
        self.z_loglum_file_centrals = z_loglum_file_centrals
        self.z_loglum_file_satellites = z_loglum_file_satellites

        self._validate_ia_depends_on()
        self.lum_centrals, self.lum_pdf_z_centrals = self._initialize_luminosity_arrays('centrals')
        self.lum_satellites, self.lum_pdf_z_satellites = self._initialize_luminosity_arrays('satellites')
        
        self._process_centrals()
        self._process_satellites()
        
    def _validate_ia_depends_on(self):
        """
        Validate the central and satellite intrinsic alignment dependencies.

        Raises:
        -------
        ValueError : If an invalid option is provided for central_ia_depends_on or satellite_ia_depends_on.
        """
        valid_options = ['constant', 'luminosity', 'halo_mass']
        if self.central_ia_depends_on not in valid_options:
            raise ValueError(f'Choose one of the following options for central_IA_depends_on: {valid_options}')
        if self.satellite_ia_depends_on not in valid_options:
            raise ValueError(f'Choose one of the following options for satellite_IA_depends_on: {valid_options}')

    def _initialize_luminosity_arrays(self, galaxy_type):
        """
        Initialize luminosity arrays based on galaxy type.

        Parameters:
        -----------
        galaxy_type : str
            Type of galaxy, either 'centrals' or 'satellites'.

        Returns:
        --------
        tuple : (lum, lum_pdf_z)
            Luminosity and luminosity PDF arrays.
        """
        
        depends_on = self.central_ia_depends_on if galaxy_type == 'centrals' else self.satellite_ia_depends_on
        z_loglum_file = self.z_loglum_file_centrals if galaxy_type == 'centrals' else self.z_loglum_file_satellites
        if depends_on == 'luminosity':
            nlbins = 10000
            lum, lum_pdf_z = self.compute_luminosity_pdf(z_loglum_file, nlbins)
        else:
            nlbins = 100000
            lum = np.ones([self.z_vec.size, nlbins])
            lum_pdf_z = np.ones([self.z_vec.size, nlbins])

        return lum, lum_pdf_z

    def mean_l_l0_to_beta(xlum, pdf, l0, beta):
        """
        Compute the mean luminosity scaling.

        Parameters:
        -----------
        xlum : array-like
            Luminosity values.
        pdf : array-like
            Probability density function values.
        l0 : float
            Pivot luminosity.
        beta : float
            Beta parameter.

        Returns:
        --------
        float : Mean luminosity scaling.
        """
        return simpson(pdf * (xlum / l0)**beta, xlum)

    def broken_powerlaw(xlum, pdf, gamma_2h_lum, l0, beta, beta_low):
        """
        Compute the broken power law.

        Parameters:
        -----------
        xlum : array-like
            Luminosity values.
        pdf : array-like
            Probability density function values.
        gamma_2h_lum : float
            Amplitude for 2-halo term.
        l0 : float
            Pivot luminosity.
        beta : float
            Beta parameter.
        beta_low : float
            Secondary beta parameter.

        Returns:
        --------
        float : Integral of the broken power law.
        """
        alignment_ampl = np.where(xlum > l0, gamma_2h_lum * (xlum / l0)**beta, gamma_2h_lum * (xlum / l0)**beta_low)
        return simpson(pdf * alignment_ampl, xlum)

    def compute_luminosity_pdf(z_loglum_file, nlbins):
        """
        Compute the luminosity PDF.

        Parameters:
        -----------
        z_loglum_file : str
            File path for galaxy luminosity data.
        nlbins : int
            Number of bins for the histogram.

        Returns:
        --------
        tuple : (bincen, pdf)
            Bin centers and probability density function values.
        """
        galfile = fits.open(z_loglum_file)[1].data
        z_gal = np.array(galfile['z'])
        loglum_gal = np.array(galfile['loglum'])

        z_bins = self.z_vec
        dz = 0.5 * (z_bins[1] - z_bins[0])
        z_edges = np.append(z_bins - dz, z_bins[-1] + dz)

        bincen = np.zeros([self.z_vec.size, nlbins])
        pdf = np.zeros([self.z_vec.size, nlbins])

        for i in range(self.z_vec.size):
            mask_z = (z_gal >= z_edges[i]) & (z_gal < z_edges[i + 1])
            loglum_bin = loglum_gal[mask_z]
            if loglum_bin.size:
                lum = 10.0**loglum_bin
                pdf_tmp, _lum_bins = np.histogram(lum, bins=nlbins, density=True)
                _dbin = (_lum_bins[-1] - _lum_bins[0]) / (1.0 * nlbins)
                bincen[i] = _lum_bins[:-1] + 0.5 * _dbin
                pdf[i] = pdf_tmp

        return bincen, pdf
        
    def _process_centrals(self):
        """
        Process the central galaxies.
        """
        if self.central_ia_depends_on == 'constant':
            self.alignment_gi = self.gamma_2h_amp * np.ones_like(self.z_vec)
        elif self.central_ia_depends_on == 'luminosity':
            if self.lpivot_cen is None:
                raise ValueError('You have chosen central luminosity scaling without providing a pivot luminosity parameter. Include lpivot_cen.')
            if self.beta_two is not None:
                mean_lscaling = np.array([self.broken_powerlaw(self.lum_centrals[i], self.lum_pdf_z_centrals[i], self.gamma_2h_amp, self.lpivot_cen, self.beta_cen, self.beta_two) for i in range(self.z_vec.size)])
            else:
                mean_lscaling = self.gamma_2h_amp * self.mean_l_l0_to_beta(self.lum_centrals, self.lum_pdf_z_centrals, self.lpivot_cen, self.beta_cen)
            self.alignment_gi = mean_lscaling
        elif self.central_ia_depends_on == 'halo_mass':
            if self.mpivot_cen is None:
                raise ValueError('You have chosen central halo-mass scaling without providing a pivot mass parameter. Include mpivot_cen.')
            if self.beta_two is not None:
                raise ValueError('A double power law model for the halo mass dependence of centrals has not been implemented.')
            self.alignment_gi = self.gamma_2h_amp * np.ones_like(self.z_vec)
            
    def _process_satellites(self):
        """
        Process the satellite galaxies.
        """
        if self.satellite_ia_depends_on == 'constant':
            self.gamma_1h_amplitude = self.gamma_1h_amp * np.ones_like(self.z_vec)
        elif self.satellite_ia_depends_on == 'luminosity':
            if self.lpivot_sat is None:
                raise ValueError('You have chosen satellite luminosity scaling without providing a pivot luminosity parameter. Include lpivot_sat.')
            mean_lscaling = self.mean_l_l0_to_beta(self.lum_satellites, self.lum_pdf_z_satellites, self.lpivot_sat, self.beta_sat)
            self.gamma_1h_amplitude = self.gamma_1h_amp * mean_lscaling
        elif self.satellite_ia_depends_on == 'halo_mass':
            if self.mpivot_sat is None:
                raise ValueError('You have chosen satellite halo-mass scaling without providing a pivot mass parameter. Include mpivot_sat.')
            self.gamma_1h_amplitude = self.gamma_1h_amp * np.ones_like(self.z_vec)


