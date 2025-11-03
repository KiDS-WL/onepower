import numpy as np
import pickle
import os
from onepower import Spectra
from scipy.integrate import trapezoid

from params_benchmark import kwargs


class OnepowerBenchmark:
    """
    Benchmark class for testing integration accuracy in power spectrum and HOD calculations.
    """

    def __init__(self, kwargs):
        """Initialize benchmark."""
        self.results = {}
        self.params = kwargs

    def _get_power_spectrum(self, spectra, spectrum_type):
        spectrum_methods = {
            'mm': spectra.power_spectrum_mm,
            'gg': spectra.power_spectrum_gg,
            'gm': spectra.power_spectrum_gm}
        return spectrum_methods[spectrum_type]

    def _test_convergence(self, quantity, values, spectra_types=['mm']):
        """
        Method to test convergence with different resolutions.

        Parameters
        ----------
        quantity : str
            Type of quantity being tested ('mass', 'k', 'nobs').
        values : list
            List of quantity values to test.
        spectra_types : list
            Types to compute. Either ['mm', 'gg', 'gm'], ['smf'], or ['hod'].

        Returns
        -------
        dict
            Results for each value and spectra type.
        """
        convergence_results = {}

        for value in values:
            modified_params = self._modify_params(self.params.copy(), quantity, value)
            spectra = Spectra(**modified_params)

            value_results = {}

            if 'smf' in spectra_types:
                value_results['smf'] = self._extract_smf_results(
                    spectra, modified_params)
            elif 'hod' in spectra_types:
                value_results['hod'] = self._extract_hod_results(
                    spectra, modified_params)
            else:
                # Power spectrum testing
                for spectrum_type in spectra_types:
                    value_results[spectrum_type] = self._extract_power_spectrum_results(
                        spectra, spectrum_type)

                    if quantity == 'mass':
                        value_results[spectrum_type].update({
                            'mass': spectra.mass,
                            'dlog10m': modified_params['dlog10m']})
                    elif quantity == 'k':
                        value_results[spectrum_type]['dlnk'] = modified_params['dlnk']

            convergence_results[value] = value_results

        return convergence_results

    def _extract_power_spectrum_results(self, spectra, spectrum_type):
        """
        Function to extract power spectrum results from objects.

        Parameters
        ----------
        spectra : Spectra object
            The spectra object containing power spectra.
        spectrum_type : str
            Type of spectrum ('mm', 'gg', 'gm').

        Returns
        -------
        dict
            Dictionary with extracted results.
        """
        ps_result = self._get_power_spectrum(spectra, spectrum_type)
        result = {
            'pk_tot': ps_result.pk_tot,
            'pk_1h': ps_result.pk_1h,
            'pk_2h': ps_result.pk_2h,
            'k_vec': spectra.k_vec}

        return result

    def _extract_smf_results(self, spectra, modified_params):
        """
        Function to extract SMF results from observable object.

        Parameters
        ----------
        spectra : Spectra object
            The spectra object containing observable.
        modified_params : dict
            Modified parameters used.

        Returns
        -------
        dict
            Dictionary with extracted results.
        """
        obs = spectra.obs

        result = {
            'stellar_mass': obs.obs[0, 0, 0, :],
            'smf_total': obs.obs_func[0, 0, :],
            'smf_cen': obs.obs_func_cen[0, 0, :],
            'smf_sat': obs.obs_func_sat[0, 0, :]}

        result['mass'] = spectra.mass
        result['dlog10m'] = modified_params['dlog10m']

        return result

    def _extract_hod_results(self, spectra, modified_params):
        """
        Function to extract HOD results from spectra object.

        Parameters
        ----------
        spectra : Spectra object
            The spectra object containing HOD.
        modified_params : dict
            Modified parameters used.

        Returns
        -------
        dict
            Dictionary with extracted results.
        """
        hod = spectra.hod

        result = {
            'mass': spectra.mass,
            'hod_cen': hod.hod_cen,
            'hod_sat': hod.hod_sat}

        result['nobs'] = modified_params['hod_settings']['nobs']

        return result

    def _modify_params(self, params, quantity, value):
        """
        Modify parameters based on quantity type.

        Parameters
        ----------
        params : dict
            Original parameters.
        quantity : str
            Type of quantity ('mass', 'k', or 'nobs').
        value : float or int
            Value to set.

        Returns
        -------
        dict
            Modified parameters.
        """
        if quantity == 'mass':
            params['dlog10m'] = value
        elif quantity == 'k':
            params['dlnk'] = value
        elif quantity == 'nobs':
            params['hod_settings']['nobs'] = value

        return params

    def create_benchmark(self, precision_params=None, filename=None):
        """
        Create high-resolution benchmark.

        Parameters
        ----------
        precision_params : dict
            Dictionary of precision parameters to override (e.g., {'dlnk': 0.000005, 'dlog10m': 0.005}).
            Must be provided.
        filename : str, optional
            Filename to save the benchmark to. If None, will auto-generate based on precision params.
        """
        if precision_params is None:
            raise ValueError("precision_params must be provided. Example: {'dlnk': 0.000005}")

        if filename is None:
            param_str = "_".join([f"{k}_{v}" for k, v in precision_params.items()])
            filename = f"high_resolution_benchmark_{param_str}.pkl"

        high_res_params = {**self.params, **precision_params}
        spectra = Spectra(**high_res_params)

        results_to_save = {
            'mm': spectra.power_spectrum_mm,
            'gg': spectra.power_spectrum_gg,
            'gm': spectra.power_spectrum_gm}

        self.save_to_file(high_res_params, results_to_save, spectra, filename)

        return filename

    def test_mass(self, dlog10m_values=[0.05, 0.01], spectra_types=['mm']):
        """Test power spectrum convergence with different mass array resolutions."""
        return self._test_convergence('mass', dlog10m_values, spectra_types)

    def test_k(self, dlnk_values=[0.001, 0.0001], spectra_types=['mm']):
        """Test power spectrum convergence with different k resolutions."""
        return self._test_convergence('k', dlnk_values, spectra_types)

    def test_mass_smf(self, dlog10m_values=[0.05, 0.005]):
        """Test SMF convergence with different halo mass resolutions."""
        return self._test_convergence('mass', dlog10m_values, spectra_types=['smf'])

    def test_nobs_hod(self, nobs_values=[300, 500]):
        """Test HOD convergence with different nobs resolutions."""
        return self._test_convergence('nobs', nobs_values, spectra_types=['hod'])

    def save_to_file(self, parameters, spectra_results, spectra, filename):
        """
        Save results to file.

        Parameters
        ----------
        parameters : dict
            Parameters used for computation.
        spectra_results : dict
            Dictionary with keys 'mm', 'gg', 'gm' containing results.
        spectra : object
            Spectra object containing k_vec, z_vec, mass.
        filename : str
            Filename to save to.
        """
        results = {
            'params': parameters,
            'spectra_results': {},
            'k_vec': spectra.k_vec,
            'z_vec': spectra.z_vec,
            'mass': spectra.mass}

        for spectrum_type, item in spectra_results.items():
            results['spectra_results'][spectrum_type] = {
                'pk_tot': item.pk_tot,
                'pk_1h': item.pk_1h,
                'pk_2h': item.pk_2h}

        with open(filename, 'wb') as f:
            pickle.dump(results, f)

        return


class SpectraWithTrapezoid(Spectra):
    """Spectra class that uses trapezoid integration instead of simpson."""

    def _replace_integration_method(self, integrand, x):
        """Helper method to use trapezoid instead of simpson."""
        return trapezoid(integrand, x, axis=-1)

    def compute_1h_term(self, profile_u, profile_v, mass, dndlnm):
        """Override to use trapezoid instead of simpson."""
        integrand = profile_u * profile_v * dndlnm / mass
        return self._replace_integration_method(integrand, mass)

    def compute_Im_term(self, mass, u_dm, b_dm, dndlnm, mean_density0):
        """Override to use trapezoid instead of simpson."""
        integrand_m = b_dm * dndlnm * u_dm * (1. / mean_density0)
        return self._replace_integration_method(integrand_m, mass)

    def compute_A_term(self, mass, b_dm, dndlnm, mean_density0):
        """Override to use trapezoid instead of simpson."""
        integrand_m1 = b_dm * dndlnm * (1.0 / mean_density0)
        A = 1.0 - self._replace_integration_method(integrand_m1, mass)
        if (A < 0.0).any():
            warnings.warn(
                'Warning: Mass function/bias correction is negative!',
                RuntimeWarning)
        return A

    def compute_Ig_term(self, profile, mass, dndlnm, b_m):
        """Override to use trapezoid instead of simpson."""
        integrand = profile * b_m * dndlnm / mass
        return self._replace_integration_method(integrand, mass)
