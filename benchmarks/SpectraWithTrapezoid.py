from onepower import Spectra
from scipy.integrate import trapezoid
import warnings

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