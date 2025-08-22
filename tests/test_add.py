import pytest
import numpy as np
from onepower import PowerSpectrumResult, Spectra, UpsampledSpectra


@pytest.fixture
def setup_data():
    z = np.linspace(0, 1, 5)
    k = np.logspace(-2, 1, 5)
    fraction_z = np.linspace(0, 1, 5)
    fraction = np.linspace(0.1, 0.9, 5)
    model_1_params = {'omega_c': 0.2, 'sigma_8': 0.8}
    model_2_params = {'omega_c': 0.3, 'sigma_8': 0.8}
    return z, k, fraction_z, fraction, model_1_params, model_2_params


@pytest.fixture
def spectra_instance(setup_data):
    z, k, fraction_z, fraction, model_1_params, model_2_params = setup_data
    return UpsampledSpectra(
        z=z,
        k=k,
        fraction_z=fraction_z,
        fraction=fraction,
        model_1_params=model_1_params,
        model_2_params=model_2_params,
    )


def test_initialization(spectra_instance, setup_data):
    z, k, fraction_z, fraction, model_1_params, model_2_params = setup_data
    spectra = spectra_instance

    assert np.allclose(spectra.z, z)
    assert np.allclose(spectra.k, k)
    assert np.allclose(spectra.fraction_z, fraction_z)
    assert np.allclose(spectra.fraction, fraction)
    assert spectra._model_1_params == model_1_params
    assert spectra._model_2_params == model_2_params


def test_frac_1_and_frac_2(spectra_instance, setup_data):
    z, k, fraction_z, _, model_1_params, model_2_params = setup_data

    spectra = spectra_instance
    assert np.allclose(spectra.frac_1, spectra.fraction)
    assert np.allclose(spectra.frac_2, 1 - spectra.fraction)

    spectra_no_fraction = UpsampledSpectra(
        z=z,
        k=k,
        fraction_z=fraction_z,
        fraction=None,
        model_1_params=model_1_params,
        model_2_params=model_2_params,
    )
    assert np.allclose(spectra_no_fraction.frac_1, np.ones_like(z))
    assert np.allclose(spectra_no_fraction.frac_2, np.zeros_like(z))


def test_power_1_and_power_2(spectra_instance, setup_data):
    z, k, fraction_z, fraction, model_1_params, _ = setup_data

    spectra = spectra_instance
    assert isinstance(spectra.power_1, Spectra)
    assert isinstance(spectra.power_2, Spectra)

    spectra_with_existing_spectra = UpsampledSpectra(
        z=z,
        k=k,
        fraction_z=fraction_z,
        fraction=fraction,
        model=Spectra(),
        model_1_params=model_1_params,
        model_2_params=None,
    )
    assert isinstance(spectra_with_existing_spectra.power_1, Spectra)
    assert spectra_with_existing_spectra.power_2 is None


def test_results_method(spectra_instance):
    spectra = spectra_instance
    requested_spectra = ['mm']
    requested_components = ['tot']
    spectra.results(requested_spectra, requested_components)
    assert hasattr(spectra, 'power_spectrum_mm')
    assert isinstance(spectra.power_spectrum_mm, PowerSpectrumResult)


def test_add_spectra(spectra_instance):
    spectra = spectra_instance
    len_z, len_k = len(spectra.z), len(spectra.k)
    rng = np.random.default_rng(seed=42)
    pk_1 = rng.random((len_z, len_k))
    pk_2 = rng.random((len_z, len_k))

    added_power_mm = spectra.add_spectra(pk_1, pk_2, 'mm')
    assert np.allclose(added_power_mm, pk_1)

    added_power_gm = spectra.add_spectra(pk_1, pk_2, 'gm')
    expected_gm = (
        spectra.frac_1[:, np.newaxis] * pk_1
        + (1.0 - spectra.frac_1[:, np.newaxis]) * pk_2
    )
    assert np.allclose(added_power_gm, expected_gm)

    added_power_other = spectra.add_spectra(pk_1, pk_2, 'other')
    expected_other = (
        spectra.frac_1[:, np.newaxis] ** 2.0 * pk_1
        + (1.0 - spectra.frac_1[:, np.newaxis]) ** 2.0 * pk_2
    )
    assert np.allclose(added_power_other, expected_other)
