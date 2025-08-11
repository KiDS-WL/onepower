import pytest

import numpy as np
from unittest.mock import MagicMock

from onepower import PowerSpectrumResult, Spectra, UpsampledSpectra

# Just some base tests for now, no specific scientific calculations tested yet!


@pytest.fixture
def setup_data():
    z = np.linspace(0, 1, 5)
    k = np.logspace(-2, 1, 5)
    fraction_z = np.linspace(0, 1, 5)
    fraction = np.linspace(0.1, 0.9, 5)
    model_1_params = {'omega_m': 0.2, 'sigma_8': 0.8}
    model_2_params = {'omega_m': 0.3, 'sigma_8': 0.8}
    return z, k, fraction_z, fraction, model_1_params, model_2_params


def test_initialization(setup_data):
    z, k, fraction_z, fraction, model_1_params, model_2_params = setup_data
    spectra = UpsampledSpectra(
        z=z,
        k=k,
        fraction_z=fraction_z,
        fraction=fraction,
        model_1_params=model_1_params,
        model_2_params=model_2_params,
    )
    assert spectra.z.all() == z.all()
    assert spectra.k.all() == k.all()
    assert spectra.fraction_z.all() == fraction_z.all()
    assert spectra.fraction.all() == fraction.all()
    assert spectra._model_1_params == model_1_params
    assert spectra._model_2_params == model_2_params


def test_frac_1_and_frac_2(setup_data):
    z, k, fraction_z, fraction, model_1_params, model_2_params = setup_data
    spectra = UpsampledSpectra(
        z=z,
        k=k,
        fraction_z=fraction_z,
        fraction=fraction,
        model_1_params=model_1_params,
        model_2_params=model_2_params,
    )
    assert np.allclose(spectra.frac_1, fraction)
    assert np.allclose(spectra.frac_2, 1 - fraction)


def test_power_1_and_power_2(setup_data):
    z, k, fraction_z, fraction, model_1_params, model_2_params = setup_data
    spectra = UpsampledSpectra(
        z=z,
        k=k,
        fraction_z=fraction_z,
        fraction=fraction,
        model_1_params=model_1_params,
        model_2_params=model_2_params,
    )

    power_1 = spectra.power_1
    assert isinstance(power_1, Spectra)

    power_2 = spectra.power_2
    assert isinstance(power_2, Spectra)


def test_results_method(setup_data):
    z, k, fraction_z, fraction, model_1_params, model_2_params = setup_data
    spectra = UpsampledSpectra(
        z=z,
        k=k,
        fraction_z=fraction_z,
        fraction=fraction,
        model_1_params=model_1_params,
        model_2_params=model_2_params,
    )

    requested_spectra = ['mm']
    requested_components = ['tot']

    spectra.results(requested_spectra, requested_components)

    assert hasattr(spectra, 'power_spectrum_mm')
    assert isinstance(spectra.power_spectrum_mm, PowerSpectrumResult)


def test_add_spectra(setup_data):
    z, k, fraction_z, fraction, model_1_params, model_2_params = setup_data
    spectra = UpsampledSpectra(
        z=z,
        k=k,
        fraction_z=fraction_z,
        fraction=fraction,
        model_1_params=model_1_params,
        model_2_params=model_2_params,
    )

    pk_1 = np.random.rand(len(z), len(k))
    pk_2 = np.random.rand(len(z), len(k))

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
