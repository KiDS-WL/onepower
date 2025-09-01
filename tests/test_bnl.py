import pytest
import numpy as np
from onepower import NonLinearBias


@pytest.fixture
def setup_data():
    mass = np.logspace(12, 15, 5)
    z_vec = np.linspace(0, 1, 5)
    k_vec = np.logspace(-3, 3, 100)
    return mass, z_vec, k_vec


@pytest.fixture
def bias_instance(setup_data):
    mass, z_vec, k_vec = setup_data
    return NonLinearBias(mass=mass, z_vec=z_vec, k_vec=k_vec)


def test_initialization(bias_instance, setup_data):
    mass, z_vec, k_vec = setup_data
    bias = bias_instance

    assert np.allclose(bias.mass, mass)
    assert np.allclose(bias.z_vec, z_vec)
    assert np.allclose(bias.k_vec, k_vec)
    assert bias.h0 == 0.7
    assert bias.sigma_8 == 0.8
    assert bias.omega_b == 0.05
    assert bias.omega_c == 0.25
    assert bias.omega_lambda == 0.7
    assert bias.n_s == 1.0
    assert bias.w0 == -1.0
    assert bias.z_dep is False


def test_as(setup_data):
    mass, z_vec, k_vec = setup_data
    bias = NonLinearBias(sigma_8=None, A_s=2.1e-9, mass=mass, z_vec=z_vec, k_vec=k_vec)
    assert bias.sigma_8 is None
    assert bias.A_s == 2.1e-9

    bias_no_as = NonLinearBias(
        sigma_8=None, A_s=None, mass=mass, z_vec=z_vec, k_vec=k_vec
    )
    with pytest.raises(ValueError):
        _ = bias_no_as.emulator


def test_ombh2(bias_instance):
    bias = bias_instance
    assert bias.ombh2 == bias.omega_b * bias.h0**2


def test_omch2(bias_instance):
    bias = bias_instance
    assert bias.omch2 == bias.omega_c * bias.h0**2


def test_emulator_initialization(bias_instance):
    bias = bias_instance
    emulator = bias.emulator
    assert hasattr(emulator, 'set_cosmology')
    assert hasattr(emulator, 'get_sigma8')


def test_bnl(bias_instance):
    bias = bias_instance
    bnl = bias.bnl
    if bias.z_dep:
        assert bnl.shape == (
            len(bias.z_vec),
            len(bias.mass),
            len(bias.mass),
            len(bias.k_vec),
        )
    else:
        assert bnl.shape == (1, len(bias.k_vec), len(bias.mass), len(bias.mass))


def test_low_k_truncation(bias_instance, setup_data):
    _, _, k_vec = setup_data
    bias = bias_instance
    k_trunc = 0.1
    truncation = bias.low_k_truncation(k_vec, k_trunc)
    assert truncation.shape == k_vec.shape


def test_high_k_truncation(bias_instance, setup_data):
    _, _, k_vec = setup_data
    bias = bias_instance
    k_trunc = 10.0
    truncation = bias.high_k_truncation(k_vec, k_trunc)
    assert truncation.shape == k_vec.shape


def test_minimum_halo_mass(bias_instance):
    bias = bias_instance
    mmin, kmin = bias.minimum_halo_mass
    assert isinstance(mmin, float)
    assert isinstance(kmin, float)


def test_rvir(bias_instance, setup_data):
    mass, _, _ = setup_data
    bias = bias_instance
    rvir = bias.rvir(mass)
    assert rvir.shape == mass.shape


def test_hl_envelopes_idx(bias_instance):
    bias = bias_instance
    rng = np.random.default_rng(seed=42)
    data = rng.random(100)
    lmin, lmax = bias.hl_envelopes_idx(data)
    assert len(lmin) > 0
    assert len(lmax) > 0


def test_create_bnl_interpolation_function(bias_instance, setup_data):
    mass, z_vec, _ = setup_data
    bias = bias_instance
    interpolation_function = bias.create_bnl_interpolation_function
    assert callable(interpolation_function)
    assert bias.bnl.shape == (1, len(bias.k_vec), len(mass), len(mass))

    bias_zdep = bias.clone()
    bias_zdep.update(z_dep=True)
    interpolation_function = bias_zdep.create_bnl_interpolation_function
    assert len(interpolation_function) == len(z_vec)
    assert bias_zdep.bnl.shape == (len(z_vec), len(bias.k_vec), len(mass), len(mass))


def test_out_of_range(bias_instance):
    bias = bias_instance
    cparam_in = np.array([0.025, 0.15, 0.9, 5.0, 1.5, -0.5])
    cparam_out = bias.test_cosmo(cparam_in)
    assert cparam_out.shape == (1, 6)
    assert np.allclose(
        cparam_out, np.array([0.0233625, 0.13178, 0.82128, 3.7128, 1.012725, -0.8])
    )

    cparam_in = np.array([0.02, 0.10, 0.5, 2.0, 0.9, -1.5])
    cparam_out = bias.test_cosmo(cparam_in)
    assert cparam_out.shape == (1, 6)
    assert np.allclose(
        cparam_out, np.array([0.0211375, 0.10782, 0.54752, 2.4752, 0.916275, -1.2])
    )
