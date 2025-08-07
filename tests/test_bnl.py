import pytest
import numpy as np
from onepower import NonLinearBias

# Just some base tests for now, no specific scientific calculations tested yet!

@pytest.fixture
def setup_data():
    mass = np.logspace(12, 15, 5)
    z_vec = np.linspace(0, 1, 5)
    k_vec = np.logspace(-3, 3, 100)
    return mass, z_vec, k_vec

def test_initialization(setup_data):
    mass, z_vec, k_vec = setup_data
    bias = NonLinearBias(mass=mass, z_vec=z_vec, k_vec=k_vec)
    assert bias.mass.all() == mass.all()
    assert bias.z_vec.all() == z_vec.all()
    assert bias.k_vec.all() == k_vec.all()
    assert bias.h0 == 0.7
    assert bias.sigma_8 == 0.8
    assert bias.omega_b == 0.05
    assert bias.omega_c == 0.25
    assert bias.omega_lambda == 0.7
    assert bias.n_s == 1.0
    assert bias.w0 == -1.0
    assert bias.z_dep == False

def test_ombh2(setup_data):
    mass, z_vec, k_vec = setup_data
    bias = NonLinearBias(mass=mass, z_vec=z_vec, k_vec=k_vec)
    assert bias.ombh2 == bias.omega_b * bias.h0**2

def test_omch2(setup_data):
    mass, z_vec, k_vec = setup_data
    bias = NonLinearBias(mass=mass, z_vec=z_vec, k_vec=k_vec)
    assert bias.omch2 == bias.omega_c * bias.h0**2

def test_emulator_initialization(setup_data):
    mass, z_vec, k_vec = setup_data
    bias = NonLinearBias(mass=mass, z_vec=z_vec, k_vec=k_vec)
    emulator = bias.emulator
    assert hasattr(emulator, 'set_cosmology')
    assert hasattr(emulator, 'get_sigma8')

def test_bnl(setup_data):
    mass, z_vec, k_vec = setup_data
    bias = NonLinearBias(mass=mass, z_vec=z_vec, k_vec=k_vec)
    bnl = bias.bnl
    if bias.z_dep:
        assert bnl.shape == (len(bias.z_vec), len(mass), len(mass), len(k_vec))
    else:
        assert bnl.shape == (1, len(mass), len(mass), len(k_vec))

def test_low_k_truncation(setup_data):
    mass, z_vec, k_vec = setup_data
    bias = NonLinearBias(mass=mass, z_vec=z_vec, k_vec=k_vec)
    k_trunc = 0.1
    truncation = bias.low_k_truncation(k_vec, k_trunc)
    assert truncation.shape == k_vec.shape

def test_high_k_truncation(setup_data):
    mass, z_vec, k_vec = setup_data
    bias = NonLinearBias(mass=mass, z_vec=z_vec, k_vec=k_vec)
    k_trunc = 10.0
    truncation = bias.high_k_truncation(k_vec, k_trunc)
    assert truncation.shape == k_vec.shape

def test_minimum_halo_mass(setup_data):
    mass, z_vec, k_vec = setup_data
    bias = NonLinearBias(mass=mass, z_vec=z_vec, k_vec=k_vec)
    mmin, kmin = bias.minimum_halo_mass
    assert isinstance(mmin, float)
    assert isinstance(kmin, float)

def test_rvir(setup_data):
    mass, z_vec, k_vec = setup_data
    bias = NonLinearBias(mass=mass, z_vec=z_vec, k_vec=k_vec)
    rvir = bias.rvir(mass)
    assert rvir.shape == mass.shape

def test_hl_envelopes_idx(setup_data):
    mass, z_vec, k_vec = setup_data
    bias = NonLinearBias(mass=mass, z_vec=z_vec, k_vec=k_vec)
    data = np.random.rand(100)
    lmin, lmax = bias.hl_envelopes_idx(data)
    assert len(lmin) > 0
    assert len(lmax) > 0

def test_compute_bnl_darkquest(setup_data):
    mass, z_vec, k_vec = setup_data
    bias = NonLinearBias(mass=mass, z_vec=z_vec, k_vec=k_vec)
    z = 0.5
    log10M1 = np.log10(mass)
    log10M2 = np.log10(mass)
    kmax = 10.0
    beta_func = bias.compute_bnl_darkquest(z, log10M1, log10M2, k_vec, kmax)
    assert beta_func.shape == (len(mass), len(mass), len(k_vec))

def test_create_bnl_interpolation_function(setup_data):
    mass, z_vec, k_vec = setup_data
    bias = NonLinearBias(mass=mass, z_vec=z_vec, k_vec=k_vec)
    interpolation_function = bias.create_bnl_interpolation_function
    if bias.z_dep:
        assert len(interpolation_function) == len(z_vec)
    else:
        assert hasattr(interpolation_function, '__call__')

def test_test_cosmo(setup_data):
    mass, z_vec, k_vec = setup_data
    bias = NonLinearBias(mass=mass, z_vec=z_vec, k_vec=k_vec)
    cparam_in = np.array([0.022, 0.12, 0.7, 3.0, 0.96, -1.0])
    cparam_out = bias.test_cosmo(cparam_in)
    assert cparam_out.shape == (1, 6)
