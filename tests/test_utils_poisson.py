import pytest

import numpy as np

from onepower.utils.poisson import Poisson, constant, power_law

# Just some base tests for now, no specific scientific calculations tested yet!

@pytest.fixture
def setup_data():
    mass = np.logspace(12, 15, 100)
    return mass

def test_constant_poisson_initialization(setup_data):
    mass = setup_data
    poisson = constant(mass=mass)
    assert poisson.mass.all() == mass.all()
    assert poisson.params['poisson'] == 1.0

def test_constant_poisson_func(setup_data):
    mass = setup_data
    poisson = constant(mass=mass)
    poisson_func = poisson.poisson_func
    assert poisson_func.shape == mass.shape
    assert np.all(poisson_func == poisson.params['poisson'])

def test_power_law_poisson_initialization(setup_data):
    mass = setup_data
    poisson = power_law(mass=mass, pivot=13.0, slope=-1.0)
    assert poisson.mass.all() == mass.all()
    assert poisson.params['poisson'] == 1.0
    assert poisson.params['pivot'] == 13.0
    assert poisson.params['slope'] == -1.0

def test_power_law_poisson_func(setup_data):
    mass = setup_data
    poisson = power_law(mass=mass, pivot=13.0, slope=-1.0)
    poisson_func = poisson.poisson_func
    assert poisson_func.shape == mass.shape

def test_power_law_poisson_func_value(setup_data):
    mass = setup_data
    poisson = power_law(mass=mass, pivot=13.0, slope=-1.0)
    poisson_func = poisson.poisson_func
    expected_poisson_func = poisson.params['poisson'] * (mass / (10.0**poisson.params['pivot']))**poisson.params['slope']
    assert np.allclose(poisson_func, expected_poisson_func)

def test_power_law_poisson_initialization_error(setup_data):
    mass = setup_data
    with pytest.raises(ValueError):
        poisson = power_law(mass=mass, pivot=None, slope=-1.0)
        poisson_func = poisson.poisson_func
    with pytest.raises(ValueError):
        poisson = power_law(mass=mass, pivot=13.0, slope=None)
        poisson_func = poisson.poisson_func
