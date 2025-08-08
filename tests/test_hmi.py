import pytest

import numpy as np

from onepower import CosmologyBase, HaloModelIngredients

# Just some base tests for now, no specific scientific calculations tested yet!

@pytest.fixture
def setup_data():
    k_vec = np.logspace(-4, 4, 100)
    z_vec = np.linspace(0.0, 3.0, 15)
    return k_vec, z_vec

def test_cosmology_base_initialization(setup_data):
    k_vec, z_vec = setup_data
    cosmo = CosmologyBase(z_vec=z_vec)
    assert cosmo.z_vec.all() == z_vec.all()
    assert cosmo.h0 == 0.7
    assert cosmo.omega_c == 0.25
    assert cosmo.omega_b == 0.05
    assert cosmo.omega_m == 0.3
    assert cosmo.w0 == -1.0
    assert cosmo.wa == 0.0
    assert cosmo.n_s == 0.9
    assert cosmo.tcmb == 2.7255
    assert cosmo.m_nu == 0.06
    assert cosmo.sigma_8 == 0.8
    assert cosmo.log10T_AGN == 7.8

def test_cosmo_model(setup_data):
    cosmo = CosmologyBase()
    cosmo_model = cosmo.cosmo_model
    assert hasattr(cosmo_model, 'H0')
    assert hasattr(cosmo_model, 'Om0')
    assert hasattr(cosmo_model, 'Ob0')
    assert hasattr(cosmo_model, 'Tcmb0')
    assert hasattr(cosmo_model, 'w0')
    assert hasattr(cosmo_model, 'wa')

def test_get_mead_growth_fnc(setup_data):
    cosmo = CosmologyBase()
    growth_fnc = cosmo.get_mead_growth_fnc
    assert callable(growth_fnc)

def test_get_mead_growth(setup_data):
    cosmo = CosmologyBase()
    growth = cosmo.get_mead_growth
    assert isinstance(growth, np.ndarray)

def test_get_mead_accumulated_growth(setup_data):
    cosmo = CosmologyBase()
    accumulated_growth = cosmo.get_mead_accumulated_growth
    assert isinstance(accumulated_growth, np.ndarray)

def test_dc_Mead(setup_data):
    cosmo = CosmologyBase()
    dc_mead = cosmo.dc_Mead
    assert isinstance(dc_mead, np.ndarray)

def test_Dv_Mead(setup_data):
    cosmo = CosmologyBase()
    dv_mead = cosmo.Dv_Mead
    assert isinstance(dv_mead, np.ndarray)

def test_halo_model_ingredients_initialization(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec)
    assert ingredients.k_vec.all() == k_vec.all()
    assert ingredients.lnk_min == np.log(10**(-4.0))
    assert ingredients.lnk_max == np.log(10**(4.0))
    assert ingredients.dlnk == (np.log(10**(4.0)) - np.log(10**(-4.0))) / 100
    assert ingredients.Mmin == 9.0
    assert ingredients.Mmax == 16.0
    assert ingredients.dlog10m == 0.05
    assert ingredients.mdef_model == 'SOMean'
    assert ingredients.hmf_model == 'Tinker10'
    assert ingredients.bias_model == 'Tinker10'
    assert ingredients.halo_profile_model_dm == 'NFW'
    assert ingredients.halo_concentration_model_dm == 'Duffy08'
    assert ingredients.halo_profile_model_sat == 'NFW'
    assert ingredients.halo_concentration_model_sat == 'Duffy08'
    assert ingredients.transfer_model == 'CAMB'
    assert ingredients.transfer_params == None
    assert ingredients.growth_model == 'CambGrowth'
    assert ingredients.growth_params == None
    assert ingredients.norm_cen == 1.0
    assert ingredients.norm_sat == 1.0
    assert ingredients.eta_cen == 0.0
    assert ingredients.eta_sat == 0.0
    assert ingredients.overdensity == 200
    assert ingredients.delta_c == 1.686

def test_halo_model_ingredients_norm_c(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    norm_c = ingredients._norm_c
    assert norm_c.shape == z_vec.shape

def test_halo_model_ingredients_norm_s(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    norm_s = ingredients._norm_s
    assert norm_s.shape == z_vec.shape

def test_halo_model_ingredients_eta_c(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    eta_c = ingredients._eta_c
    assert eta_c.shape == z_vec.shape

def test_halo_model_ingredients_eta_s(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    eta_s = ingredients._eta_s
    assert eta_s.shape == z_vec.shape

def test_halo_model_ingredients_delta_c_mod(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    delta_c_mod = ingredients._delta_c_mod
    assert delta_c_mod.shape == z_vec.shape

def test_halo_model_ingredients_mdef_mod(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    mdef_mod = ingredients._mdef_mod
    assert isinstance(mdef_mod, str)

def test_halo_model_ingredients_hmf_mod(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    hmf_mod = ingredients._hmf_mod
    assert isinstance(hmf_mod, str)

def test_halo_model_ingredients_bias_mod(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    bias_mod = ingredients._bias_mod
    assert isinstance(bias_mod, str)

def test_halo_model_ingredients_halo_concentration_mod_dm(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    halo_concentration_mod_dm = ingredients._halo_concentration_mod_dm
    assert callable(halo_concentration_mod_dm)

def test_halo_model_ingredients_halo_concentration_mod_sat(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    halo_concentration_mod_sat = ingredients._halo_concentration_mod_sat
    assert callable(halo_concentration_mod_sat)

def test_halo_model_ingredients_mdef_params(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    mdef_params = ingredients.mdef_params
    assert isinstance(mdef_params, list)

def test_halo_model_ingredients_halo_profile_params(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    halo_profile_params = ingredients.halo_profile_params
    assert isinstance(halo_profile_params, dict)

def test_halo_model_ingredients_scale_factor(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    scale_factor = ingredients.scale_factor
    assert isinstance(scale_factor, np.ndarray)

def test_halo_model_ingredients_disable_mass_conversion(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    disable_mass_conversion = ingredients.disable_mass_conversion
    assert isinstance(disable_mass_conversion, bool)

def test_halo_model_ingredients_K(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    K = ingredients.K
    assert isinstance(K, np.ndarray)

def test_halo_model_ingredients_hmf_generator(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    hmf_generator = ingredients._hmf_generator
    assert isinstance(hmf_generator, tuple)

def test_halo_model_ingredients_hmf_cen(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    hmf_cen = ingredients._hmf_cen
    assert isinstance(hmf_cen, list)

def test_halo_model_ingredients_hmf_sat(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    hmf_sat = ingredients._hmf_sat
    assert isinstance(hmf_sat, list)

def test_halo_model_ingredients_mass(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    mass = ingredients.mass
    assert isinstance(mass, np.ndarray)

def test_halo_model_ingredients_power(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    power = ingredients.power
    assert isinstance(power, np.ndarray)

def test_halo_model_ingredients_nonlinear_power(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    nonlinear_power = ingredients.nonlinear_power
    assert isinstance(nonlinear_power, np.ndarray)

def test_halo_model_ingredients_kh(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    kh = ingredients.kh
    assert isinstance(kh, np.ndarray)

def test_halo_model_ingredients_halo_overdensity_mean(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    halo_overdensity_mean = ingredients.halo_overdensity_mean
    assert isinstance(halo_overdensity_mean, np.ndarray)

def test_halo_model_ingredients_nu(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    nu = ingredients.nu
    assert isinstance(nu, np.ndarray)

def test_halo_model_ingredients_dndlnm(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    dndlnm = ingredients.dndlnm
    assert isinstance(dndlnm, np.ndarray)

def test_halo_model_ingredients_mean_density0(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    mean_density0 = ingredients.mean_density0
    assert isinstance(mean_density0, np.ndarray)

def test_halo_model_ingredients_mean_density_z(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    mean_density_z = ingredients.mean_density_z
    assert isinstance(mean_density_z, np.ndarray)

def test_halo_model_ingredients_rho_halo(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    rho_halo = ingredients.rho_halo
    assert isinstance(rho_halo, np.ndarray)

def test_halo_model_ingredients_halo_bias(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    halo_bias = ingredients.halo_bias
    assert isinstance(halo_bias, np.ndarray)

def test_halo_model_ingredients_neff(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    neff = ingredients.neff
    assert isinstance(neff, np.ndarray)

def test_halo_model_ingredients_sigma8_z(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    sigma8_z = ingredients.sigma8_z
    assert isinstance(sigma8_z, np.ndarray)

def test_halo_model_ingredients_fnu(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    fnu = ingredients.fnu
    assert isinstance(fnu, np.ndarray)

def test_halo_model_ingredients_conc_cen(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    conc_cen = ingredients.conc_cen
    assert isinstance(conc_cen, np.ndarray)

def test_halo_model_ingredients_nfw_cen(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    nfw_cen = ingredients.nfw_cen
    assert isinstance(nfw_cen, np.ndarray)

def test_halo_model_ingredients_u_dm(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    u_dm = ingredients.u_dm
    assert isinstance(u_dm, np.ndarray)

def test_halo_model_ingredients_r_s_cen(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    r_s_cen = ingredients.r_s_cen
    assert isinstance(r_s_cen, np.ndarray)

def test_halo_model_ingredients_rvir_cen(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    rvir_cen = ingredients.rvir_cen
    assert isinstance(rvir_cen, np.ndarray)

def test_halo_model_ingredients_conc_sat(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    conc_sat = ingredients.conc_sat
    assert isinstance(conc_sat, np.ndarray)

def test_halo_model_ingredients_nfw_sat(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    nfw_sat = ingredients.nfw_sat
    assert isinstance(nfw_sat, np.ndarray)

def test_halo_model_ingredients_u_sat(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    u_sat = ingredients.u_sat
    assert isinstance(u_sat, np.ndarray)

def test_halo_model_ingredients_r_s_sat(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    r_s_sat = ingredients.r_s_sat
    assert isinstance(r_s_sat, np.ndarray)

def test_halo_model_ingredients_rvir_sat(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    rvir_sat = ingredients.rvir_sat
    assert isinstance(rvir_sat, np.ndarray)

def test_halo_model_ingredients_growth_factor(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)
    growth_factor = ingredients.growth_factor
    assert isinstance(growth_factor, np.ndarray)
