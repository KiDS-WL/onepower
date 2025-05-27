import warnings
import numpy as np
from astropy.cosmology import Flatw0waCDM
import halo_model_utility as hmu
import hmf
from halomod import get_halomodel
from halomod.halo_model import DMHaloModel
from halomod.concentration import make_colossus_cm, interp_concentration
import halomod.profiles as profile_classes
import halomod.concentration as concentration_classes
import time

# Silencing a warning from hmf for which the nonlinear mass is still correctly calculated
warnings.filterwarnings("ignore", message="Nonlinear mass outside mass range")


class HaloModelIngredients:
    def __init__(self,
            **kwargs
        ):
        self.k_vec = k_vec
        self.z_vec = z_vec
        self.lnk_min = lnk_min
        self.lnk_max = lnk_max
        self.dlnk = dlnk
        self.Mmin = Mmin
        self.Mmax = Mmax
        self.dlog10m = dlog10m
        self.disable_mass_conversion = disable_mass_conversion
        self.halo_profile_model = halo_profile_model
        self.halo_concentration_model = halo_concentration_model
        self.omega_c = omega_c
        self.omega_m = omega_m
        self.omega_b = omega_b
        self.w0 = w0
        self.wa = wa
        self.h0 = h0
        self.n_s = n_s
        self.tcmb = tcmb
        self.mnu = mnu
        self.sigma_8 = sigma_8
        self.log10T_AGN = log10T_AGN
        self.transfer_model = transfer_model
        self.transfer_params = transfer_params
        self.growth_model = growth_model
        self.growth_params = growth_params
        self.mead_correction = mead_correction

        self.halo_profile_params = {'cosmo': self.cosmo_model}

        if self.mead_correction in ['feedback', 'nofeedback']:
            self._setup_mead_correction(norm_sat, eta_sat, norm_cen)
        else:
            self._setup_default(hmf_model, bias_model, cm_model, mdef_model, overdensity, norm_sat, eta_sat, norm_cen, eta_cen)

    def _setup_mead_correction(self, norm_sat, eta_sat, norm_cen):
        self.disable_mass_conversion = True
        self.hmf_model = 'ST'
        self.bias_model = 'ST99'
        self.halo_concentration_model = 'Bullock01'
        growth = hmu.get_growth_interpolator(self.cosmo_model)
        a = self.cosmo_model.scale_factor(self.z_vec)
        g = growth(a)
        G = np.array([hmu.get_accumulated_growth(ai, growth) for ai in a])
        delta_c_mead = hmu.dc_Mead(a, self.cosmo_model.Om(self.z_vec) + self.cosmo_model.Onu(self.z_vec),
                            self.cosmo_model.Onu0 / (self.cosmo_model.Om0 + self.cosmo_model.Onu0), g, G)
        halo_overdensity_mead = hmu.Dv_Mead(a, self.cosmo_model.Om(self.z_vec) + self.cosmo_model.Onu(self.z_vec),
                                        self.cosmo_model.Onu0 / (self.cosmo_model.Om0 + self.cosmo_model.Onu0), g, G)
        self.delta_c = delta_c_mead
        self.mdef_model = hmu.SOVirial_Mead
        self.mdef_params = [{'overdensity': halo_overdensity_mead[i]} for i, _ in enumerate(self.z_vec)]

        self.norm_sat = norm_sat
        self.eta_sat = eta_sat

        if self.mead_correction == 'nofeedback':
            self.norm_cen = 5.196 * np.ones_like(self.z_vec)
        elif self.mead_correction == 'feedback':
            theta_agn = self.log10T_AGN - 7.8
            self.norm_cen = (5.196 / 4.0) * ((3.44 - 0.496 * theta_agn) * np.power(10.0, self.z_vec * (-0.0671 - 0.0371 * theta_agn)))

    def _setup_default(self, hmf_model, bias_model, cm_model, mdef_model, overdensity, norm_sat, eta_sat, norm_cen, eta_cen):
        self.disable_mass_conversion = False
        self.hmf_model = hmf_model
        self.bias_model = bias_model
        try:
            self.halo_concentration_model = interp_concentration(getattr(concentration_classes, cm_model))
        except:
            self.halo_concentration_model = interp_concentration(make_colossus_cm(cm_model))
        self.delta_c = (3.0 / 20.0) * (12.0 * np.pi) ** (2.0 / 3.0) * (1.0 + 0.0123 * np.log10(self.cosmo_model.Om(self.z_vec)))
        self.mdef_model = mdef_model
        self.mdef_params = [{} if self.mdef_model == 'SOVirial' else {'overdensity': overdensity} for z in self.z_vec]
        self.norm_cen = norm_cen * np.ones_like(self.z_vec)
        self.norm_sat = norm_sat * np.ones_like(self.z_vec)
        self.eta_cen = eta_cen * np.ones_like(self.z_vec)
        self.eta_sat = eta_sat * np.ones_like(self.z_vec)
            
    @cached_property
    def cosmo_model(self):
        # Update the cosmological parameters
        return Flatw0waCDM(
            H0=self.h0*100.0,
            Ob0=self.omega_b,
            Om0=self.omega_m,
            m_nu=[0, 0, self.mnu],
            Tcmb0=self.tcmb,
            w0=self.w0,
            wa=self.wa
        )
      
    @cached_property
    def hmf_init(self):
        hmf = DMHaloModel(
            lnk_min=self.lnk_min,
            lnk_max=self.lnk_max,
            dlnk=self.dlnk,
            Mmin=self.Mmin,
            Mmax=self.Mmax,
            dlog10m=self.dlog10m,
            hmf_model=self.hmf_model,
            mdef_model=self.mdef_model,
            mdef_params=self.mdef_params,
            disable_mass_conversion=self.disable_mass_conversion,
            bias_model=self.bias_model,
            halo_profile_model=self.halo_profile_model,
            halo_profile_params=self.halo_profile_params,
            halo_concentration_model=self.halo_concentration_model,
            cosmo_model=self.cosmo_model,
            sigma_8=self.sigma_8,
            n=self.n_s,
            transfer_model=self.transfer_model,
            transfer_params=self.transfer_params,
            growth_model=self.growth_model,
            growth_params=self.growth_params
        )
        hmf.ERROR_ON_BAD_MDEF = False
        return hmf
        
    @cached_property
    def hmf(self):
        # First we clone the initialised hmf instance n-times, where n is len(z_vec)
        hmf_z = [self.hmf_init.clone() for z in self.z_vec]
        
        # Then update the each cloned instance with redshift dependent params
        return [hmf_z[i].update(
            z=z[i],
            mdef_params=self.mdef_params[i],
            delta_c=self.delta_c[i]
        ) for i in range(len(self.z_vec))]

    @cached_property
    def hmf_cen(self):
        hmf = self.hmf
        if self.mead_correction in ['feedback', 'nofeedback']:
            self.eta_cen = [0.1281 * hmf[i].sigma8_z**(-0.3644) for i, _ in enumerate(self.z_vec)]
        # Then update the each cloned instance with redshift dependent params
        return [hmf[i].update(
            halo_profile_params={'eta_bloat': self.eta_cen[i]},
            halo_concentration_params={'norm': self.norm_cen[i]}
        ) for i in range(len(self.z_vec))]
        
    @cached_property
    def hmf_sat(self):
        hmf = self.hmf
        # Then update the each cloned instance with redshift dependent params
        return [hmf[i].update(
            halo_profile_params={'eta_bloat': self.eta_cen[i]},
            halo_concentration_params={'norm': self.norm_sat[i]}
        ) for i in range(len(self.z_vec))]
        
    @property
    def mass(self):
        return self.hmf_init.m

    @property
    def halo_overdensity_mean(self):
        return np.array([self.hmf[i].halo_overdensity_mean for i, _ in enumerate(self.z_vec)])

    @property
    def nu(self):
        return np.array([self.hmf[i].nu**0.5 for i, _ in enumerate(self.z_vec)])
           
    @property
    def dndlnm(self):
        return np.array([self.hmf[i].dndlnm for i, _ in enumerate(self.z_vec)])
    
    @property
    def mean_density0(self):
        return np.array([self.hmf[i].mean_density0 for i, _ in enumerate(self.z_vec)])

    @property
    def mean_density_z(self):
        return np.array([self.hmf[i].mean_density for i, _ in enumerate(self.z_vec)])

    @property
    def rho_halo(self):
        return np.array([self.hmf[i].halo_overdensity_mean * self.hmf[i].mean_density0 for i, _ in enumerate(self.z_vec)])

    @property
    def b_nu(self):
        return np.array([self.hmf[i].halo_bias for i, _ in enumerate(self.z_vec)])

    @property
    def neff(self):
        return np.array([self.hmf[i].n_eff_at_collapse for i, _ in enumerate(self.z_vec)])

    @property
    def sigma8_z(self):
        return np.array([self.hmf[i].sigma8_z for i, _ in enumerate(self.z_vec)])
       
    @property
    def fnu(self):
        return np.array([self.cosmo_model.Onu0 / self.cosmo_model.Om0 for _ in self.z_vec])

    @property
    def conc_cen(self):
        return np.array([self.hmf_cen[i].cmz_relation for i, _ in enumerate(self.z_vec)])
        
    @property
    def nfw_cen(self):
        return np.array([self.hmf_cen[i].halo_profile.u(self.k_vec, self.hmf_cen[i].m) for i, _ in enumerate(self.z_vec)])
            
    @property
    def u_dm(self):
        return self.nfw_cen / np.expand_dims(self.nfw_cen[0, :], 0)
        
    @property
    def r_s_cen(self):
        return np.array([self.hmf_cen[i].halo_profile._rs_from_m(self.hmf_cen[i].m) for i, _ in enumerate(self.z_vec)])
            
    @property
    def rvir_cen(self):
        return np.array([self.hmf_cen[i].halo_profile.halo_mass_to_radius(self.hmf_cen[i].m) for i, _ in enumerate(self.z_vec)])


    @property
    def conc_sat(self):
        return np.array([self.hmf_sat[i].cmz_relation for i, _ in enumerate(self.z_vec)])
    
    @property
    def nfw_sat(self):
        return np.array([self.hmf_sat[i].halo_profile.u(self.k_vec, self.hmf_sat[i].m) for i, _ in enumerate(self.z_vec)])

    @property
    def u_sat(self):
        return self.nfw_sat / np.expand_dims(self.nfw_sat[0, :], 0)
        
    @property
    def r_s_sat(self):
        return np.array([self.hmf_sat[i].halo_profile._rs_from_m(self.hmf_sat[i].m) for i, _ in enumerate(self.z_vec)])
            
    @property
    def rvir_sat(self):
        return np.array([self.hmf_sat[i].halo_profile.halo_mass_to_radius(self.hmf_sat[i].m) for i, _ in enumerate(self.z_vec)])
       
        
    # Maybe implement at some point?
    # Rnl = DM_hmf.filter.mass_to_radius(DM_hmf.mass_nonlinear, DM_hmf.mean_density0)
    # neff[jz] = -3.0 - 2.0*DM_hmf.normalised_filter.dlnss_dlnm(Rnl)

    # Only used for mead_corrections
    # pk_cold = DM_hmf.power * hmu.Tk_cold_ratio(DM_hmf.k, g, block[cosmo_params, 'ommh2'], block[cosmo_params, 'h0'], this_cosmo_run.Onu0/this_cosmo_run.Om0, this_cosmo_run.Neff, T_CMB=tcmb)**2.0
    # sigma8_z[jz] = hmu.sigmaR_cc(pk_cold, DM_hmf.k, 8.0)

