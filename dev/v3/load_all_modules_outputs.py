from cosmosis.datablock import names, option_section
import sys
import numpy as np
import os

def setup(options):
    ref_run = "/Volumes/Samsung_T5/referee_paperI/mc_uncertainties_run/ref_run"	
    return ref_run
    
def execute(block,config):
    
    ref_run = config 
    
    # load matter lin    
    k_h_lin = np.loadtxt(os.path.join(ref_run, "matter_power_lin/k_h.txt"))
    pk_lin = np.loadtxt(os.path.join(ref_run, "matter_power_lin/p_k.txt"))
    z_lin = np.loadtxt(os.path.join(ref_run, "matter_power_lin/z.txt"))    
    block.put_grid("matter_power_lin", "z", z_lin, "k_h", k_h_lin, "p_k", pk_lin)
    
    # load distances
    a = np.loadtxt(os.path.join(ref_run, "distances/a.txt"))
    d_a = np.loadtxt(os.path.join(ref_run, "distances/d_a.txt"))
    d_l = np.loadtxt(os.path.join(ref_run, "distances/d_l.txt"))
    d_m = np.loadtxt(os.path.join(ref_run, "distances/d_m.txt"))
    h = np.loadtxt(os.path.join(ref_run, "distances/h.txt"))
    mu = np.loadtxt(os.path.join(ref_run, "distances/mu.txt"))
    z = np.loadtxt(os.path.join(ref_run, "distances/z.txt"))
    
    block.put_double_array_1d("distances", "a", a)
    block.put_double_array_1d("distances", "d_a", d_a)
    block.put_double_array_1d("distances", "d_l", d_l)
    block.put_double_array_1d("distances", "d_m", d_m)
    block.put_double_array_1d("distances", "h", h)
    block.put_double_array_1d("distances", "mu", mu)
    block.put_double_array_1d("distances", "z", z)

    
    # load matter nl
    k_h_nl = np.loadtxt(os.path.join(ref_run, "matter_power_nl/k_h.txt"))
    pk_nl = np.loadtxt(os.path.join(ref_run, "matter_power_nl/p_k.txt"))
    z_nl = np.loadtxt(os.path.join(ref_run, "matter_power_nl/z.txt"))    
    block.put_grid("matter_power_nl", "z", z_nl, "k_h", k_h_nl, "p_k", pk_nl)
    
    # load hmf
    dndlnmh = np.loadtxt(os.path.join(ref_run, "hmf/dndlnmh.txt"))
    m_h = np.loadtxt(os.path.join(ref_run, "hmf/m_h.txt"))
    z_hmf = np.loadtxt(os.path.join(ref_run, "hmf/z.txt"))
    block.put_grid("hmf", "z", z_hmf, "m_h", m_h, "dndlnmh", dndlnmh)
        
    # load halobias 
    b_hb = np.loadtxt(os.path.join(ref_run, "halobias/b_hb.txt"))    
    m_hb = np.loadtxt(os.path.join(ref_run, "halobias/m_h.txt"))    
    z_hb = np.loadtxt(os.path.join(ref_run, "halobias/z.txt"))    
    block.put_grid("halobias", "z", z_hb, "m_h", m_hb, "b_hb", b_hb)

    # load hod red
    central_fraction_red = np.loadtxt(os.path.join(ref_run, "hod_red/central_fraction.txt"))
    satellite_fraction_red = np.loadtxt(os.path.join(ref_run, "hod_red/satellite_fraction.txt"))
    n_cen_red = np.loadtxt(os.path.join(ref_run, "hod_red/n_cen.txt"))
    n_sat_red = np.loadtxt(os.path.join(ref_run, "hod_red/n_sat.txt"))
    number_density_cen_red = np.loadtxt(os.path.join(ref_run, "hod_red/number_density_cen.txt"))
    number_density_sat_red = np.loadtxt(os.path.join(ref_run, "hod_red/number_density_sat.txt"))
    mass_red = np.loadtxt(os.path.join(ref_run, "hod_red/mass.txt"))
    z_red = np.loadtxt(os.path.join(ref_run, "hod_red/z.txt"))
    
    suffix = "_red"
    block.put_grid("hod" + suffix, "z", z_red, "mass", mass_red, "n_sat", n_sat_red)
    block.put_grid("hod" + suffix, "z", z_red, "mass", mass_red, "n_cen", n_cen_red)
    block.put_double_array_1d("hod" + suffix, "redshifts", z_red)
    block.put_double_array_1d("hod" + suffix, "number_density_cen", number_density_cen_red)
    block.put_double_array_1d("hod" + suffix, "number_density_sat", number_density_sat_red)
    block.put_double_array_1d("hod" + suffix, "central_fraction", central_fraction_red)
    block.put_double_array_1d("hod" + suffix, "satellite_fraction", satellite_fraction_red)
    
    # load hod_blue
    central_fraction_blue = np.loadtxt(os.path.join(ref_run, "hod_blue/central_fraction.txt"))
    satellite_fraction_blue = np.loadtxt(os.path.join(ref_run, "hod_blue/satellite_fraction.txt"))
    n_cen_blue = np.loadtxt(os.path.join(ref_run, "hod_blue/n_cen.txt"))
    n_sat_blue = np.loadtxt(os.path.join(ref_run, "hod_blue/n_sat.txt"))
    number_density_cen_blue = np.loadtxt(os.path.join(ref_run, "hod_blue/number_density_cen.txt"))
    number_density_sat_blue = np.loadtxt(os.path.join(ref_run, "hod_blue/number_density_sat.txt"))
    mass_blue = np.loadtxt(os.path.join(ref_run, "hod_blue/mass.txt"))
    z_blue = np.loadtxt(os.path.join(ref_run, "hod_blue/z.txt"))
    
    suffix = "_blue"
    block.put_grid("hod" + suffix, "z", z_blue, "mass", mass_blue, "n_sat", n_sat_blue)
    block.put_grid("hod" + suffix, "z", z_blue, "mass", mass_blue, "n_cen", n_cen_blue)
    block.put_double_array_1d("hod" + suffix, "redshifts", z_blue)
    block.put_double_array_1d("hod" + suffix, "number_density_cen", number_density_cen_blue)
    block.put_double_array_1d("hod" + suffix, "number_density_sat", number_density_sat_blue)
    block.put_double_array_1d("hod" + suffix, "central_fraction", central_fraction_blue)
    block.put_double_array_1d("hod" + suffix, "satellite_fraction", satellite_fraction_blue)

    # fourier transform of the nfw profile
    z_nfw = np.loadtxt(os.path.join(ref_run, "fourier_nfw_profile/z.txt"))
    m_h_nfw = np.loadtxt(os.path.join(ref_run, "fourier_nfw_profile/m_h.txt"))
    k_h_nfw = np.loadtxt(os.path.join(ref_run, "fourier_nfw_profile/k_h.txt"))
    u_dm_nfw = np.loadtxt(os.path.join(ref_run, "fourier_nfw_profile/ukm.txt"))
    block.put_double_array_1d("fourier_nfw_profile", "z", z_nfw)
    block.put_double_array_1d("fourier_nfw_profile", "m_h", m_h_nfw)
    block.put_double_array_1d("fourier_nfw_profile", "k_h", k_h_nfw)
    block.put_double_array_nd("fourier_nfw_profile", "ukm", u_dm_nfw)

    # concentration
    c_nfw = np.loadtxt(os.path.join(ref_run, "concentration/c.txt"))
    m_h_nfw = np.loadtxt(os.path.join(ref_run, "concentration/m_h.txt"))
    z_nfw = np.loadtxt(os.path.join(ref_run, "concentration/z.txt"))
    block.put_grid("concentration", "z", z_nfw, "m_h", m_h_nfw, "c", c_nfw)

    # scale radius    
    r_s = np.loadtxt(os.path.join(ref_run, "nfw_scale_radius/rs.txt"))
    block.put_grid("nfw_scale_radius", "z", z_nfw, "m_h", m_h_nfw, "rs", r_s)
    
    # virial radius
    rvir = np.loadtxt(os.path.join(ref_run, "virial_radius/rvir.txt"))
    block.put_double_array_1d("virial_radius", "m_h", m_h_nfw)
    block.put_double_array_1d("virial_radius", "rvir", rvir)
         
    return 0