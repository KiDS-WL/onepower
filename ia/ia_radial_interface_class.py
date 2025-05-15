from cosmosis.datablock import option_section
import numpy as np
from ia_radial_lib_class import SatelliteAlignment

def setup(options):
    # Set up the resolution for redshift, mass, and k grid for calculating w(k,z|m)
    # function which is slow: the lower the resolution the better
    # and we'll interpolate over this later to get the
    # right resolution for the power spectrum calculation

    # The set of default parameters here are fast and reasonably accurate
    # Needs more testing to be sure what the optimal defaults are though
    # TODO:  I have not yet tested the resolution in the z-dimension

    nmass = options.get_int(option_section, 'nmass', default=5)
    kmin = options.get_double(option_section, 'kmin', default=1e-3)
    kmax = options.get_double(option_section, 'kmax', default=1e3)
    nk = options.get_int(option_section, 'nk', default=10)
    k_vec = np.logspace(np.log10(kmin), np.log10(kmax), nk)

    # Are we calculating the alignment for say red or blue galaxies?
    name = options.get_string(option_section, 'output_suffix', default='').lower()
    suffix = f'_{name}' if name else ''
    
    ell_max = options.get_int(option_section, 'ell_max', default=6)
    n_hankel = options.get_int(option_section, 'N_hankel', default=350)

    return k_vec, nmass, suffix, n_hankel, ell_max

def execute(block, config):
    
    k_vec, nmass, suffix, n_hankel, ell_max = config
    
    
    # Load slope of the power law that describes the satellite alignment
    gamma_1h_slope = block[f'intrinsic_alignment_parameters{suffix}', 'gamma_1h_radial_slope']
        
    # This already contains the luminosity dependence if there
    gamma_1h_amplitude = block[f'ia_small_scale_alignment{suffix}', 'alignment_1h']
        
    # Also load the redshift dimension
    z = block['concentration_m', 'z']

    # Now I want to load the high resolution halo parameters calculated with the halo model module
    # and then downsample them to a lower resolution grid for the radial IA calculation
    # When downsampling we note that this doesn't need to be perfect, our final resolution does not need to
    # perfectly match the user input value - just as close as possible
    mass_halo = block['concentration_m', 'm_h']
    c_halo = block['concentration_m', 'c']
    r_s_halo = block['nfw_scale_radius_m', 'rs']
    rvir_halo = block['virial_radius', 'rvir_m']
    
    align_params = {}
    align_params.update({
        'mass': mass_halo,
        'k_vec': k_vec,
        'z_vec': z,
        'c': c_halo,
        'r_s': r_s_halo,
        'rvir': rvir_halo,
        'nmass': nmass,
        'n_hankel': n_hankel,
        'ell_max': ell_max,
        'gamma_1h_slope': gamma_1h_slope,
        'gamma_1h_amplitude': gamma_1h_amplitude
    })
    
    satellite_alignment = SatelliteAlignment(**align_params)
    wkm, z, mass, k = satellite_alignment.wkm()

    for jz in range(z.size):
        block.put_grid(
            'wkm', f'mass_{jz}{suffix}', mass, f'k_h_{jz}{suffix}', k,
            f'w_km_{jz}{suffix}', wkm[jz, :, :]
        )
    block.put_double_array_1d('wkm', f'z{suffix}', z)

    return 0

def cleanup(config):
    pass
