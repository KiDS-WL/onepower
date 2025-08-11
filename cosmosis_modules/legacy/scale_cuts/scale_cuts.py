import configparser as cfp
import numpy as np
import twopoint
import wrapper_twopoint as wtp
from cosmosis.datablock import option_section
from cosmosis.datablock.cosmosis_py import errors

# New scale cuts supporting 1pt statistics

def setup(options):
    config = {}

    ## Read inputs from datablock
    config['output_section_name'] = options.get_string(option_section, 'output_section_name', default='likelihood')
    try:
        config['data_and_covariance_fits_filename'] = options.get_string(option_section, 'data_and_covariance_fits_filename')
    except errors.BlockNameNotFound:
        raise NameError('data_and_covariance_fits_filename cannot be empty')

    ## Read extension names for data outputs
    config['wt_extension_name']                       = options.get_string(option_section, 'wt_extension_name', default='wtheta')
    config['gt_extension_name']                       = options.get_string(option_section, 'gt_extension_name', default='gammat')
    config['xi_plus_extension_name']                  = options.get_string(option_section, 'xi_plus_extension_name', default='xip')
    config['xi_minus_extension_name']                 = options.get_string(option_section, 'xi_minus_extension_name', default='xim')
    config['bandpower_clustering_extension_name']     = options.get_string(option_section, 'bandpower_clustering_extension_name', default='Pnn')
    config['bandpower_ggl_extension_name']            = options.get_string(option_section, 'bandpower_ggl_extension_name', default='PneE')
    config['bandpower_e_cosmic_shear_extension_name'] = options.get_string(option_section, 'bandpower_e_cosmic_shear_extension_name', default='PeeE')
    config['bandpower_b_cosmic_shear_extension_name'] = options.get_string(option_section, 'bandpower_b_cosmic_shear_extension_name', default='PeeB')
    config['psi_stats_gg_extension_name']             = options.get_string(option_section, 'psi_stats_gg_extension_name', default='Psi_gg')
    config['psi_stats_gm_extension_name']             = options.get_string(option_section, 'psi_stats_gm_extension_name', default='Psi_gm')
    config['cosebis_extension_name']                  = options.get_string(option_section, 'cosebis_extension_name', default='En')
    config['onepoint_extension_name']                 = options.get_string(option_section, 'onepoint_extension_name', default='1pt')

    ## Read section names for theory outputs
    config['wt_section_name']                       = options.get_string(option_section, 'wt_section_name', default='galaxy_xi')
    config['gt_section_name']                       = options.get_string(option_section, 'gt_section_name', default='galaxy_shear_xi')
    config['xi_plus_section_name']                  = options.get_string(option_section, 'xi_plus_section_name', default='shear_xi_plus')
    config['xi_minus_section_name']                 = options.get_string(option_section, 'xi_minus_section_name', default='shear_xi_minus')
    config['bandpower_clustering_section_name']     = options.get_string(option_section, 'bandpower_clustering_section_name', default='bandpower_clustering')
    config['bandpower_ggl_section_name']            = options.get_string(option_section, 'bandpower_ggl_section_name', default='bandpower_ggl')
    config['bandpower_e_cosmic_shear_section_name'] = options.get_string(option_section, 'bandpower_e_cosmic_shear_section_name', default='bandpower_e_cosmic_shear')
    config['bandpower_b_cosmic_shear_section_name'] = options.get_string(option_section, 'bandpower_b_cosmic_shear_section_name', default='bandpower_b_cosmic_shear')
    config['psi_stats_gg_section_name']             = options.get_string(option_section, 'psi_stats_gg_section_name', default='psi_gg')
    config['psi_stats_gm_section_name']             = options.get_string(option_section, 'psi_stats_gm_section_name', default='psi_gm')
    config['cosebis_section_name']                  = options.get_string(option_section, 'cosebis_section_name', default='cosebis')
    config['onepoint_section_name']                 = options.get_string(option_section, 'onepoint_section_name', default='observable_function')

    ## Read scale cuts
    try:
        ## Try to load scale cuts file if specified
        config['scale_cuts_filename'] = options.get_string(option_section, 'scale_cuts_filename')
        try:
            parser = cfp.ConfigParser()
            parser.read(config['scale_cuts_filename'])
        except:
            raise OSError('\"%s\" not found' % config['scale_cuts_filename'])

        config['scale_cuts_option'] = options.get_string(option_section, 'scale_cuts_option', default='scale_cuts_none')
        try:
            ## Make scale cuts arguments
            scDict = parser[config['scale_cuts_option']]
        except:
            raise KeyError('Bad scale cuts option: \"%s\"' % config['scale_cuts_option'])

    except errors.BlockNameNotFound:
        ## Otherwise read inline options
        key_list   = [k for _, k in options.keys(option_section) if k.split('_')[0] in ['use', 'cut', 'keep']]
        value_list = [str(options[option_section, k]).strip(' []') for k in key_list]
        scDict = {k : v for k, v in zip(key_list, value_list)}

    ## Load data & cov file
    try:
        TP_data = wtp.TwoPointWrapper.from_fits(config['data_and_covariance_fits_filename'], covmat_name='COVMAT')
    except:
        raise OSError('\"%s\" not found' % config['data_and_covariance_fits_filename'])

    labConv = wtp.LabelConvention(w=config['wt_extension_name'],
                                  gamma_t=config['gt_extension_name'],
                                  xi_p=config['xi_plus_extension_name'],
                                  xi_m=config['xi_minus_extension_name'],
                                  P_nn=config['bandpower_clustering_extension_name'],
                                  P_ne_E=config['bandpower_ggl_extension_name'],
                                  P_ee_E=config['bandpower_e_cosmic_shear_extension_name'],
                                  E_n=config['cosebis_extension_name'],
                                  Psi_gg=config['psi_stats_gg_extension_name'],
                                  Psi_gm=config['psi_stats_gm_extension_name'],
                                  onept=config['onepoint_extension_name'])
    statsList, scArgs = labConv.makeScaleCutsArgs(scDict) ## Here, we convert the keys from the default ones to the custom ones.
    config['scale_cuts_arguments'] = scArgs
    config['label_convention']     = labConv

    ## Do scale cuts to data and cov
    TP_data.cutScales(cutCross=scArgs[0], statsTag_tomoInd_tomoInd_list=scArgs[1], statsTag_binIndList_dict=scArgs[2])
    TP_data.keepScales(statsTag_tomoInd1_tomoInd2__angMin_angMax_dict=scArgs[3], statsTag__angMin_angMax_dict=scArgs[4])
    #print('  Did scale cuts to data & cov')

    ## Don't put this before scale cuts, because TP_data.choose_data_sets does
    ## not modify the covmat_info attribute but only the covmat attribute.
    ## While one can cut a correctly cut covariance, the scale cuts on this
    ## covariance, which relies on covmat_info, can result in errors.
    statsList_c           = [labConv.defaultToCustomStatsTag(stats) for stats in statsList]
    config['use_stats']   = statsList
    config['use_stats_c'] = statsList_c
    TP_data.choose_data_sets(statsList_c)

    ## Extract the vector & matrix & put in config dict
    config['data']       = TP_data.makeMeanVector()
    config['covariance'] = TP_data.covmat
    config['inv_covariance'] = np.linalg.inv(TP_data.covmat)

    config['simulate'] = options.get_bool(option_section, 'simulate', default=False)
    config['simulate_with_noise'] = options.get_bool(option_section, 'simulate_with_noise', default=True)
    config['number_of_simulations'] = options.get_int(option_section,'number_of_simulations',default=1)
    config['mock_filename'] = options.get_string(option_section, 'mock_filename', default="")
    if config['simulate']:
        config["TP_data"] = TP_data

    return config

def execute(block, config):
    #print('Gathering theory outputs to make a vector')

    ## Save data and cov in the data block
    output_section_name = config['output_section_name']
    block[output_section_name, 'data']       = config['data']
    block[output_section_name, 'covariance'] = config['covariance']
    block[output_section_name, 'inv_covariance'] = config['inv_covariance']

    ## Define some things first
    labConv    = config['label_convention']
    nbTomo_max = 1000
    spectra    = []

    ## Don't change the order of this list
    ## Read as: [section_name, extension_name, angle_name, isGGL]
    sectionNameList = [
        [config['wt_section_name'],                       config['wt_extension_name'],                       'theta',         False],
        [config['gt_section_name'],                       config['gt_extension_name'],                       'theta',         True],
        [config['xi_plus_section_name'],                  config['xi_plus_extension_name'],                  'theta',         False],
        [config['xi_minus_section_name'],                 config['xi_minus_extension_name'],                 'theta',         False],
        [config['bandpower_clustering_section_name'],     config['bandpower_clustering_extension_name'],     'ell',           False],
        [config['bandpower_ggl_section_name'],            config['bandpower_ggl_extension_name'],            'ell',           True],
        [config['bandpower_e_cosmic_shear_section_name'], config['bandpower_e_cosmic_shear_extension_name'], 'ell',           False],
        [config['psi_stats_gg_section_name'],             config['psi_stats_gg_extension_name'],             'index_bin_1_1',     False],
        [config['psi_stats_gm_section_name'],             config['psi_stats_gm_extension_name'],             'index_bin_1_1',     True],
        [config['cosebis_section_name'],                  config['cosebis_extension_name'],                  'n',     False],
    ]

    for line in sectionNameList:
        section_name, extension_name, angle_name, isGGL = line

        if extension_name not in config['use_stats']:
            #print('  Skipped %s' % extension_name)
            continue

        elif not block.has_section(section_name):
            raise AssertionError('You specified \"{}\" in use_stats. When I was computing for \"{}\", I tried to\n\
                grab theory values from \"{}\" but could not find anything.'.format(config['use_stats'], extension_name, section_name))

        ## Initialize!
        sBuilder = wtp.SpectrumBuilder()

        ## Is it BP? If yes, then ell needs to be calculated instead of calling directly from data block.
        if angle_name == 'ell':
            lower = block[section_name, 'l_min_vec']
            upper = block[section_name, 'l_max_vec']
            angle = 0.5 * (np.log10(lower) + np.log10(upper))
            angle = 10**angle
        else:
            angle = block[section_name, angle_name]

        ## Is it GGL? If yes, the indices i & j run differently from the others.
        if isGGL:
            for i in range(nbTomo_max):
                if not block.has_value(section_name, 'bin_%d_%d' % (i+1, 1)):
                    #print('  %s stops at lens bin %d' % (section_name, i))
                    break

                for j in range(nbTomo_max):
                    value_name  = 'bin_%d_%d' % (i+1, j+1)
                    try:
                        value = block[section_name, value_name]
                        sBuilder.addTomo(i, j, angle, value)
                    except:
                        break
        else:
            for i in range(nbTomo_max):
                if not block.has_value(section_name, 'bin_%d_%d' % (i+1, i+1)):
                    #print('  %s stops at bin %d' % (section_name, i))
                    break

                for j in range(i, nbTomo_max):
                    value_name  = 'bin_%d_%d' % (j+1, i+1)
                    try:
                        value = block[section_name, value_name]
                        sBuilder.addTomo(i, j, angle, value)
                    except:
                        break

        ## These are some necessary components for generating the mean vector (spectrum).
        line  = labConv.kernelTypeDict[extension_name]
        ker1  = line[0]
        ker2  = line[1]
        type1 = line[2]
        type2 = line[3]
        unit  = line[4]

        spec  = sBuilder.makeSpectrum(extension_name, (type1, type2), unit, kernels=(ker1, ker2))
        spectra.append(spec)

    if config['onepoint_extension_name'] in config['use_stats']:
        if not block.has_section(config['onepoint_section_name']):
            raise AssertionError('You specified \"{}\" in use_stats. When I was computing for \"{}\", I tried to\n\
                grab theory values from \"{}\" but could not find anything.'.format(config['use_stats'], config['onepoint_extension_name'], config['onepoint_section_name']))
        else:
            onept_theory = [twopoint.OnePointMeasurement.from_block(block, config['onepoint_section_name'], output_name='1pt')]
    else:
        onept_theory = None

    ## Make. Ta da!
    TP_theory = wtp.TwoPointWrapper.from_spectra(spectra, nobs=onept_theory, kernels=None, covmat_info=None)

    ## Do scale cuts to theory
    scArgs = config['scale_cuts_arguments']
    TP_theory.cutScales(cutCross=scArgs[0], statsTag_tomoInd_tomoInd_list=scArgs[1], statsTag_binIndList_dict=scArgs[2])
    TP_theory.keepScales(statsTag_tomoInd1_tomoInd2__angMin_angMax_dict=scArgs[3], statsTag__angMin_angMax_dict=scArgs[4])
    #print('  Did scale cuts to theory')

    ## Extract the vector & put in block as output_section_name
    block[output_section_name, 'theory'] = TP_theory.makeMeanVector()

    if config["simulate"]:
        mu = block[output_section_name, 'theory']
        cov = block[output_section_name, 'covariance']
        if config["simulate_with_noise"]:
            if(config["number_of_simulations"]>1):
                for n in range(config["number_of_simulations"]):
                    s = np.random.multivariate_normal(mu, cov)
                    config["TP_data"].replaceMeanVector(s)
                    if config["mock_filename"] != "":
                        config["TP_data"].to_fits(config["mock_filename"]+str(n+1)+".fits", overwrite=True)
            else:
                s = np.random.multivariate_normal(mu, cov)
                config["TP_data"].replaceMeanVector(s)
                if config["mock_filename"] != "":
                    config["TP_data"].to_fits(config["mock_filename"]+".fits", overwrite=True)
        else:
            s = mu
            config["TP_data"].replaceMeanVector(s)
            if config["mock_filename"] != "":
                config["TP_data"].to_fits(config["mock_filename"]+".fits", overwrite=True)

    return 0
