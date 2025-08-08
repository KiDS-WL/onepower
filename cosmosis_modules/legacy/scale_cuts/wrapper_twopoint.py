##############################
    ##  wrapper_twopoint.py    ##
    ##  Chieh-An Lin        ##
    ##  Version 2020.02.21    ##
    ##############################

import collections as clt
import numpy as np
import twopoint

###############################################################################
## Functions related to label convention

class LabelConvention:
    """
    There can be 3 different names for 1 statistics.
    Say for example, xi_+:
    In the theory pipeline, it is something like "shear_xi_plus"
    In the scale cuts module, it's "xiP" (cannot have underscore for the string-splitting reason).
    In the measurement pipeline, it depends on user's choice.
    This class deals with the conversion.
    """

    def __init__(self, w='wtheta', gamma_t='gammat', gamma_x='gammax', xi_p='xip', xi_m='xim',
                  P_nn='Pnn', P_ne_E='PneE', P_ne_B='PneB', P_ee_E='PeeE', P_ee_B='PeeB',
                  E_n='En', B_n='Bn', Psi_gm='Psi_gm', Psi_gg='Psi_gg', onept='1pt'):

        self.prefix  = 'twoPoint'
        self.lens    = 'NZ_LENS'
        self.source  = 'NZ_SOURCE'
        self.obs     = 'NZ_OBS'

        self.onept   = '1PT'.lower()

        self.w       = 'wtheta'.lower()
        self.gamma_t = 'gammat'.lower()
        self.gamma_x = 'gammax'.lower()
        self.xi_p    = 'xip'.lower()
        self.xi_m    = 'xim'.lower()
        self.P_nn    = 'Pnn'.lower()
        self.P_ne_E  = 'PneE'.lower()
        self.P_ne_B  = 'PneB'.lower()
        self.P_ee_E  = 'PeeE'.lower()
        self.P_ee_B  = 'PeeB'.lower()
        self.E_n     = 'En'.lower()
        self.B_n     = 'Bn'.lower()
        self.Psi_gm  = 'Psi_gm'.lower()
        self.Psi_gg  = 'Psi_gg'.lower()

        self.labDict = {
            self.w:        w,
            self.gamma_t:  gamma_t,
            self.gamma_x:  gamma_x,
            self.xi_p:     xi_p,
            self.xi_m:     xi_m,
            self.P_nn:     P_nn,
            self.P_ne_E:   P_ne_E,
            self.P_ne_B:   P_ne_B,
            self.P_ee_E:   P_ee_E,
            self.P_ee_B:   P_ee_B,
            self.E_n:      E_n,
            self.B_n:      B_n,
            self.Psi_gm:   Psi_gm,
            self.Psi_gg:   Psi_gg,
            self.onept:    onept,
            'all':         'all'
        }

        self.revLabDict = {}
        for key, value in self.labDict.items():
            self.revLabDict[value] = key

        tpType1 = twopoint.Types.galaxy_position_real       ## GPR
        tpType2 = twopoint.Types.galaxy_shear_plus_real     ## G+R
        tpType3 = twopoint.Types.galaxy_shear_minus_real    ## G-R
        tpType4 = twopoint.Types.galaxy_position_fourier    ## GPF
        tpType5 = twopoint.Types.galaxy_shear_emode_fourier ## GEF
        tpType6 = twopoint.Types.galaxy_shear_bmode_fourier ## GBF

        self.kernelTypeDict = clt.OrderedDict()
        ## Don't touch the order of the assertion
        self.kernelTypeDict[w]       = [self.lens,   self.lens,   tpType1, tpType1, 'arcmin']
        self.kernelTypeDict[gamma_t] = [self.lens,   self.source, tpType1, tpType2, 'arcmin']
        self.kernelTypeDict[gamma_x] = [self.lens,   self.source, tpType1, tpType3, 'arcmin']
        self.kernelTypeDict[xi_p]    = [self.source, self.source, tpType2, tpType2, 'arcmin']
        self.kernelTypeDict[xi_m]    = [self.source, self.source, tpType3, tpType3, 'arcmin']
        self.kernelTypeDict[P_nn]    = [self.lens,   self.lens,   tpType4, tpType4, None]
        self.kernelTypeDict[P_ne_E]  = [self.lens,   self.source, tpType4, tpType5, None]
        self.kernelTypeDict[P_ne_B]  = [self.lens,   self.source, tpType4, tpType6, None]
        self.kernelTypeDict[P_ee_E]  = [self.source, self.source, tpType5, tpType5, None]
        self.kernelTypeDict[P_ee_B]  = [self.source, self.source, tpType6, tpType6, None]
        self.kernelTypeDict[E_n]     = [self.source, self.source, tpType5, tpType5, None]
        self.kernelTypeDict[B_n]     = [self.source, self.source, tpType6, tpType6, None]
        self.kernelTypeDict[Psi_gm]  = [self.lens,   self.source, tpType4, tpType5, None]
        self.kernelTypeDict[Psi_gg]  = [self.lens,   self.lens,   tpType4, tpType4, None]
        return

    def defaultToCustomStatsTag(self, statsTag):
        statsList   = statsTag.split('+')
        statsList_c = [self.labDict[stats.lower()] for stats in statsList]
        statsTag_c  = '+'.join(statsList_c)
        return statsTag_c

    def customToDefaultStatsTag(self, statsTag_c):
        statsList_c = statsTag_c.split('+')
        statsList   = [self.revLabDict[stats_c] for stats_c in statsList_c]
        statsTag    = '+'.join(statsList)
        return statsTag

    def makeScaleCutsArgs(self, scDict):
        """
        Convert some lines of ASCII characters into a tuple of 5 elements defining various scale cuts
        scDict is a dictionary, constructed by:
        scDict[scale_cut_option] = list of strings
        """
        statsList = None
        cutCross = False
        statsTag_tomoInd_tomoInd_list = []
        statsTag_binIndList_dict = {}
        statsTag_tomoInd1_tomoInd2__angMin_angMax_dict = {}
        statsTag__angMin_angMax_dict = {}

        for key, value in scDict.items():
            keySplit = key.split('_')

            if key == 'use_stats':
                statsList = value.split()

            elif key == 'cut_cross':
                cutCross = bool(value)

            elif 'cut_pair' in key:
                statsTag = self.labDict[keySplit[2]]
                for pair in value.split():
                    pair = pair.split('+')
                    tomoInd1 = int(pair[0])
                    tomoInd2 = int(pair[1])
                    statsTag_tomoInd_tomoInd_list.append((statsTag, tomoInd1, tomoInd2))

            elif 'cut_bin' in key:
                statsTag   = self.labDict[keySplit[2]]
                binIndList = [int(ind) for ind in value.split()]
                statsTag_binIndList_dict[statsTag] = binIndList

            elif 'keep_ang' in key and len(keySplit) > 4:
                statsTag = self.labDict[keySplit[2]]
                tomoInd1 = int(keySplit[3])
                tomoInd2 = int(keySplit[4])
                value    = value.split()
                angMin   = float(value[0])
                angMax   = float(value[1])
                statsTag_tomoInd1_tomoInd2__angMin_angMax_dict[(statsTag, tomoInd1, tomoInd2)] = (angMin, angMax)

            elif 'keep_ang' in key and len(keySplit) < 4:
                statsTag = self.labDict[keySplit[2]]
                value    = value.split()
                angMin   = float(value[0])
                angMax   = float(value[1])
                statsTag__angMin_angMax_dict[statsTag] = (angMin, angMax)

        scArgs = cutCross, statsTag_tomoInd_tomoInd_list, statsTag_binIndList_dict, \
            statsTag_tomoInd1_tomoInd2__angMin_angMax_dict, statsTag__angMin_angMax_dict
        return statsList, scArgs

###############################################################################
## Functions related to SpectrumMeasurement builder

class SpectrumBuilder():

    def __init__(self):
        self.tIL1    = []
        self.tIL2    = []
        self.aIL     = []
        self.angList = []
        self.valList = []
        self.nobs = []
        self.obs = []
        self.obs_name = []
        self.nbin = []
        return

    def addTomo(self, tomoInd1, tomoInd2, angle, value):
        N_ang  = len(angle)
        angInd = np.arange(N_ang, dtype=int)
        self.tIL1.append([tomoInd1+1] * N_ang)
        self.tIL2.append([tomoInd2+1] * N_ang)
        self.aIL.append(angInd + 1)
        self.angList.append(angle)
        self.valList.append(value)
        return

    def add_one_point(self, obs_name, nbin, x, angbin, values):
        self.obs_name.append(obs_name)
        self.nbin.append(nbin)
        self.obs.append(x)
        self.nobs.append(values)
        #self.angbin.append(angbin)

    def makeSpectrum(self, name, types, angle_unit, kernels=(None, None)):
        self.tIL1    = np.concatenate(self.tIL1)
        self.tIL2    = np.concatenate(self.tIL2)
        self.aIL     = np.concatenate(self.aIL)
        self.angList = np.concatenate(self.angList)
        self.valList = np.concatenate(self.valList)
        spec = twopoint.SpectrumMeasurement(name, (self.tIL1, self.tIL2), types, kernels, 'SAMPLE', self.aIL, self.valList, angle=self.angList, angle_unit=angle_unit)
        return spec

###############################################################################
## Functions related to TwoPointFile wrapper

class TwoPointWrapper(twopoint.TwoPointFile):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ## Default arguments are: spectra, kernels, nobs, windows, covmat_info
        return

    @classmethod
    def from_spectra(cls, spectra, kernels=None, nobs=None, covmat_info=None):
        TP = cls(spectra, kernels, nobs, 'SAMPLE', covmat_info) ## windows = 'SAMPLE'
        return TP

    @classmethod
    def from_fits(cls, filename, covmat_name=None):
        TP = super().from_fits(filename, covmat_name=covmat_name)
        return TP

    ## to_fits(self, filename, overwrite=False, clobber=False) can be used directly as
    ## TP.to_fits(filename, overwrite=True)

    def makeMeanVector(self):
        theory = []
        for spectrum in self.spectra:
            theory.append(spectrum.value)
        if self.nobs is not None:
            for obs in self.nobs:
                theory.append(np.array(obs.nobs).flatten())
        theory = np.concatenate(theory)
        return theory

    def replaceMeanVector(self, data):
        idx = []
        for spectrum in self.spectra:
            idx.append(len(spectrum.value))
        if self.nobs is not None:
            for obs in self.nobs:
                idx.append(len(np.array(obs.nobs).flatten()))

        if sum(idx) != len(data):
            raise ValueError(f"Size of provided data vector incompatible with spectra: {len(data)} vs {sum(idx)}.")

        idx.insert(0, 0)
        idx = np.cumsum(idx)

        for i, spectrum in enumerate(self.spectra):
            spectrum.value = data[idx[i]:idx[i+1]]
        if self.nobs is not None:
            for i, obs in enumerate(self.nobs):
                obs.nobs[i] = np.array(data[idx[len(self.spectra)+i]:idx[len(self.spectra)+i+1]]).flatten()

    def cutScales(self, cutCross=False, statsTag_tomoInd_tomoInd_list=[], statsTag_binIndList_dict={}):
        if cutCross:
            self.mask_cross() ## Cut cross pair bins, but not cross-corr between auto pair bins; e.g. C_12_12 masked, C_11_22 preserved

        if len(statsTag_tomoInd_tomoInd_list) > 0:
            self.mask_scales(bin_cuts=statsTag_tomoInd_tomoInd_list) ## Cut the whole tomo pair bin

        for statsTag, binIndList in statsTag_binIndList_dict.items():
            self.mask_indices(statsTag, binIndList) ## Cut by bin indices, 0-indexing
        return

    def keepScales(self, statsTag_tomoInd1_tomoInd2__angMin_angMax_dict={}, statsTag__angMin_angMax_dict={}):
        if len(statsTag_tomoInd1_tomoInd2__angMin_angMax_dict) > 0:
            self.mask_scales(cuts=statsTag_tomoInd1_tomoInd2__angMin_angMax_dict) ## Keep an angular range for specific tomo bins

        for statsTag, angMin_angMax in statsTag__angMin_angMax_dict.items():
            self.mask_scale(statsTag, min_scale=angMin_angMax[0], max_scale=angMin_angMax[1]) ## Keep an angular range for a specific tomo bin
        return

###############################################################################
