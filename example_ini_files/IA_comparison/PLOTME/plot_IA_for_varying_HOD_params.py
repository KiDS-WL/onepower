import numpy as np
import os
import matplotlib.pyplot as plt

# This script runs the cosmosis test sampler for a range of input HOD parameters and plots the IA models

# These are the parameters that I want to change and the range that I want to vary them over
# I'm using the prior ranges here from Dvornik et al 2023 https://arxiv.org/pdf/2210.03110.pdf
hod_params={
'norm_c':[0.1,1.0]}#,
#'log_ml_0':[7.0,13.0],
#'log_ml_1':[9.0,14.0],
#'g1':[2.5,15.0],
#'g2':[0.1,10.0],
#'scatter':[0.0,2.0],
#'norm_s':[0.1,1.0],
#'pivot': [10.0,14.0],
#'b0':[-5.0,5.0],
#'alpha_s':[-5.0,5.0],
#'b1':[-5.0,5.0],
#'b2':[-5.0,5.0]
#}

# A plotting module
datadir='output/HM_IA/'
def plot_cl_GIII_over_GG(ax,iz,jz,colour):
    # Read in the power spectra files
    shearcl='shear_cl_gg'
    ellvals = np.loadtxt(datadir+shearcl+'/ell.txt')
    GGvals = np.loadtxt(datadir+shearcl+'/bin_'+str(iz)+'_'+str(jz)+'.txt')
    #II and GI
    #IIvals = np.loadtxt(datadir+'shear_cl_ii/bin_'+str(iz)+'_'+str(jz)+'.txt')
    #GIvals = np.loadtxt(datadir+'shear_cl_gi/bin_'+str(iz)+'_'+str(jz)+'.txt')
    #ratio = (IIvals + GIvals + GGvals)/GGvals
    shearcl='shear_cl'  #This is GG+GI(ij) + GI(ji) +II
    clvals = np.loadtxt(datadir+shearcl+'/bin_'+str(iz)+'_'+str(jz)+'.txt')
    ratio = clvals/GGvals
    print(clvals[1],ratio[1])
    ax.plot(ellvals, ratio, color=colour,linestyle='solid')



#How many samples do you want to take in addition to testing the minimum and maximum values?
isample=2
#Set up a nice range of colours to plot the samples in
colours = plt.cm.jet(np.linspace(0,1,isample+2))
fig, ax = plt.subplots(12, 3,figsize=(6,24),sharex='all',gridspec_kw=dict(hspace=0))

ip=-1
for param, value_range in hod_params.items():
    ip=ip+1
    iv=-1
    for val in np.linspace(value_range[0],value_range[1],num=2+isample):
        iv=iv+1
        command='hod_parameters_red.'+param+'=%.5e '%(val)
        command+='hod_parameters_blue.'+param+'=%.5e '%(val)
        print(command)
        test_sampler_run='cosmosis ../create_mock_with_HaloModel_IA.ini -v '+command 
        test_sample_exit_status=os.system(test_sampler_run)
        plot_cl_GIII_over_GG(ax[ip,0],1,1,colours[iv])  #tomo bin 1-1
        plot_cl_GIII_over_GG(ax[ip,1],5,1,colours[iv])  #tomo bin 1-5
        plot_cl_GIII_over_GG(ax[ip,2],5,5,colours[iv])  #tomo bin 5-5
        ax[ip,2].text(100.0,1.0,param)


ax[0,0].set_xscale('log')
plt.tight_layout()
fig.savefig('Varying_HOD_params.png')