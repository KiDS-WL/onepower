import matplotlib.pyplot as plt
import numpy as np
import os

# This script runs the cosmosis test sampler for a range of input HOD parameters and plots the IA models

# These are the parameters that I want to change and the range that I want to vary them over
# I'm using the prior ranges here from Dvornik et al 2023 https://arxiv.org/pdf/2210.03110.pdf
hod_params={
'norm_c':[0.1,1.0],
'log_ml_0':[7.0,13.0],
'log_ml_1':[9.0,14.0],
'g1':[2.5,15.0],
'g2':[0.1,10.0],
'scatter':[0.0,2.0],
'norm_s':[0.1,1.0],
'pivot': [10.0,14.0],
'b0':[-5.0,5.0],
'alpha_s':[-5.0,5.0],
'b1':[-5.0,5.0],
'b2':[-5.0,5.0]
}

isample=10
gamma_hat_params={
'nmass':np.linspace(5,50,num=2+isample).astype(int),
'kmin':np.geomspace(0.0001,0.1,num=2+isample),
'kmax':np.geomspace(10,10000,num=2+isample),
'nk':np.linspace(10,50,num=2+isample).astype(int)
}

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


def plot_cl_shear_over_default(shearcl,ax,iz,jz,colour):
    # Read in the power spectra files
    ellvals = np.loadtxt(datadir+shearcl+'/ell.txt')
    clvals = np.loadtxt(datadir+shearcl+'/bin_'+str(iz)+'_'+str(jz)+'.txt')
    default_vals = np.loadtxt('../'+datadir+shearcl+'/bin_'+str(iz)+'_'+str(jz)+'.txt')
    ratio = clvals/default_vals -1.0
    print(clvals[1],ratio[1])
    ax.plot(ellvals, ratio, color=colour,linestyle='solid')
    ax.yaxis.get_major_formatter().set_useOffset(False)


def plot_tomobins(ip,iv,shearcl,ax):
    plot_cl_shear_over_default(shearcl,ax[ip,0],1,1,colours[iv])  #tomo bin 1-1
    plot_cl_shear_over_default(shearcl,ax[ip,1],5,1,colours[iv])  #tomo bin 1-5
    plot_cl_shear_over_default(shearcl,ax[ip,2],5,5,colours[iv])  #tomo bin 5-5

def put_exponent_in_top_right_corner(ax):
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.get_yaxis().get_offset_text().set_visible(False)
    ax_max = max(ax.get_yticks())
    exponent_axis = np.floor(np.log10(ax_max)).astype(int)
    ax.text(0.01,0.01,r'$\times$10$^{%i}$'%(exponent_axis),
             ha='left', va='bottom',transform = ax.transAxes,fontsize=12)

def tidy_my_plot(ax,ip):
    # The exponent ends up overlapping with the upper panel plot :(
    # So we're going to use this slightly messy hack to remove it
    for i in range(3):
        put_exponent_in_top_right_corner(ax[ip,i])
    #Put the title in the bottom left corner
    ax[ip,0].text(0.01,1.1,param,ha='left', va='top',transform = ax[ip,0].transAxes,fontsize=12)
    ax[0,0].set_xscale('log')
    ax[ip,0].set_ylabel(r'$\Delta C(\ell)/C(\ell)$',fontsize=12)



#How many samples do you want to take in addition to testing the minimum and maximum values?
#Set up a nice range of colours to plot the samples in
colours = plt.cm.jet(np.linspace(0,1,isample+2))

#We're going to make separate figures for GG+GI+II,  GI and II
figall, axall = plt.subplots(4, 3,figsize=(6,8),sharex='all')#,gridspec_kw=dict(hspace=0))
figGI, axGI = plt.subplots(4, 3,figsize=(6,8),sharex='all')#,gridspec_kw=dict(hspace=0))
figII, axII = plt.subplots(4, 3,figsize=(6,8),sharex='all')#,gridspec_kw=dict(hspace=0))

#12 for HOD figsize=(6,24)
#4 for gamma_hat figsize=(6,8)

ip=-1
for param, value_range in gamma_hat_params.items():
    ip=ip+1
    iv=-1
    for val in value_range:
        iv=iv+1
        #command='hod_parameters_red.'+param+'=%.5e '%(val)
        #command+='hod_parameters_blue.'+param+'=%.5e '%(val)  #use -v when changing values
        command='radial_satellite_alignment_red.'+param+'=%.5e '%(val)
        command+='radial_satellite_alignment_blue.'+param+'=%.5e '%(val) #use -p when changing parameters
        print(command)
        test_sampler_run='cosmosis ../create_mock_with_HaloModel_IA.ini -p '+command
        test_sample_exit_status=os.system(test_sampler_run)
        plot_tomobins(ip,iv,'shear_cl',axall)
        plot_tomobins(ip,iv,'shear_cl_gi',axGI)
        plot_tomobins(ip,iv,'shear_cl_ii',axII)

    tidy_my_plot(axall,ip)
    tidy_my_plot(axGI,ip)
    tidy_my_plot(axII,ip)

figall.tight_layout()
figGI.tight_layout()
figII.tight_layout()

figall.savefig('Shear_All_varying_radial_calc_grid_params.png')
figGI.savefig('Shear_GI_varying_radial_calc_grid_params.png')
figII.savefig('Shear_II_varying_radial_calc_grid_params.png')
