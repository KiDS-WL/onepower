import matplotlib.pyplot as plt
import numpy as np

datadir='../output/HM_IA/'
def plot_cl(ax,shearcl,iz,jz,colour,linestyle):
    # Read in the power spectra files
    ellvals = np.loadtxt(datadir+shearcl+'/ell.txt')
    clvals = np.loadtxt(datadir+shearcl+'/bin_'+str(iz+1)+'_'+str(jz+1)+'.txt')
    ax.set_axis_on()
    ax.plot(ellvals, np.abs(clvals),label=shearcl,color=colour,linestyle=linestyle)

def plot_ratio_cl(ax,shearcl,cl_base,iz,jz,colour,linestyle):
    # Read in the power spectra files
    # Read in the power spectra files
    ellvals = np.loadtxt(datadir+shearcl+'/ell.txt')
    clvals = np.loadtxt(datadir+shearcl+'/bin_'+str(iz+1)+'_'+str(jz+1)+'.txt')
    ax.set_axis_on()
    ax.plot(ellvals,(clvals)/cl_base,label=shearcl,color=colour,linestyle=linestyle)

#First plot the Cl's
figone, ax = plt.subplots(5, 5,figsize=(10,10),sharey='all',sharex='all',gridspec_kw=dict(wspace=0,hspace=0))
for jz in range (5):
    for iz in range (5):
        if iz < jz:
            ax[iz,jz].set_axis_off()
        else:
            print (iz,jz)
            plot_cl(ax[iz,jz],'shear_cl',iz,jz,'black','solid')
            plot_cl(ax[iz,jz],'shear_cl_gg',iz,jz,'blue','dashed')
            plot_cl(ax[iz,jz],'shear_cl_gi',iz,jz,'cyan','solid')
            plot_cl(ax[iz,jz],'shear_cl_ii',iz,jz,'magenta','solid')
            ax[iz,jz].set_yscale('log')
            ax[iz,jz].set_xscale('log')
            if iz==4:
                ax[iz,jz].set_xlabel(r'$\ell$',fontsize=16)
            if jz==0:
                ax[iz,jz].set_ylabel(r'$C(\ell)$',fontsize=16)

ax[4,4].legend(loc='upper right',bbox_to_anchor=(1.0, 1.7))
plt.tight_layout()
figone.savefig('Cl_GG_GI_II_comparison.png')

# Now plot ratios to the NL matter power spectrum
figtwo, ax = plt.subplots(5, 5,figsize=(10,10),sharey='row',sharex='all',gridspec_kw=dict(wspace=0,hspace=0))
for jz in range (5):
    for iz in range (5):
        if iz < jz:
            ax[iz,jz].set_axis_off()
        else:
            print (iz,jz)
            cl_base = np.loadtxt(datadir+'shear_cl_gg/bin_'+str(iz+1)+'_'+str(jz+1)+'.txt')
            plot_ratio_cl(ax[iz,jz],'shear_cl',cl_base,iz,jz,'black','solid')
            plot_ratio_cl(ax[iz,jz],'shear_cl_gi',cl_base,iz,jz,'cyan','solid')
            plot_ratio_cl(ax[iz,jz],'shear_cl_ii',cl_base,iz,jz,'magenta','solid')
            ax[iz,jz].set_xscale('log')
            ax[iz,jz].hlines(y=1.0, xmin=0, xmax=3000, linestyle='dotted', color='grey')
            if iz==4:
                ax[iz,jz].set_xlabel(r'$\ell$',fontsize=16)
            if jz==0:
                ax[iz,jz].set_ylabel('$C(\\ell)/C_{\\rm GG}(\\ell)$',fontsize=16)

ax[4,4].legend(loc='upper right',bbox_to_anchor=(1.0, 1.7))
plt.tight_layout()
figtwo.savefig('Cl_GG_GI_II_ratio_comparison.png')
