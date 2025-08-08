import matplotlib.pyplot as plt
import numpy as np

datadir='../output/HM_IA/'
fig, ((ax1, ax2)) = plt.subplots(1, 2,figsize=(10,4))

def plot_pow_spec(ax,pow_spec,iz,colour,linestyle):
    # Read in the power spectra files
    kvals = np.loadtxt(datadir+pow_spec+'/k_h.txt')
    pkvals = np.loadtxt(datadir+pow_spec+'/p_k.txt')
    zvals = np.loadtxt(datadir+pow_spec+'/z.txt')
    ax.plot(kvals, np.abs(pkvals[iz,:]),label=pow_spec,color=colour,linestyle=linestyle)
#    ax.plot(kvals, kvals*kvals*pkvals[iz,:],label=pow_spec,color=colour,linestyle=linestyle)
    print(np.max(pkvals[iz,:]),np.min(pkvals[iz,:]),zvals[iz])

def plot_ratio_pow_spec(ax,pow_spec,pkvals_base,iz,colour,linestyle):
    # Read in the power spectra files
    kvals = np.loadtxt(datadir+pow_spec+'/k_h.txt')
    pkvals = np.loadtxt(datadir+pow_spec+'/p_k.txt')
    zvals = np.loadtxt(datadir+pow_spec+'/z.txt')
    ax.plot(kvals, pkvals[iz,:]/pkvals_base[iz,:],label=pow_spec,color=colour,linestyle=linestyle)
    print(np.max(pkvals[iz,:]),zvals[iz])

# Choose redshift of interest
iz=2  #z=0.75
plot_pow_spec(ax1,'matter_power_nl',iz,'black','solid')
plot_pow_spec(ax1, 'galaxy_power_red',iz,'red','solid')
plot_pow_spec(ax1, 'galaxy_power_blue',iz,'blue','solid')
#plot_pow_spec(ax1,'matter_intrinsic_power',iz,'grey','solid')
#plot_pow_spec(ax1,'matter_intrinsic_power_blue',iz,'blue','solid')
#plot_pow_spec(ax1,'matter_intrinsic_power_red',iz,'red','solid')
#plot_pow_spec(ax1,'intrinsic_power',iz,'grey','dashed')
#plot_pow_spec(ax1,'intrinsic_power_blue',iz,'blue','dashed')
#plot_pow_spec(ax1,'intrinsic_power_red',iz,'red','dashed')

ax1.legend(loc='upper right',bbox_to_anchor=(1.2, 1.2),fontsize=8)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_xlabel('k h/Mpc')
ax1.set_ylabel('P(k,z=0.75)')

# Now plot ratios to the NL matter power spectrum
pkvals_base = np.loadtxt(datadir+'matter_power_nl/p_k.txt')

#plot_ratio_pow_spec(ax2,'matter_intrinsic_power',pkvals_base,iz,'grey','solid')
#plot_ratio_pow_spec(ax2,'matter_intrinsic_power_blue',pkvals_base,iz,'blue','solid')
#plot_ratio_pow_spec(ax2,'matter_intrinsic_power_red',pkvals_base,iz,'red','solid')
#plot_ratio_pow_spec(ax2,'intrinsic_power',pkvals_base,iz,'grey','dashed')
#plot_ratio_pow_spec(ax2,'intrinsic_power_blue',pkvals_base,iz,'blue','dashed')
#plot_ratio_pow_spec(ax2,'intrinsic_power_red',pkvals_base,iz,'red','dashed')

ax2.set_xscale('log')
ax2.set_xlabel('k h/Mpc')
ax2.set_ylabel('P(k,z=0.75)/PNL(k,z=0.75)')


plt.tight_layout()
fig.savefig('3D_powerspectra_comparison.png')
