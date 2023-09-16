import numpy as np
import matplotlib.pyplot as plt

datadir='../output/HM_IA/'

def make_the_plot(pow_spec,colour,linestyle):
    # Read in the power spectra files
    kvals = np.loadtxt(datadir+pow_spec+'/k_h.txt')
    pkvals = np.loadtxt(datadir+pow_spec+'/p_k.txt')
    zvals = np.loadtxt(datadir+pow_spec+'/z.txt')
    plt.plot(kvals, np.abs(pkvals[5,:]),label=pow_spec,color=colour,linestyle=linestyle)
    print(np.max(pkvals[5,:]),zvals[5])

make_the_plot('matter_power_nl','black','solid')
make_the_plot('matter_intrinsic_power','grey','solid')
make_the_plot('matter_intrinsic_power_blue','blue','solid')
make_the_plot('matter_intrinsic_power_red','red','solid')
make_the_plot('intrinsic_power','grey','dashed')
make_the_plot('intrinsic_power_blue','blue','dashed')
make_the_plot('intrinsic_power_red','red','dashed')


plt.legend(loc='upper right',bbox_to_anchor=(1.2, 1.2),fontsize=8)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('k h/Mpc')
plt.ylabel('P(k,z=0.75)')
plt.tight_layout()
plt.show()