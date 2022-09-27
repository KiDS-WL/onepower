import numpy as np
from scipy.integrate import simps
from math import factorial
from scipy.special import legendre, binom

def I_x(a,b):
	eps = 1e-10
	x = np.linspace(-1.+eps,1.-eps,500)
	int = simps((1.-x**2.)**(a/2.)*x**b,x)
	return int
	
def legendre_coefficients(l,m):
	# note that scipy.special.legendre returns an array with the coefficients of the legendre polynomials 
	return legendre(l)[m]


def g(n):
	g_vec = np.zeros(11)
	g_vec[2] = np.pi/2.
	g_vec[4] = np.pi/2.
	g_vec[6] = np.pi*15./32.
	g_vec[8] = np.pi*7./16.
	g_vec[10] = np.pi*105./256.
	return g_vec[n]


# see Fortuna et al. 2020, Appendix B	
def another_fell(theta_k, phi_k, l, gamma_b):
	phase = np.cos(2.*phi_k) + 1j*np.sin(2.*phi_k)
	sum1 = 0.
	for m in range(0,l+1):
		sum2=0.
		for j in range(0,m+1):
			sum2 += binom(m,j) * g(j) * np.sin(theta_k)**(j) * np.cos(theta_k)**(m-j) * I_x(j+gamma_b,m-j)
		sum1 += binom(l,m)*binom(0.5*(l+m-1.),l)*sum2		
	return 2.**l * sum1*phase 

			
def compare_leg_coeff(l,m):	
	print(legendre_coefficients(l,m))
	print(2.**l * binom(l,m)*binom(0.5*(l+m-1.),l))
	return
