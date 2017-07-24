from __future__ import division

import numpy as np
from scipy import special
from numpy import log, exp, sin ,cos, pi, log10, sqrt
from scipy.integrate import quad, dblquad, cumtrapz
from matplotlib import pyplot as plt
import time

import CDM_SubHalo_Potential
import FDM_SubHalo_Potential

#integral precision
p = 2

#num plot points
num = 50

#fluc, normedfluc, fourierfluc, sqfourierfluc,tidalvar 
var = "sqfourierfluc"

#calc, test
mode = "calc"

#radius for test
Rtest = 10**3

params = { 
   'axes.labelsize': 24, 
   'axes.titlesize': 22,  
   'legend.fontsize': 20, 
   'xtick.labelsize': 24, 
   'ytick.labelsize': 24, 
   'text.usetex': True,
   'figure.figsize': [10,8], # instead of 4.5, 4.5
   'lines.linewidth': 2,
   'xtick.major.pad': 15, 
   'ytick.major.pad': 15, 
   'figure.subplot.bottom': 0.12,
   'figure.subplot.top': 0.95,
   'figure.subplot.left': 0.2,
   #'font.size': 22
   }   
plt.rcParams.update(params)


ep = np.logspace(1, p, num)
D = np.logspace(0, 5, num)

if(var == "fluc"):
	CDMfunc = CDM_SubHalo_Potential.Fluc
elif(var == "normedfluc"):
	CDMfunc = CDM_SubHalo_Potential.NormalizedFluc
elif(var == "fourierfluc"):
	CDMfunc = CDM_SubHalo_Potential.NormedFourierMagInt
elif(var == "sqfourierfluc"):
	CDMfunc = CDM_SubHalo_Potential.IntegSpectralPower
elif(var == "tidalvar"):
	CDMfunc = CDM_SubHalo_Potential.TidalVariance

def CDM_Calculate():
	if(mode == "calc"):
		return map(lambda x: CDMfunc(x, int(10**p)), D)
	elif(mode == "test"):
		return map(lambda x: CDMfunc(Rtest, int(x)), ep)
		
if(var == "fluc"):
	FDMfunc = FDM_SubHalo_Potential.Fluc
elif(var == "normedfluc"):
	FDMfunc = FDM_SubHalo_Potential.NormalizedFluc
elif(var == "fourierfluc"):
	FDMfunc = FDM_SubHalo_Potential.NormedFourierMagInt
elif(var == "sqfourierfluc"):
	FDMfunc = FDM_SubHalo_Potential.IntegSpectralPower
elif(var == "tidalvar"):
	FDMfunc = FDM_SubHalo_Potential.TidalVariance
		
def FDM_Calculate(set_m22):
	FDM_SubHalo_Potential.m22 = set_m22

	if(mode == "calc"):
		return map(lambda x: FDMfunc(x, int(10**p)), D)
	elif(mode == "test"):
		return map(lambda x: FDMfunc(Rtest, int(x)), ep)

def main():
	lines = []
	
	t = time.time()
	lines.append(plt.loglog(D, CDM_Calculate(), label = '$CDM$', linestyle = '--'))
	print "done CDM in " + str(time.time() - t)
	
	log_axion_masses = [6,4,2,1,0,-1]
	for logm in log_axion_masses:
		
		t = time.time()
		lines.append(plt.loglog(D, FDM_Calculate(10**logm), label = r'$m_{a} = 10^{' + str(logm-22) +'} eV$'))
		print "done FDM log(m22) = " + str(logm) + " in " + str(time.time() - t)
		
	plt.xlabel(r'$r(pc)$')
	if(var == "fluc"):
		plt.ylabel(r'$ \left < \left (\frac{\partial \phi}{\partial t} \right )^2 \right > (km^4 s^{-4} Myr^{-2})$')
	elif(var == "normedfluc"):
		plt.ylabel(r'$ \left < \left (\frac{\partial \phi}{\partial t} \right )^2 \right > \left (\frac{\Omega}{\phi} \right )^2$')
	elif(var == "fourierfluc"):
		plt.ylabel(r'\[ \frac{1}{\phi} \int_{\Omega}^{\infty} | \tilde{\phi}(\omega) | d\omega \]')
	elif(var == "sqfourierfluc"):
		plt.ylabel(r'\[ \frac{\Omega}{\phi^2} \int_{\Omega}^{\infty} | \tilde{\phi}(\omega) |^2 d\omega \]')
	elif(var == "tidalvar"):
		plt.ylabel(r'$\sigma_{T}^2$')
	plt.legend(loc='lower right')
	plt.show()
    
if __name__ == "__main__":
    main()

    

