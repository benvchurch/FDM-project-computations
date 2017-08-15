from __future__ import division

import numpy as np
from scipy import special
from numpy import log, exp, sin ,cos, pi, log10, sqrt
from scipy.integrate import quad, dblquad, cumtrapz
from matplotlib import pyplot as plt
import time

import CDM_SubHalo_Potential
from CDM_SubHalo_Potential import c, G, MaxRadius, MFreeNFW

#integral precision
p = 2

#num plot points
num = 50

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


D = np.logspace(0, 9, num)


def main():
	#plt.figure()
	#plt.loglog(D, map(lambda x : CDM_SubHalo_Potential.VelocityDispersion(x, int(10**p), 10**15), D))
	#plt.loglog(D, map(lambda x : CDM_SubHalo_Potential.FlucWalk(x, int(10**p), 10**15), D))
	#plt.xlabel(r'$r(pc)$')
	#plt.ylabel(r'$ \Delta v \: (km/s)$')
	#plt.show()
	
	Ms = np.logspace(9, 20, 22)
	Vs, Fs = map(lambda x : max(map(lambda y : CDM_SubHalo_Potential.VelocityDispersion(y, int(10**p), x), D)), Ms), map(lambda x : max(map(lambda y : CDM_SubHalo_Potential.FlucWalk(y, int(10**p), x), D)), Ms)
	plt.figure()
	plt.loglog(Ms, Vs, label = "Dispersion")
	plt.loglog(Ms, Fs, label = "Fluctation")
	plt.xlabel(r'$M_p (M_\odot)$')
	plt.ylabel(r'$ \Delta v \: (km/s)$')
	plt.legend()
	
	Vmaxes = map(lambda x : sqrt(MFreeNFW(2.1625 / c * MaxRadius(x), x) * G/(2.1625 / c * MaxRadius(x))), Ms)
	
	plt.figure()
	plt.loglog(Vmaxes, Vs, label = "Dispersion")
	plt.loglog(Vmaxes, Fs, label = "Fluctation")
	plt.xlabel(r'$ v_{max} \: (km/s)$')
	plt.ylabel(r'$ \Delta v \: (km/s)$')
	plt.legend()
	plt.show()

if __name__ == "__main__":
    main()

    

