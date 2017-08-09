from __future__ import division

import numpy as np
from scipy import special
from numpy import log, exp, sin ,cos, pi, log10, sqrt
from scipy.integrate import quad, dblquad, cumtrapz
from matplotlib import pyplot as plt
import time


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
   'figure.subplot.left': 0.225,
   #'font.size': 22
   }   
plt.rcParams.update(params)

def readData(fileName):
	file_obj = open(fileName, "r")
	lines = file_obj.read().splitlines()
	to_remove = []
	for l in lines:
		if "nan" in l or "inf" in l:
			to_remove.append(l)
			print "Remoivng: " + str(l)
	
	for l in to_remove:
		lines.remove(l)
		
	x, y = list(map(lambda x : 10**float(x.split(" ")[0]), lines)), list(map(lambda x : 10**float(x.split(" ")[1]), lines))
	file_obj.close()
	return x, y 

def main():	
	Dens_x, Dens_y = readData("Density.txt")
	Flucs_x, Flucs_y = readData("Flucs.txt")
	
	plt.figure()
	plt.loglog(Dens_x, Dens_y)
	plt.xlabel(r'$ r(pc)$')
	plt.ylabel(r'$ \rho \quad (M_{\odot} pc^{-3})$')
	
	plt.figure()
	plt.loglog(Flucs_x, Flucs_y)
	plt.xlabel(r'$ r(pc)$')
	plt.ylabel(r'$ \sqrt{\left < \left (\frac{\partial \phi}{\partial t} \right )^2 \right >} \quad ((km/s)^{2} Myr^{-1})$')
	plt.show()
	
    
if __name__ == "__main__":
    main()

    

