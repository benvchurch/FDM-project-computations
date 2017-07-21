from __future__ import division

import numpy as np
from scipy import special
from numpy import log, exp, sin ,cos, pi, log10, sqrt
from scipy.integrate import quad, dblquad, cumtrapz
from matplotlib import pyplot as plt
import time

import CDM_SubHalo_Potential
import FDM_SubHalo_Potential

p = 7

def main():
	ep = np.linspace(2,9,10)
	D = np.linspace(0, 5, 40)
	t = time.time()
	F = map(lambda x: log10(CDM_SubHalo_Potential.Fluc(10**x, int(2**p))), D)
	print "done CDM in " + str(time.time() - t)
	lines = []
	lines.append(plt.plot(D, F, label = 'CDM', linestyle = '--'))
	axion_masses = [4,3,2,1,0,-1]
	for m in axion_masses:
		FDM_SubHalo_Potential.m22 = 10**m
		t = time.time()
		F = map(lambda x: log10(FDM_SubHalo_Potential.Fluc(10**x, int(2**p))), D)
		print "done FDM log(m) = " + str(m) + " in " + str(time.time() - t)
		lines.append(plt.plot(D, F, label = r'$m_{a} = 10^{' + str(m-22) +'} eV$'))
		
	plt.xlabel('log(r/pc)')
	plt.ylabel('log(Potential Fluctuation)')
	plt.legend(loc='lower right')
	plt.show()
    
if __name__ == "__main__":
    main()

    

