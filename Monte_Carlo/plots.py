from __future__ import division

import numpy as np
from scipy import special
from numpy import log, exp, sin ,cos, pi, log10, sqrt
from scipy.integrate import quad, dblquad, cumtrapz
from matplotlib import pyplot as plt
import time
import sys
from statsmodels.nonparametric.smoothers_lowess import lowess


params = {
   'text.usetex': True
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
    file_obj.close()

    get_coord = lambda i : np.array(map(lambda x : 10**float(x.split(" ")[i]), lines))
    return map(get_coord, [0, 1, 2])

def main():
    Dens_x, Dens_y, Dens_errs = readData("data_files/Density" + sys.argv[1] + '_' + sys.argv[2] + ".txt")
    Flucs_x, Flucs_y, Flucs_errs = readData("data_files/Flucs" + sys.argv[1] + '_' + sys.argv[2] + ".txt")

    filter_func = lambda d : lowess(d, Dens_x, is_sorted = False, frac = 0.2, it = 0)

    plt.figure()
    plt.loglog(filter_func(Dens_y)[:,0], filter_func(Dens_y)[:,1])
    plt.fill_between(filter_func(Dens_y / Dens_errs)[:,0], filter_func(Dens_y / Dens_errs)[:,1], filter_func(Dens_y * Dens_errs)[:,1],  alpha = 0.5,  edgecolor='#1B2ACC', facecolor='#089FFF', linewidth = 2)
    plt.xlabel(r'$ r(pc)$')
    plt.ylabel(r'$ \rho \quad (M_{\odot} pc^{-3})$')
    plt.savefig('../../Results/Monte_Carlo/Density' + sys.argv[1] + '_' + sys.argv[2] + '.pdf')

    filter_func = lambda d : lowess(d, Flucs_x, is_sorted = False, frac = 0.2, it = 0)

    plt.figure()
    plt.loglog(filter_func(Flucs_y)[:,0], filter_func(Flucs_y)[:,1])
    plt.fill_between(filter_func(Flucs_y / Flucs_errs)[:,0], filter_func(Flucs_y / Flucs_errs)[:,1], filter_func(Flucs_y * Flucs_errs)[:,1],  alpha = 0.5,  edgecolor='#1B2ACC', facecolor='#089FFF', linewidth = 2)
    plt.xlabel(r'$ r(pc)$')
    plt.ylabel(r'$ \sqrt{\left < \left (\frac{\partial \phi}{\partial t} \right )^2 \right >} \quad ((km/s)^{2} Myr^{-1})$')
    plt.savefig('../../Results/Monte_Carlo/Flucs' + sys.argv[1] + '_' + sys.argv[2] + '.pdf')

if __name__ == "__main__":
	main()
