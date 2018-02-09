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
    file_obj.close()

    get_coord = lambda i : np.array(map(lambda x : 10**float(x.split(" ")[i]), lines))
    return map(get_coord, [0, 1, 2])

def main():
    x,y, errs = readData("data_files/" + sys.argv[1])

    filter_func = lambda d : lowess(d, x, is_sorted = False, frac = 0.2, it = 0)

    plt.figure()
    plt.loglog(filter_func(y)[:,0], filter_func(y)[:,1])
    plt.fill_between(filter_func(y / errs)[:,0], filter_func(y / errs)[:,1], filter_func(y * errs)[:,1],  alpha = 0.5,  edgecolor='#1B2ACC', facecolor='#089FFF', linewidth = 2)
    plt.xlabel(r'$ r(pc)$')
    plt.ylabel(r'$ \rho \quad (M_{\odot} pc^{-3})$')
    plt.savefig('../../Results/Monte_Carlo/' + sys.argv[1].split(".")[0])

if __name__ == "__main__":
	main()
