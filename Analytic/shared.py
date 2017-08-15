from __future__ import division
import numpy as np
from mpmath import meijerg
from scipy import special, interpolate
from numpy import log, exp, sin ,cos, pi, log10, sqrt

crit_density = 1.3211775*10**-7; 
f = 0.1;
p = 1.9;
c = 10.0;
G = 0.0045;
k = 2;
Mprimary = 10**12;
T_age = 10**4

def MaxRadius(M):
    return pow(3*M/(4 * pi * 200 * crit_density), 1/3)

def DFreeNFW(r, M):
    Rmax = MaxRadius(M)
    Rc = Rmax/c
    if(r < Rmax):
        return 200/3.0 * crit_density / (log(1+c) - c/(1+c))*c**3 * 1/(r/Rc*(1+r/Rc)**2)
    else:
        return 0

def MFreeNFW(r, M):
    Rmax = MaxRadius(M)
    Rc = Rmax/c
    if(r < Rmax):
        return M*(log(1+r/Rc)-r/(r+Rc))/(log(1+c) - c/(1+c))
    else:
        return M

def PhiFreeNFW(r, M):
    Rmax = MaxRadius(M)
    Rc = Rmax/c
    if(r < Rmax):
        return -M*G*((Rmax/r * log(1+r/Rc) - log(1+c))/(log(1+c) - c/(1+c)) + 1)/Rmax
    else:
        return -M*G/r

def TidalRadius(m, R):
    Rt = R*pow(m/(2*MFreeNFW(R, Mprimary)), 1/3)
    return Rt
      
def FourierF(k):
	return sqrt(2/pi) * special.kv(0, abs(k))
	
def FourierIntegral(k):
	return sqrt(pi/2) * (1/k - special.kv(0, k) * special.modstruve(-1,k) - special.kv(0,k) * special.modstruve(0,k))	    

interpolation_NUM = 1000

interpolation_points_x = np.logspace(-10, 1, interpolation_NUM)
interpolation_points_y = map(lambda k : 1/sqrt(pi) * (5.568327996831708 - float(meijerg([[1],[1]],[[1/2,1/2,1/2], [0]],k,1/2)))/k, interpolation_points_x)  
interpolated_function = interpolate.interp1d(interpolation_points_x, interpolation_points_y, fill_value = 'extrapolate')

def SqFourierIntegral(k):
	if(k < 10):
		return interpolated_function(k)
	else:
		return 0


#from matplotlib import pyplot as plt
#plt.loglog(interpolation_points_x, interpolation_points_y)
#plt.show()

#x_new = np.logspace(-10, 2, 10**5)
#plt.loglog(x_new, interpolated_function(x_new))
#plt.loglog(x_new, map(SqFourierIntegral, x_new))
#plt.show()
  
def trapz2d(z, x = None,y = None):
    ''' Integrates a regularly spaced 2D grid using the composite trapezium rule. 
    IN:
       z : 2D array
       x : (optional) grid values for x (1D array)
       y : (optional) grid values for y (1D array)
       dx: if x is not supplied, set it to the x grid interval
       dy: if y is not supplied, set it to the x grid interval
    '''
    
    sum = np.sum
    dx = (x[-1]-x[0])/(np.shape(x)[0]-1)
    dy = (y[-1]-y[0])/(np.shape(y)[0]-1)    
    
    s1 = z[0,0] + z[-1,0] + z[0,-1] + z[-1,-1]
    s2 = sum(z[1:-1,0]) + sum(z[1:-1,-1]) + sum(z[0,1:-1]) + sum(z[-1,1:-1])
    s3 = sum(z[1:-1,1:-1])
    
    return 0.25*dx*dy*(s1 + 2*s2 + 4*s3)

