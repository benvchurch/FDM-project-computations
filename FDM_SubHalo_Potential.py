from __future__ import division

import numpy as np
from scipy import special
from numpy import log, exp, sin ,cos, pi, log10, sqrt
from scipy.integrate import quad, dblquad, cumtrapz
from matplotlib import pyplot as plt

def trapz2d(z,x = None,y = None):
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


crit_density = 1.5*10**-7 
f = 0.1
p = 1.9
c = 10.0
G = 0.0045
k = 2
Mprimary = 10**12
m22 = 10**3
ksol = 9.1*10**-2

def cuttoff(axion_mass):
	return 10**8.7 * pow(axion_mass, -3/2)

def MaxRadius(M):
    return pow(3*M/(4 * pi * 200 * crit_density), 1/3)
MpriMax = MaxRadius(Mprimary)

def SolitonMass(M):
    return 2.7 * 10**8 / m22 * pow(M/10**10, 1/3)
    
def HalfSolitonRadius(M):
    return 33*pi**2/(1024*pow(ksol, 3/2)) * 1.9*10**10 / SolitonMass(M) * m22**-2

def SolitonProfile(r, M):
    Msol = SolitonMass(M)
    R12 = HalfSolitonRadius(M)
    return (1.9*10**10) * m22**-2 * (1/R12)**4/(1+ksol*(r/R12)**2)**8

def SolitonMassProfile(r, M):
    Msol = SolitonMass(M)
    R = HalfSolitonRadius(M)/sqrt(ksol)
    return Msol * np.arctan(r/R) * 2/pi + Msol/(3465*pi/2*(R**2+r**2)**7)*(3465*R*r**13 + 23100*R**3*r**11 + 65373*R**5*r**9 + 101376*R**7*r**7 + 92323*R**9*r**5 + 48580*R**11*r**3 - 3465*R**13*r)    
        
def NFWProfile(r, M):
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

def RhoFree(r, M):
    R12 = HalfSolitonRadius(M)
    if(r < MaxRadius(M)):
        if(r < R12):
            return SolitonProfile(r, M) + NFWProfile(r, M)
        else:
            return SolitonProfile(r, M) + NFWProfile(R12, M)
    else:
        return 0
        
def MFree(r, M):
    R12 = HalfSolitonRadius(M)
    Rmax = MaxRadius(M)
    if(r < R12):
        return SolitonMassProfile(r, M) + 4/3*pi*r**3 * NFWProfile(R12, M)
    elif(r < Rmax):
        return SolitonMassProfile(r, M) + MFreeNFW(r, M) - MFreeNFW(R12, M) + 4/3*pi*R12**3*NFWProfile(R12, M) 
    else:
		return SolitonMassProfile(Rmax, M) + MFreeNFW(Rmax, M) - MFreeNFW(R12, M) + 4/3*pi*R12**3*NFWProfile(R12, M)

def TidalRadius(m, R):
    Rt = R*pow(m/(2*MFreeNFW(R, Mprimary)), 1/3)
    return Rt
        
def Truncate(m, R):
        return MFreeNFW(TidalRadius(m, R), m)

def NumericTruncate(m, R):
    if(R > MpriMax):
        return m
    else:
        return m * pow(R,0.6)/pow(10,3.2)

def MSubHalo(r, M, D):
    Rt = TidalRadius(M, D)
    if(r < Rt):
		return MFree(r, M)
    else:
        return MFree(Rt, M)

def TidalLimit(M):
    Msol = SolitonMass(M)
    R = HalfSolitonRadius(M)/sqrt(ksol)
    return -2*Msol*G/R**3 * (48580/3465 - 1/3 + 7) * 2/pi + 4/3*pi*NFWProfile(R*sqrt(ksol), M)*2*G - 4*pi*G*RhoFree(0,M)
    
def SubHaloTidalForce(r, M, D):
    if(r < 10**-2):
        return TidalLimit(M)
    else:   
        Rt = TidalRadius(M, D)
        if(r < Rt):
            return 2*G*MSubHalo(r, M, D)/r**3 - 4*pi*RhoFree(r, M)
        else:
            return 2*G*MSubHalo(r, M, D)/r**3  

def Nhalo(m, R):
    return pow(m, -p)*NFWProfile(R, Mprimary)*(2-p)*pow(f,p-1)*pow(Mprimary, p-2)

def HaloDensity(R):
    return quad(lambda x: Truncate(x, R) * Nhalo(x, R), 0, f*Mprimary)[0]

def PotChange(m, R, r):
    return MSubHalo(r, m, R)*G/r**2

def TidalVariance(D, N):
    func = lambda m, r: 10**(m+r)*10**(2*r) * Nhalo(10**m, D) * SubHaloTidalForce(10**r, 10**m, D)**2
    M = np.linspace(log10(cuttoff(m22)), log10(f*Mprimary), num = N)
    R = np.linspace(-2, log10(D/2), num = N)
    Z = np.empty((N, N), dtype=object)
    for i in range(N):
        for j in range(N):
            Z[i,j] = func(M[i], R[j])
     
    ans = trapz2d(Z, M, R)
    return ans*4*pi/(2*MFreeNFW(D, Mprimary)*G/D**3 - 4*pi*G*NFWProfile(D, Mprimary))**2 * log(10)**2

def Fluc(D, N):
    func = lambda m, r: 10**(m+r)*10**(2*r)  * Nhalo(10**m, D) * PotChange(10**m, D, 10**r)**2
    M = np.linspace(log10(cuttoff(m22)), log10(f*Mprimary), num = N)
    R = np.linspace(-1, log10(D/2), num = N)
    Z = np.empty((N, N), dtype=object)
    for i in range(N):
        for j in range(N):
            Z[i,j] = func(M[i], R[j])
     
    ans = trapz2d(Z, M, R)
    return ans*4*pi*(-PhiFreeNFW(D, Mprimary)/3) * log(10)**2    

def NormalizedFluc(D, N):
	return sqrt(Fluc(D, N)/(PhiFreeNFW(D, Mprimary)**2 *4*pi*MFreeNFW(D, Mprimary)/D**3))

def FourierF(k):
    return sqrt(2/pi) * special.kv(0, abs(k))
    
def FourierIntegral(k):
    return pi/2 * (1/k - special.kv(0, k) * special.modstruve(-1,k) - special.kv(0,k) * special.modstruve(0,k)) 

def NormedFourierMagInt(D, N):
    gravParam = MFreeNFW(D, Mprimary)*G
    phi = -PhiFreeNFW(D, Mprimary)
    func = lambda m, r: 10**(m+r)*10**(2*r)  * Nhalo(10**m, D) * MSubHalo(10**r, 10**m, D)*G/sqrt(2*pi) * FourierIntegral((10**r)*sqrt(gravParam/D**3)/sqrt(phi/3))
    M = np.linspace(log10(cuttoff(m22)), log10(f*Mprimary), num = N)
    R = np.linspace(-1, log10(D/2), num = N)
    Z = np.empty((N, N), dtype=object)
    for i in range(N):
        for j in range(N):
            Z[i,j] = func(M[i], R[j])
            
    ans = trapz2d(Z, M, R)
    return ans*4*pi/phi * log(10)**2 * sqrt(gravParam/D**3)/sqrt(phi/3)

def main():
    ep = np.linspace(2,9,10)
    D = np.linspace(0, 5, 40)
    F = map(lambda x: log10(NormalizedFluc(10**2, int(2**x))), ep)
    plt.plot(ep, F)
    plt.xlabel('log(r/pc)')
    plt.ylabel('log(Normalized Potential Fluctuation)')
    plt.show()
    
    
    
if __name__ == "__main__":
    main()

    

