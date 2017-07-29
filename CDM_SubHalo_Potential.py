from __future__ import division

import numpy as np
from numpy import pi, log10, sqrt
from shared import *

def Truncate(m, R):
        return MFreeNFW(TidalRadius(m, R), m)

def NumericTruncate(m, R):
    if(R > MpriMax):
        return m
    else:
        return m * pow(R,0.6)/pow(10,3.2)
        
def MHaloNFW(r, M, R):
    Rmax = min(MaxRadius(M), TidalRadius(M, R))
    Rc = MaxRadius(M)/c
    if(r < Rmax):
        return M*(log(1+r/Rc)-r/(r+Rc))/(log(1+c) - c/(1+c))
    else:
        return M*(log(1+Rmax/Rc)-Rmax/(Rmax+Rc))/(log(1+c) - c/(1+c))

def Nhalo(m, R):
    return pow(m, -p)*DFreeNFW(R, Mprimary)*(2-p)*pow(f,p-1)*pow(Mprimary, p-2)

def HaloDensity(R):
    return quad(lambda x: Truncate(x, R) * Nhalo(x, R), 0, f*Mprimary)[0]

def PotChange(m, R, r):
    return MHaloNFW(r, m, R)*G/r**2

def SubHaloTidalForce(r, M, D):
    Rmax = min(MaxRadius(M), TidalRadius(M, D))
    Rc = MaxRadius(M)/c

    if(r < 10**-2):
            return M*G*(k-1)*2/(3*Rc**3)/(log(1+c) - c/(1+c))
    if(r < Rmax):
            return M*G*(k-1)*((2*log(1+r/Rc) - r*(3*r+2*Rc)/(r+Rc)**2)/(log(1+c) - c/(1+c)))/r**3
    else:
            return 2*Truncate(M, D)*G*(k-1)/r**3
    
def TidalVariance(D, N):
    func = lambda m, r: 10**(m+r)*10**(2*r) * Nhalo(10**m, D) * SubHaloTidalForce(10**r, 10**m, D)**2
    M = np.linspace(-2, log10(f*Mprimary), num = N)
    R = np.linspace(0, log10(D/2), num = N)
    Z = np.empty((N, N), dtype=object)
    for i in range(N):
        for j in range(N):
            Z[i,j] = func(M[i], R[j])
     
    ans = trapz2d(Z, M, R)
    return ans*4*pi/(2*MFreeNFW(D, Mprimary)*G/D**3 - 4*pi*G*DFreeNFW(D, Mprimary))**2 * log(10)**2

def Fluc(D, N):
    func = lambda m, r: 10**(m+r)*10**(2*r)  * Nhalo(10**m, D) * PotChange(10**m, D, 10**r)**2
    M = np.linspace(-1, log10(f*Mprimary), num = N)
    R = np.linspace(-1, log10(D/2), num = N)
    Z = np.empty((N, N), dtype=object)
    for i in range(N):
        for j in range(N):
            Z[i,j] = func(M[i], R[j])
     
    ans = trapz2d(Z, M, R)
    return sqrt(ans*4*pi*(-PhiFreeNFW(D, Mprimary)/3) * log(10)**2)    

def NormalizedFluc(D, N):
	return Fluc(D, N)/sqrt(PhiFreeNFW(D, Mprimary)**2 *4*pi*MFreeNFW(D, Mprimary)/D**3)

def NormedFourierMagInt(D, N):
	gravParam = MFreeNFW(D, Mprimary)*G
	phi = -PhiFreeNFW(D, Mprimary)
	#change the power of mass
	func = lambda m, r: 10**(m+r)*10**(2*r)  * Nhalo(10**m, D) * MHaloNFW(10**r, 10**m, D) * G * FourierIntegral((10**r)*sqrt(gravParam/D**3)/sqrt(phi/3))
	M = np.linspace(-1, log10(f*Mprimary), num = N)
	R = np.linspace(-1, log10(D/2), num = N)
	Z = np.empty((N, N), dtype=object)
	for i in range(N):
		for j in range(N):
			Z[i,j] = func(M[i], R[j])
			
	ans = trapz2d(Z, M, R)
	return ans*4*pi/phi * log(10)**2 * sqrt(gravParam/D**3)/sqrt(phi/3)
       
def IntegSpectralPower(D, N):
    gravParam = MFreeNFW(D, Mprimary)*G
    phi = -PhiFreeNFW(D, Mprimary)
    #change the power of mass
    func = lambda m, r: 10**(m+r)*10**(2*r)  * Nhalo(10**m, D) * (MHaloNFW(10**r, 10**m, D))**2 * G * SqFourierIntegral((10**r)*sqrt(gravParam/D**3)/sqrt(phi/3))
    M = np.linspace(-1, log10(f*Mprimary), num = N)
    R = np.linspace(-1, log10(D/2), num = N)
    Z = np.empty((N, N), dtype=object)
    for i in range(N):
        for j in range(N):
            Z[i,j] = func(M[i], R[j])
            
    ans = trapz2d(Z, M, R)
    return ans*4*pi/phi**2 * log(10)**2 * (gravParam/D**3)/(phi/3) * G**2

def FlucWalk(D, N):
    func = lambda m, r: 10**(m+r)*10**(2*r)  * Nhalo(10**m, D) * PotChange(10**m, D, 10**r)**2
    M = np.linspace(-1, log10(f*Mprimary), num = N)
    R = np.linspace(-1, log10(D/2), num = N)
    Z = np.empty((N, N), dtype=object)
    for i in range(N):
        for j in range(N):
            Z[i,j] = func(M[i], R[j])
     
    ans = trapz2d(Z, M, R)
    return sqrt(sqrt(ans*4*pi*(-PhiFreeNFW(D, Mprimary)/3) * log(10)**2)*T_age)
    
def VelocityDispersion(D, N):
    func = lambda m, r: 10**(m+r)*10**(2*r)  * Nhalo(10**m, D) * PotChange(10**m, D, 10**r)**2 * 10**r
    M = np.linspace(-1, log10(f*Mprimary), num = N)
    R = np.linspace(-1, log10(D/2), num = N)
    Z = np.empty((N, N), dtype=object)
    for i in range(N):
        for j in range(N):
            Z[i,j] = func(M[i], R[j])
     
    ans = trapz2d(Z, M, R)
    return sqrt(ans*4*pi/sqrt(-PhiFreeNFW(D, Mprimary)/3) * log(10)**2*T_age)        

    
def FlucWalkMassDownTo(D, M, N):
    func = lambda m, r: 10**(m+r)*10**(2*r)  * Nhalo(10**m, D) * PotChange(10**m, D, 10**r)**2
    M = np.linspace(log10(f*Mprimary-M), log10(f*Mprimary), num = N)
    R = np.linspace(-1, log10(D/2), num = N)
    Z = np.empty((N, N), dtype=object)
    for i in range(N):
        for j in range(N):
            Z[i,j] = func(M[i], R[j])
     
    ans = trapz2d(Z, M, R)
    return sqrt(sqrt(ans*4*pi*(-PhiFreeNFW(D, Mprimary)/3) * log(10)**2)*T_age)
    
def VelocityDispersionMassDownTo(D, M, N):
    func = lambda m, r: 10**(m+r)*10**(2*r)  * Nhalo(10**m, D) * PotChange(10**m, D, 10**r)**2 * 10**r
    M = np.linspace(log10(f*Mprimary-M), log10(f*Mprimary), num = N)
    R = np.linspace(-1, log10(D/2), num = N)
    Z = np.empty((N, N), dtype=object)
    for i in range(N):
        for j in range(N):
            Z[i,j] = func(M[i], R[j])
     
    ans = trapz2d(Z, M, R)
    return sqrt(ans*4*pi/sqrt(-PhiFreeNFW(D, Mprimary)/3) * log(10)**2*T_age)        
