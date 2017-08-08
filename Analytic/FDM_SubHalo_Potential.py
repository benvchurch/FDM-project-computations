from __future__ import division

import numpy as np
from numpy import pi, log10, sqrt
from shared import *

import cosmolopy
import hmf_module_sharpk as modhmf
import hmf_module_sharpk_analyt as modhmf_a

m22 = 1
ksol = 2.2944716 * 10**12

def cutoff(axion_mass):
    return 10**8.7 * pow(axion_mass, -3/2)

def SolitonMass(M):
    return 2.7 * 10**8 / m22 * pow(M/10**10, 1/3)
    
def SolitonRadius(M):
    return 33*pi**2/1024 * ksol/ SolitonMass(M) * m22**-2
    
def SolitonHalfRadius(M):
	return SolitonRadius(M)*sqrt(pow(2,1/8)-1)

def SolitonProfile(r, M):
    Msol = SolitonMass(M)
    R = SolitonRadius(M)
    return ksol * m22**-2 * (1/R)**4/(1+(r/R)**2)**8

def SolitonMassProfile(r, M):
    Msol = SolitonMass(M)
    R = SolitonRadius(M)
    u = r/R
    return Msol * np.arctan(u) * 2/pi + Msol/(3465*pi/2*(1+u**2)**7)*(3465*u**13 + 23100*u**11 + 65373*u**9 + 101376*u**7 + 92323*u**5 + 48580*u**3 - 3465*u)    
        

def RhoFree(r, M):
    R12 = SolitonHalfRadius(M)
    if(r < MaxRadius(M)):
        if(r < R12):
            return SolitonProfile(r, M) + DFreeNFW(r, M)
        else:
            return SolitonProfile(r, M) + DFreeNFW(R12, M)
    else:
        return 0
        
def MFree(r, M):
    R12 = SolitonHalfRadius(M)
    Rmax = MaxRadius(M)
    if(r < R12):
        return SolitonMassProfile(r, M) + 4/3*pi*r**3 * DFreeNFW(R12, M)
    elif(r < Rmax):
        return SolitonMassProfile(r, M) + MFreeNFW(r, M) - MFreeNFW(R12, M) + 4/3*pi*R12**3*DFreeNFW(R12, M) 
    else:
        return SolitonMassProfile(Rmax, M) + MFreeNFW(Rmax, M) - MFreeNFW(R12, M) + 4/3*pi*R12**3*DFreeNFW(R12, M)
        
def Truncate(m, R):
        return MFree(TidalRadius(m, R), m)

def NumericTruncate(m, R):
    if(R > MpriMax):
        return m
    else:
        return m * pow(R,0.6)/pow(10,3.2)

def MSubHalo(r, M, D):
    Rt = TidalRadius(M, D)
    if(Rt < SolitonHalfRadius(M)):
		return 0
    if(r < Rt):
        return MFree(r, M)
    else:
        return MFree(Rt, M)

def TidalLimit(M):
    Msol = SolitonMass(M)
    R = SolitonRadius(M)
    R12 = SolitonHalfRadius(M)
    return -2*Msol*G/R**3 * (48580/3465 - 1/3 + 7) * 2/pi + 4/3*pi*DFreeNFW(R12, M)*2*G - 4*pi*G*RhoFree(0,M)
    
def SubHaloTidalForce(r, M, D):
	Rt = TidalRadius(M, D)
	if(Rt < SolitonHalfRadius(M)):
		return 0
	if(r < 10**-2):
		return TidalLimit(M)
	else:   
		if(r < Rt):
			return 2*G*MSubHalo(r, M, D)/r**3 - 4*pi*RhoFree(r, M)
		else:
			return 2*G*MSubHalo(r, M, D)/r**3  

HMF_Memoized = dict()


def HMF_FDM(M):
	if(not ((M, m22) in HMF_Memoized)):
		rho_c = 2.7755536e11 #msol/Mpc^-3 h^2
		cosmo = {'N_nu': 0, 'h': 0.696, 'omega_M_0': 0.284, 'omega_b_0': 0.045, 'omega_lambda_0': 0.716, 'omega_n_0': 0, 'n': 0.962, 'sigma_8': 0.818, 'm22':m22}
		deltac = 1.686
		k = np.logspace(-3,4,4000)
		R = np.linspace(1, pow(M/(4*np.pi*rho_c*cosmo['omega_M_0']*cosmo['h']**2/3.0),1/3.0),2)
		Rks = R/2.5
		Mks = 4*np.pi*Rks**3*rho_c*cosmo['omega_M_0']*cosmo['h']**2/3.0
		z = 0.
		ppp = cosmolopy.perturbation.power_spectrum(k,z,**cosmo)
		sigg = modhmf.signumvec_unnorm(k,Rks,ppp)
		ddd = modhmf_a.dlnsigmabydlnm(Rks,sigg,z,**cosmo)
		HMF_Memoized[(M, m22)] =  (3*modhmf.fitfunc_ST(sigg,deltac*1.195)*ddd/(4*np.pi*R**3))[1]/M
	return HMF_Memoized[(M, m22)]


def NhaloCDM(m, R):
    return pow(m, -p)*DFreeNFW(R, Mprimary)*(2-p)*pow(f,p-1)*pow(Mprimary, p-2)


def Nhalo(m, R):
	return DFreeNFW(R, Mprimary)*(2-p)*pow(f,p-1)*pow(Mprimary, -2)*HMF_FDM(m)/HMF_FDM(Mprimary)


def HaloDensity(R):
    return quad(lambda x: Truncate(x, R) * Nhalo(x, R), 0, f*Mprimary)[0]

def PotChange(m, R, r):
    return MSubHalo(r, m, R)*G/r**2

def TidalVariance(D, N):
    func = lambda m, r: 10**(m+r)*10**(2*r) * Nhalo(10**m, D) * SubHaloTidalForce(10**r, 10**m, D)**2
    M = np.linspace(log10(cutoff(m22)), log10(f*Mprimary), num = N)
    R = np.linspace(0, log10(D/2), num = N)
    Z = np.empty((N, N), dtype=object)
    for i in range(N):
        for j in range(N):
            Z[i,j] = func(M[i], R[j])
     
    ans = trapz2d(Z, M, R)
    return ans*4*pi/(2*MFreeNFW(D, Mprimary)*G/D**3 - 4*pi*G*DFreeNFW(D, Mprimary))**2 * log(10)**2

def Fluc(D, N):
    func = lambda m, r: 10**(m+r)*10**(2*r)  * Nhalo(10**m, D) * PotChange(10**m, D, 10**r)**2
    M = np.linspace(log10(cutoff(m22)), log10(f*Mprimary), num = N)
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
    func = lambda m, r: 10**(m+r)*10**(2*r)  * Nhalo(10**m, D) * MSubHalo(10**r, 10**m, D) *G * FourierIntegral((10**r)*sqrt(gravParam/D**3)/sqrt(phi/3))
    M = np.linspace(log10(cutoff(m22)), log10(f*Mprimary), num = N)
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
    func = lambda m, r: 10**(m+r)*10**(2*r)  * Nhalo(10**m, D) * (MSubHalo(10**r, 10**m, D))**2 * SqFourierIntegral((10**r)*sqrt(gravParam/D**3)/sqrt(phi/3))
    M = np.linspace(log10(cutoff(m22)), log10(f*Mprimary), num = N)
    R = np.linspace(-1, log10(D/2), num = N)
    Z = np.empty((N, N), dtype=object)
    for i in range(N):
        for j in range(N):
            Z[i,j] = func(M[i], R[j])
            
    ans = trapz2d(Z, M, R)
    return sqrt(ans*4*pi/phi**2 * log(10)**2 * (gravParam/D**3)/(phi/3) * G**2)

def FlucWalk(D, N):
    func = lambda m, r: 10**(m+r)*10**(2*r)  * Nhalo(10**m, D) * PotChange(10**m, D, 10**r)**2
    M = np.linspace(log10(cutoff(m22)), log10(f*Mprimary), num = N)
    R = np.linspace(-1, log10(D/2), num = N)
    Z = np.empty((N, N), dtype=object)
    for i in range(N):
        for j in range(N):
            Z[i,j] = func(M[i], R[j])
     
    ans = trapz2d(Z, M, R)
    return sqrt(sqrt(ans*4*pi*(-PhiFreeNFW(D, Mprimary)/3) * log(10)**2)*T_age)
    
def VelocityDispersion(D, N):
    func = lambda m, r: 10**(m+r)*10**(2*r)  * Nhalo(10**m, D) * PotChange(10**m, D, 10**r)**2 * 10**r
    M = np.linspace(log10(cutoff(m22)), log10(f*Mprimary), num = N)
    R = np.linspace(-1, log10(D/2), num = N)
    Z = np.empty((N, N), dtype=object)
    for i in range(N):
        for j in range(N):
            Z[i,j] = func(M[i], R[j])
     
    ans = trapz2d(Z, M, R)
    return sqrt(ans*4*pi/sqrt(-PhiFreeNFW(D, Mprimary)/3) * log(10)**2*T_age)        

    

