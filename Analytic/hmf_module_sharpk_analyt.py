import numpy as np
import pylab as pl
import cosmolopy
import scipy.integrate as si
#import cosmolopy.perturbation

rho_c=2.7755536e11  #in units of h^2 M_sum Mpc^-3
cosmo={'N_nu': 0, 'h': 0.6933, 'omega_M_0': 0.288, 'omega_b_0': 0.045, 'omega_lambda_0': 0.712, 'omega_n_0': 0, 'n': 0.971, 'sigma_8': 0.830, 'm22':1e12}
#cosmo={'N_nu': 0, 'h': 0.6933, 'omega_M_0': 0.288, 'omega_b_0': 0.045, 'omega_lambda_0': 0.712, 'omega_n_0': 0, 'n': 0.971, 'sigma_8': 0.830*0.0035412141/0.3, 'm22':1e12}
#k=np.logspace(-4,4,100)
deltac=1.686

def transfunc(k,**kwargs):
    return cosmolopy.perturbation.transfer_function_EH(k*kwargs['h'],**kwargs)[0]

def unnorm_pspec(z,k,**kwargs):
    return k**kwargs['n'] 
    #return k**kwargs['n']*transfunc(k,**kwargs)**2*cosmolopy.perturbation.fgrowth(float(z),kwargs['omega_M_0'])**2

#def unnorm_pspec(z,k,**kwargs):
#    return k**3

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def dlnsigmabydlnm(R_arr,sig,z=0.,smooth_n=5,**kwargs):
    pp=cosmolopy.perturbation.power_spectrum(1./R_arr,z,**kwargs)
    return pp/(12*np.pi**2*R_arr**3*sig**2)

def dlnsigmabydlnm2(R_arr,sig,pp):
    #pp=cosmolopy.perturbation.power_spectrum(1./R_arr,z,**kwargs)
    return pp/(12*np.pi**2*R_arr**3*sig**2)

def fitfunc_ST(sig,deltac=deltac):
    A=0.3222
    a=0.707
    p=0.3
    func=A*np.sqrt(2*a/np.pi)*(1+(sig**2/(a*deltac**2))**p)*(deltac/sig)*np.exp(-a*deltac**2/(2*sig**2))
    return func

def fitfunc_PS(sig,deltac=deltac):
    return np.sqrt(2/np.pi)*(deltac/sig)*np.exp(-0.5*(deltac/sig)**2)

def fitfunc_PS_fdm(sig,R,z=0):
    deltaccc=deltac/deltacc(np.pi/R,z)[0]
    return np.sqrt(2/np.pi)*(deltaccc/sig)*np.exp(-0.5*(deltaccc/sig)**2)

def hmf(R_arr,sig,fit,z=0.,deltac=deltac,**kwargs):
    if fit=='PS':
        return (3./(4*np.pi*R_arr**3))*fitfunc_PS(sig,deltac)*dlnsigmabydlnm(R_arr,sig,z,**kwargs)
    elif fit=='ST':
        return (3./(4*np.pi*R_arr**3))*fitfunc_ST(sig,deltac)*dlnsigmabydlnm(R_arr,sig,z,**kwargs)

def hmf_fdm(R_arr,R,sig,fit,**kwargs):
    if fit=='PS':
        return (3./(4*np.pi*R_arr**3))*fitfunc_PS_fdm(sig,R)*dlnsigmabydlnm(R_arr,sig,z,**kwargs)
    elif fit=='ST':
        return (3./(4*np.pi*R_arr**3))*fitfunc_ST_fdm(sig,R)*dlnsigmabydlnm(R_arr,sig,z,**kwargs)

#def W(x):
#    return 3*(np.sin(x)-x*np.cos(x))/(x**3)

#def W(x):
#    return 4.*np.exp(-0.5*x**2)-3.*np.exp(-1.5*x**2)

#def W(x):
#    return a*np.exp(-0.5*alpha*x**2)-(a-1.)*np.exp(-0.5*beta*x**2)

def W(k, R):
    return np.where(k<1/R,1.,0.)

#def integrand(k,R):
#    return k**2*cosmolopy.perturbation.power_spectrum(k,0,**cosmo)*(W(k*R)**2)/(2*np.pi**2)

def integrand(k,R,pspec):
    return k**2*pspec*(W(k,R)**2)/(2*np.pi**2)

def sig2_num(k,R,pspec):
    return np.sqrt(si.simps(integrand(k,R,pspec),k))

def hmf_from_pspec_b(k,pspec,**cosmo):
    """This takes k in Mpc^-1 and returns Mass array (in M_sun) and the baryon mass function dn/dlnM (Mpc^-3)"""
    R=np.logspace(-3,2,100)
    M=4*np.pi*R**3*rho_c*cosmo['omega_b_0']*cosmo['h']**2/3.
    sig=signumvec(k,R,pspec,**cosmo)
    hmf_ST=hmf(M,sig,'ST',**cosmo)
    return M, hmf_ST
    
def hmf_from_pspec_dm(k,pspec,**cosmo):
    """This takes k in Mpc^-1 and returns Mass array (in M_sun) and the dark matter halo mass function dn/dlnM (Mpc^-3)"""
    R=np.logspace(-3,2,100)
    M=4*np.pi*R**3*rho_c*cosmo['omega_M_0']*cosmo['h']**2/3.
    sig=signumvec(k,R,pspec,**cosmo)
    hmf_ST=hmf(M,sig,'ST',**cosmo)
    return M, hmf_ST

def signumvec(k,R,pspec,**cosmo):
    return np.array([sig2_num(k,i,pspec) for i in R])*(cosmo['sigma_8']/sig2_num(k,8./cosmo['h'],pspec))

def signumvec_unnorm(k,R,pspec):
    return np.array([sig2_num(k,i,pspec) for i in R])

def fdm_evo(k,a,m=1e-22):
    """k should be an array with values in units of Mpc^-1"""
    H0=69.77*u.km/u.s/u.Mpc
    x=np.array((hbar*(k/u.Mpc)**2*c**2/((m*u.eV)*H0*np.sqrt(a))).decompose())
    y=np.abs(((3-x**2)*np.cos(x)+3*x*np.sin(x))/x**2)
    yy=np.where(y<1,1.,y)
    return yy

def deltacc(k,z=0.):
    x = fdm_evo(k,1./(1+z))*fdm_evo(0.002,1./1101)/(fdm_evo(k,1./1101)*fdm_evo(0.002,1/(1+z)))
    return np.exp(np.convolve(np.log(x),np.ones(5)/5.,'same'))

##dsq=norm_power(**cosmo)
##cosmo['deltasq']=dsq
#
#R=np.logspace(-3,2,100)
#M=4*np.pi*R**3*rho_c*cosmo['omega_M_0']/3.
##sig=cosmolopy.perturbation.sigma_r(R/cosmo['h'],0,**cosmo)[0]
#sig=signumvec(R/cosmo['h'])
#hmf_ST=hmf(M,sig,'ST',**cosmo)
#hmf_PS=hmf(M,sig,'PS',**cosmo)

