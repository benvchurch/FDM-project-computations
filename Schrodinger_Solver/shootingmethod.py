#Schrodinger 1D equation solved in an infinite potential well by Numerov's algorithm and the shooting method
#Not generalized to any potential yet as I'm not sure about the boundary conditions.
from __future__ import division
import numpy as np
import pylab as plt


hbar = 1
m = 1

xmin = -1
xmax = 1
N = 10000
Tolerance = 1e-3

def v(x):
    return 1/2*(x*10)**2
def f(x, E):
    return (-E+v(x))*2*m/(hbar*hbar)


X, dx = np.linspace(xmin, xmax, N, retstep=True)

def Boundary(E):
    Psi = np.zeros(N)
    Psi[1]=1
    F=f(X,E)

    for i in range(1,N-1):
        Psi[i+1]=(2*Psi[i]*(1-dx*dx*F[i]/12) - Psi[i-1]*(1-dx*dx*F[i-1]/12) +dx*dx*F[i]*Psi[i])/(1-dx*dx*F[i+1]/12)
    return Psi[-1]

def Wavefunction(E):
    Psi = np.zeros(N)
    Psi[1]=1
    F=f(X,E)

    for i in range(1,N-1):
        Psi[i+1]=(2*Psi[i]*(1-dx*dx*F[i]/12) - Psi[i-1]*(1-dx*dx*F[i-1]/12) +dx*dx*F[i]*Psi[i])/(1-dx*dx*F[i+1]/12)
    return Psi


Energies = [(10,20)]
for E in Energies:
    E1, E2 = E

    B1 = Boundary(E1)
    B2 = Boundary(E2)
    accuracy = min((abs(B1), abs(B2)))
    print(B1, B2, accuracy)
    if(B1 > B2):
        Ebuffer = E1
        E1 = E2
        E2 = Ebuffer
    while(accuracy > Tolerance):
        Emid = (E1 + E2)/2
        Bmid = Boundary(Emid)
        if(Bmid > 0):
            #print("Overshooting")
            E2 = Emid
        if(Bmid < 0):
            #print("Undershooting")
            E1 = Emid
        if(Bmid == 0):
            print("WHY IS THIS HAPPENING WITH FLOATS")
        accuracy = min((abs(Boundary(E1)), abs(Boundary(E2))))
        print(E1, E2, accuracy)
    print(Emid)
    Psi = Wavefunction(Emid)
    plt.plot(X,Psi, 'k-', label="$\Psi$, E=%.2f" % round(Emid,2))
    plt.legend()
    plt.grid()
    plt.savefig("%.2fNumerovAlgorithm.png" % round(Emid,2),)
