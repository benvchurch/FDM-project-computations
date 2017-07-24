#Schrodinger 1D equation solved by finding eigenvectors of the hamiltonian
from __future__ import division
import numpy as np
import scipy
import scipy.integrate
import scipy.linalg
import scipy.misc
import scipy.special
import pylab as plt

l = 100
N = 2048
Large_val = 10000000

def v(x):
	return -1/x
	
X, dx = np.linspace(0.001, l, N, retstep=True)

H = np.diag(v(X))
H[0,0] = Large_val

for i in range(N):
    H[i,i] += 2/dx**2
    if(i > 0):
        H[i-1, i] -= 1/dx**2
    if i<N-1:
        H[i+1, i] -= 1/dx**2

num_energies = 10
E, psi_E = scipy.linalg.eigh(H)
plt.plot(np.linspace(0,N,N)[:num_energies], E[:num_energies], "bo", label="Calculated energies")
plt.title("Energies for the first" +  " $" + str(num_energies) + "$ states")
plt.xlabel("State index")
plt.ylabel("Energy")
plt.legend()
plt.show()

for n in range(10):
    psi = psi_E[:,n]/X
    norm = np.sqrt(scipy.integrate.simps(psi**2, X))
    psi/=norm
    plt.plot(X, psi, label="$n=%i$" % n)
    plt.title("wavefunction #" + str(n))
    plt.xlabel("$x$ position")
    plt.ylabel("Normalized wavefunction amplitude $\psi$")
    plt.grid()
    plt.legend()
    #plt.savefig(str(n) + "harmonicOscillatorEigenvectors.png")
    plt.show()
