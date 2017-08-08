#Schrodinger 1D equation solved by finding eigenvectors of the hamiltonian
from __future__ import division
import numpy as np
import scipy
import scipy.integrate
import scipy.linalg
import scipy.misc
import scipy.special
from matplotlib import pyplot as plt
from scipy import integrate
from matplotlib import animation

l = 100
N = 2**10
Large_val = 10000000

def f_M(u):
	return (np.arctan(u) * 2/np.pi + 1/(3465*np.pi/2*(1+u**2)**7)*(3465*u**13 + 23100*u**11 + 65373*u**9 + 101376*u**7 + 92323*u**5 + 48580*u**3 - 3465*u))/u**2

def v(x):
	u = x/4.13337988676
	ans, err = integrate.quad(f_M, u, np.inf)
	return -ans
	
X, dx = np.linspace(0.01, l, N, retstep=True)

H = np.diag(map(v, X))
H[0,0] = Large_val

for i in range(N):
    H[i,i] += 2/dx**2
    if(i > 0):
        H[i-1, i] -= 1/dx**2
    if i<N-1:
        H[i+1, i] -= 1/dx**2

num_energies = 10
E, psi_E = scipy.linalg.eigh(H)
#plt.plot(np.linspace(0,N,N)[:num_energies], E[:num_energies], "bo", label="Calculated energies")
#plt.title("Energies for the first" +  " $" + str(num_energies) + "$ states")
#plt.xlabel("State index")
#plt.ylabel("Energy")
#plt.legend(loc='lower right')
#plt.show()

psi = []

for n in range(10):
	psi.append(psi_E[:,n]/X)
	norm = np.sqrt(scipy.integrate.simps(psi[n]**2, X))
	psi[n]/=norm
	plt.figure()
	plt.loglog(X, psi[n]**2, label="$n=%i$" % n)
	plt.title("eigenstate #" + str(n))
	plt.xlabel("$r$ position")
	plt.ylabel("Normalized amplitude $\psi^2$")
	plt.loglog(X, map(lambda x: psi[n][1]**2/(1+(x/4.13337988676)**2)**8, X), label = "soliton fit")
	plt.grid()
	plt.legend(loc='lower left')
	plt.xlim(0.1, l)
	plt.savefig("SolitonEigenvector" + str(n) + ".png")
	plt.close()
   
lin_comb = [1, 0, 0, 0, 0, 0, 1, 1]

def phi_sum(t):
	s = np.zeros(N)
	for i in range(len(lin_comb)):
		for j in range(len(lin_comb)):
			s += lin_comb[i]*lin_comb[j]*psi[i]*psi[j]*np.cos((E[i]-E[j])*t)
	return s 

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0.15, l), ylim=(10**-6, 1))
line, = ax.loglog(X, phi_sum(0))

# initialization function: plot the background of each frame
def init():
    line.set_data(X, phi_sum(0))
    return line,

# animation function.  This is called sequentially
def animate(i):
    x = X
    y = phi_sum(i*0.1)
    line.set_data(x, y)
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames = 10**4, interval=10, blit=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()
