import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import matplotlib.animation
from matplotlib.animation import FuncAnimation
import time 
from scipy.integrate import ode

t0 = time.time()

########## Set Parameters ##########
Chebyshev_order = 50 # Order of the Chebyshev polynormials and the Bessel functions
no_site = 300  #1335 # Number of molecules (grids)
no_timestep = 50  # Number of total timestep
delt_t = 33.073157824 # Size of timestep in a.u. 0.4134144728d0(Jiffy)=0.01 fs
h_bar = 1.0 # Reduce Plank constant in a.u.


mass = 1.0 # Mass of particle: mass of the electron in a.u. = 9.10939*10^(-31) [kg]
grid_size=1.0 #0.188973 Grid size in a.u. (Bohr radius, a0)= 0.1[A]
Coupling_energy=0.0036749843813164 # 0.1 eV
Electric_field=0.000036749843813164 # 0.001 eV/grid
ormaga=0.024 # Frequency of harmonic oscillator in a.u.(1/Jiffy)= 1.0 [1/fs]


########## Initialize Wave Function and Probability ##########
##### Chebyshev #####
wave = np.zeros((no_timestep, no_site), dtype=complex)
wave[0, 149] = 1.0
probability = np.zeros((no_timestep, no_site))
probability[0, 149] = 1.0

##### Exact Solution #####
wave_exact = np.zeros((no_timestep, no_site), dtype=complex)
wave_exact[0, 149] = 1.0
probability_exact = np.zeros((no_timestep, no_site))
probability_exact[0, 149] = 1.0

##### 4th-Order Runge-Kutta #####
wave_RK4 = np.zeros((no_timestep, no_site), dtype=complex)
wave_RK4[0, 149] = 1.0
probability_RK4 = np.zeros((no_timestep, no_site))
probability_RK4[0, 149] = 1.0

##### Scipy ODE Solver #####
wave_ODE = np.zeros((no_timestep, no_site), dtype=complex)
wave_ODE[0, 149] = 1.0
ode_wave = np.zeros((no_site), dtype=complex)
ini_wave = np.zeros((no_site), dtype=complex)
ini_wave[149] = 1.0
probability_ODE = np.zeros((no_timestep, no_site))
probability_ODE[0, 149] = 1.0



########## Define Identity Operator Imaginary Factor, the Phase Factor & Bessel Function ##########
##### Identity operator #####
I_oper = np.zeros((no_site, no_site), dtype= complex)
for i in range(no_site):
  I_oper[i,i] = 1

##### Imaginary factor: i #####
Im_factor = complex(0, 1.0)

##### Chebyshev: Maximal and minimal energy #####
E_max = 0.00734956855769941  #max(real(Potential_op(:,:)))+(0.wave5d0*pi*pi/(mass*grid_size*grid_size))
E_min= -0.00734956855769941  #min(real(Potential_op(:,:)))
delta_E=E_max-E_min
b=0.5*delta_E
a=0.5*(E_min+E_max)

##### Chebyshev phase factor: exp(i*theta)=cos(theta)+i*sin(theta) #####
theta=(-a*delt_t)/h_bar
phase_factor = complex(np.cos(theta),+np.sin(theta))

##### Chebyshev: Variable of Bessel function #####
Bessel_variable = -(b*delt_t)/h_bar

##### Chebyshev: Bessel function of the first kind #####
Bessel_function = sp.jv(np.arange(Chebyshev_order), Bessel_variable)
#print(Bessel_function)


########## Define Hamiltonian Operators (combine with ML in the future) ##########
H = np.zeros((no_site, no_site), dtype= complex)

##### Offdiagonal elements: electronic coupling #####
for i in range(no_site-1):
  H[i,i+1] = Coupling_energy

for i in range(1, no_site):
  H[i,(i-1)] = Coupling_energy
  
##### Normalize Hamiltonian #####
"""
This step normalizes the Hamiltonian operator on 667the interval [-1,1]
We have to check whether the normalization of Hamiltonian needs to be implemented every timestep 
"""
H_norm = (H/b)-(a/b)*I_oper

##### Exact Solution: Diagonize Hamiltonian #####
eigen_w, eigen_v = np.linalg.eig(H)
idx = np.argsort(eigen_w)
eigen_w = eigen_w[idx]
eigen_v = eigen_v[:,idx]

########## Exact Solution: Calculate the Coeffiecient of each Eigenstate ##########
coeff = np.zeros(no_site, dtype= complex)
check_wave = np.zeros(no_site, dtype= complex)

for i in range(no_site):
    coeff[i] = np.dot(eigen_v[:,i], wave[0, :].T)



########## Scipy ODE Solver: define ordinary differential equation #########
def dY_dt(q, ode_wave):
  dydt = (-Im_factor/h_bar)*np.dot(H, ode_wave[:]) 
  return dydt

########## Scipy ODE Solver: setup ODE method  ##########
t0=0
r = ode(dY_dt).set_integrator('zvode', method='bdf') 
r.set_initial_value(ini_wave, t0)

########## Time Evolution ##########
for t in range(no_timestep-1):
  print('Timestep: %s' %(t+1))


  ##### Time propagation: exact solution #####
  for i in range(no_site):
    theta_exact = -eigen_w[i]*(t+1)*delt_t/h_bar
    phase_factor_exact = complex(np.cos(theta_exact),+np.sin(theta_exact))
    wave_exact[(t+1), :] += phase_factor_exact*coeff[i]*eigen_v[:,i] 

  ##### Normalize wave function at time t+1 #####  
  normalization = np.sum(abs(wave_exact[(t+1), :])*grid_size)
  probability_exact[(t+1), :] = (wave_exact.conjugate()[(t+1), :]*wave_exact[(t+1), :])*grid_size



  ##### Time propagation: exact solution #####
  r.integrate(r.t+delt_t)
  wave_ODE[(t+1), :] = r.y
  ##### Normalize wave function at time t+1 #####  
  normalization = np.sum(abs(wave_ODE[(t+1), :])*grid_size)
  probability_ODE[(t+1), :] = (wave_ODE.conjugate()[(t+1), :]*wave_ODE[(t+1), :])*grid_size



  ##### Time propagation: 4th-Order Runge-Kutta #####    
  RK4_vector = np.zeros((4, no_site), dtype= complex)
  mutiply_factor = (-Im_factor*delt_t)/h_bar

  ### K_1 ###
  RK4_vector[0, :] = np.dot(H, wave[t,:])
  RK4_vector[0, :] = mutiply_factor*RK4_vector[0, :]

  ### K_2 ###
  RK4_vector[1, :] = np.dot(H, (wave[t,:]+0.5*RK4_vector[0, :]))
  RK4_vector[1, :] = mutiply_factor*RK4_vector[1, :]

  ### K_3 ###
  RK4_vector[2, :] = np.dot(H, (wave[t,:]+0.5*RK4_vector[1, :]))
  RK4_vector[2, :] = mutiply_factor*RK4_vector[2, :]

  ### K_4 ###
  RK4_vector[3, :] = np.dot(H, (wave[t,:]+RK4_vector[2, :]))
  RK4_vector[3, :] = mutiply_factor*RK4_vector[3, :]

  wave_RK4[(t+1), :] = wave_RK4[t, :] + (RK4_vector[0, :]+2*RK4_vector[1, :]+2*RK4_vector[2, :]+RK4_vector[3, :])/6.0

  ##### Normalize wave function at time t+1 #####  
  normalization_RK4 = np.sum(abs(wave_RK4[(t+1), :])*grid_size)
  probability_RK4[(t+1), :] = (wave_RK4.conjugate()[(t+1), :]*wave_RK4[(t+1), :])*grid_size



  ##### Time propagation: Chebyshev #####  
  Chebyshev_vector = np.zeros((Chebyshev_order, no_site), dtype= complex)
  ### T_0 ###
  Chebyshev_vector[0, :] = np.dot(I_oper, wave[t,:])
  mutiply_factor = 1.0*(Im_factor**0)*Bessel_function[0]
  wave[(t+1), :] = wave[(t+1), :] + mutiply_factor*Chebyshev_vector[0, :]

  ### T_1 ###
  Chebyshev_vector[1, :] = np.dot(H_norm, wave[t,:])
  mutiply_factor = 2.0*(Im_factor**1)*Bessel_function[1]
  wave[(t+1), :] = wave[(t+1), :] + mutiply_factor*Chebyshev_vector[1, :]
  
  ### T_n ###
  for n in range(2, Chebyshev_order):
    Chebyshev_vector[n, :] = np.dot((2.0*H_norm), Chebyshev_vector[(n-1), :])-Chebyshev_vector[(n-2), :]
    mutiply_factor = 2.0*(Im_factor**n)*Bessel_function[n]
    wave[(t+1), :] = wave[(t+1), :] + mutiply_factor*Chebyshev_vector[n, :]

  wave[(t+1), :] = wave[(t+1), :]*phase_factor
  ##### Normalize wave function at time t+1 #####  
  normalization = np.sum(abs(wave[(t+1), :])*grid_size)
  probability[(t+1), :] = (wave.conjugate()[(t+1), :]*wave[(t+1), :])*grid_size
  #print(probability[(t+1), :].real)
  """
  1. The grid size is a fixed number but in a real molecular system it varies as the distance btw molecules changes 
  2. Check the value of normalization factor, and think about the problem of norm conservation 
  """
  #print(normalization)
  
  
  #print(Chebyshev_vector[:n+1, :])
  #print(mutiply_factor)
  #print(wave.real[(t+1), :])
  #print(wave.imag[(t+1), :])


########## Generate Animation ##########

pos = np.arange(no_site)

"""
fig, ax = plt.subplots()
line, = plt.plot([], [])

def init():
    ax.set_xlim(0,300)
    ax.set_ylim(-0.01,1.0)
    return line,

def update(i):
    line.set_data(pos[:], probability[i, :])
    label = 'Time [fs]: {:.2f}'.format(i*0.8)
    line.set_label(label)
    legend = plt.legend()
    return line, legend
"""
fig = plt.figure()
ax = fig.add_axes([0.1, 0.15, 0.8,0.8])

def update(i):
  ax.clear()
  time_label = 'Time [fs]: {:.2f}'.format(i*0.8)
  line, = ax.plot([], [], ' ', label=time_label)
  line, = ax.plot(pos[:], probability[i, :], 'r*', label='Chebyshev')
  line, = ax.plot(pos[:], probability_exact[i, :], 'b+', label='Exact solution')
  line, = ax.plot(pos[:], probability_RK4[i, :], color='g', marker='D', label='4th-order RK')
  line, = ax.plot(pos[:], probability_ODE[i, :], color='k', marker='>', label='Scipy ODE')
  ax.set_xlim(0,300)
  ax.set_ylim(-0.01,0.15)


  ax.legend()  
  #line.set_label(time_label)
  #plt.legend()
  ax.lines

#ani = FuncAnimation(fig=fig, func=update, frames=no_timestep, init_func=init, interval=50, blit=True)
ani = FuncAnimation(fig=fig, func=update, frames=no_timestep, interval=10, blit=False)
writer = matplotlib.animation.PillowWriter()
ani.save("Propagation_Comparison.gif", writer=writer)


# Set up formatting for the movie files
#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=10000)

#ani.save('Propagation_PE.mp4', writer=writer)

#os.system("ffmpeg -i Propagation_PE.mp4 Propagation_PE.gif")

plt.show()


print('Total computing time is %f seconds' %(time.time() - t0))

"""
plt.figure(1)
x = np.linspace(-20, 20, 500)
y0 = sp.jv(0, x)
y1 = sp.jv(1, x)
y2 = sp.jv(2, x)
y3 = sp.jv(3, x)

plt.plot(x, y0, color="red", linewidth=1.0, linestyle="--")
plt.plot(x, y1, color="blue", linewidth=1.0, linestyle="--")
plt.plot(x, y2, color="darkgrey", linewidth=1.0, linestyle="--")
plt.plot(x, y3, color="green", linewidth=1.0, linestyle="--")
"""