"""
This script propagate the wave function by given the Hamiltonian. 
The unit of time & energy are fs & eV, respectively.
The required input variables are:
    (1) hamiltonian: the Hamiltonian (unit in eV)
    (2) ini_wave: The initial wave function coefficient (1-D array)
    (3) qm_delt_t: Time interval of each QM times step (unit in fs)
    (4) qm_timestep: Number of total timestep in QM region
 
Currently, this script include two algorithms (functions) to propagte a wavefunction:
    (1) wave_propagation.Exact_solution():
        This function diagonizes the Hamiltonian, and time evolution is based on the eigen states.
        The return arrays include:
          (a) wave: wave function at all the time step in QM domain [(qm_timestep+1), no_site]
          (b) probability: the corresponing probability density accroding to the wave function.
          
    (2) wave_propagation.Scipy_ODE_solver():
        This function propagates the wave function by using the ODE solver in SciPy.
        The return arrays include:
          (a) wave: wave function at all the time step in QM domain [(qm_timestep+1), no_site]
          (b) probability: the corresponing probability density accroding to the wave function

It's worth noting that the unit of "qm_delt_t" is femto-second (fs), but the unit of "md_delt_t" is pico-second (ps).
The main program runs a simple OpenMM file to establish "wave_propagatio" class & to test "Scipy_ODE_solver"
Reference: ProgressReport20200213.pptx
"""

import numpy as np
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
from scipy.spatial import distance
from scipy.integrate import ode
from scipy import integrate
from subroutine_Molecule_List import molecule_list
from subroutine_Feature_Generation import generate_CM
from subroutine_ML_Prediction import ml_coupling
from subroutine_Hamiltonian import hamiltonian
import torch
import torch.nn as nn
from torch.autograd import Variable


class wave_propagation(): #Unit: eV
    def __init__(self, hamiltonian, ini_wave, qm_delt_t, qm_timestep):
      self.no_site = hamiltonian.shape[0] # Number of molecules (grids) in quantum mechanics region
      self.qm_timestep = qm_timestep # Number of total timestep in quantum mechanics region
      self.delt_t = qm_delt_t #Time interval of each QM times step: unit in fs (femto-second)
      self.h_bar = 4.135667662/(2*np.pi)  # Reduce Plank constant in in unit of eV.fs
      self.Im_factor = complex(0, 1.0) # Imaginary factor: i
      ### Hamiltonian ###
      self.hamiltonian = np.zeros((self.no_site, self.no_site), dtype= complex)
      self.hamiltonian.real = hamiltonian
      ### Site-based wave function ###
      self.wave = np.zeros(((self.qm_timestep+1), self.no_site), dtype=complex)
      self.wave[0,:] = ini_wave
      ### Probability ###
      self.probability = np.zeros(((self.qm_timestep+1), self.no_site))
      self.probability[0,:] = (np.absolute(self.wave[0, :]))**2
      
      
    def Exact_solution(self):
      ########## Diagonize Hamiltonian ##########
      eigen_w, eigen_v = np.linalg.eig(self.hamiltonian)
      idx = np.argsort(eigen_w)
      eigen_w = eigen_w[idx]    #eigenvalue
      eigen_v = eigen_v[:,idx]  #eigenvector
      
      ########## Calculate the Coeffiecient of each Eigenstate ##########
      coeff = np.zeros(self.no_site, dtype= complex)
      #check_wave = np.zeros(self.no_site, dtype= complex)
      
      for i in range(self.no_site):
        coeff[i] = np.dot(eigen_v[:,i], self.wave[0, :].T)
        #check_wave = check_wave + (coeff[i] *eigen_v[:,i])

      ########## Time Evolution ##########
      for t in range(self.qm_timestep):
        #print('Timestep: %s' %(t+1))
      
        ##### Time propagation of wave function#####
        for i in range(self.no_site):
          theta_exact = -eigen_w[i]*(t+1)*self.delt_t/self.h_bar
          phase_factor_exact = complex(np.cos(theta_exact),+np.sin(theta_exact))
          self.wave[(t+1), :] += phase_factor_exact*coeff[i]*eigen_v[:,i]   
        ##### Probability density #####
        self.probability[(t+1), :] = (np.absolute(self.wave[(t+1), :]))**2      
      return(self.wave, self.probability)


    def Scipy_ODE_solver(self):
      ########## Diagonize Ordinary Differential Equation ##########
      def dY_dt(q, ode_wave):
        dydt = (-self.Im_factor/self.h_bar)*np.dot(self.hamiltonian, ode_wave[:]) 
        return dydt

      ########## Scipy ODE Solver: setup ODE method  ##########
      t0=0
      r = ode(dY_dt).set_integrator('zvode', method='bdf') 
      r.set_initial_value(self.wave[0, :], t0)

      ########## Time Evolution ##########
      for t in range(self.qm_timestep):
        ##### Time propagation:  Scipy ODE Solver #####
        r.integrate(r.t+self.delt_t)
        self.wave[(t+1), :] = r.y
        ##### Probability density #####  
        self.probability[(t+1), :] = (np.absolute(self.wave[(t+1), :]))**2
      return(self.wave, self.probability)


def main():   
    ########## Assign System Parameters ##########
    time_step = 10 #Number of total timestep in MD region
    md_delt_t = 0.001 #time interval of each MD times step: unit in ps (pico-second)
    qm_timestep = 500 # Number of total timestep in QM region
    qm_delt_t = 0.5 #md_delt_t*1000/qm_timestep #time interval of each QM times step: unit in fs (femto-second)
    charged_mol_indx = 1 # begin at zero, the molecule initially be assigned charge
    
    ########## Read Initial Configuration ##########
    conf = GromacsGroFile('../MD_Simulation_OpenMM/CONFIG.gro') #GROMACS format
    #conf = PDBFile('input.pdb') #PDB format ###
    
    ########## Setup Force Field & System Information #########
    top = GromacsTopFile('../MD_Simulation_OpenMM/Ethylene.top', periodicBoxVectors=conf.getPeriodicBoxVectors()) # GROMACS format
    #top = ForceField('*.xml') # OpenMM default format
    ### Total Number of Molecules ###
    total_no_mol = top.topology.getNumResidues()
    ### Total Number of Atoms per Molecule ###
    no_atom_per_mol = int(len(conf.atomNames)/total_no_mol)
    ### Box Size ###
    box_size = conf.getPeriodicBoxVectors()._value[0][0]
    ### Atom Mass ###
    mol_mass = []
    for i in range(no_atom_per_mol):
      mol_mass.append(top._currentMoleculeType.atoms[i][7])
    mol_mass = np.array(mol_mass, dtype=float)
    ### Nuclear Charge: To Generate ML Features ###
    nucl_charge = np.zeros(no_atom_per_mol)
    for i in range(no_atom_per_mol):
      nucl_charge[i] = float(top._atomTypes[top._currentMoleculeType.atoms[i][1]][2])
    
    ########## Determine Computing Platform ##########
    platform = Platform.getPlatformByName('CUDA') #'CUDA', 'CPU', 'OpenCL'
    properties = {'DeviceIndex': '0,1'}
    
    ########## Setup Simulation Algorithms ##########
    ###  Nonbonded Interaction ###
    system = top.createSystem(nonbondedMethod=PME, nonbondedCutoff=1.5*nanometer, ewaldErrorTolerance=0.00001)
    ### Integrator ###
    integrator = VerletIntegrator(md_delt_t*picoseconds) #Leapfrog Verlet integrator
    
    ########## Create Simulation Object  ##########
    simulation = Simulation(top.topology, system, integrator, platform, properties)
    ### Asign the initial configuration ###
    simulation.context.setPositions(conf.positions)
    
    ########## Run MD Simulation ##########
    simulation.step(time_step)
    
    ########## On-the-fly Trajectory (Unit: nm) ##########
    ### Extract trajectory values ###
    on_the_fly_traj = simulation.context.getState(getPositions=True).getPositions(asNumpy=True) #[all_atoms,xyz]
    on_the_fly_traj = on_the_fly_traj._value
    ### Reshape the dimension of trajectory ###
    on_the_fly_traj = np.reshape(on_the_fly_traj, (total_no_mol,no_atom_per_mol,3)) #[molecules,atoms,xyz]
    
    ########## List of Molecule Index Adjacent to the Charged Molecule ##########
    list_indx, COM, dis_COM, list_pair, pair_traj, pair_COM_dis = molecule_list(on_the_fly_traj,total_no_mol,no_atom_per_mol,charged_mol_indx,box_size,mol_mass)
    #dis_COM: distance between the centers of mass (the charged one vs. all molecules)
    print(list_indx)
    print(list_pair)
    print(pair_COM_dis)

    ########## List of Molecule Index Adjacent to the Charged Molecule ##########
    CM_intra_inter_atom, CM_inter = generate_CM(no_atom_per_mol,pair_traj,nucl_charge)
    #dis_COM: distance between the centers of mass (the charged one vs. all molecules)
    #print(CM_intra_inter_atom.shape)
    #np.savetxt('CM_inter.txt', CM_inter)
    #np.savetxt('CM_intra_inter_atom.txt', CM_intra_inter_atom)
    
    ########## Predict Electronic Coupling (Hij) by ML Model ##########
    Hij = ml_coupling(CM_intra_inter_atom, pair_COM_dis)
    print(Hij)
    print(Hij.shape)
    
    ########## Define the Hamiltonian ##########
    Hii = np.arange(list_indx.shape[0])
    H = hamiltonian(list_indx.shape[0], Hii, Hij)
    
    ########## Define the Arraies of Wave Function & Probability Density ##########    
    wave = np.zeros(((qm_timestep+1), list_indx.shape[0]), dtype=complex)
    probability = np.zeros(((qm_timestep+1), list_indx.shape[0]))
    wave[0, charged_mol_indx] = 1.0
    
    
    ########## Wave Function Propagation ##########
    propagation = wave_propagation(H, wave[0, :], qm_delt_t, qm_timestep)
    #wave, probability = propagation.Exact_solution()
    wave, probability = propagation.Scipy_ODE_solver()
    print("Wave Function Propagation")

    
if __name__ == '__main__':
    main()    