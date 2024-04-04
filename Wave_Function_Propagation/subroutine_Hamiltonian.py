"""
This script defines a class called "hamiltonian" along with several functions (under developing).
The required input variables are:
    (1) H_size: Determining the size of a Hamiltonian matrix [H_size,H_size]
    (2) Hii: Diagonal elements of a Hamiltonian matrix
    (3) Hij: Off-diagonal elements of a Hamiltonian matrix 

The unit of electronic coupling obtained from ML model is eV!!!
In the future this class will also update the nonadiabatic couping (NAC).

The main program runs a simple OpenMM file to test "hamiltonian"
"""

import numpy as np
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
from scipy.spatial import distance
from subroutine_Molecule_List import molecule_list
from subroutine_Feature_Generation import generate_CM
from subroutine_ML_Prediction import ml_coupling
import torch
import torch.nn as nn
from torch.autograd import Variable

class hamiltonian(): #Unit: eV
    def __init__(self, H_size, Hii, Hij):
      self.Hii = Hii
      self.Hij = Hij
      ########## Define Matrix Elements lof Hamiltonian ##########
      self.hamiltonian = np.zeros((H_size,H_size), dtype=float)
      ### Off-diagonal elements ###
      self.iu_idx = np.triu_indices(H_size,k=1)
      self.il_idx = np.tril_indices(H_size,k=-1)
      self.hamiltonian[self.iu_idx] = Hij
      self.hamiltonian = self.hamiltonian + self.hamiltonian.T
      ### Diagonal elements ###
      np.fill_diagonal(self.hamiltonian, Hii)
      
    def test(self):
      print(self.hamiltonian)
      print(self.Hii)
      print(self.Hij)

def main():   
    ########## Assign System Parameters ##########
    time_step = 10 #Number of total timestep in MD region
    md_delt_t = 0.001 #time interval of each MD times step: unit in ps (pico-second)
    qm_timestep = 10 # Number of total timestep in QM region
    qm_delt_t = md_delt_t*1000/qm_timestep #time interval of each MD times step: unit in fs (femto-second)
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
    x = hamiltonian(list_indx.shape[0], Hii, Hij)
    x.test()
    
if __name__ == '__main__':
    main()    

