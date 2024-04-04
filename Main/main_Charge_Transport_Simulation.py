"""
This script is the main script for charge transport simulation. 
The overall processing flow are list as below:

0. Determine the initial charged center & update force field (FF) (only for t = 0 )
1. Reshape on-the-fly trajectory (OpenMM data format >>> NumPy array)
2. Update molecular list
3. ﻿Define Wave Function & Probability Density & QM-region weighting list
4. Calculate site energy (remove FF & recover FF)
5. Calculate electronic coupling
6. Build up Hamiltonian
7. Propagate wave packet
8. Update QM-region weighting list & force field list
9. Update charged state force field
10. ﻿Define the new charged center
11. Run MD simulation
12. For loop from step 1 to step 11

All the subroutines and the ML model file (*.pth) must be in the same directory.
In "Overall_Scope.pptx", you will see the connection between each subroutines.
Currently, the results of charge transport have not be saved in an effective way.
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
from subroutine_Site_Energy import site_energy_calculation
from subroutine_Hamiltonian import hamiltonian
from subroutine_Force_Field_Update import force_field
from subroutine_Wave_Propagation import wave_propagation
from subroutine_Report_Molecular_Structure import report_molecular_structure
import torch
import torch.nn as nn
from torch.autograd import Variable
import time



def main():
    init_time = time.time()
    ########## Assign System Parameters ##########
    time_step = 1 #Number of total timestep in MD region
    md_delt_t = 0.001 #time interval of each MD times step: unit in ps (pico-second)
    qm_timestep = 100 # Number of total timestep in QM region per MD timstep
    qm_delt_t = md_delt_t*1000/qm_timestep #time interval of each MD times step: unit in fs (femto-second)
    charged_mol_indx = 1281 # begin at zero, the molecule initially be assigned charge
    #charged_top_file = '../MD_Simulation_OpenMM/Ethylene_qforce_charged.top'
    #neutral_top_file = '../MD_Simulation_OpenMM/Ethylene_qforce_neutral.top'
    charged_top_file = '../MD_Simulation_OpenMM/Ethylene_qforce_charged_1706.top'
    neutral_top_file = '../MD_Simulation_OpenMM/Ethylene_qforce_neutral_1706.top'
    #neutral_top_file = '../MD_Simulation_OpenMM/Ethylene.top'
    output_filename = 'conservation_check.txt'


    output_file = open(output_filename, 'w')
    output_file.write('MD_time [fs]     Normalization     Energy [eV]\n')
    
    ########## Read Initial Configuration ##########
    #conf = GromacsGroFile('../MD_Simulation_OpenMM/CONFIG.gro') #GROMACS format
    conf = GromacsGroFile('../MD_Simulation_OpenMM/CONFIG_1706.gro') #GROMACS format
    #conf = GromacsGroFile('../MD_Simulation_OpenMM/CONFIG_1706_overlap.gro') #GROMACS format
    #conf = PDBFile('input.pdb') #PDB format ###
    
    ########## Setup Force Field & System Information #########
    top = GromacsTopFile(neutral_top_file, periodicBoxVectors=conf.getPeriodicBoxVectors()) # GROMACS format
    #top = ForceField('*.xml') # OpenMM default format
    ### Total Number of Molecules ###
    total_no_mol = top.topology.getNumResidues()
    ### Total Number of Atoms per Molecule ###
    no_atom_per_mol = int(len(conf.atomNames)/total_no_mol)
    ### Box Size ###
    box_size = conf.getPeriodicBoxVectors()._value[0][0] #cubic box
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
    #properties = {'DeviceIndex': '0,1'} #Don't activate this variable because it triggers a bug of OpenMM
    
    ########## Setup Simulation Algorithms ##########
    ###  Nonbonded Interaction ###
    system = top.createSystem(nonbondedMethod=PME, nonbondedCutoff=1.5*nanometer, ewaldErrorTolerance=0.00001)
    ### Specify energy (force field) group index: Don't Move ###
    for i in range(system.getNumForces()):
      system.getForce(i).setForceGroup((i+1))
    ### Integrator ###
    integrator = VerletIntegrator(md_delt_t*picoseconds) #Leapfrog Verlet integrator
    
    ########## Create Simulation Object  ##########
    #simulation = Simulation(top.topology, system, integrator, platform, properties)  #Don't activate this variable because it triggers a bug of OpenMM
    simulation = Simulation(top.topology, system, integrator, platform)
    #simulation = Simulation(top.topology, system, integrator)
    ### Asign the initial configuration ###
    simulation.context.setPositions(conf.positions)
    
    ########## Create Updating Force Field Object  ##########
    ff = force_field(conf, charged_top_file, neutral_top_file, system, charged_mol_indx)
    list_QM_weight_ff_update = np.array(1.0000)
    ff.update_ff(charged_mol_indx, list_QM_weight_ff_update, simulation.context, save_list=True)
    
    """
    Start MD for loop
    """
    update_list_indx = []
    update_dict_wave = {}
    
    for md_t in range(time_step):
      ########## On-the-fly Trajectory (Unit: nm) ##########
      ### Extract trajectory values ###
      on_the_fly_traj = simulation.context.getState(getPositions=True).getPositions(asNumpy=True) #[all_atoms,xyz]
      on_the_fly_traj = on_the_fly_traj._value
      ### Reshape the dimension of trajectory ###
      on_the_fly_traj = np.reshape(on_the_fly_traj, (total_no_mol,no_atom_per_mol,3)) #[molecules,atoms,xyz]
    
      ########## List of Molecule Index Adjacent to the Charged Molecule ##########
      list_indx, COM, dis_COM, list_pair, pair_traj, pair_COM_dis = molecule_list(on_the_fly_traj,total_no_mol,no_atom_per_mol,charged_mol_indx,box_size,mol_mass)
      #dis_COM: distance between the centers of mass (the charged center vs. all molecules)
      #print(list_indx)
      #print(list_pair)
      #print(pair_COM_dis)
    
      ########## Report Molecular Structure ##########
      #report_structure = report_molecular_structure(on_the_fly_traj,list_indx,no_atom_per_mol,box_size)
      ##### Save as *.xyz format #####
      #report_structure.save_xyz(top)
      #print(np.argwhere(list_indx==charged_mol_indx))

      ########## Define Wave Function, Probability Density & QM-Region Weighting List ##########    
      print("Charged center locates on %s at %s time-step." %(charged_mol_indx, md_t))
      ##### Wave Function #####
      wave = np.zeros(((qm_timestep+1), list_indx.shape[0]), dtype=complex)
      ### When md_t = 0 ###
      if (md_t==0):
        wave[0, np.argwhere(list_indx==charged_mol_indx).flatten()] = 1.0
      ### When md_t > 0, update QM weight corresponding to previous step ###
      else:
        for i, mol_indx in np.ndenumerate(list_indx):
          if (mol_indx in update_dict_wave):
            wave[0, i] = update_dict_wave[mol_indx]
          else:
            print("Warning: the %s molecule is not considered in the current Hamiltonian" %(mol_indx))
            print("Please check the QM weight of the %s molecule" %(mol_indx))
            print("If the value is small enough, we reasonably exclude this molecule from the Hamiltonian.")        
      #print(wave[0, :])

      ##### Probability Density #####
      probability = np.zeros(((qm_timestep+1), list_indx.shape[0]))
      ##### QM-Region Weighting List #####
      list_QM_weight = np.absolute(wave[0, :])**2
    
      ########## Calculate Site Energy (Hii) ########## 
      Hii = site_energy_calculation(system, simulation, ff, list_indx, list_QM_weight)
      #print('Site energy:')
      #print(Hii) #unit in eV)
    
      ########## Predict Electronic Coupling (Hij) by ML Model ##########
      ##### Generate ML Features #####
      CM_intra_inter_atom, CM_inter = generate_CM(no_atom_per_mol,pair_traj,nucl_charge)
      ##### Predict Electronic Coupling #####
      Hij = ml_coupling(CM_intra_inter_atom, pair_COM_dis)
      #print('Electronic coupling')
      #print(Hij)
    
      ########## Define the Hamiltonian ##########
      H = hamiltonian(list_indx.shape[0], Hii, Hij)
      #print(H.hamiltonian)
      #print(H.hamiltonian.shape)
      #Add NAC here!!! need r dot, velocity
      #Refer to "simulation.context.getState(getPositions=True)"
    
      ########## Wave Function Propagation ##########
      #propagation = wave_propagation(H.hamiltonian, wave[0, :], qm_delt_t, qm_timestep)  #Nucleus are moving
      propagation = wave_propagation(H.hamiltonian, wave[0, :], 0.1, qm_timestep) #Nucleus are fixed
      #print("Wave Function Propagating...")
      #wave, probability = propagation.Scipy_ODE_solver() #wave[time, site]
      wave, probability = propagation.Exact_solution() #wave[time, site]
      #print('t=0')
      #print(wave[0,:])
      #print('t=t')
      #print(wave[-1,:])

      ########## Update QM-Region Weighting List & Force Field List ##########
      ##### Define QM-Region Weighting List #####
      list_QM_weight = np.absolute(wave[-1, :])**2 #[time, site]
      ##### Update Force Field List #####
      update_list_indx = list_indx[np.argwhere(list_QM_weight!=0.000)].flatten()
      print('List of charge molecules after wave propagation: %s' %(update_list_indx))
      ##### QM-Region Weighting List for Changing Force Field #####
      list_QM_weight_ff_update = np.zeros(update_list_indx.shape)
      list_QM_weight_ff_update = list_QM_weight[np.argwhere(list_QM_weight!=0.000)].flatten()
      print('Weights of charge molecules after wave propagation:')
      print(list_QM_weight_ff_update)
    
      print('Normalization:')
      for t in range(wave.shape[0]):
        print(np.abs(wave[t,:].dot(wave[t,:])))
      print(np.abs(wave[-1,:].dot(wave[-1,:])))

      print('Energy conservation:')
      for t in range(wave.shape[0]):
        print(np.abs(wave[t,:].dot(H.hamiltonian.dot(wave[t,:]))))  
      print(np.abs(wave[-1,:].dot(H.hamiltonian.dot(wave[-1,:]))))
      
      output_file.write('%s     %f     %f\n' %(md_t, (np.abs(wave[-1,:].dot(wave[-1,:]))), (np.abs(wave[-1,:].dot(H.hamiltonian.dot(wave[-1,:]))))))
      
      ########## Set Initial Wave for the Next Round by Defining a Python Dictionary ##########
      update_dict_wave = {}
      for i, mol_indx in np.ndenumerate(update_list_indx):
        update_dict_wave[mol_indx] = wave[-1,(np.argwhere(list_indx==mol_indx))].flatten()[0]
      #print(update_dict_wave.items())
        
      ########## Update Charged State Force Field ##########
      ff.intialize_neutral_ff([], simulation.context)
      ff.update_ff(update_list_indx, list_QM_weight_ff_update, simulation.context, save_list=True)
    
      ########## Define the New Charged Center  ##########
      charged_mol_indx = list_indx[np.argwhere(list_QM_weight==list_QM_weight.max())].flatten()
    
      ########## Run MD Simulation ##########
      simulation.step(1) #Rub MD simulation for one time-step
    
    print('Total computing time [min]: %f' %((time.time()-init_time)/60.0))
if __name__ == '__main__':
    main()    

