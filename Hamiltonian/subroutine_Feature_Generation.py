"""
This script provides a function, "generate_CM", 
that generates two types of Coulomb Matrix (CM) representations for machine learning.
The two types of CM representations are complete CM and CM only considering intermolecular elements.
It's worth noting that the unit of length in OpenMM is nanometer,
but the uinit of length of ML features is angstrom.
Therefore, the input trajcetory ("pair_traj") must change its unnit!
This script also calculates the center-of-mass distance of each pair in the unit of nm
The return values are 
(1) CM_intra_inter_atom: complete CM [pairs, CM]
(2) CM_inter: CM only considering intermolecular elements [pairs, CM]

The main program runs a simple OpenMM file to test "generate_CM"
"""


import numpy as np
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
from scipy.spatial import distance
from subroutine_Molecule_List import molecule_list


def generate_CM(no_atom_per_mol,pair_traj,nucl_charge):
    ########### Change unit of length from nm to angstrom ###########
    """
    The unit of length in OpenMM is nm,
    but the uinit of length of ML features is angstrom
    """
    pair_traj = pair_traj*10.00000
    
    ########### Define the dimension of ML features ###########
    #Machine learning feature: Couomb matrix only considering intermolecular elements [pair, CM]
    dim = int(no_atom_per_mol*no_atom_per_mol) #Dimesion of Ml feature
    CM_inter = np.zeros((pair_traj.shape[0], dim))
    #print(CM_inter.shape)

    #Machine learning feature: complete Couomb matrix  [pair index, timesteps, CM]
    dim = int((no_atom_per_mol*2)*(no_atom_per_mol*2+1)/2) #Dimesion of Ml feature
    CM_intra_inter_atom = np.zeros((pair_traj.shape[0], dim))
    #print(CM_intra_inter_atom.shape)
    
    ########### Calculate distance between center of mass of molecular pair ###########
    for pair in range (pair_traj.shape[0]): #[pair index, pair=2, atoms, xyz=3]
      ########### Calculate the diagonal element of Coulomb matrix ###########
      CM = np.zeros((no_atom_per_mol*2, no_atom_per_mol*2), float) #12*12
      cm_diagonal = 0.5*nucl_charge**2.4 #6*1
      cm_diagonal = np.concatenate([cm_diagonal,cm_diagonal], axis=None) #12*1
      np.fill_diagonal(CM, cm_diagonal)

      ########### Calculate the off-diagonal element of Coulomb matrix ###########
      pair_xyz = np.concatenate([pair_traj[pair,0,:,:],pair_traj[pair,1,:,:]]) #12*3
      pair_charge = np.concatenate([nucl_charge,nucl_charge]) #12*1
      for i in range(no_atom_per_mol*2):
        for j in range(i+1, no_atom_per_mol*2):
          # Calculate pairwise distance
          dst = distance.euclidean(pair_xyz[i], pair_xyz[j])
          CM[i,j] = pair_charge[i]*pair_charge[j]/dst
      #print(CM)
      #np.savetxt('CM.txt', CM.reshape((no_atom_per_mol*2,no_atom_per_mol*2)))

      ########### Generate different CM features ###########
      ##### CM_intra_inter_atom ####
      iu_idx = np.triu_indices(no_atom_per_mol*2)
      CM_intra_inter_atom[pair, :] = CM[iu_idx]
      #print(CM_intra_inter_atom)

      ##### CM_inter #####
      CM_inter_partial = []
      for i in range(no_atom_per_mol):
        CM_inter_partial = np.concatenate([CM_inter_partial, CM[i, no_atom_per_mol:no_atom_per_mol*2]])
            
      CM_inter[pair, :] = np.array(CM_inter_partial)
    #return(CM_inter) 
    return(CM_intra_inter_atom, CM_inter)


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

    ########## List of Molecule Index Adjacent to the Charged Molecule ##########
    CM_intra_inter_atom, CM_inter = generate_CM(no_atom_per_mol,pair_traj,nucl_charge)
    #dis_COM: distance between the centers of mass (the charged one vs. all molecules)
    print(CM_intra_inter_atom)
    print(CM_inter)
    #np.savetxt('CM_inter.txt', CM_inter)
    #np.savetxt('CM_intra_inter_atom.txt', CM_intra_inter_atom)
    
if __name__ == '__main__':
    main()    
