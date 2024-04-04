"""
This script generate a list, which extracts molecules index around the charged molecule
while "MD_OpenMM" is runing.
The return arrays include:
    (1) list_indx: a1-D array of molecule index
    (2) COM: a 2-D array of center of mass (COM) of each molecule [molecule,xyz]
    (3) dis: a 1-D array showing the COM distance between the charged molecule & all molecule [molecule]
    (4) list_pair: a 2-D array listing all the pairs based on "list_indx"
    (5) pair_traj: a 4-D array saving the trajectory of the "list pair",
        and its four dimension corresponds to [pair, 2, atom, xyz]
    (6) pair_COM_dis: the center-of-mass distance of each pair in the unit of nm [pairs]
Please specify cut-off distance (in the unit of nm) denoted as "list_cutoff"
that lists molucules within the criteria region around the charged molecule.
The unit of length is nanometer!!!

The main program runs a simple OpenMM file to test "molecule_list"

In the future, the function "molecule_list" should be changed into a class,
and all the return arraies will become the attributes of the class
"""

import numpy as np
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
from scipy.spatial import distance

def molecule_list(traj,total_no_mol,no_atom_per_mol,charged_mol_indx,box_size,mol_mass):
    list_cutoff = 2.50000 #unit: nm
    """
    list_cutoff: Specifing the cut-off distance (in the unit of nm) 
    that lists molucules within the criteria region around the charged molecule.
    """
    
    ############ Deal with Periodic Boundary Condition ############
    traj_init = traj
    traj_pbc = [] #[molecules,atoms,xyz]
    for i in range(total_no_mol):
      ref_atom = np.full((no_atom_per_mol,3), traj_init[i,0,:]) #Regard the first atom as the reference
      check = np.absolute(traj_init[i,:,:]-ref_atom)
      traj_pbc.append(np.where((check>0.5*box_size), traj_init[i,:,:]-np.sign((traj_init[i,:,:]-ref_atom))*box_size, traj_init[i,:,:]))
      #debug_check.append(np.where((check>0.5*box_size), 99999, 0))  
    traj_pbc = np.array(traj_pbc)
    #debug_check = np.array(debug_check)
    
    ############ Calculate Center of Mass ############
    ### Create mass matrix ###
    mass_matrix = np.full((total_no_mol,no_atom_per_mol), mol_mass)
    ### Center of mass ###
    COM = np.zeros((total_no_mol,3))
    mx = mass_matrix*traj_pbc[:,:,0]
    my = mass_matrix*traj_pbc[:,:,1]
    mz = mass_matrix*traj_pbc[:,:,2]
    COM[:,0] = np.sum(mx, axis=1)/mol_mass.sum()
    COM[:,1] = np.sum(my, axis=1)/mol_mass.sum()
    COM[:,2] = np.sum(mz, axis=1)/mol_mass.sum()
    
    ############ Calculate Distance between the Charged Molecule & Others ##########
    ### Assign COM of the charged molecule ###
    ref_COM = np.full((total_no_mol,3), COM[charged_mol_indx,:])
    ### Calculate distance ###
    ref_COM = (COM - ref_COM)**2
    dis = np.sum(ref_COM, axis=1)
    dis = dis**0.50000
    
    ############ List Molcule Adjacent to the Charged Molecule  ##########
    list_indx = np.argwhere((dis < list_cutoff) & (dis > 0.00))  # Separation distance < list_cutoff
    list_indx = list_indx.flatten()
    ### Add the index of the charged molecule ###
    list_indx = np.append(list_indx, charged_mol_indx)
    ### Sort the list: small to large ###
    list_indx.sort()
    
    ############ List All the Pairs Based on "list_indx"  ##########
    list_pair = []
    for i in range(list_indx.shape[0]):
      for j in range(i+1,list_indx.shape[0]):
        list_pair.append((np.append(list_indx[i],list_indx[j])))
    list_pair = np.array(list_pair)

    ############ Generate Trajectory Array of the Listed Pairs ############
    pair_traj = np.zeros((list_pair.shape[0],2,no_atom_per_mol,3)) #[pair, 2, atom, xyz]
    for i, pair_idx in enumerate(list_pair):
      ### Select molecules ###
      mol_1 = traj_pbc[pair_idx[0],:,:]
      mol_2 = traj_pbc[pair_idx[1],:,:]
      ### Assign trajectory array ###
      pair_traj[i,:,:,:] = np.reshape(np.concatenate((mol_1, mol_2)), (2,no_atom_per_mol,3))
    
    ############ Calculate the COM Distance of the Listed Pairs  ##########
    pair_COM_dis = np.zeros((list_pair.shape[0])) #[pair]; unit:nm
    for i, pair_idx in enumerate(list_pair):
      pair_COM_dis[i] = distance.euclidean(COM[pair_idx[0],:], COM[pair_idx[1],:])
    
    
    return(list_indx,COM,dis,list_pair,pair_traj,pair_COM_dis)


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
    print(pair_traj)
    print(on_the_fly_traj[list_indx,:,:])

    
if __name__ == '__main__':
    main()    