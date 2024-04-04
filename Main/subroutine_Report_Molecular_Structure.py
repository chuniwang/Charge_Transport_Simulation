"""
This script calculate the site energy (diagonal elements in hamiltonian) by the function called "site_energy_calculation"
The site energy of each molecule is obtained by subtracting the total energy from the energy without considering the 
force field (FF) of the site molecules. Therefore, we have to remove the FF of the target site molecule to get the energy.
Then, we recover the FF, and repeat these preocesses for the remaining site molecules.
The required input variables are:
    (1) system: one of the OpenMM major classes
    (2) simulation: one of the OpenMM major classes
    (3) ff: the force field class that is defined by "subroutine_Force_Filed_Update.py"
    (4) list_indx: a 1-D array of molecule index to build the hamiltonian, which is obtained by "subroutine_Molecule_List.py"
    (5) list_QM_weight: a 1-D array defining the weight of QM-region

The returned variable is "site_energy" 
Its original unit of energy is kJ/mol. Please multiply a factor of 0.01036410 to change the unit into eV!!!
The main program runs a simple OpenMM file to test "site_energy_calculation"
"""

import numpy as np
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from subroutine_Molecule_List import molecule_list

class report_molecular_structure():
    def __init__(self,on_the_fly_traj,list_indx,no_atom_per_mol,box_size):
      self.no_list_mol = list_indx.shape[0] #number of molecules in the list
      self.no_list_atom = self.no_list_mol*no_atom_per_mol #total number of atoms in the list
      self.box_size = box_size
      self.no_atom_per_mol = no_atom_per_mol
      
      ########## Extract the Trajectory of the List Molecule ##########
      self.traj = np.zeros((self.no_list_mol,no_atom_per_mol,3)) #trajectory before PBC [molecule, atom, xyz]
      self.traj_pbc = np.zeros((self.no_list_mol,no_atom_per_mol,3)) #trajectory after PBC[molecule, atom, xyz]
      for i, mol_idx in np.ndenumerate(list_indx):
        self.traj[i,:,:] = on_the_fly_traj[mol_idx,:,:]
        
      ############ Deal with Periodic Boundary Condition ############
      traj_init = self.traj
      traj_pbc = [] #[molecules,atoms,xyz]
      for i in range(self.no_list_mol):
        ref_atom = np.full((no_atom_per_mol,3), traj_init[i,0,:]) #Regard the first atom as the reference
        check = np.absolute(traj_init[i,:,:]-ref_atom)
        traj_pbc.append(np.where((check>0.5*box_size), traj_init[i,:,:]-np.sign((traj_init[i,:,:]-ref_atom))*box_size, traj_init[i,:,:]))
        #debug_check.append(np.where((check>0.5*box_size), 99999, 0))  
      self.traj_pbc = np.array(traj_pbc)*10.00 #change unit from nm to angstrom
      #debug_check = np.array(debug_check)

    def save_xyz(self,top):
      ########## Save Molecular Structure as *.xyz Format ##########
      filename = top._currentMoleculeType.atoms[0][3] + '_' + str(self.no_list_mol) + 'mers.xyz'
      output_file = open(filename, 'w')
      output_file.write('%d\n' %(self.no_list_atom))
      output_file.write('\n')
      
      for m in range(self.no_list_mol):
        for n in range(self.no_atom_per_mol):
          atom_symbol = top._currentMoleculeType.atoms[n][1][0]
          output_file.write('%s     %.4f     %.4f     %.4f\n' %(atom_symbol,self.traj_pbc[m,n,0],self.traj_pbc[m,n,1],self.traj_pbc[m,n,2]))
      print('The strucure of the charged multimers has been saved as %s' %(filename))


def main():   
    ########## Assign System Parameters ##########
    time_step = 10 #Number of total timestep in MD region
    md_delt_t = 0.001 #time interval of each MD times step: unit in ps (pico-second)
    qm_timestep = 10 # Number of total timestep in QM region
    qm_delt_t = md_delt_t*1000/qm_timestep #time interval of each MD times step: unit in fs (femto-second)
    charged_mol_indx = 1 # begin at zero, the molecule initially be assigned charge
    
    charged_top_file = '../MD_Simulation_OpenMM/Ethylene_qforce_charged.top'
    neutral_top_file = '../MD_Simulation_OpenMM/Ethylene_qforce_neutral.top'
    #neutral_top_file = '../MD_Simulation_OpenMM/Ethylene.top'
    
    ########## Read Initial Configuration ##########
    conf = GromacsGroFile('../MD_Simulation_OpenMM/CONFIG.gro') #GROMACS format
    #conf = PDBFile('input.pdb') #PDB format ###
    
    ########## Setup Force Field & System Information #########
    top = GromacsTopFile(neutral_top_file, periodicBoxVectors=conf.getPeriodicBoxVectors()) # GROMACS format
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
    #platform = Platform.getPlatformByName('CUDA') #'CUDA', 'CPU', 'OpenCL'
    #properties = {'DeviceIndex': '0,1'}
    
    ########## Setup Simulation Algorithms ##########
    ###  Nonbonded Interaction ###
    system = top.createSystem(nonbondedMethod=PME, nonbondedCutoff=1.5*nanometer, ewaldErrorTolerance=0.00001)
    ### Specify energy group index ###
    for i in range(system.getNumForces()):
      system.getForce(i).setForceGroup((i+1))
    ### Integrator ###
    integrator = VerletIntegrator(md_delt_t*picoseconds) #Leapfrog Verlet integrator
    
    ########## Create Simulation Object  ##########
    #simulation = Simulation(top.topology, system, integrator, platform, properties)
    #simulation = Simulation(top.topology, system, integrator, platform)
    simulation = Simulation(top.topology, system, integrator)
    ### Asign the initial configuration ###
    simulation.context.setPositions(conf.positions)
    
    ########## Create Updating Force Field Object  ##########
    #ff = force_field(conf, charged_top_file, neutral_top_file, system, charged_mol_indx)

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
    #print(list_pair)
    #print(pair_COM_dis)

    ########## Report Molecular Structure ##########
    report_structure = report_molecular_structure(on_the_fly_traj,list_indx,no_atom_per_mol,box_size)
    report_structure.save_xyz(top)
    #print(report_structure.traj)
    ##### Save as *.xyz format #####

    ########## Define Wave Function, Probability Density & QM-Region Weighting List ##########    
    ##### Wave Function #####
    ##### Probability Density #####
    ##### QM-Region Weighting List #####
    
    ########## Calculate Site Energy (Hii) ########## 
    
    ########## Predict Electronic Coupling (Hij) by ML Model ##########
    
    ########## Define the Hamiltonian ##########
    
    ########## Wave Function Propagation ##########

    ########## Update QM-Region Weighting List & Force Field List ##########
    ##### Define QM-Region Weighting List #####
    ##### Update Force Field List #####
    ##### QM-Region Weighting List for Changing Force Field #####

    ########## Update Charged State Force Field ##########

    ########## Extract the Molecule Index of the New Charged Center  ##########

    ########## Run MD Simulation ##########
    simulation.step(time_step)
    
    
if __name__ == '__main__':
    main()    

