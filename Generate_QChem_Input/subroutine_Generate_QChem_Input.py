"""
This script provides a function, "generate_QChem", 
that create QChem input file of the molecular pairs. 

The input trajectory (pair_traj) is in the unit of nm,
but the QChem file is in the unit of angstrom.
The transformation of unit has been included among this script.

User need to specify:
    (1) atom_symbol: lsiting the atom type of each atom of a molecule,
        the order of atom type must be consistant with the order in OpenMM
    (2) Molecule_type: this variable is part of the name of QChem input file
    (3) QChem_reference: Provide details of the $rem section of QChem input file

The main program runs a simple OpenMM file to test "generate_QChem"
"""

import numpy as np
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
from subroutine_Molecule_List import molecule_list


def generate_QChem(no_atom_per_mol,pair_traj,list_pair):
    atom_symbol = ['C','C','H','H','H','H'] #Ethylene
    Molecule_type = "Ethlene_"
    QChem_reference = "QChem_Ref_Ethylene.inp" #Provide details of the $rem section of QChem input file
    
    ############### Read QChem reference file ###############
    qchem_ref = open(QChem_reference)
    lines = qchem_ref.read()
    
    for i, pair_idx in enumerate(list_pair):
      ########## Define QChem input file name ##########
      QChem_filename = Molecule_type + str(pair_idx[0]) + '_' + str(pair_idx[1])
      QChem_filename = QChem_filename + '.inp'
      print('Create %s' %(QChem_filename))

      ########## $comment section ##########
      output_file = open(QChem_filename, 'w')
      output_file.write('$comment\n')
      output_file.write('  Pair index in OpenMM: %s and %s\n' %(pair_idx[0], pair_idx[1]))
      output_file.write('$end\n \n')

      ########## $molecule section ##########
      output_file.write('$molecule\n')
      output_file.write('  0  1\n')

      for m in range(2):  #Two molecule xyz
        output_file.write('--\n')
        output_file.write('  0  1\n')

        ##### Write atom type and xyz coordinates #####
        #pair_traj: [pair index, pair=2,  atoms, xyz=3]
        for j in range(no_atom_per_mol):
            atom_xyz = pair_traj[i,m,j,:]*10 #transform unit from nm to angstron
            output_file.write('%1s %12.7f %12.7f %12.7f\n' %(atom_symbol[j], atom_xyz[0],atom_xyz[1],atom_xyz[2]))

      output_file.write('$end\n \n')
      ########## $rem section ##########
      output_file.write(lines)
      output_file.close()


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
    
    ########## Generate QChem Input File ##########
    generate_QChem(no_atom_per_mol,pair_traj,list_pair)
    
    
if __name__ == '__main__':
    main()    
