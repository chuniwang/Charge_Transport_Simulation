import mdtraj as md
import numpy as np
import time

Trajectory_file = "traj_comp.xtc" #Trajectory file from GROMACS
Topology_reference = "CONFIG.gro" #The file format *.top, whiich is used in GROMACS, cannot be topology in MDTraj.
mole_pair_indx = [1, 8667] #Specify index of a molecular pair
QChem_reference = "QChem_Ref_Ethylene_NAC.inp" #Provide details of the $rem section of QChem input file
no_frames = 3 #Number of trajectory (or frames) for the QChem input file 
Molecule_type = "Ethlene_"

############### Read QChem reference file ###############
qchem_ref = open(QChem_reference)
lines = qchem_ref.read()

############### Read trajectory file ###############
t0 = time.time()
print('Loading GROMACS trajectory file......')
traj = md.load(Trajectory_file, top=Topology_reference)
t1 = time.time()
print('Time for loading GROMACS trajectory file: %f min' %((t1-t0)/60.0))
print(traj)

############### Assign topology information ###############
topo = traj.topology
#print(topo)
print('Molecule index in GROMACS: %s' %(mole_pair_indx))
mole_pair_indx[0] = mole_pair_indx[0]-1 #make the index to begin at zero
mole_pair_indx[1] = mole_pair_indx[1]-1 #make the index to begin at zero
print('Molecule index in Python fromat: %s' %(mole_pair_indx))

##### Extract atom index of the molecule #####
mol_atom_indx = []
for m in range(2):
  extract_indx = [a.index for a in topo.residue(mole_pair_indx[m]).atoms]
  print('Atom index of the %sth molecule: %s (python index rule)' %(mole_pair_indx[m], extract_indx))
  mol_atom_indx = mol_atom_indx + extract_indx

n_atom_per_mol = traj.topology.residue(0).n_atoms
mol_atom_indx = [mol_atom_indx[i:i+n_atom_per_mol] for i in range(0, len(mol_atom_indx), n_atom_per_mol)]
#print(mol_atom_indx)
#print(mol_atom_indx[0][:])
#print(mol_atom_indx[1][:])


############### Write QChem input file ###############
for t in range(no_frames):
  ########## Define QChem input file name ##########
  time_frame = "%05d" %t
  QChem_filename = Molecule_type + str(mole_pair_indx[0]+1) + '_' + str(mole_pair_indx[1]+1)
  QChem_filename = QChem_filename + '-' + time_frame + '.inp'
  print('Create %s' %(QChem_filename))


  ########## $comment section ##########
  output_file = open(QChem_filename, 'w')
  output_file.write('$comment\n')
  output_file.write('  Molecule index in GROMACS: %s and %s\n' %((mole_pair_indx[0]+1), (mole_pair_indx[1]+1)))
  output_file.write('  Trajectory of the time step: %s\n' %(t))
  output_file.write('$end\n \n')

  ########## $molecule section ##########
  output_file.write('$molecule\n')
  output_file.write('  0  1\n')

  for m in range(2):
    output_file.write('--\n')
    output_file.write('  0  1\n')

    ##### Write atom type and xyz coordinates #####
    for i, indx in enumerate(mol_atom_indx[m][:]):
        atom_symbol = traj.topology.atom(indx).element.symbol
        atom_xyz = traj.xyz[t, indx, :]*10 #transform unit from nm to angstron
        output_file.write('%1s %12.7f %12.7f %12.7f\n' %(atom_symbol, atom_xyz[0], atom_xyz[1], atom_xyz[2]))

  output_file.write('$end\n \n')

  ########## $rem section ##########
  output_file.write(lines)
  output_file.close()

t2 = time.time()
print('Total processing time: %f min' %((t2-t0)/60.0))

"""
Testing scrypt
#print('%s' %[i for i in topo.residue(4).atoms])
#x_ele = [a.element for a in topo.residue(4).atoms]
#x_ele[0].symbol
#traj.topology.atom(4).element.symbol
#for i in range [traj.topology.residue(0).n_atoms]:
"""
