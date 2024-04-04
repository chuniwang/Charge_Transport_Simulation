from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import time,os
import numpy as np


########## Assign System Parameters ##########
time_step = 5000000 #simulation time steps
state_data_interval = 50000 #interval steps for recording state data into StateData.log
traj_interval = 50000 #interval steps for recording trajectories into traj.dcd
checkpoint_interval = 50000 #interval steps for saving checkpoint file
continue_simulation = 'no' #yes: continue the simulation from previous results; no: begin a new simulation based on CONFIG.gro
continue_conf = 'config_eq_final_1706.xml' #the input file you continue the simulation
charged_mol_indx = 1 # begin at zero, the index of a molecule is initially assigned charge

########## Determine Computing Platform ##########
platform = Platform.getPlatformByName('CUDA') #'CUDA', 'CPU', 'OpenCL'
#properties = {'DeviceIndex': '0,1', 'Precision': 'double'}
properties = {'DeviceIndex': '0,1'}


########## Read Initial Configuration ##########
#conf = GromacsGroFile('CONFIG.gro') #GROMACS format
conf = GromacsGroFile('CONFIG_1706.gro') #GROMACS format
#conf = PDBFile('input.pdb') #PDB format ###


########## Setup Force Field & System Information ##########
#top = GromacsTopFile('Ethylene.top', periodicBoxVectors=conf.getPeriodicBoxVectors()) # GROMACS format
top = GromacsTopFile('Ethylene_qforce_neutral_1706.top', periodicBoxVectors=conf.getPeriodicBoxVectors()) # GROMACS format
#,includeDir='/usr/local/gromacs/share/gromacs/top')
#top = ForceField('*.xml') # OpenMM default format
### Total Number of Molecules ###
total_no_mol = top.topology.getNumResidues()
## Total Number of Atoms per Molecule ###
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

########## Setup Simulation Algorithms ##########
###  Nonbonded Interaction ###
system = top.createSystem(nonbondedMethod=PME, nonbondedCutoff=1.5*nanometer, ewaldErrorTolerance=0.00001)

### Integrator ###
integrator = VerletIntegrator(0.001*picoseconds) #Leapfrog Verlet integrator
#integrator = LangevinIntegrator(130*kelvin, 1.5/picosecond, 0.001*picoseconds) #Langevin integrator

### Temperature & pressure coupling ###
system.addForce(AndersenThermostat(130*kelvin, 1/picosecond)) #Andersen thermostat
system.addForce(MonteCarloBarostat(1*bar, 130*kelvin)) #Monte Carlo barostat

### Specify energy group index ###
for i in range(system.getNumForces()):
  system.getForce(i).setForceGroup((i+1))

########## Create Simulation Object  ##########
#simulation = Simulation(top.topology, system, integrator)
simulation = Simulation(top.topology, system, integrator, platform, properties)
### Asign the initial configuration ###
if continue_simulation == 'yes':
  if not os.path.exists(continue_conf):
    raise ValueError('%s does not exist!' % continue_conf)
  print('The initial configuration is based on %s' %(continue_conf))
  simulation.loadState(continue_conf)
elif continue_simulation == 'no':
  simulation.context.setPositions(conf.positions)
  print('The initial configuration is based on CONFIG.gro' )

print('Total potential energy of the initial configuration: %s kJ/mol' %(simulation.context.getState(getEnergy=True).getPotentialEnergy()))
for i in range(system.getNumForces()):
  print(simulation.context.getState(getEnergy=True, groups={(i+1)}).getPotentialEnergy())

########## Energy Minimization ##########
#If the initial configuration has not been equilibrated, impplenting energy minimization is highly suggested.
#simulation.minimizeEnergy(tolerance=0.1*kilojoule/mole, maxIterations=500)
#positions = simulation.context.getState(getPositions=True).getPositions()
#PDBFile.writeFile(simulation.topology, positions, open('config_energy_minize.pdb', 'w'))


########## Assign Reporters ##########
#simulation.reporters.append(PDBReporter('output.pdb', 1000))
### Print on the screen ###
simulation.reporters.append(StateDataReporter(stdout, state_data_interval, step=True,
    kineticEnergy=True, potentialEnergy=True, totalEnergy=True, temperature=True, separator=' '))

### Record state information ###
simulation.reporters.append(StateDataReporter('StateData_1706.log', state_data_interval, step=True, volume=True,
    kineticEnergy=True, potentialEnergy=True, totalEnergy=True, temperature=True, separator=' '))

### Record trajectories ###
simulation.reporters.append(DCDReporter('traj_1706.dcd', traj_interval)) 

### Record checkpoint file ###
simulation.reporters.append(CheckpointReporter('checkpoint.chk', checkpoint_interval)) 


########## Run MD Simulation ##########
ini_time=time.time()
simulation.step(time_step)
print("Computation Time: %.4f [sec]" %(time.time()-ini_time))


########## Save the Last Trajectory ##########
final_positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
PDBFile.writeFile(simulation.topology, final_positions, open('config_final1706.pdb', 'w'))
simulation.saveState('config_final_1706.xml')
#print(final_positions)
#print(State())


########## Remove Checkpoint File ##########
print("MD simulation is done!")
os.remove('checkpoint.chk')
print("The checkpoint file has been removed.")
