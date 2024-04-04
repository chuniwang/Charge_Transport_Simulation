"""
This script defines a class called "force_field" along with three functions, which change the parameters of force field (FF). 
The three functions in this class are "update_ff", "intialize_neutral_ff", and "ff_site_energy_calculation"
When you first initialize this class, the required input variables are:
    (1) conf: the initial MD configure, which must be obtained from ***.gro
    (2) charged_top_file: path of the charged top file 
    (3) neutral_top_file: path of the neutral top file
    (4) system_update: one of the OpenMM major classes (system)
    (5) charged_mol_indx: the index of molecule, which initially be assigned charge

"update_ff" changes the FF parameters from neutral state into charged state.
This function can be implemented to update the FF for MD simulation or to calculate the site energy
The required input variables are:
    (1) update_list_indx: a list saves the molecule index whose FF needs to change to charged state
    (2) list_QM_weight: a lsist corresponds to the "update_list_indx", and save the weighting of chare state
    (3) context: one of the OpenMM major object (simulation.context)
    (4) save_list: bool value 
        If its value is True, the update_list_indx will be saved and automatically become the input varaiable of "intialize_neutral_ff"
        If its value is False, the update_list_indx won't be saved

"intialize_neutral_ff" changes the FF parameters from charged state into neutral state.
The required input variables are:
    (1) update_list_indx: a list saves the molecule index whose FF needs to change back to neutral state
        If update_list_indx is [] (empty list), the function will use the molecule list from "update_ff"
    (2) context: one of the OpenMM major object (simulation.context)

"ff_site_energy_calculation" is only implemented during the calculation of site energy.
This function sets all the FF of a target molcule as zeros.
    (1) charge_idx: the index of the target molecule whose FF will be set zeros
    (2) context: one of the OpenMM major object (simulation.context)

Notice:
(1) The unit of electronic coupling obtained from ML model is eV!
(2) Each group of FF parameters belong to individual Python class so each of them has their own functions.
    Please check the reference URLs in each group of FF when you try to modify this script!
    http://docs.openmm.org/latest/api-python/library.html#forces
(3) In the main progrm, each FF group must be assigned an index number before any calculation.
    Otherwise OpenMM will report wrong energy for each FF group.
    Please check the keyword "Specify FF Group Index" in this script to check how to define FF group index.

The main program runs a simple OpenMM file to test "force_field
"""

import numpy as np
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
from scipy.spatial import distance
from subroutine_Molecule_List import molecule_list
#from subroutine_Feature_Generation import generate_CM
#from subroutine_ML_Prediction import ml_coupling
#import torch
#import torch.nn as nn
#from torch.autograd import Variable

class force_field(): #Unit: eV
    def __init__(self, conf, charged_top_file, neutral_top_file, system_update, charged_mol_indx):
      ########## Initialize Reference TOP & System for Charged & Neurtral States ##########
      self.conf = conf
      self.top_charged = GromacsTopFile(charged_top_file, periodicBoxVectors=self.conf.getPeriodicBoxVectors()) # GROMACS format
      self.top_neutral = GromacsTopFile(neutral_top_file, periodicBoxVectors=self.conf.getPeriodicBoxVectors()) # GROMACS format
      self.system_charged = self.top_charged.createSystem(nonbondedMethod=PME, nonbondedCutoff=1.5*nanometer, ewaldErrorTolerance=0.00001)
      self.system_neutral = self.top_neutral.createSystem(nonbondedMethod=PME, nonbondedCutoff=1.5*nanometer, ewaldErrorTolerance=0.00001)
      self.list_indx_origin = [charged_mol_indx]
      
      ########## Define Force Objects ##########
      #print(system_update.getForces())
      ##### Nonbond Force #####
      self.nonbond_update = system_update.getForce(0)
      self.nonbond_charged = self.system_charged.getForce(0) #Charged-state reference
      self.nonbond_neutral = self.system_neutral.getForce(0) #Neutral-state reference
      print('Nonbond force group: %d' % (self.nonbond_update.getForceGroup()))
      ##### Bond Force #####
      self.bond_update = system_update.getForce(1)
      self.bond_charged = self.system_charged.getForce(1) #Charged-state reference
      self.bond_neutral = self.system_neutral.getForce(1) #Neutral-state reference
      print('Bond force group: %d' % (self.bond_update.getForceGroup()))
      ##### Angle Force #####
      self.angle_update = system_update.getForce(2)
      self.angle_charged = self.system_charged.getForce(2) #Charged-state reference
      self.angle_neutral = self.system_neutral.getForce(2) #Neutral-state reference
      print('Angle force group: %d' % (self.angle_update.getForceGroup()))
      ##### Torsion Force #####
      self.torsion_update = system_update.getForce(3)
      self.torsion_charged = self.system_charged.getForce(3) #Charged-state reference
      self.torsion_neutral = self.system_neutral.getForce(3) #Neutral-state reference
      print('Torsion force group: %d' % (self.torsion_update.getForceGroup()))
  
      ########## Set the Number of Potential Functions for a Molecule ##########
      ### Total Number of Molecules ###
      total_no_mol = self.top_neutral.topology.getNumResidues()
      ### Nonbond Force: Atoms ###
      self.no_nonbond = self.nonbond_update.getNumParticles()/total_no_mol
      if ((self.nonbond_update.getNumParticles())!=(self.nonbond_charged.getNumParticles())):
        raise ValueError('The number of nonbond forces is not consistant! Please check both TOP files.')
      ### Nonbond Force: Exception ###
      self.no_nonbond_excp = self.nonbond_update.getNumExceptions()/total_no_mol
      if ((self.nonbond_update.getNumExceptions())!=(self.nonbond_charged.getNumExceptions())):
        raise ValueError('The number of nonbond forces is not consistant! Please check both TOP files.')      
      ### Bond Force ###
      self.no_bond = self.bond_update.getNumBonds()/total_no_mol
      if ((self.bond_update.getNumBonds())!=(self.bond_charged.getNumBonds())):
        raise ValueError('The number of bond forces is not consistant! Please check both TOP files.')
      ### Angle Force ###
      self.no_angle = self.angle_update.getNumAngles()/total_no_mol
      if ((self.angle_update.getNumAngles())!=(self.angle_charged.getNumAngles())):
        raise ValueError('The number of angle forces is not consistant! Please check both TOP files.')
      ### Torsion Force ###
      self.no_torsion = self.torsion_update.getNumTorsions()/total_no_mol
      if ((self.torsion_update.getNumTorsions())!=(self.torsion_charged.getNumTorsions())):
        raise ValueError('The number of torsion forces is not consistant! Please check both TOP files.')


    def update_ff(self, update_list_indx, list_QM_weight, context, save_list):
	  ########## Update The Charged State ##########
      if (save_list==True):
        self.list_indx_origin = update_list_indx
      elif (save_list==False):
        self.list_indx_origin = self.list_indx_origin
      else:
        print("You set the wrong variable. Please specify save_list as True or False!")
          
      for i, charge_idx in np.ndenumerate(update_list_indx):
        ##### Define the Scaling Factor for the Delocalized Charged State  #####
        QM_weight = list_QM_weight[i]
        
        ##### Nonbond Force  #####
        # The following link introduces the Python functions & their parameters. 
        # http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.NonbondedForce.html          
        ### Nonbond force: atom ###
        start_idx = int(charge_idx * self.no_nonbond)
        end_idx = int((charge_idx+1) * self.no_nonbond)
        for n in range(start_idx, end_idx):
          update_ref = self.nonbond_charged.getParticleParameters(n)
          chargeProd = update_ref[0]._value * QM_weight
          #print(self.nonbond_update.getParticleParameters(n))
          self.nonbond_update.setParticleParameters(n, chargeProd, update_ref[1]._value, update_ref[2]._value)
          #print(self.nonbond_update.getParticleParameters(n))
        self.nonbond_update.updateParametersInContext(context)     

        ### Nonbond Force: 1-4 Exception ###
        # The following links introduce the 1-4 exculsion and corresponding setting in MD simulation. 
        # https://manual.gromacs.org/documentation/current/reference-manual/topologies/molecule-definition.html#excl
        # https://manual.gromacs.org/documentation/current/reference-manual/topologies/topology-file-formats.html
        # https://manual.gromacs.org/documentation/current/reference-manual/topologies/parameter-files.html
        start_idx = int(charge_idx * self.no_nonbond_excp)
        end_idx = int((charge_idx+1) * self.no_nonbond_excp)
        fudge_QQ = float(self.top_charged._defaults[4]) #Rescaling factor for intra-molecular electrostatic interaction
        fudge_QQ = fudge_QQ * QM_weight * QM_weight #Scaling factor for the deloclalized charged state
        for n in range(start_idx, end_idx):
          update_ref = self.nonbond_charged.getExceptionParameters(n)
          #print(self.nonbond_update.getExceptionParameters(n))
          if((self.nonbond_update.getExceptionParameters(n)[2]._value) != 0.0 ):
            chargeProd = fudge_QQ*(self.nonbond_charged.getParticleParameters(update_ref[0])[0]._value)*(self.nonbond_charged.getParticleParameters(update_ref[1])[0]._value)
            self.nonbond_update.setExceptionParameters(n, update_ref[0], update_ref[1], chargeProd, update_ref[3], update_ref[4])
        self.nonbond_update.updateParametersInContext(context)
        
        ##### Bond Force  #####
        # The following link introduces the Python functions & their parameters. 
        # http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.HarmonicBondForce.html
        start_idx = int(charge_idx * self.no_bond)
        end_idx = int((charge_idx+1) * self.no_bond)
        for n in range(start_idx, end_idx):
          update_ref = self.bond_charged.getBondParameters(n)
          bond_eq = self.bond_neutral.getBondParameters(n)[2]._value + QM_weight * (update_ref[2]._value - self.bond_neutral.getBondParameters(n)[2]._value)
          self.bond_update.setBondParameters(n, update_ref[0], update_ref[1], bond_eq, update_ref[3]._value)
        self.bond_update.updateParametersInContext(context)

        ##### Angle Force  #####
        # The following link introduces the Python functions & their parameters. 
        # http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.HarmonicAngleForce.html
        start_idx = int(charge_idx * self.no_angle)
        end_idx = int((charge_idx+1) * self.no_angle)
        for n in range(start_idx, end_idx):
          update_ref = self.angle_charged.getAngleParameters(n)
          angle_eq = self.angle_neutral.getAngleParameters(n)[3]._value + QM_weight * (update_ref[3]._value - self.angle_neutral.getAngleParameters(n)[3]._value)
          self.angle_update.setAngleParameters(n, update_ref[0], update_ref[1], update_ref[2], angle_eq, update_ref[4]._value)
        self.angle_update.updateParametersInContext(context)

        ##### Torsion Force: Improper Dihedrals from Q-Force  #####
        # OpenMM defines the improper dihedral potential in CustomTorsionForce intead of general TorsionForce
        # The following links introduce the Python functions & their parameters.
        # http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomTorsionForce.html
        # http://docs.openmm.org/latest/userguide/application.html#customtorsionforce
        start_idx = int(charge_idx * self.no_torsion)
        end_idx = int((charge_idx+1) * self.no_torsion)
        for n in range(start_idx, end_idx):
          update_ref = self.torsion_charged.getTorsionParameters(n)
          self.torsion_update.setTorsionParameters(n, update_ref[0], update_ref[1], update_ref[2], update_ref[3], (update_ref[4][0], update_ref[4][1]))
        self.torsion_update.updateParametersInContext(context)

    def intialize_neutral_ff(self, update_list_indx, context):    
      ########## Define the Initialization List ##########
      if (update_list_indx==[]):
        initialization_list = self.list_indx_origin
      else:
        initialization_list = update_list_indx
      ########## Change to the Neutral State ##########
      for i, charge_idx in np.ndenumerate(initialization_list):
        ##### Nonbond Force  #####
        # The following link introduces the Python functions & their parameters. 
        # http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.NonbondedForce.html          
        ### Nonbond force: atom ###
        start_idx = int(charge_idx * self.no_nonbond)
        end_idx = int((charge_idx+1) * self.no_nonbond)
        for n in range(start_idx, end_idx):
          update_ref = self.nonbond_neutral.getParticleParameters(n)
          #print(n,self.nonbond_update.getParticleParameters(n))
          self.nonbond_update.setParticleParameters(n, update_ref[0]._value, update_ref[1]._value, update_ref[2]._value)
          #print(n,self.nonbond_update.getParticleParameters(n))
        self.nonbond_update.updateParametersInContext(context)  #Necessary step   

        ### Nonbond Force: 1-4 Exception ###
        # The following links introduce the 1-4 exculsion and corresponding setting in MD simulation. 
        # https://manual.gromacs.org/documentation/current/reference-manual/topologies/molecule-definition.html#excl
        # https://manual.gromacs.org/documentation/current/reference-manual/topologies/topology-file-formats.html
        # https://manual.gromacs.org/documentation/current/reference-manual/topologies/parameter-files.html
        start_idx = int(charge_idx * self.no_nonbond_excp)
        end_idx = int((charge_idx+1) * self.no_nonbond_excp)
        fudge_QQ = float(self.top_charged._defaults[4]) #Rescaling factor for intra-molecular electrostatic interaction
        for n in range(start_idx, end_idx):
          update_ref = self.nonbond_neutral.getExceptionParameters(n)
          if((self.nonbond_update.getExceptionParameters(n)[2]._value) != 0.0 ):
            chargeProd = fudge_QQ*(self.nonbond_neutral.getParticleParameters(update_ref[0])[0]._value)*(self.nonbond_neutral.getParticleParameters(update_ref[1])[0]._value)
            self.nonbond_update.setExceptionParameters(n, update_ref[0], update_ref[1], chargeProd, update_ref[3], update_ref[4])
          #print(self.nonbond_update.getExceptionParameters(n))
        self.nonbond_update.updateParametersInContext(context)  #Necessary step 
        
        ##### Bond Force  #####
        # The following link introduces the Python functions & their parameters. 
        # http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.HarmonicBondForce.html
        start_idx = int(charge_idx * self.no_bond)
        end_idx = int((charge_idx+1) * self.no_bond)
        for n in range(start_idx, end_idx):
          update_ref = self.bond_neutral.getBondParameters(n)
          self.bond_update.setBondParameters(n, update_ref[0], update_ref[1], update_ref[2]._value, update_ref[3]._value)
        self.bond_update.updateParametersInContext(context)  #Necessary step 

        ##### Angle Force  #####
        # The following link introduces the Python functions & their parameters. 
        # http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.HarmonicAngleForce.html
        start_idx = int(charge_idx * self.no_angle)
        end_idx = int((charge_idx+1) * self.no_angle)
        for n in range(start_idx, end_idx):
          update_ref = self.angle_neutral.getAngleParameters(n)
          self.angle_update.setAngleParameters(n, update_ref[0], update_ref[1], update_ref[2], update_ref[3]._value, update_ref[4]._value)
        self.angle_update.updateParametersInContext(context)  #Necessary step 

        ##### Torsion Force: Improper Dihedrals from Q-Force  #####
        # OpenMM defines the improper dihedral potential in CustomTorsionForce intead of general TorsionForce
        # The following links introduce the Python functions & their parameters.
        # http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomTorsionForce.html
        # http://docs.openmm.org/latest/userguide/application.html#customtorsionforce
        start_idx = int(charge_idx * self.no_torsion)
        end_idx = int((charge_idx+1) * self.no_torsion)
        for n in range(start_idx, end_idx):
          update_ref = self.torsion_neutral.getTorsionParameters(n)
          self.torsion_update.setTorsionParameters(n, update_ref[0], update_ref[1], update_ref[2], update_ref[3], (update_ref[4][0], update_ref[4][1]))
        self.torsion_update.updateParametersInContext(context)  #Necessary step 


    def ff_site_energy_calculation(self, charge_idx, context):    
        ########## Remove Force Field ##########
        ##### Nonbond Force  #####
        # The following link introduces the Python functions & their parameters. 
        # http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.NonbondedForce.html          
        ### Nonbond force: atom ###
        start_idx = int(charge_idx * self.no_nonbond)
        end_idx = int((charge_idx+1) * self.no_nonbond)
        for n in range(start_idx, end_idx):
          update_ref = self.nonbond_neutral.getParticleParameters(n)
          #print(self.nonbond_update.getParticleParameters(n))
          self.nonbond_update.setParticleParameters(n, 0.000, update_ref[1]._value, 0.000)
          #print(self.nonbond_update.getParticleParameters(n))
        self.nonbond_update.updateParametersInContext(context)      #Necessary step  

        ### Nonbond Force: 1-4 Exception ###
        # The following links introduce the 1-4 exculsion and corresponding setting in MD simulation. 
        # https://manual.gromacs.org/documentation/current/reference-manual/topologies/molecule-definition.html#excl
        # https://manual.gromacs.org/documentation/current/reference-manual/topologies/topology-file-formats.html
        # https://manual.gromacs.org/documentation/current/reference-manual/topologies/parameter-files.html
        excp_list = []
        start_idx = int(charge_idx * self.no_nonbond_excp)
        end_idx = int((charge_idx+1) * self.no_nonbond_excp)
        for n in range(start_idx, end_idx):
          update_ref = self.nonbond_neutral.getExceptionParameters(n)
          if((self.nonbond_update.getExceptionParameters(n)[2]._value) != 0.0 ):
            self.nonbond_update.setExceptionParameters(n, update_ref[0], update_ref[1], 0.000, update_ref[3], 0.000)
            excp_list.append(n)
          #print(self.nonbond_update.getExceptionParameters(n))
        self.nonbond_update.updateParametersInContext(context)  #Necessary step 
        
        ##### Bond Force  #####
        # The following link introduces the Python functions & their parameters. 
        # http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.HarmonicBondForce.html
        start_idx = int(charge_idx * self.no_bond)
        end_idx = int((charge_idx+1) * self.no_bond)
        for n in range(start_idx, end_idx):
          update_ref = self.bond_neutral.getBondParameters(n)
          self.bond_update.setBondParameters(n, update_ref[0], update_ref[1], 0.000, 0.000)
        self.bond_update.updateParametersInContext(context)  #Necessary step 

        ##### Angle Force  #####
        # The following link introduces the Python functions & their parameters. 
        # http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.HarmonicAngleForce.html
        start_idx = int(charge_idx * self.no_angle)
        end_idx = int((charge_idx+1) * self.no_angle)
        for n in range(start_idx, end_idx):
          update_ref = self.angle_neutral.getAngleParameters(n)
          self.angle_update.setAngleParameters(n, update_ref[0], update_ref[1], update_ref[2], 0.000, 0.000)
        self.angle_update.updateParametersInContext(context)  #Necessary step 

        ##### Torsion Force: Improper Dihedrals from Q-Force  #####
        # OpenMM defines the improper dihedral potential in CustomTorsionForce intead of general TorsionForce
        # The following links introduce the Python functions & their parameters.
        # http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomTorsionForce.html
        # http://docs.openmm.org/latest/userguide/application.html#customtorsionforce
        start_idx = int(charge_idx * self.no_torsion)
        end_idx = int((charge_idx+1) * self.no_torsion)
        for n in range(start_idx, end_idx):
          update_ref = self.torsion_neutral.getTorsionParameters(n)
          self.torsion_update.setTorsionParameters(n, update_ref[0], update_ref[1], update_ref[2], update_ref[3], (0.000, 0.000))
        self.torsion_update.updateParametersInContext(context)  #Necessary step 
        
        return(excp_list)


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
    ### Specify FF Group Index ###
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
    ff = force_field(conf, charged_top_file, neutral_top_file, system, charged_mol_indx)

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

    ########## Define Wave Function, Probability Density & QM-Region Weighting List ##########    
    ##### Wave Function #####
    wave = np.zeros(((qm_timestep+1), list_indx.shape[0]), dtype=complex)
    wave[0, np.argwhere(list_indx==charged_mol_indx).flatten()] = 1.0
    print(wave[0, :])
    """
    How to update the wave function when list_indx changes its shape.
    """
    ##### Probability Density #####
    probability = np.zeros(((qm_timestep+1), list_indx.shape[0]))
    ##### QM-Region Weighting List #####
    list_QM_weight = np.absolute(wave[0, :])**2
    
    ########## Calculate Site Energy (Hii) ########## 
    
    ########## Predict Electronic Coupling (Hij) by ML Model ##########
    
    ########## Define the Hamiltonian ##########
    
    ########## Wave Function Propagation ##########
    

    ########## Update QM-Region Weighting List & Force Field List ##########
    ##### Define QM-Region Weighting List #####
    list_QM_weight = np.absolute(wave[-1, :])**2
    ##### Update Force Field List #####
    update_list_indx = list_indx[np.argwhere(list_QM_weight!=0.000)].flatten()
    ##### QM-Region Weighting List for Changing Force Field #####
    list_QM_weight_ff_update = np.zeros(update_list_indx.shape)
    list_QM_weight_ff_update = list_QM_weight[np.argwhere(list_QM_weight!=0.000)].flatten()
    
    ########## Update Charged State Force Field ##########
    ##### Check Force Group #####
    print(system.getForces())
    freeGroups = set(range(32)) - set(force.getForceGroup() for force in system.getForces())
    occupyGroups = set(force.getForceGroup() for force in system.getForces())
    print(freeGroups)
    print(occupyGroups)
    
    energy = simulation.context
    print('--------------------')
    print("Energy before modification:")
    print('total energy: %s' % (energy.getState(getEnergy=True).getPotentialEnergy()))
    print('nonbond: %s' % (energy.getState(getEnergy=True, groups={1}).getPotentialEnergy()))
    print('bond: %s' % (energy.getState(getEnergy=True, groups={2}).getPotentialEnergy()))
    print('angle: %s' % (energy.getState(getEnergy=True, groups={3}).getPotentialEnergy()))
    print('dihedral: %s' % (energy.getState(getEnergy=True, groups={4}).getPotentialEnergy()))
    print('--------------------')
    #for n in range(system.getForce(0).getNumParticles()):
     # print(system.getForce(0).getParticleParameters(n))
    #for n in range(system.getForce(0).getNumExceptions()):
     # print(system.getForce(0).getExceptionParameters(n))
    
    ff.intialize_neutral_ff([], simulation.context)
    print("Energy after initialization:")
    print('total energy: %s' % (energy.getState(getEnergy=True).getPotentialEnergy()))
    print('nonbond: %s' % (energy.getState(getEnergy=True, groups={1}).getPotentialEnergy()))
    print('bond: %s' % (energy.getState(getEnergy=True, groups={2}).getPotentialEnergy()))
    print('angle: %s' % (energy.getState(getEnergy=True, groups={3}).getPotentialEnergy()))
    print('dihedral: %s' % (energy.getState(getEnergy=True, groups={4}).getPotentialEnergy()))
    #for n in range(system.getForce(0).getNumParticles()):
     # print(system.getForce(0).getParticleParameters(n))
    #for n in range(system.getForce(0).getNumExceptions()):
      #print(system.getForce(0).getExceptionParameters(n))
    print('--------------------')

    update_list_indx = np.array([1])
    list_QM_weight_ff_update = np.array([1.000])
    ff.update_ff(update_list_indx, list_QM_weight_ff_update, simulation.context, save_list=True)
    print("Energy after update:")
    print('total energy: %s' % (energy.getState(getEnergy=True).getPotentialEnergy()))
    print('nonbond: %s' % (energy.getState(getEnergy=True, groups={1}).getPotentialEnergy()))
    print('bond: %s' % (energy.getState(getEnergy=True, groups={2}).getPotentialEnergy()))
    print('angle: %s' % (energy.getState(getEnergy=True, groups={3}).getPotentialEnergy()))
    print('dihedral: %s' % (energy.getState(getEnergy=True, groups={4}).getPotentialEnergy())) 
    #for n in range(system.getForce(0).getNumParticles()):
     # print(system.getForce(0).getParticleParameters(n))
    #for n in range(system.getForce(0).getNumExceptions()):
    #  print(system.getForce(0).getExceptionParameters(n))

    
    ########## Extract the Molecule Index of the New Charged Center  ##########
    charged_mol_indx = list_indx[np.argwhere(list_QM_weight==list_QM_weight.max())].flatten()
    
    ########## Run MD Simulation ##########
    simulation.step(time_step)

    
if __name__ == '__main__':
    main()    

