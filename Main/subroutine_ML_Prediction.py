"""
This script uses a well-trained ML model to predict the electronic couling (off-diagonal elements of the Hamiltonian).
When you implement the ML model, please make sure:
    (1) Defining the "Model" class, which describes the ML architecture 
        and must be consistent with the architecture training the ML model,
    (2) Specifing the path of file, denoted as "ML_model_filename", 
        which saves the parameters of the well-trained ML model,
    (3) Specifing the computation devieces (CPU or GPU) denoted as "device",
        which is required by Pytorch 
    (4) The input ML feature is a required variable denoted as "X" [pairs, CM],
        the type of ML feature must be consistent with the features training the ML model.

The return arrays include:
    (1) EC_predict: a 1-D array of electronic couling [pair]

Being aware that:
    (1)The unit of electronic coupling obtained from ML model is eV!!!
    (2) Using CUDA may be slower than using CPU because it takes time on arranging GPU memory.
    (3) Because the ML model fails predicting coupling value for pair distacne larger than 0.8 nm,
        the coupling is set to zero for the pairs whose distacne is larger than 0.8 nm.

The main program runs a simple OpenMM file to test "ml_coupling"
"""

import numpy as np
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
from scipy.spatial import distance
from subroutine_Molecule_List import molecule_list
from subroutine_Feature_Generation import generate_CM
import torch
import torch.nn as nn
from torch.autograd import Variable

########### Define neural network architecture ###########
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        ##### Set actativation function #####
        self.linear_in = torch.nn.Linear(78, 78)
        self.linear_hid_JCTC_1 = torch.nn.Linear(78, 36)
        self.linear_hid_JCTC_2 = torch.nn.Linear(36, 36)
        self.linear_hid_JCTC_3 = torch.nn.Linear(36, 36)
        self.linear_hid_JCTC_4 = torch.nn.Linear(36, 36)
        self.linear_out = torch.nn.Linear(36, 1)
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)
        self.batchnorm = torch.nn.BatchNorm1d(78)
        self.batchnorm_JCTC_1 = torch.nn.BatchNorm1d(78)
        self.batchnorm_JCTC_2 = torch.nn.BatchNorm1d(36)
        self.batchnorm_JCTC_3 = torch.nn.BatchNorm1d(36)
        self.batchnorm_JCTC_4 = torch.nn.BatchNorm1d(36)

    def forward(self, x):
        ##### Input Layer #####
        H1 = self.leaky_relu(self.linear_in(self.batchnorm(x)))
        ##### Hidden Layer #####
        H2 = self.leaky_relu(self.linear_hid_JCTC_1(self.batchnorm_JCTC_1(H1)))
        H3 = self.leaky_relu(self.linear_hid_JCTC_2(self.batchnorm_JCTC_2(H2)))
        H4 = self.leaky_relu(self.linear_hid_JCTC_3(self.batchnorm_JCTC_3(H3)))
        H5 = self.leaky_relu(self.linear_hid_JCTC_4(self.batchnorm_JCTC_4(H4)))
        ##### Output Layer #####
        y_pred = self.linear_out(H5)
        return y_pred


def ml_coupling(X, pair_COM_dis):    
    ########### Set ML parameters ###########
    device = "cpu" # cuda or cpu
    ML_model_filename = "MSE_160K_CM_BN_H5_BSize500_Constant00060_Best_Model.pth"

    ########### Setup PyTorch Data Format ###########
    X = Variable(torch.from_numpy(np.array(X))).float()
    
    if device == "cuda":
    #  print('Data are processed on GPU.')
      X = X.cuda() 
    #elif device == "cpu":
    #  print('Data are processed on CPU.')
    #else:
    #  raise ValueError('Invalid device name %s, please type CUDA or CPU!' % device)

    ########### Loading Model ###########
    model = Model()
    model.load_state_dict(torch.load(ML_model_filename, map_location=torch.device(device)))
    model.to(torch.device(device))

    ########### Print model information ###########
    #for param_tensor in model.state_dict():
        #print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        #print(param_tensor, "\t", model.state_dict()[param_tensor])
    #k = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(k)

    ########### Preditcion electronic coupling (EC) by ML Model ###########
    model.eval()
    EC_predict = model.forward(X).detach()
    
    ########### Transform into Numpy format ###########
    EC_predict = EC_predict.cpu().numpy().flatten()
    #print(EC_predict)
    
    ########### Define the coupling value as zero for long-distance pairs ###########
    long_dis_index = np.argwhere(pair_COM_dis > 0.80)
    EC_predict[long_dis_index] = 0.0000000000

    ########### Save predicted coupling value ###########
    #np.savetxt('ML_EC.txt', EC_predict.cpu().numpy())
    return(EC_predict)
    

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
    print(CM_intra_inter_atom.shape)
    #np.savetxt('CM_inter.txt', CM_inter)
    #np.savetxt('CM_intra_inter_atom.txt', CM_intra_inter_atom)
    
    ########## Predict Electronic Coupling (Hij) by ML Model ##########
    Hij = ml_coupling(CM_intra_inter_atom, pair_COM_dis)
    print(Hij)
    print(Hij.shape)
    
    
if __name__ == '__main__':
    main()    

