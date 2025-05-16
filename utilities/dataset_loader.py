import numpy as np
import torch
from torch.utils.data import Dataset
import pickle

class GroundTruthDataset(Dataset):
    def __init__(self, sys_name, device, dtype):
        # Load Ground Truth data
        
        if (sys_name == '1DBurgers'):
            self.py_data = pickle.load(open("./data/1DBG/1DBG_ns441_nx200_nt201.p", "rb"))
            self.dt = (self.py_data['data'][0]['t'][1] - self.py_data['data'][0]['t'][0]).item()
            self.dx = 0.03
            
            if dtype == 'double':
                self.z1 = torch.from_numpy(self.py_data['data'][0]['x']).double()
                self.dz = torch.from_numpy(self.py_data['data'][0]['dx']).double()
                self.mu = torch.from_numpy(np.array(self.py_data['param'])).double()
            elif dtype == 'float':
                self.z1 = torch.from_numpy(self.py_data['data'][0]['x']).float()
                self.dz = torch.from_numpy(self.py_data['data'][0]['dx']).float()
                self.mu = torch.from_numpy(np.array(self.py_data['param'])).float()
            
            self.dim_t = self.z1.shape[0]
            self.dim_z = self.z1.shape[1]
            self.len = self.dim_t - 1
            self.dim_mu = self.mu.shape[1]
            
            if device == 'gpu':
                self.dz = self.dz.to(torch.device("cuda"))
                self.mu = self.mu.to(torch.device("cuda"))
                
        elif (sys_name == 'Vlasov1D1V'):
            self.py_data = pickle.load(open("./data/Vlasov1D1V/Vlasov_ns441_nx1024_nt251_tstop5.00_dataset1.p", "rb"))
            self.dt = (self.py_data['data'][0]['t'][1] - self.py_data['data'][0]['t'][0]).item()
            self.dx = 14/32
            
            if dtype == 'double':
                self.z1 = torch.from_numpy(self.py_data['data'][0]['x']).double()
                self.dz = torch.from_numpy(self.py_data['data'][0]['dx']).double()
                self.mu = torch.from_numpy(np.array(self.py_data['param'])).double()
            elif dtype == 'float':
                self.z1 = torch.from_numpy(self.py_data['data'][0]['x']).float()
                self.dz = torch.from_numpy(self.py_data['data'][0]['dx']).float()
                self.mu = torch.from_numpy(np.array(self.py_data['param'])).float()
            
            self.dim_t = self.z1.shape[0]
            self.dim_z = self.z1.shape[1]
            self.len = self.dim_t - 1
            self.dim_mu = self.mu.shape[1]
            
            if device == 'gpu':
                self.dz = self.dz.to(torch.device("cuda"))
                self.mu = self.mu.to(torch.device("cuda"))
                

    def __getitem__(self, snapshot):
        z = self.z[snapshot, :]
        return z

    def __len__(self):
        return self.len

# def load_dataset(sys_name,dset_dir,device,dtype):
#     # Create Dataset instance
#     dataset = GroundTruthDataset(sys_name,device,dtype)

#     return dataset
