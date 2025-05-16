import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
from utilities import max_err_heatmap, eval_model

def get_cmap(n, name='tab20'):
    return plt.cm.get_cmap(name, n)
cmap = get_cmap(10)
plt.rcParams["text.usetex"] = False
plt.style.use("default")

# Check GPU status
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    device = torch.device("cuda:0")  # Change the index to select a different GPU
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU.")

#%% Load Data
folder = "BG_tLaSDI-pGFINN_REC1e-01_JAC1e-09_MOD1e-07_iter1000"
param_dim = 2 # dimension of parameterization

amp_train = np.linspace(0.7, 0.9, 3)
width_train = np.linspace(0.9, 1.1, 3)
amp_test = np.linspace(0.7, 0.9, 21)
width_test = np.linspace(0.9, 1.1, 21)
num_train = amp_train.size * width_train.size
num_test = amp_test.size * width_test.size

data = pickle.load(open("./data/1DBG/1DBG_ns441_nx200_nt201.p", "rb"))
nt = data['data'][0]['x'].shape[0]

# Indices of training samples
path = "./outputs/" + folder + "/train_indices.p"
temp =  torch.load(open(path, "rb"))
train_indices = temp['train_indices']
# print(train_indices)

#%% Load trained model
path = "./outputs/" + folder + "/model_best.pkl"
model_best =  torch.load(open(path, "rb"))
path = "./outputs/" + folder + "/model_best_AE.pkl"
model_best_AE =  torch.load(open(path, "rb"))
net = model_best
AE = model_best_AE

#%% Evaluate Trained Model
x_AE,z_tlasdi_all,x_tlasdi_all,S_net,dSdt_net,dEdt_net,z_tt_all = eval_model(AE,net,data,param_dim,device)

#%% Plot Loss History
path = "./outputs/" + folder + "/loss_history_value.p"
loss =  torch.load(open(path, "rb"))
loss_history = np.array(loss['loss_history'])
loss_GFINNs_history = np.array(loss['loss_GFINNs_history'])
loss_AE_history = np.array(loss['loss_AE_history'])
loss_AE_jac_history = np.array(loss['loss_AE_jac_history'])
loss_dx_history = np.array(loss['loss_dx_history'])
loss_dz_history = np.array(loss['loss_dz_history'])

plt.rc('text', usetex=False)
fig = plt.figure(figsize=(10,6))
plt.plot(loss_history[:,0], loss_history[:,1], label='Total', zorder=3)
plt.plot(loss_GFINNs_history[:,0], loss_GFINNs_history[:,1], label='pGFINN', linewidth=2, zorder=3)
plt.plot(loss_AE_history[:,0], loss_AE_history[:,1], label='AE_recon', linewidth=2, zorder=3)
plt.plot(loss_AE_jac_history[:,0], loss_AE_jac_history[:,1], label='AE_jac', linewidth=2, zorder=3)
plt.plot(loss_dx_history[:,0], loss_dx_history[:,1], label='dx', linewidth=2, zorder=3)
plt.plot(loss_dz_history[:,0], loss_dz_history[:,1], label='dz', linewidth=2, zorder=3)
plt.xlabel('Epochs', fontsize=16)
plt.yscale('log')
plt.xlim(left=0)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.minorticks_on()
plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
plt.grid(which='major', color='black', linestyle='-', linewidth=0.5)
plt.legend(fontsize=16, loc='best')
plt.tight_layout()
plt.savefig("./outputs/" + folder + "/loss.png")

#%% Plot Latent Dynamics
idx = 0
xt = data['data'][0]['t']
tidx = np.arange(0,nt,5)

# Encoder Representation of Latent Dynamics
x_AE_i = x_AE[idx*nt:(idx+1)*nt,:].cpu().detach().numpy()

# pGFINN-predicted Latent Dynamics
x_tlasdi = x_tlasdi_all[idx*nt:(idx+1)*nt,:].cpu().numpy()

# Normalization
z_max = 0
for i in range(x_tlasdi.shape[1]):
    z_max_i = x_AE_i[:,i].max()
    if z_max_i > z_max:
        z_max = z_max_i

plt.rc('text', usetex=False)
fig = plt.figure(figsize=(6,4))
for i in range(x_tlasdi.shape[1]):
    plt.plot(xt,x_AE_i[:,i] / z_max, '-', lw=2, c=cmap(i), zorder=3)
    plt.plot(xt[tidx],x_tlasdi[tidx,i] / z_max, '*--', lw=2, markersize=5, c=cmap(i), zorder=3)
    
plt.xlabel('Time', fontsize=14)
plt.ylabel('z', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(left=0)
plt.minorticks_on()
plt.grid(which='major', color='gray', linestyle=':', linewidth=0.5)
plt.legend(['Encoder', 'pGFINN'], loc='best', frameon=False, fontsize=18)
plt.tight_layout()
plt.savefig("./outputs/" + folder + "/latent_dynamics.png")

#%% Plot Entropy
plt.rc('text', usetex=False)
fig = plt.figure(figsize=(6,4))
line, = plt.plot(xt, S_net[idx*nt:(idx+1)*nt].detach().numpy(), 'r', lw=2, zorder=3)
plt.xlabel('Time')
plt.ylabel('S')
plt.xlim(left=0)
plt.minorticks_on()
plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
plt.grid(which='major', color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.savefig("./outputs/" + folder + "/entropy.png")

#%% Plot Entropy Production Rate
fig = plt.figure(figsize=(6,4))
plt.plot(xt, dSdt_net[idx*nt:(idx+1)*nt].detach().numpy(), 'r', lw=2, zorder=3)
plt.xlabel('Time')
plt.ylabel('dS/dt')
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.minorticks_on()
plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
plt.grid(which='major', color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.savefig("./outputs/" + folder + "/entropy_rate.png")

#%% Plot Energy Rate
plt.rc('text', usetex=False)
fig = plt.figure(figsize=(6,4))
plt.plot(xt, dEdt_net[idx*nt:(idx+1)*nt].detach().numpy(), 'b', lw=2, zorder=3)
plt.xlabel('Time')
plt.ylabel('dE/dt')
plt.xlim(left=0)
plt.ylim([-0.1,0.1])
plt.minorticks_on()
plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
plt.grid(which='major', color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.savefig("./outputs/" + folder + "/energy_rate.png")

#%% Plot Max. Relative Errors Across the Parameter Space
grid2, grid1 = np.meshgrid(width_train, amp_train)
train_param = np.hstack((grid1.flatten().reshape(-1, 1), grid2.flatten().reshape(-1, 1)))
grid2, grid1 = np.meshgrid(width_test, amp_test)
test_param = np.hstack((grid1.flatten().reshape(-1, 1), grid2.flatten().reshape(-1, 1)))

mu = torch.from_numpy(np.array(data['param'])).float()
mu_tr1 = mu[train_indices,:]

a_grid, w_grid = np.meshgrid(amp_test, width_test)
param_list = np.hstack([a_grid.flatten().reshape(-1,1), w_grid.flatten().reshape(-1,1)])
a_grid, w_grid = np.meshgrid(np.arange(amp_test.size), np.arange(width_test.size))
idx_list = np.hstack([a_grid.flatten().reshape(-1,1), w_grid.flatten().reshape(-1,1)])

idx_param = []
for i,ip in enumerate(mu_tr1.cpu().numpy()):
    idx = np.argmin(np.linalg.norm(param_list-ip, axis=1))
    idx_param.append((idx, np.array([param_list[idx,0], param_list[idx,1]])))
    
max_err = np.zeros([len(amp_test), len(width_test)])
count = 0
for i,a in enumerate(amp_test):
    for j,w in enumerate(width_test):
        # Max error of all time steps
        max_array_tmp = (np.linalg.norm(z_tt_all[count*nt:(count+1)*nt].cpu() - z_tlasdi_all[count*nt:(count+1)*nt].cpu(), axis=1) / np.linalg.norm(z_tt_all[count*nt:(count+1)*nt].cpu(), axis=1)*100)
        max_array = np.expand_dims(max_array_tmp, axis=0)
        max_err[i,j] = max_array.max()
        count += 1
        
plt.rcParams["text.usetex"] = False
data_path = './outputs/' + folder + "/"
max_err_heatmap(max_err, amp_test, width_test, data_path, idx_list, idx_param, 
                      xlabel='k', ylabel='T', dtype='float')