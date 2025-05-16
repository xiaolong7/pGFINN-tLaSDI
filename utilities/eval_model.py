import torch
import numpy as np

def eval_model(AE,net,data,param_dim,device):
    z_tt_all = torch.from_numpy(np.array([]))
    pred_indices = np.arange(len(data['data'])) 
    nt = data['data'][0]['x'].shape[0]
    
    for j in pred_indices:
        z_tt_all = torch.cat((z_tt_all, torch.from_numpy(data['data'][j]['x'])), 0)
                
    z_tt_all = z_tt_all.to(dtype=torch.float32).to(device)
    
    mu1 = torch.from_numpy(np.array(data['param'])).to(dtype=torch.float32).to(device)
    mu_tt_all = mu1[pred_indices, :]
    
    mu_pred_all = torch.repeat_interleave(mu_tt_all, nt, dim=0)
    
    z0 = z_tt_all[::nt, :]
    
    mu0 = mu_pred_all[::nt, :]
    
    chunk_size = int(z_tt_all.shape[0]/10)
    z_tt_chunks = torch.chunk(z_tt_all, chunk_size, dim=0)
    mu_chunks = torch.chunk(mu_pred_all, chunk_size, dim=0)
    
    z_ae_chunks = []
    x_all_chunks = []
    for z_tt_chunk, mu_chunk in zip(z_tt_chunks,mu_chunks):
        with torch.no_grad():
            z_ae_chunk, x_all_chunk = AE(z_tt_chunk.detach())
            z_ae_chunks.append(z_ae_chunk)
            x_all_chunks.append(x_all_chunk)
    x_all_all = torch.cat(x_all_chunks,dim=0)
        
    # Encode Initial Condition
    _, x0 = AE(z0) 
    
    # Latent Space Dynamics Prediction
    x_net = torch.zeros(x_all_all.shape).to(dtype=torch.float32).to(device)
    x_net[::nt,:] = x0
    dSdt_net = torch.zeros(x_all_all.shape[0]).to(dtype=torch.float32)
    dEdt_net = torch.zeros(x_all_all.shape[0]).to(dtype=torch.float32)
    S_net = torch.zeros(x_all_all.shape[0]).to(dtype=torch.float32)
    E_net = torch.zeros(x_all_all.shape[0]).to(dtype=torch.float32)
    
    if (param_dim > 0):
        x0 = torch.cat((x0, mu0), dim=1)
    dE, M = net.netE(x0)
    E = net.netE.E(x0)
    
    dS, L = net.netS(x0)
    S = net.netS.S(x0)
    
    dE = dE.unsqueeze(1)
    dS = dS.unsqueeze(1)
    
    dEdt = torch.sum(dE.squeeze()* ((dE @ L) + (dS @ M)).squeeze(),1)
    dSdt = torch.sum(dS.squeeze()* ((dE @ L) + (dS @ M)).squeeze(),1)
    S = S.squeeze()
    E = E.squeeze()
    
    dEdt_net[::nt] = dEdt
    dSdt_net[::nt] = dSdt
    S_net[::nt] = S
    E_net[::nt] = E
    
    for snapshot in range(nt - 1):
        x1_net = net.integrator2(x0.detach())
        x_net[snapshot + 1::nt, :] = x1_net[:,:x_net.shape[1]]
        
        x0 = x1_net
        dE, M = net.netE(x0)
        dS, L = net.netS(x0)
    
        S = net.netS.S(x0)
        S = S.squeeze()
        E = net.netE.E(x0)
        E = E.squeeze()
        
        dE = dE.unsqueeze(1)
        dS = dS.unsqueeze(1)
    
        dEdt = torch.sum(dE.squeeze() * ((dE @ L) + (dS @ M)).squeeze(), 1)
        dSdt = torch.sum(dS.squeeze() * ((dE @ L) + (dS @ M)).squeeze(), 1)
    
        dEdt_net[snapshot+1::nt] = dEdt
        dSdt_net[snapshot+1::nt] = dSdt
        S_net[snapshot+1::nt] = S
        E_net[snapshot+1::nt] = E
        
    x_tlasdi_all = x_net
    x_net = None
    x_tlasdi_all = x_tlasdi_all.detach()
    chunk_size = int(x_tlasdi_all.shape[0]/10)
    x_tlasdi_chunks = torch.chunk(x_tlasdi_all, chunk_size, dim=0)
    mu_chunks = torch.chunk(mu_pred_all, chunk_size, dim=0)
    
    # Decode Latent Dynamics to Physical Dynamics
    z_tlasdi_chunks = []
    for x_tlasdi_chunk, mu_chunk in zip(x_tlasdi_chunks,mu_chunks):
        with torch.no_grad():
            z_tlasdi_chunk = AE.decode(x_tlasdi_chunk.detach())
            z_tlasdi_chunks.append(z_tlasdi_chunk)
    z_tlasdi_all = torch.cat(z_tlasdi_chunks,dim=0)
    
    # Encoder Representation of Latent Dynamics
    _,x_AE = AE(z_tt_all)
    
    return x_AE,z_tlasdi_all,x_tlasdi_all,S_net,dSdt_net,dEdt_net,z_tt_all