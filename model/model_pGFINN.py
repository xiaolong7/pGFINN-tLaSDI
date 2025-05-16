import learner as ln
import torch
from learner.utils import mse, grad
from learner.integrator import RK

class LNN(ln.nn.Module):
    def __init__(self, ind, K, layers=2, width=50, activation='relu', xi_scale=0.01, param_dim=0):

        super(LNN, self).__init__()
        self.S = ln.nn.FNN(ind + param_dim, 1, layers, width, activation)
        self.ind = ind
        self.K = K
        self.sigComp = ln.nn.FNN(ind + param_dim, K**2 , layers, width, activation)
        self.xi_scale = xi_scale
        self.param_dim = param_dim
        self.__init_params() # skew symmetric matrices
        
    def forward(self, x):
        sigComp = self.sigComp(x).reshape(-1, self.K, self.K)
        sigma = sigComp - torch.transpose(sigComp, -1, -2) # B

        x = x.requires_grad_(True)
        S = self.S(x)
        dS = grad(S, x)
        if self.param_dim > 0:
            dS = dS[...,:-self.param_dim]
        dS = dS.reshape([-1,self.ind])
        ddS = dS.unsqueeze(-2)
        B = []
        for i in range(self.K):
            xi = torch.triu(self.xi[i], diagonal = 1)
            xi = xi - torch.transpose(xi, -1,-2)
            B.append(ddS@xi)
        B = torch.cat(B, dim = -2)
        L = torch.transpose(B,-1,-2) @ sigma @ B
        if len(dS.size()) == 1:
            dS = dS.unsqueeze(0)
        return dS, L
        
    def __init_params(self):
        self.xi = torch.nn.Parameter((torch.randn([self.K, self.ind, self.ind])*self.xi_scale).requires_grad_(True)) 

class MNN(ln.nn.Module):
    def __init__(self, ind, K, layers=2, width=50, activation='relu', xi_scale=0.01, param_dim=0):
        super(MNN, self).__init__()
        self.E = ln.nn.FNN(ind + param_dim, 1, layers, width, activation)
        self.ind = ind
        self.K = K
        self.sigComp = ln.nn.FNN(ind + param_dim, K**2 , layers, width, activation)
        self.xi_scale = xi_scale
        self.param_dim = param_dim
        self.__init_params() # skew symmetric matrices
        
    def forward(self, x):
        sigComp = self.sigComp(x).reshape(-1, self.K, self.K)
        sigma = sigComp @ torch.transpose(sigComp, -1, -2)  # B

        x = x.requires_grad_(True)
        E = self.E(x)
        dE = grad(E, x)
        if self.param_dim > 0:
            dE = dE[...,:-self.param_dim]
        dE = dE.reshape([-1,self.ind])
        ddE = dE.unsqueeze(-2)
        B = []
        for i in range(self.K):
            xi = torch.triu(self.xi[i], diagonal = 1)
            xi = xi - torch.transpose(xi, -1,-2)
            B.append(ddE@xi)
        
        B = torch.cat(B, dim = -2)
        M = torch.transpose(B,-1,-2) @ sigma @ B
        
        if len(dE.size()) == 1:
            dE = dE.unsqueeze(0)
        return dE, M
        
    def __init_params(self):
        self.xi = torch.nn.Parameter((torch.randn([self.K, self.ind, self.ind])*self.xi_scale).requires_grad_(True))

class pGFINN(ln.nn.LossNN):

    def __init__(self, netS, netE, dt, order=1, iters=1, kb=1, b_dim=1, fluc=False, param_dim=0):
        super(pGFINN, self).__init__()
        self.netS = netS
        self.netE = netE
        self.dt = dt
        self.iters = iters
        self.fluc = fluc
        self.b_dim = b_dim
        self.integrator = RK(self.f, order=order, iters=iters)
        self.loss = mse
        self.param_dim = param_dim
        
    def f(self, x):
        dE, M = self.netE(x)
        dS, L = self.netS(x)
        dE = dE.unsqueeze(1)
        dS = dS.unsqueeze(1)
        y = -(dE @ L).squeeze() + (dS @ M).squeeze()
        if len(y.shape) < 2:
            y = y.view(1,-1)
        y = torch.cat((y, torch.zeros((y.shape[0],self.param_dim), device=x.device)), axis=1)
        return y
    
    def g(self, x):
        return self.netE.B(x)

    def criterion(self, X, y):      
        X_next = self.integrator.solve(X, self.dt)
        loss = self.loss(X_next, y)
        return loss

    def integrator2(self, X):
        X_next = self.integrator.solve(X, self.dt)
        return X_next

    def predict(self, x0, k, return_np=False):
        x = torch.transpose(self.integrator.flow(x0, self.dt, k - 1), 0, 1)
        if return_np:
            x = x.detach().cpu().numpy()
        return x
