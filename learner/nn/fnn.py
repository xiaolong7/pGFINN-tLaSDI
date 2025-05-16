import torch.nn as nn
import torch
import torch.nn.functional as F

from .module import StructureNN

class FNN(StructureNN):
    '''Fully connected neural networks.
    '''
    def __init__(self, ind, outd, layers=2, width=50, activation='relu', initializer='default', softmax=False):
    #def __init__(self, ind, outd, layers=2, width=50, activation='relu', initializer='kaiming_uniform', softmax=False):

        super(FNN, self).__init__()
        self.ind = ind
        self.outd = outd
        self.layers = layers
        self.width = width
        self.activation = activation
        self.initializer = initializer
        self.softmax = softmax
        
        self.modus = self.__init_modules()
        self.__initialize()
        
        self.alpha = nn.Parameter(torch.tensor(1.0))

        self.activation_vec = (self.layers - 1) * [self.activation] 

    def activation_function(self, x, activation):
        if activation == 'linear': x = x
        elif activation == 'sigmoid': x = torch.sigmoid(x)
        elif activation == 'relu': x = F.relu(x)
        elif activation == 'rrelu': x = F.rrelu(x)
        elif activation == 'tanh': x = torch.tanh(x)
        elif activation == 'sin': x = torch.sin(x)
        elif activation == 'elu': x = F.elu(x)
        elif activation == 'gelu':x = F.gelu(x)
        elif activation == 'silu':x = F.silu(x)
        elif activation == 'Ad10_tanh':x = torch.tanh(10*self.alpha*x)
        elif activation == 'Ad1_tanh':x = torch.tanh(1*self.alpha*x)
        elif activation == 'Ad5_tanh':x = torch.tanh(5*self.alpha*x)
        elif activation == 'Ad01_tanh':x = torch.tanh(.1*self.alpha*x)
        elif activation == 'Ad_sin':x = torch.sin(10*self.alpha*x)
        else: raise NotImplementedError
        return x
        
    def forward(self, x):
        idx = 0
        for i in range(1, self.layers):
            LinM = self.modus['LinM{}'.format(i)]
            x = self.activation_function(LinM(x), self.activation_vec[idx])
            idx += 1
        x = self.modus['LinMout'](x)
        if self.softmax:
            x = nn.functional.softmax(x, dim=-1)
        return x
    
    def __init_modules(self):
        modules = nn.ModuleDict()
        if self.layers > 1:
            modules['LinM1'] = nn.Linear(self.ind, self.width)
            for i in range(2, self.layers):
                modules['LinM{}'.format(i)] = nn.Linear(self.width, self.width)
            modules['LinMout'] = nn.Linear(self.width, self.outd)
        else:
            modules['LinMout'] = nn.Linear(self.ind, self.outd)
            
        return modules
    
    def __initialize(self):
        for i in range(1, self.layers):
            self.weight_init_(self.modus['LinM{}'.format(i)].weight)
            nn.init.constant_(self.modus['LinM{}'.format(i)].bias, 0)
        self.weight_init_(self.modus['LinMout'].weight)
        nn.init.constant_(self.modus['LinMout'].bias, 0)