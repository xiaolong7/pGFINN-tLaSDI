import torch
import torch.nn as nn

#%% SiLU activation
def silu(input):
    return input * torch.sigmoid(input)

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return silu(input)
    
#%% Standard AutoEncoder
class AutoEncoder(nn.Module):
    """
    Autoencoders with symmetric architecture for the encoder and the decoder. 
    Linear activation is applied to the embedding layer and the output layer.
    All other hidden layers have the same activation.
    
    input args:
        - layer_vec: list, contains the neuron numbers in each layer 
                of the Encoder, including the input layer.
        - activation: int, activation for the hidden layers of the Encoder
                and the Decoder; ReLU, Tanh, Sigmoid, SiLU
    """

    def __init__(self, layer_vec, activation='relu'):
        super(AutoEncoder,self).__init__()
        self.encoder = self._make_layers_encoder(layer_vec, activation)
        self.decoder = self._make_layers_decoder(layer_vec, activation)
        
    def _make_layers_encoder(self, layer_vec, activation):
        layers = []
        for i in range(len(layer_vec)-2):
            layers.append(nn.Linear(layer_vec[i],layer_vec[i+1]))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'silu':
                layers.append(SiLU())
        layers.append(nn.Linear(layer_vec[-2],layer_vec[-1]))
        return nn.Sequential(*layers)
    
    def _make_layers_decoder(self, layer_vec, activation):
        layers = []
        for i in range(len(layer_vec)-2):
            layers.append(nn.Linear(layer_vec[-1-i],layer_vec[-2-i]))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'silu':
                layers.append(SiLU())
        layers.append(nn.Linear(layer_vec[1],layer_vec[0]))
        return nn.Sequential(*layers)
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
        
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
    
    def JVP(self, z, x, dz, dx, trunc_period):

        dim_z = z.shape[1]

        idx_trunc = range(0, dim_z - 1, trunc_period)  # 3 for VC, 10 for BG

        def decode_trunc(xx):
            xx = self.decode(xx)
            return xx[:,idx_trunc]

        def jvp_de(xa, dxa):
            decode_wrapper = lambda xx: decode_trunc(xx)
            J_f_x = torch.autograd.functional.jvp(decode_wrapper, xa, dxa,create_graph=True)
            J_f_x_v = J_f_x[1]
            J_f_x = None
            return J_f_x_v
        
        def jvp_en(za, dza):
            encode_wrapper = lambda zz: self.encode(zz)
            J_f_x = torch.autograd.functional.jvp(encode_wrapper, za, dza,create_graph=True)
            J_f_x_v = J_f_x[1]
            J_f_x = None
            return J_f_x_v

        J_dV = jvp_de(x, dx)
        J_eV = jvp_en(z, dz)
        J_edV = jvp_de(x, J_eV)

        return J_edV, J_eV, J_dV, idx_trunc
    
if __name__ == '__main__':
    pass
