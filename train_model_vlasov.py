# 1D/1V Vlasov-Poisson Equation

import argparse
import torch
from model import LNN, MNN, pGFINN
import learner as ln
from utilities import str2bool, GroundTruthDataset
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = False

def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    problem = 'Vlasov1D1V'

    #-----------------------------------------------------------------------------
    # Autoencoder parameters
    latent_dim = args.latent_dim 
    AE_activation = args.AE_activation
    layer_vec_AE = [1024,500,200,100,latent_dim]
    
    #-----------------------------------------------------------------------------
    # pGFINN parameters
    layers = args.layers  # pGFINNs structure
    width = args.width
    pgfinn_activation = args.pgfinn_activation # pGFINN activation
    param_dim = args.param_dim # dimension of parameterization
    order = args.order
    iters = 1
    extraD_L = args.extraD_L
    extraD_M = args.extraD_M
    xi_scale = args.xi_scale

    #-----------------------------------------------------------------------------
    # Training parameters
    device = args.device  # 'cpu' or 'gpu'
    dtype = args.dtype
    print_every = 200
    batch_size = args.batch_size
    update_epochs = args.update_epochs
    n_train_max = args.n_train_max
    lr = args.lr # initial learning rate
    epochs = args.epochs
    gamma_lr = args.gamma_lr
    miles_lr = args.miles_lr
    load_model = args.load_model
    load_epochs = args.load_epochs
    weight_decay_GFINNs = args.weight_decay_GFINNs
    weight_decay_AE = args.weight_decay_AE
    lambda_r_AE = args.lambda_r_AE
    lambda_jac_AE = args.lambda_jac_AE
    lambda_dx = args.lambda_dx
    lambda_dz = args.lambda_dz
    trunc_period = args.trunc_period
    
    #--------------------------------------------------------------------------------
    # Load data
    # dataset = load_dataset('Vlasov1D1V','data',device,dtype)
    dataset = GroundTruthDataset('Vlasov1D1V',device,dtype)
    
    #--------------------------------------------------------------------------------
    # Load model
    if args.load_model:
        AE_name = '_REC'+"{:.0e}".format(lambda_r_AE) + '_JAC'+ "{:.0e}".format(lambda_jac_AE) + '_MOD'+"{:.0e}".format(lambda_dx) + '_iter'+str(epochs+load_epochs)
    else:
        AE_name = '_REC'+"{:.0e}".format(lambda_r_AE) + '_JAC'+ "{:.0e}".format(lambda_jac_AE) + '_MOD'+"{:.0e}".format(lambda_dx) + '_iter'+str(epochs)

    load_path =  problem + '_tLaSDI-pGFINN' + '_REC'+"{:.0e}".format(lambda_r_AE) + '_JAC'+ "{:.0e}".format(lambda_jac_AE) + '_MOD'+"{:.0e}".format(lambda_dx) + '_iter'+str(load_epochs)

    path = problem + '_tLaSDI-pGFINN' + AE_name

    #--------------------------------------------------------------------------------
    # Create pGFINN
    netS = LNN(latent_dim, extraD_L, layers=layers, width=width, activation=pgfinn_activation, xi_scale=xi_scale, param_dim=param_dim)
    netE = MNN(latent_dim, extraD_M, layers=layers, width=width, activation=pgfinn_activation, xi_scale=xi_scale, param_dim=param_dim)
    net = pGFINN(netS, netE, dataset.dt / iters, order=order, iters=iters, param_dim=param_dim)

    #--------------------------------------------------------------------------------
    # Train and test tLaSDI
    args2 = {
        'net': net,
        'sys_name':'Vlasov1D1V',
        'output_dir': 'outputs',
        'save_plots': True,
        'criterion': None,
        'optimizer': 'adam',
        'lr': lr,
        'epochs': epochs,
        'AE_name': AE_name,
        'dset_dir': 'data',
        'output_dir_AE': 'outputs',
        'save_plots_AE': True,
        'layer_vec_AE': layer_vec_AE,
        'activation_AE': AE_activation,
        'num_sensor': param_dim,
        'lr_AE': 1e-4,
        'lambda_r_AE': lambda_r_AE,
        'lambda_jac_AE': lambda_jac_AE,
        'lambda_dx': lambda_dx,
        'lambda_dz': lambda_dz,
        'miles_lr': miles_lr,
        'gamma_lr': gamma_lr,
        'weight_decay':weight_decay_GFINNs,
        'weight_decay_AE':weight_decay_AE,
        'path': path,
        'load_path': load_path,
        'batch_size': batch_size,
        'update_epochs':update_epochs,
        'print_every': print_every,
        'save': True,
        'load': load_model,
        'callback': None,
        'dtype': dtype,
        'device': device,
        'tol': 1e-3,
        'tol2': 2,
        'adaptive': 'reg_max',
        'n_train_max': n_train_max,
        'subset_size_max': 80,
        'trunc_period': trunc_period
    }

    ln.Brain_tLaSDI_pGFINN.Init(**args2)

    ln.Brain_tLaSDI_pGFINN.Run()

    ln.Brain_tLaSDI_pGFINN.Restore()

    ln.Brain_tLaSDI_pGFINN.Output()

    ln.Brain_tLaSDI_pGFINN.Test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tLaSDI')

    parser.add_argument('--seed', default=0, type=int, help='random seed')

    parser.add_argument('--device', type=str, choices=["gpu", "cpu"], default="gpu",
                        help='deviced used')

    parser.add_argument('--dtype', type=str, choices=["float", "double"], default="float",
                        help='data type used')

    #--------------------------------------------------------------------------------
    # Autoencoder parameters

    parser.add_argument('--latent_dim', type=int, default=5,
                        help='latent space dimension')

    parser.add_argument('--AE_activation', default='relu', type=str,
                        help='activation function for autoencoder')
    
    #--------------------------------------------------------------------------------
    # pGFINN parameters

    parser.add_argument('--param_dim', type=int, default=2,
                        help='dimension of parameterization')
    
    parser.add_argument('--pgfinn_activation', default='tanh', type=str,
                        help='activation function for pGFINN')

    parser.add_argument('--layers', type=int, default=5,
                        help='# of layers for pGFINN')

    parser.add_argument('--width', type=int, default=40,
                        help='width of pGFINN')

    parser.add_argument('--extraD_L', type=int, default=9,
                        help='# of skew-symmetric matrices generated to construct L')
    
    parser.add_argument('--extraD_M', type=int, default=9,
                        help='# of skew-symmetric matrices generated to construct M')

    parser.add_argument('--xi_scale', type=float, default=.3333,
                        help='initialization scale of skew-symmetric matrices of pGFINN')

    parser.add_argument('--order', type=int, default=1,
                        help='time integrator for pGFINN, 1:Euler, 2:RK23, 4:RK45')
    
    #--------------------------------------------------------------------------------
    # Training parameters

    parser.add_argument('--batch_size', default=50, type=int,
                        help='batch size')
    
    parser.add_argument('--load_model', default=False, type=str2bool,
                        help='load previously trained model')

    parser.add_argument('--epochs', type=int, default=50000,
                        help='number of training epochs')
    
    parser.add_argument('--load_epochs', type=int, default=50000,
                        help='number of epochs for loaded network')

    parser.add_argument('--lambda_r_AE', type=float, default=1e-1,
                        help='weight for Reconstruction loss')

    parser.add_argument('--lambda_jac_AE', type=float, default=1e-9,
                        help='weight for Jacobian loss')

    parser.add_argument('--lambda_dx', type=float, default=1e-7,
                        help='weight for physical dynamics consistency of Model loss')

    parser.add_argument('--lambda_dz', type=float, default=1e-6,
                        help='weight for latent dynamics consistency of Model loss')

    parser.add_argument('--trunc_period', type=int, default=1,
                        help='truncation indices for Jacobian computations') # only consider every 'trunc_period'th index in Jacobian computation
    
    
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='initial learning rate')
    
    parser.add_argument('--miles_lr',  type=int, default=2000,
                        help='learning rate decay frequency')

    parser.add_argument('--gamma_lr', type=float, default=.99,
                        help='rate of learning rate decay')

    parser.add_argument('--weight_decay_GFINNs', type=float, default=0,
                        help='weight decay rate for pGFINN')
    
    parser.add_argument('--weight_decay_AE', type=float, default=0,
                        help='weight decay rate for AE')

    parser.add_argument('--update_epochs', type=int, default=10000000,
                        help='greedy sampling frequency')

    parser.add_argument('--n_train_max', type=int, default=9,
                        help='max number of training samples for adaptive sampling')

    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    main(args)