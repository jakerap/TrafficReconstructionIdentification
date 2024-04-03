
import logging
import os

# Delete some warning messages from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# import tensorflow as tf
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn.init as init

import matplotlib.pyplot as plt



class NeuralNetwork(nn.Module):

    def __init__(self, t, x, u, v, X_f, t_g, u_v,
                 layers_density, layers_trajectories, layers_speed, max_speed=None,
                 init_density=[[], []], init_trajectories=[[], []], init_speed=[[], []],
                 beta=0.05, N_epochs=1000, N_lambda=10, sigmas=[], opt=0):
        super(NeuralNetwork, self).__init__()
     
        '''
        Initialize a neural network for regression purposes.

        Parameters
        ----------
        t : list of N numpy array of shape (?,)
            standardized time coordinate of training points.
        x : list of N numpy array of shape (?,)
            standardized space coordinate of training points.
        u : list of N numpy array of shape (?,)
            standardized density values at training points.
        v : list of N numpy array of shape (?,)
            standardized velocity values at training points.
        X_f : 2D numpy array of shape (N_F, 2)
            standardized (space, time) coordinate of F physics training points.
        t_g : 1D numpy array of shape (N_G, 1)
            standardized time coordinate of G physics training points.
        u_v : 1D numpy array of shape (N_v, 1)
            standardized u coordinate of dV physics training points.
        layers_density : list of int (size N_L)
            List of integers corresponding to the number of neurons in each
            for the neural network Theta.
        layers_trajectories : list of int
            List of integers corresponding to the number of neurons in each 
            layer for the neural network Phi.
        layers_speed : list of int
            List of integers corresponding to the number of neurons in each 
            layer for the neural network V.
        init_density : list of two lists, optional
            Initial values for the weight and biases of Theta. 
            The default is [[], []].
        init_trajectories : list of two lists, optional
            Initial values for the weight and biases of Phi. 
            The default is [[], []].
        init_speed : list of two lists, optional
            Initial values for the weight and biases of V. 
            The default is [[], []].

        Returns
        -------
        None.

        '''
        
        # Parameters
        self.beta = beta # Regularization parameter
        self.N_epochs = N_epochs # Number of epochs
        self.N_lambda = N_lambda # Number of epochs before updating the lambdas
        self.opt = opt # Optimization method
        self.sigmas = sigmas # Weights for the loss functions / soft-hard constraints

        # data points
        self.t = t # time data points 16*[162, 1] there are 16 PVs
        self.x = x # space data points 16*[162, 1], actually each consecutive PV has more data points
        self.u = u # density data points 16*[162, 1]
        self.v = v # speed data points 16*[162, 1]


        # physics points
        self.x_f = torch.tensor(X_f[:, 0:1], dtype=torch.float32) # space data points for the PDE [500, 1]
        self.t_f = torch.tensor(X_f[:, 1:2], dtype=torch.float32) # time data points for the PDE [500, 1]
        self.t_g = torch.tensor(t_g, dtype=torch.float32) # time data points for the PDE 16*[50, 1]
        self.u_v = torch.tensor(u_v, dtype=torch.float32) # density data points for the PDE [50, 1]
        
        self.N = len(self.x)  # Number of agents
        self.max_speed = max_speed
        

        self.gamma_var = torch.tensor(1e-2, dtype=torch.float32, requires_grad=True) # Viscosity parameter
        self.noise_rho_bar = [torch.tensor(torch.randn(1, 1) * 0.001, dtype=torch.float32, requires_grad=True) 
                              for _ in range(self.N)] # Noise for the density


        # build density network
        self.density_network = self.build_network(layers_density, act=nn.Tanh())

        # build trajectories network
        self.trajectories_networks = [self.build_network(layers_trajectories, act=nn.Tanh()) for _ in range(self.N)]

        # build velocity network
        self.velocity_network = self.build_network(layers_speed, act=nn.Tanh())


    def calculate_losses(self, t, x, u, v, X_f, t_g, u_v):
        """
        Calculate and return the model losses based on inputs and physics.
        Parameters:
            t, x, u, v: Tensors representing time, space, density, and velocity at training points.
            X_f, t_g, u_v: Tensors representing physics-informed points and conditions.
        Returns:
            A dictionary of various loss components.
        """
        # Predictions
        u_pred = torch.stack([self.net_u(t[i], x[i]) for i in range(len(t))]) - torch.stack(self.noise_rho_bar)
        v_pred = self.net_v(u)
        x_pred = torch.stack([self.net_x_pv(t_g[i], i) for i in range(len(t_g))])

        # Data Losses
        MSEu1 = F.mse_loss(u, u_pred)  # MSE for Density Predictions (rho_pred - rho_true)^2
        MSEu2 = F.mse_loss(u, torch.stack(self.u_pred))  # MSE for Adjusted Density Predictions (rho_pred - rho_true - noise)^2
        
        MSEtrajectories = F.mse_loss(x, x_pred)  # MSE for Trajectories Predictions (x_pred - x_true)^2

        MSEv1 = F.mse_loss(v, v_pred)  # MSE for Speed Predictions (v_pred - v_true)^2
        MSEv2 = F.mse_loss(v, self.net_v(torch.stack(self.u_pred)))  # Placeholder, adjust as needed


        f_pred = self.net_f(X_f[:, 1], X_f[:, 0])  # Assuming X_f is [x, t]
        g_pred = self.net_g(t_g)

        MSEf = torch.mean(f_pred**2)  # Residual of PDE Predictions flux function
        MSEg = torch.mean(g_pred**2)  # Residual of PDE Predictions g function
        MSEv = torch.mean(torch.relu(self.net_ddf(u_v))**2)  # Placeholder, adjust as needed
        return {
            "MSEu1": MSEu1, "MSEu2": MSEu2, "MSEf": MSEf, 
            "MSEtrajectories": MSEtrajectories, "MSEg": MSEg,
            "MSEv1": MSEv1, "MSEv2": MSEv2, "MSEv": MSEv
        }
        

    def calculate_data_losses(self, t, x, u, v):
        # Predictions
        u_pred = [self.net_u(t[i], x[i]) - self.noise_rho_bar[i]*0 for i in range(len(t))] # list of tensors, since each tensor has a different shape
        v_pred = [self.net_v(u[i]) for i in range(len(t))]
        # x_pred = [self.net_x_pv(t[i], i) for i in range(len(t))]  # data points or physics points?
        # x_pred0 = self.net_x_pv(t[0], 0)
        # x_pred = torch.stack([self.net_x_pv(t_g[i], i) for i in range(len(t_g))]) # data points or physics points?

        # Data Losses
        MSEu1_list = [F.mse_loss(u[i], u_pred[i]) for i in range(len(u))] # L_2
        MSEu1 = torch.mean(torch.stack(MSEu1_list))  # MSE for Adjusted Density Predictions (rho_pred - rho_true - noise)^2

        # # MSEu2_l
        # MSEtrajectories_list = [F.mse_loss(x[i], x_pred[i]) for i in range(len(x))]
        # MSEtrajectories = torch.mean(torch.stack(MSEtrajectories_list))  # MSE for Trajectories Predictions (x_pred - x_true)^2
        # MSEtrajectories0 = F.mse_loss(x[0].squeeze(0), x_pred0.squeeze(0))  # MSE for Trajectories Predictions (x_pred - x_true)^2


        MSEv1_list = [F.mse_loss(v[i], v_pred[i]) for i in range(len(v))] # L_3
        MSEv1 = torch.mean(torch.stack(MSEv1_list))  # MSE for Adjusted Density Predictions (rho_pred - rho_true - noise)^2

        MSEv2_list = [F.mse_loss(v[i], self.net_v(u_pred[i])) for i in range(len(v))] # L_5
        MSEv2 = torch.mean(torch.stack(MSEv2_list))  # MSE for Adjusted Density Predictions (rho_pred - rho_true - noise)^2

        return {
            "MSEu1": MSEu1,
            # "MSEtrajectories": MSEtrajectories,
            # "MSEtrajectories": MSEtrajectories0,
            "MSEv1": MSEv1,
            "MSEv2": MSEv2
        }


    def calculate_physics_losses(self):
        f_pred = self.net_f(self.t_f, self.x_f)
        # g_pred = self.net_g(self.t_g)
        MSEf = torch.mean(f_pred**2)  # Residual of PDE Predictions flux function
        # MSEg = torch.mean(g_pred**2)  # Residual of PDE Predictions g function
        MSEv = torch.mean(torch.relu(self.net_ddf(self.u_v))**2)  # Placeholder, adjust as needed

        return {
            "MSEf": MSEf,
            # "MSEg": MSEg,
            "MSEv": MSEv
        }

        
    def preprocess_datapoints_flatten(self, x):
        flattened_arrays = [arr.flatten() for arr in x] # Step 1: Flatten each ndarray
        concatenated_array = np.concatenate(flattened_arrays) # Concatenate the flattened arrays
        torch_tensor = torch.tensor(concatenated_array, dtype=torch.float32) # Step 3: Convert to a PyTorch tensor
        return torch_tensor
    
    def preprocess_datapoints(self, x):
        return [torch.from_numpy(arr).float() for arr in x]

    # create a function to train the model with torch
    def train(self):
        '''
        Train the neural network
        '''
        # Convert the data to torch tensors
        t = self.preprocess_datapoints(self.t) # 3267 data points in total (all PV data combined)
        x = self.preprocess_datapoints(self.x)
        u = self.preprocess_datapoints(self.u)
        v = self.preprocess_datapoints(self.v)

        breakpoint()


        X_f = torch.tensor(self.x_f, dtype=torch.float32)
        t_g = torch.tensor(self.t_g, dtype=torch.float32)
        u_v = torch.tensor(self.u_v, dtype=torch.float32)
        
        # Define the optimizer
        optimizer = optim.Adam(self.parameters(), lr=0.001)
       
        # Train the model
        self.N_epochs = 1000
        with tqdm(total=self.N_epochs, desc="Training Progress") as pbar:
            for epoch in tqdm(range(self.N_epochs)):
                
                optimizer.zero_grad()
                losses_data = self.calculate_data_losses(t, x, u, v)
                losses_physics = self.calculate_physics_losses()
                # print(losses['MSEtrajectories'])
                loss = 0.01*sum(losses_data.values()) + sum(losses_physics.values())
                # loss = losses["MSEtrajectories"]
                # loss = sum([self.sigmas[i]*losses[key] for i, key in enumerate(losses.keys())])
                # loss = sum([self.sigmas[i]*losses[key] for i, key in enumerate(losses.keys())])
                pbar.set_postfix({'loss': loss.item()})
                pbar.update(1)

                loss.backward()
                optimizer.step()
                if epoch % self.N_lambda == 0:
                    pass
                    # self.gamma_var = self.gamma_var - 0.1 * self.gamma_var.grad
                    # self.noise_rho_bar = [self.noise_rho_bar[i] - 0.1 * self.noise_rho_bar[i].grad for i in range(self.N)]
            return loss
     
    # build pytorch network    
    def build_network(self, layers, act=nn.Tanh()):
        net = []
        for i in range(len(layers)-1):
            linear_layer = nn.Linear(layers[i], layers[i+1])
            init.xavier_normal_(linear_layer.weight)
            net.append(linear_layer)
            net.append(act)
        return nn.Sequential(*net)

    # predictors
    def net_v(self, rho):
        '''
        Standardized velocity
        '''
        v_tanh = torch.square(self.velocity_network(rho))
        if self.max_speed is None:
            return v_tanh*(1-rho)
        else:
            return (v_tanh*(1+rho)/2 + self.max_speed)*(1-rho)/2
    
    def net_ddf(self, rho):
        '''
        Standardized second derivative of the flux
        N_v[v] := f_rhorho = (rho*v(rho))_rhorho = 2*v_rho + rho*v(rho)_rhorho
        '''
        rho.requires_grad_(True)

        f = rho*self.net_v(rho)
        f_drho = torch.autograd.grad(f, rho, torch.ones_like(f), create_graph=True)[0]
        f_drhorho = torch.autograd.grad(f_drho, rho, torch.ones_like(f_drho), create_graph=True)[0]
        return f_drhorho
    
    def net_F(self, rho):
        '''
        Characteristic speed
        '''
        rho.requires_grad_(True)
        v_pred = self.net_v(rho)
        v_pred_du = torch.autograd.grad(v_pred, rho, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0]
        return v_pred + (rho + 1) * v_pred_du
        # return -self.max_speed * rho

    def net_u(self, t, x):
        '''
        return the standardized value of rho hat at position (t, x)
        '''
        rho_pred = self.density_network(torch.cat([t,x],1)) # TODO: were the encoders needed?
        return rho_pred

    def net_f(self, t, x):
        '''
        return the physics function f at position (t,x)
        N_rho[rho, v] := rho_t + F(rho) * rho_x - gamma^2 * rho_xx
        '''
        t.requires_grad_(True)
        x.requires_grad_(True)
        rho = self.net_u(t, x)

        # Assuming rho is not scalar and needs grad_outputs for proper gradient computation
        grad_outputs = torch.ones_like(rho)

        rho_dt = torch.autograd.grad(rho, t, grad_outputs=grad_outputs, create_graph=True)[0]
        rho_dx = torch.autograd.grad(rho, x, grad_outputs=grad_outputs, create_graph=True)[0]
        rho_dxx = torch.autograd.grad(rho_dx, x, grad_outputs=grad_outputs, create_graph=True)[0]

        f = rho_dt + self.net_F(rho) * rho_dx - self.gamma_var**2 * rho_dxx
        return f
    
    def net_x_pv(self, t, i=0):
        x_tanh = self.trajectories_networks[i](t)
        return x_tanh
    
    def net_x(self, t):
        '''
        Return the standardized position of each agent.
        '''
        output = [self.net_x_pv(t[i], i) for i in range(self.N)]
        return output
    
    def net_g(self, t):
        '''
        return the physics function g for all agents at time t
        N_y[y_i] := x_t - v(rho(t, x(t)))
        '''
        t.requires_grad_(True)
        x_trajectories = self.net_x(t)

        g = []
        for i in range(len(x_trajectories)):
            grad_outputs = torch.ones_like(x_trajectories[i])
            x_dt = torch.autograd.grad(x_trajectories[i], t[i], grad_outputs=grad_outputs, create_graph=True)[0]
            rho = self.net_u(t[i], x_trajectories[i])
            g.append(x_dt - self.net_v(rho))
        return g
    
    def predict_speed(self, u):
        '''
        Return the standardized estimated speed at u
        '''
        u = torch.tensor(u, dtype=torch.float32).requires_grad_(True)
        return self.net_v(u).detach().numpy()
    
    def predict_F(self, u):
        '''
        Return the standardized estimated characteristic speed at u
        '''
        u = torch.tensor(u, dtype=torch.float32).requires_grad_(True)
        return self.net_F(u).detach().numpy()
    
    def predict_trajectories(self, t):
        '''
        Return the standardized estimated agents' locations at t
        Parameters
        '''        
        predictions = []  # Define the variable "predictions"
        t = torch.tensor(t, dtype=torch.float32)

        predictions = self.net_x(t)

        return predictions
        tf_dict = {}
        i = 0
        for t in zip(self.t):
            tf_dict[k] = torch.tensor(v, dtype=torch.float32)
            i = i+1
        return self.net_x(torch.stack([tf_dict[k] for k in self.t_f])).detach().numpy()
    