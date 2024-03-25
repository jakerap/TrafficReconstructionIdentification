
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
from torch.autograd import Variable
import numpy as np

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

        self.t = t # time data points 16*[162, 1] there are 16 PVs
        self.x = x # space data points 16*[162, 1], actually each consecutive PV has more data points
        self.u = u # density data points 16*[162, 1]
        self.v = v # speed data points 16*[162, 1]

        self.x_f = X_f[:, 0:1] # space data points for the PDE [500, 1]
        self.t_f = X_f[:, 1:2] # time data points for the PDE [500, 1]
        self.t_g = t_g # time data points for the PDE 16*[50, 1]
        self.u_v = u_v # density data points for the PDE [50, 1]
        
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


    def calculate_losses(self, t, x, u, v, X_f, t_g, u_v, only_data=False):
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


        if only_data:
            return {
                "MSEu1": MSEu1, "MSEu2": MSEu2,
                "MSEtrajectories": MSEtrajectories,
                "MSEv1": MSEv1, "MSEv2": MSEv2
            }
        else:
            # Physics Losses
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
        
    def preprocess_datapoints(self, x):
        flattened_arrays = [arr.flatten() for arr in x] # Step 1: Flatten each ndarray
        concatenated_array = np.concatenate(flattened_arrays) # Concatenate the flattened arrays
        torch_tensor = torch.tensor(concatenated_array, dtype=torch.float32) # Step 3: Convert to a PyTorch tensor
        return torch_tensor

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

        X_f = torch.tensor(self.X_f, dtype=torch.float32)
        t_g = torch.tensor(self.t_g, dtype=torch.float32)
        u_v = torch.tensor(self.u_v, dtype=torch.float32)
        
        # Define the optimizer
        optimizer = optim.Adam(self.parameters(), lr=0.01)

        breakpoint()
       
        # Train the model
        for epoch in range(self.N_epochs):
            print(f"epoch {epoch}")
            optimizer.zero_grad()
            losses = self.calculate_losses(t, x, u, v, X_f, t_g, u_v, only_data=True)
            loss = sum([self.sigmas[i]*losses[key] for i, key in enumerate(losses.keys())])
            loss.backward()
            optimizer.step()
            if epoch % self.N_lambda == 0:
                self.gamma_var = self.gamma_var - 0.1 * self.gamma_var.grad
                self.noise_rho_bar = [self.noise_rho_bar[i] - 0.1 * self.noise_rho_bar[i].grad for i in range(self.N)]
        return losses
     
    # build pytorch network    
    def build_network(self, layers, act=nn.Tanh()):
        net = []
        for i in range(0, len(layers)-1):
            net.append(nn.Linear(layers[i], layers[i+1]))
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
        f = rho*self.net_v(rho)
        f_drho = torch.autograd.grad(f, rho, create_graph=True)[0]
        f_drhorho = torch.autograd.grad(f_drho, rho, create_graph=True)[0]
        return f_drhorho
    
    def net_F(self, rho):
        '''
        Characteristic speed
        '''
        v_pred = self.net_v(rho)
        v_pred_du = torch.autograd.grad(v_pred, rho, create_graph=True)[0]
        return v_pred + (rho+1)*v_pred_du
        # return -self.max_speed * u

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
        rho = self.net_u(t, x)
        rho_dt = torch.autograd.grad(rho, t, create_graph=True)[0]
        rho_dx = torch.autograd.grad(rho, x, create_graph=True)[0]
        rho_dxx = torch.autograd.grad(rho_dx, x, create_graph=True)[0]
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
        
        x_trajectories = self.net_x(t) 
        g = []
        for i in range(len(x_trajectories)):
            x_dt = torch.autograd.grad(x_trajectories[i], t[i], create_graph=True)[0]
            rho = self.net_u(t[i], x_trajectories[i])
            g.append(x_dt - self.net_v(rho))
        return g
    
    def predict_speed(self, u):
        '''
        Return the standardized estimated speed at u
        '''
        u = torch.tensor(u, dtype=torch.float32)
        return self.net_v(u).detach().numpy()
    
    def predict_F(self, u):
        '''
        Return the standardized estimated characteristic speed at u
        '''
        u = torch.tensor(u, dtype=torch.float32)
        return self.net_F(u).detach().numpy()
    
    def predict_trajectories(self, t):
        '''
        Return the standardized estimated agents' locations at t
        Parameters
      
        '''
        tf_dict = {}
        i = 0
        for k, v in zip(self.t_tf, t):
            tf_dict[k] = torch.tensor(v, dtype=torch.float32)
            i = i+1
        return self.net_x(torch.stack([tf_dict[k] for k in self.t_tf])).detach().numpy()
    