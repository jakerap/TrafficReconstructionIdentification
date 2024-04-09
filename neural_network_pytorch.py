
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


def print_data_ranges(x, name=""):
    if isinstance(x, list):
        print(f"range of values {name}: [{min([min(xs) for xs in x])}, - {max([max(xs) for xs in x])}")
    else:
        print(f"range of values {name}: [{x.min()}, - {x.max()}]")
      


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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'
        print(f"Device: {self.device}")

        # data points
        self.t = self.preprocess_datapoints(t) # time data points 16*[162, 1] there are 16 PVs
        self.x = self.preprocess_datapoints(x) # space data points 16*[162, 1], actually each consecutive PV has more data points
        self.u = self.preprocess_datapoints(u)
        self.v = self.preprocess_datapoints(v)

        # physics points
        self.x_f = torch.tensor(X_f[:, 0:1], dtype=torch.float32, requires_grad=True).to(self.device) # space data points for the PDE [500, 1]
        self.t_f = torch.tensor(X_f[:, 1:2], dtype=torch.float32, requires_grad=True).to(self.device) # time data points for the PDE [500, 1]
        self.t_g = torch.tensor(t_g, dtype=torch.float32, requires_grad=True).to(self.device) # time data points for the PDE 16*[50, 1]
        self.u_v = torch.tensor(u_v, dtype=torch.float32, requires_grad=True).to(self.device) # density data points for the PDE [50, 1]

        # weights of the losses
        self.lambdas = {
            "MSEf": torch.tensor(1, dtype=torch.float32, requires_grad=True).to(self.device),
            "MSEg": torch.tensor(1, dtype=torch.float32, requires_grad=True).to(self.device),
            "MSEv": torch.tensor(1, dtype=torch.float32, requires_grad=True).to(self.device),
            "MSEgamma": torch.tensor(1, dtype=torch.float32, requires_grad=True).to(self.device),
            "MSEu1": torch.tensor(1, dtype=torch.float32, requires_grad=True).to(self.device),
            "MSEu2": torch.tensor(1, dtype=torch.float32, requires_grad=True),
            "MSEtrajectories": torch.tensor(1, dtype=torch.float32, requires_grad=True).to(self.device),
            "MSEv1": torch.tensor(1, dtype=torch.float32, requires_grad=True).to(self.device),
            "MSEv2":torch.tensor(1, dtype=torch.float32, requires_grad=True).to(self.device)
        }


        self.N = len(self.x)  # Number of agents
        self.max_speed = max_speed

        # print_data_ranges(self.t, "t")
        # print_data_ranges(self.x, "x")
        # print_data_ranges(self.u, "u")
        # print_data_ranges(self.v, "v")
        # print_data_ranges(self.x_f, "x_f")
        # print_data_ranges(self.t_f, "t_f")
        # print_data_ranges(self.t_g, "t_g")
        # print_data_ranges(self.u_v, "u_v")

        # learnable parameters
        self.gamma_var = torch.nn.Parameter(torch.tensor(1e-2, dtype=torch.float32), requires_grad=True).to(self.device)  # Correct use of dtype # Viscosity parameter
        self.noise_rho_bar = [torch.nn.Parameter(torch.randn(1, 1).to(self.device) * 0.001, requires_grad=True) for _ in range(self.N)] # Noise for the density


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

    def plot_density_loss_on_trajectory(self):
        plt.figure(figsize=(12, 6))  # Set figure size, adjust as needed
        
        # First subplot for predicted u values
        plt.subplot(3, 3, 1)  # 1 row, 2 columns, subplot 1
        for (t, x, u) in zip(self.t, self.x, self.u):  # No need to iterate over u for prediction plot
            u_pred = self.net_u(t, x)
            u_pred = u_pred.detach().numpy()
            plt.scatter(t.detach().numpy(), x.detach().numpy(), c=u_pred, cmap='rainbow', vmin=-1, vmax=1, s=5)
        plt.xlabel(r'Time [min]')
        plt.ylabel(r'Position [km]')
        plt.colorbar(label='Predicted u')
        # plt.title('Predicted Values at data points')
        plt.title('Predicted at data points')
        
        # Second subplot for actual u values
        plt.subplot(3, 3, 2)  # 1 row, 2 columns, subplot 2
        for (t, x, u) in zip(self.t, self.x, self.u):
            plt.scatter(t.detach().numpy(), x.detach().numpy(), c=u.detach().numpy(), cmap='rainbow', vmin=-1, vmax=1, s=5)
        plt.xlabel(r'Time [min]')
        plt.ylabel(r'Position [km]')
        plt.colorbar(label='Actual u')
        plt.title('Actual Values')

        plt.subplot(3, 3, 3)  # 1 row, 2 columns, subplot 1
        for (t, x, u) in zip(self.t, self.x, self.u):  # No need to iterate over u for prediction plot
            u_pred = self.net_u(t, x)
            u_pred = u_pred.detach().numpy()
            plt.scatter(t.detach().numpy(), x.detach().numpy(), c=u_pred-u.detach().numpy(), cmap='rainbow', vmin=0, vmax=1, s=5)
        plt.xlabel(r'Time [min]')
        plt.ylabel(r'Position [km]')
        plt.colorbar(label='Predicted u')
        # plt.title('Predicted Values at data points')
        plt.title('Prediction error at data points')

        plt.subplot(3, 3, 4)  # 1 row, 2 columns, subplot 2
        u_pred = self.net_u(self.t_f, self.x_f)
        plt.scatter(self.t_f.detach().numpy(), self.x_f.detach().numpy(), c=u_pred.detach().numpy(), cmap='rainbow', vmin=-1, vmax=1, s=15)
        plt.xlabel(r'Time [min]')
        plt.ylabel(r'Position [km]')
        plt.colorbar(label='Predicted u')
        plt.title('Predicted Values at physics points')

        t_min, t_max = self.t_f.min().item(), self.t_f.max().item()  # Replace with actual min/max of time
        x_min, x_max = self.x_f.min().item(), self.x_f.max().item()  # Replace with actual min/max of position
        t_grid, x_grid = np.meshgrid(np.linspace(t_min, t_max, 100), np.linspace(x_min, x_max, 100))

        # Flatten the grid for model evaluation and convert to tensor if necessary
        t_flat = t_grid.flatten()
        x_flat = x_grid.flatten()
        # Assuming your model expects PyTorch tensors; adjust as necessary
        t_flat_tensor = torch.tensor(t_flat, dtype=torch.float32).unsqueeze(1)
        x_flat_tensor = torch.tensor(x_flat, dtype=torch.float32).unsqueeze(1)

        # Evaluate the model on the grid points
        # Note: You might need to adjust this part based on your model's input format
        u_pred_flat = self.net_u(t_flat_tensor, x_flat_tensor).detach().numpy()

        # Reshape the flat predictions back into the grid shape for plotting
        u_pred_grid = u_pred_flat.reshape(t_grid.shape)

        plt.subplot(3, 3, 5)  
        plt.pcolormesh(t_grid, x_grid, u_pred_grid, cmap='rainbow', shading='auto', vmin=-1, vmax=1)
        plt.xlabel(r'Time [min]')
        plt.ylabel(r'Position [km]')
        plt.colorbar(label='Predicted u')
        plt.title('Predicted Values at Physics Points')

        # fundamental diagram
        plt.subplot(3, 3, 6)
        rho = np.linspace(-1, 1, 100)
        rho = torch.tensor(rho, dtype=torch.float32).unsqueeze(1)
        v_pred = self.predict_speed(rho)
        F_pred = self.predict_F(rho)
        for (t, x, u, v) in zip(self.t, self.x, self.u, self.v):
            plt.scatter(u.detach().numpy(), v.detach().numpy(), c='blue', s=5)
        rho = np.linspace(-1,1,100)
        plt.plot(rho, v_pred, c='red')
        plt.plot(rho, F_pred, c='green')
        # plt.plot(rho, ((rho+1)/2)*v_pred, c='purple')
        
        plt.legend(['v', 'F', 'rho*v'])

        # density over position at different times
        plt.subplot(3, 3, 7)  # 1 row, 2 columns, subplot 1
        x = np.linspace(-1, 1, 100)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        u_pred = self.net_u(torch.ones_like(x)*-1, x)
        x = x.detach().numpy()
        u_pred = u_pred.detach().numpy()
        plt.plot(x, u_pred, c='red')
        plt.xlabel(r'Position [km]')
        plt.ylabel(r'Density')
        plt.grid()
        plt.title('Density prediction at time -1')

        plt.subplot(3, 3, 8)  # 1 row, 2 columns, subplot 1
        x = np.linspace(-1, 1, 100)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        u_pred = self.net_u(torch.ones_like(x)*0, x)
        u_pred = u_pred.detach().numpy()
        plt.plot(x.detach().numpy(), u_pred, c='red')
        plt.xlabel(r'Position [km]')
        plt.ylabel(r'Density')
        plt.grid()
        plt.title('Density prediction at time 0')

        plt.subplot(3, 3, 9)  # 1 row, 2 columns, subplot 1
        x = np.linspace(-1, 1, 100)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        u_pred = self.net_u(torch.ones_like(x), x)
        u_pred = u_pred.detach().numpy()
        plt.plot(x.detach().numpy(), u_pred, c='red')
        plt.xlabel(r'Position [km]')
        plt.ylabel(r'Density')
        plt.grid()
        plt.title('Density prediction at time 1')        

        plt.tight_layout()  # Adjust layout
        plt.show()


    def check_network_device(self):
        # Example for checking the device of the density network
        density_network_device = next(self.density_network.parameters()).device
        print(f'Density Network Device: {density_network_device}')

        # Example for checking the device of the first trajectories network
        # Assuming self.trajectories_networks is not empty
        if self.trajectories_networks:
            trajectories_network_device = next(self.trajectories_networks[0].parameters()).device
            print(f'First Trajectories Network Device: {trajectories_network_device}')

        # Example for checking the device of the velocity network
        velocity_network_device = next(self.velocity_network.parameters()).device
        print(f'Velocity Network Device: {velocity_network_device}')

    def calculate_data_losses(self):

        # Predictions
        u_pred = [self.net_u(self.t[i], self.x[i]) - self.noise_rho_bar[i]*0 for i in range(len(self.t))] # list of tensors, since each tensor has a different shape
        x_pred = [self.net_x_pv(self.t[i], i) for i in range(len(self.t))]  
        u_pred_pv_position = [self.net_u(self.t[i], x_pred[i]) - self.noise_rho_bar[i]*0 for i in range(len(self.t))]
        v_pred = [self.net_v(self.u[i]) for i in range(len(self.t))]

        # L_2
        MSEu1_list = [F.mse_loss(self.u[i], u_pred[i]) for i in range(len(self.u))] # L_2
        MSEu1 = torch.mean(torch.stack(MSEu1_list))  # MSE for Adjusted Density Predictions (rho_pred - rho_true - noise)^2

        # L_4
        MSEu2_list = [F.mse_loss(self.u[i], u_pred_pv_position[i]) for i in range(len(self.u))] # L_4
        MSEu2 = torch.mean(torch.stack(MSEu2_list))  

        # L_1
        MSEtrajectories_list = [F.mse_loss(self.x[i], x_pred[i]) for i in range(len(self.x))]
        MSEtrajectories = torch.mean(torch.stack(MSEtrajectories_list))  # MSE for Trajectories Predictions (x_pred - x_true)^2

        # L_3
        MSEv1_list = [F.mse_loss(self.v[i], v_pred[i]) for i in range(len(self.v))] # L_3
        MSEv1 = torch.mean(torch.stack(MSEv1_list))  # MSE for Adjusted Density Predictions (rho_pred - rho_true - noise)^2

        # L_5
        MSEv2_list = [F.mse_loss(self.v[i], self.net_v(u_pred[i])) for i in range(len(self.v))] # L_5
        MSEv2 = torch.mean(torch.stack(MSEv2_list))  # MSE for Adjusted Density Predictions (rho_pred - rho_true - noise)^2

        return {
            "MSEu1": MSEu1,
            # "MSEu2": MSEu2,
            "MSEtrajectories": MSEtrajectories,
            "MSEv1": MSEv1,
            "MSEv2": MSEv2
        }


    def calculate_physics_losses(self):
        f_pred = self.net_f(self.t_f, self.x_f)
        g_pred = self.net_g(self.t_g)
        MSEf = torch.mean(f_pred**2)  # Residual of PDE Predictions flux function
        MSEg = torch.mean(torch.stack(g_pred)**2) # Residual of PDE Predictions g function
        MSEv = torch.mean(torch.relu(self.net_ddf(self.u_v))**2)  # Placeholder, adjust as needed
        MSEgamma = self.gamma_var**2

        return {
            # "MSEf": MSEf/100,
            # "MSEg": MSEg/100,
            "MSEv": MSEv,
            # "MSEgamma": MSEgamma
        }

        
    def preprocess_datapoints_flatten(self, x):
        flattened_arrays = [arr.flatten() for arr in x] # Step 1: Flatten each ndarray
        concatenated_array = np.concatenate(flattened_arrays) # Concatenate the flattened arrays
        torch_tensor = torch.tensor(concatenated_array, dtype=torch.float32, requires_grad=True) # Step 3: Convert to a PyTorch tensor with required gradient
        return torch_tensor
    
    def preprocess_datapoints(self, x):
        return [torch.from_numpy(arr).float().requires_grad_(True).to(self.device) for arr in x]

    # create a function to train the model with torch
    def train(self):
        '''
        Train the neural network
        '''

        # losses tracking
        loss_history = {}
        gamma_var_history = []
        noise_rho_bar_history = []
        lambdas_history = {key: [] for key in self.lambdas.keys()}


        # Define the optimizer
        # all_params = list(self.density_network.parameters()) + list(self.velocity_network.parameters())
        optimizer = optim.Adam(self.parameters(), lr=0.005)
        optimizer = optim.Adam([
            {'params': self.density_network.parameters()},
            {'params': self.velocity_network.parameters()},
            {'params': [p for traj_net in self.trajectories_networks for p in traj_net.parameters()]}, # added this way such that it is optimized as well
            # {'params': self.noise_rho_bar},  # Assuming noise_rho_bar is a list of parameters/tensors
            # {'params': [self.gamma_var]},
            # {'params': self.lambdas.values()}
        ], lr=0.005)

        
        # optimizer = optim.Adam(list(self.parameters()) + [self.gamma_var] + [*self.noise_rho_bar], lr=0.005)
        # optimizer = optim.LBFGS(self.parameters(), lr=0.005)
       
        # Train the model
        self.N_epochs = 5000
        # self.plot_density_loss_on_trajectory()
        with tqdm(total=self.N_epochs, desc="Training Progress") as pbar:
            for epoch in range(self.N_epochs):

                if epoch > self.N_epochs:
                    optimizer = optim.LBFGS(self.parameters(), lr=0.1, max_iter=20, history_size=100)

                    def closure():
                        optimizer.zero_grad()
                        losses_data = self.calculate_data_losses()
                        losses_physics = self.calculate_physics_losses()
                        losses = {**losses_data, **losses_physics}

                        loss = sum(losses.values())
                        loss.backward()
                        return loss
                    
                    optimizer.step(closure)
                    
                    loss = closure()
                
                else:
                
                    optimizer.zero_grad()
                    losses_data = self.calculate_data_losses()
                    losses_physics = self.calculate_physics_losses()
                    losses = {**losses_data, **losses_physics}
                    # weighted_loss = {key: losses[key] * torch.sigmoid(self.lambdas[key]) for key in losses}

                    # loss = sum(losses_data.values()) + sum(losses_physics.values())
                    loss = sum(losses.values())
                    # if epoch < 4000:
                    #     loss = sum(losses_data.values())
                    # else:
                    #     loss = sum(losses.values())
                        # loss = sum(weighted_loss.values())
                    # loss = sum([self.sigmas[i]*losses[key] for i, key in enumerate(losses.keys())])

                    loss.backward()
                    optimizer.step()


                # print(self.gamma_var, self.noise_rho_bar[0])

                pbar.set_postfix({'loss': loss.item()})
                pbar.update(1)

                combined_losses = {**losses_data, **losses_physics}
                for key, value in combined_losses.items():
                    if key not in loss_history:
                        loss_history[key] = []
                    # Assuming the loss values are tensors, use .item() to get Python numbers
                    loss_history[key].append(value.item())
                

                gamma_var_history.append(self.gamma_var.item())
                noise_rho_bar_history.append([tensor.item() for tensor in self.noise_rho_bar])

                for key, value in self.lambdas.items():
                    lambdas_history[key].append(torch.sigmoid(value).item())


            self.plot_density_loss_on_trajectory()
            return loss_history, gamma_var_history, noise_rho_bar_history, lambdas_history
        
     
    # build pytorch network    
    def build_network(self, layers, act=nn.Tanh()):
        net = []
        for i in range(len(layers)-1):
            linear_layer = nn.Linear(layers[i], layers[i+1])
            init.xavier_normal_(linear_layer.weight)
            net.append(linear_layer)
            net.append(act)
        return nn.Sequential(*net).to(self.device)

    # predictors
    def net_v(self, rho):
        '''
        Standardized velocity
        '''
        # v_tanh = torch.square(self.velocity_network(rho))
        v_tanh = self.velocity_network(rho)
        return (v_tanh+1)/2 * self.max_speed * (1-rho) # scaled
        if self.max_speed is not None:
            return v_tanh*(1-rho)
        else:
            return (v_tanh*(1+rho)/2 + self.max_speed)*(1-rho)/2
    
    def net_ddf(self, rho):
        '''
        Standardized second derivative of the flux
        N_v[v] := f_rhorho = (rho*v(rho))_rhorho = 2*v_rho + rho*v(rho)_rhorho
        '''

        f = rho*self.net_v(rho)
        f_drho = torch.autograd.grad(f, rho, torch.ones_like(f), create_graph=True)[0]
        f_drhorho = torch.autograd.grad(f_drho, rho, torch.ones_like(f_drho), create_graph=True)[0]
        return f_drhorho
    
    def net_F(self, rho):
        '''
        Characteristic speed
        '''
        # rho.requires_grad_(True)
        v_pred = self.net_v(rho)
        v_pred_drho = torch.autograd.grad(v_pred, rho, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0]
        return v_pred + (rho + 1)/2 * v_pred_drho
        # return v_pred + (rho + 1) * v_pred_drho # good?
        return v_pred + rho * v_pred_drho
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
        rho = self.net_u(t, x)

        rho_dt = torch.autograd.grad(rho, t, grad_outputs=torch.ones_like(rho), create_graph=True)[0]
        rho_dx = torch.autograd.grad(rho, x, grad_outputs=torch.ones_like(rho), create_graph=True)[0]
        rho_dxx = torch.autograd.grad(rho_dx, x, grad_outputs=torch.ones_like(rho_dx), create_graph=True)[0]

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
    
    # def net_g(self, t):
    #     '''
    #     return the physics function g for all agents at time t
    #     N_y[y_i] := x_t - v(rho(t, x(t)))
    #     '''
    #     x_trajectories = self.net_x(t)

    #     g = []
    #     for i in range(len(x_trajectories)):
    #         grad_outputs = torch.ones_like(x_trajectories[i])
    #         x_dt = torch.autograd.grad(x_trajectories[i], t[i], grad_outputs=grad_outputs, create_graph=True)[0]
    #         rho = self.net_u(t[i], x_trajectories[i])
    #         g.append(x_dt - self.net_v(rho))
    #     return g
    

    def net_g(self, t):
        '''
        return the physics function g for all agents at time t
        N_y[y_i] := x_t - v(rho(t, x(t)))
        '''

        g = []

        for i in range(len(t)):

            x_trajectories = self.net_x_pv(t[i], i)# - self.noise_rho_bar[i] # TODO: is the noise needed?

            x_dt = torch.autograd.grad(x_trajectories, t, grad_outputs=torch.ones_like(x_trajectories), create_graph=True)[0] # have to use the whole t and then select the i-th element
            # print(x_dt[i])
            rho = self.net_u(t[i], x_trajectories)
            g.append(x_dt[i] - self.net_v(rho))

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
    