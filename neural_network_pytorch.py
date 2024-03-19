
import logging
import os

# Delete some warning messages from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf
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
        
        
        self.beta = beta # Regularization parameter
        self.N_epochs = N_epochs # Number of epochs
        self.N_lambda = N_lambda # Number of epochs before updating the lambdas
        self.opt = opt # Optimization method
        self.sigmas = sigmas # Weights for the loss functions / soft-hard constraints

        self.t = t # time data points
        self.x = x # space data points
        self.u = u # density data points
        self.v = v # speed data points

        self.x_f = X_f[:, 0:1] # space data points for the PDE
        self.t_f = X_f[:, 1:2] # time data points for the PDE
        self.t_g = t_g # time data points for the PDE
        self.u_v = u_v # density data points for the PDE
        
        self.N = len(self.x)  # Number of agents
        self.max_speed = max_speed


        self.gamma_var = torch.tensor(1e-2, dtype=torch.float32, requires_grad=True) # Viscosity parameter
        self.noise_rho_bar = [torch.tensor(torch.randn(1, 1) * 0.001, dtype=torch.float32, requires_grad=True) 
                              for _ in range(self.N)] # Noise for the density

        # build density network
        self.density_network = self.build_network(layers_density, act=nn.Tanh())

        # build trajectories network
        self.trajectories_network = [self.build_network(layers_trajectories, act=nn.Tanh()) for _ in range(self.N)]

        # build velocity network
        self.velocity_network = self.build_network(layers_speed, act=nn.Tanh())


        # PDE part     
        self.t_tf = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(self.N)]
        self.x_tf = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(self.N)]
        self.u_tf = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(self.N)]
        self.v_tf = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(self.N)]
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])
        self.u_v_tf = tf.placeholder(tf.float32, shape=[None, self.u_v.shape[1]])
        
        self.u_pred = [self.net_u(self.t_tf[i], self.net_x_pv(self.t_tf[i], i)) - self.noise_rho_bar[i]
                   for i in range(self.N)] 
        self.f_pred = self.net_f(self.t_f_tf, self.x_f_tf)        
        
        # Agents part
        self.t_g_tf = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(self.N)]
        
        self.x_pred = self.net_x(self.t_tf)
        self.g_pred = self.net_g(self.t_g_tf)

        # MSE part
        self.MSEu1 = torch.mean((torch.cat(self.u_tf, dim=0) - self.net_u(torch.cat([self.t_tf, self.x_tf], dim=1)))**2)    # MSE for Density Predictions (rho_pred - rho_true)^2
        self.MSEu2 = torch.mean((torch.cat(self.u_tf, dim=0) - torch.cat(self.u_pred, dim=0))**2)                           # MSE for Adjusted Density Predictions (rho_pred - rho_ture - noise)^2
        self.MSEf =  torch.mean(self.f_pred**2)                                                                             # Residual of PDE Predictions flux function 
        
        self.MSEtrajectories = torch.mean((torch.cat(self.x_tf, dim=0) - torch.cat(self.x_pred, dim=0))**2)                 # MSE for Trajectories Predictions (x_pred - x_true)^2
        self.MSEg = torch.mean(torch.cat(self.g_pred, dim=0)**2)                                                            # Residual of PDE Predictions g function 
            
        self.MSEv1 = torch.mean((torch.cat(self.v_tf, dim=0) - self.net_v(torch.cat(self.u_tf, dim=0)))**2)                 # MSE for Speed Predictions (v_pred - v_true)^2
        self.MSEv2 = torch.mean((torch.cat(self.v_tf, dim=0) - self.net_v(torch.cat(self.u_pred, dim=0)))**2)               # MSE for Adjusted Speed Predictions (v_pred - v_true - noise)^2
        self.MSEv = torch.mean(torch.relu(self.net_ddf(self.u_v_tf))**2)                                                    # MSE for PDE Predictions speed function 

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
        f_pred = self.net_f(X_f[:, 1], X_f[:, 0])  # Assuming X_f is [x, t]
        x_pred = torch.stack([self.net_x_pv(t_g[i], i) for i in range(len(t_g))])
        g_pred = self.net_g(t_g)

        # Losses
        MSEu1 = F.mse_loss(u, u_pred)  # MSE for Density Predictions (rho_pred - rho_true)^2
        MSEu2 = F.mse_loss(u, torch.stack(self.u_pred))  # MSE for Adjusted Density Predictions (rho_pred - rho_true - noise)^2
        MSEf = torch.mean(f_pred**2)  # Residual of PDE Predictions flux function
        MSEtrajectories = F.mse_loss(x, x_pred)  # MSE for Trajectories Predictions (x_pred - x_true)^2
        MSEg = torch.mean(g_pred**2)  # Residual of PDE Predictions g function
        MSEv1 = F.mse_loss(v, v_pred)  # MSE for Speed Predictions (v_pred - v_true)^2
        # For MSEv2, adjust the code to compute adjusted speed predictions as needed
        MSEv2 = F.mse_loss(v, self.net_v(torch.stack(self.u_pred)))  # Placeholder, adjust as needed
        # For MSEv, you will need to implement net_ddf method
        MSEv = torch.mean(torch.relu(self.net_ddf(u_v))**2)  # Placeholder, adjust as needed
        
        return {
            "MSEu1": MSEu1, "MSEu2": MSEu2, "MSEf": MSEf, 
            "MSEtrajectories": MSEtrajectories, "MSEg": MSEg,
            "MSEv1": MSEv1, "MSEv2": MSEv2, "MSEv": MSEv
        }
     
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
        x_tanh = self.trajectories_network[i](t)
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


    def train(self):
        '''
        Train the neural networks

        Returns
        -------
        None.
        '''
        
        tf_dict = { }
        
        for k, v in zip(self.x_tf, self.x):
            tf_dict[k] = v
            
        for k, v in zip(self.t_tf, self.t):
            tf_dict[k] = v
            
        for k, v in zip(self.u_tf, self.u):
            tf_dict[k] = v
            
        for k, v in zip(self.v_tf, self.v):
            tf_dict[k] = v
            
        for k, v in zip(self.t_g_tf, self.t_g):
            tf_dict[k] = v
            
        tf_dict[self.t_f_tf] = self.t_f
        tf_dict[self.x_f_tf] = self.x_f
        tf_dict[self.u_v_tf] = self.u_v
        
        if self.opt == 1 or self.opt == 2 or self.opt == 9:
            tf_dict[self.c_tf] = 1e-2
        
        for i in range(len(self.optimizer)):
            for j in range(len(self.lambdas_tf)):
                tf_dict[self.lambdas_tf[j]] = self.lambdas_init[j]
            print('---> STEP %.0f' % (i+1))
            self.epoch = 1
            self.saved_lambdas = self.optimizer[i].train(tf_dict, i+1)    
    
    
    def predict_speed(self, u):
        '''
        Return the standardized estimated speed at u
        '''
        u = np.float32(u)
        return self.sess.run(self.net_v(u))
        # return self.net_v(u)
    
    def predict_F(self, u):
        '''
        Return the standardized estimated characteristic speed at u
        '''
        u = np.float32(u)
        u_tf = tf.placeholder(tf.float32, shape=[None, 1])
        return self.sess.run(self.net_F(u_tf), feed_dict={u_tf: u})
    
    def predict_trajectories(self, t):
        '''
        Return the standardized estimated agents' locations at t
        Parameters
      
        '''
        tf_dict = {}
        i = 0
        for k, v in zip(self.t_tf, t):
            tf_dict[k] = v
            i = i+1
        return self.sess.run(self.x_pred, tf_dict)
    
class OptimizationProcedure():
    
    def __init__(self, mother, loss, N_epochs, options, var_list=None, opt=0):
        self.loss = loss
        self.opt = opt
        
        if self.opt >= 2: # We use ADAM
                self.optimizer_gd = tf.train.AdamOptimizer().minimize(loss, var_list=var_list)
        
        if self.opt == 0 or self.opt == 1 or self.opt >= 6:
            # Define BFGS
            self.optimizer_BFGS = tf.contrib.opt.ScipyOptimizerInterface(loss, var_list=var_list,
                                                                             method='L-BFGS-B', 
                                                                             options=options)
        self.mother = mother
        self.N_epochs = N_epochs
        
    def train(self, tf_dict, step=1):
        
        mother = self.mother
        # N_f = len(tf_dict[mother.t_f_tf])
        saved_lambdas = [[tf_dict[mother.lambdas_tf[i]]] for i in range(len(mother.lambdas_tf))]
        
        if self.opt >= 2: # Use of first order scheme
                
            L = mother.sess.run(mother.losses, tf_dict)

            for epoch in range(self.N_epochs):
                
                mother.epoch = epoch + 1
                #mother.loss_callback(L, mother.sess.run(self.loss, tf_dict))
                
                mother.sess.run(self.optimizer_gd, tf_dict)
                
                if self.opt == 4 or self.opt == 6: # ADAMBU or pretraining BFGS: Update of lambda
                    if epoch % mother.N_lambda == 0:
                        lambdas = mother.sess.run(mother.lambdas, tf_dict)
                        for i in range(len(saved_lambdas)):
                            if len(mother.sigmas) > 0:
                                tf_dict[mother.lambdas_tf[i]] = min(lambdas[i], mother.sigmas[i])
                            else:
                                tf_dict[mother.lambdas_tf[i]] = lambdas[i]
                        # t_f, x_f = latin_hypercube_sampling(N_f)
                        # tf_dict[mother.t_f_tf] = t_f
                        # tf_dict[mother.x_f_tf] = x_f
                elif self.opt == 3 or self.opt == 7: # Primal-dual
                    if epoch % mother.N_lambda == 0:
                        for i in range(len(saved_lambdas)):
                            if mother.constraints[i] == 0:
                                continue
                            if mother.sigmas[i] <= 0: # This is a hard constraint
                                lr = 1e-3 * 1 / (1 + epoch / self.N_epochs)
                                Li = mother.sess.run(mother.losses[i], tf_dict)
                                new_lambda = tf_dict[mother.lambdas_tf[i]] + lr * Li
                                tf_dict[mother.lambdas_tf[i]] = new_lambda
                            else: # This is a soft constraint
                                lr = 1e-2 * (2 * mother.sigmas[i] - tf_dict[mother.lambdas_tf[i]]) / mother.sigmas[i]
                                Li = mother.sess.run(mother.losses[i], tf_dict)
                                new_lambda = tf_dict[mother.lambdas_tf[i]] + lr * Li
                                tf_dict[mother.lambdas_tf[i]] = min(new_lambda, mother.sigmas[i])
                                
                elif self.opt == 2 or self.opt == 9: # ADMM
                    if epoch % mother.N_lambda == 0:
                        for i in range(len(saved_lambdas)):
                            if mother.constraints[i] == 0:
                                continue
                            if mother.sigmas[i] <= 0: # This is a hard constraint
                                Li = mother.sess.run(mother.losses[i], tf_dict)
                                new_lambda = tf_dict[mother.lambdas_tf[i]] + tf_dict[mother.c_tf] * Li
                                tf_dict[mother.lambdas_tf[i]] = new_lambda
                            else: # this is a soft constraint
                                tf_dict[mother.c_tf] = 1e-2 * (2 * mother.sigmas[i] - tf_dict[mother.lambdas_tf[i]]) / mother.sigmas[i]
                                Li = mother.sess.run(mother.losses[i], tf_dict)
                                new_lambda = tf_dict[mother.lambdas_tf[i]] + tf_dict[mother.c_tf] * Li
                                tf_dict[mother.lambdas_tf[i]] = min(new_lambda, mother.sigmas[i])                                
                                
                for i in range(len(saved_lambdas)):
                    saved_lambdas[i].append(tf_dict[mother.lambdas_tf[i]])
                        
                Lnew = mother.sess.run(mother.losses, tf_dict)
                coef_L = 0.
                for i in range(len(Lnew)):
                    coef_L = max(coef_L, abs(L[i] - Lnew[i]) / max(L[i], Lnew[i], 1))
                    
                dL = mother.sess.run(mother.dL, tf_dict)
                max_dL = 0.
                for i in range(len(dL)):
                    max_dL = max(max_dL, np.amax(abs(dL[i])))
                    
                if np.max(Lnew) <= 1e-6:
                    print('Low value of the cost.')
                    break
                
                if coef_L <= 1e-8:
                    print('No evolution of the cost.')
                    break
                
                if max_dL <= 1e-8:
                    print('Gradient is almost zero.')
                    break
                
                L = Lnew
                
        if self.opt == 0 or self.opt >= 6:
            self.optimizer_BFGS.minimize(mother.sess,
                                    feed_dict=tf_dict,
                                    fetches=[mother.losses, self.loss])
            
        if self.opt == 1: # Real penalty method
            for i in range(self.N_epochs):
                self.optimizer_BFGS.minimize(mother.sess,
                                        feed_dict=tf_dict,
                                        fetches=[mother.losses, self.loss])
                
                max_difference_lambdas = 0
                for i in range(len(saved_lambdas)):
                    if mother.constraints[i] == 0:
                        saved_lambdas[i].append(1.)
                        continue
                    old_lambdai = tf_dict[mother.lambdas_tf[i]]
                    Li = mother.sess.run(mother.losses[i], tf_dict)
                    tf_dict[mother.lambdas_tf[i]] = old_lambdai + tf_dict[mother.c_tf] * Li
                    saved_lambdas[i].append(tf_dict[mother.lambdas_tf[i]])
                    relative_diff = abs(old_lambdai - tf_dict[mother.lambdas_tf[i]]) / max(old_lambdai, tf_dict[mother.lambdas_tf[i]], 1)
                    max_difference_lambdas = max(max_difference_lambdas, relative_diff)
                    
                tf_dict[mother.c_tf] = 1.1 * tf_dict[mother.c_tf]
                    
                if max_difference_lambdas <= 1e-3:
                    break
            
        return saved_lambdas