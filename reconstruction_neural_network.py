# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:15:37 2020

@author: barreau
"""

import numpy as np
#import matplotlib
#matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from time import time
from pyDOE import lhs
from neural_network import NeuralNetwork
        
def hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    h, m, s = int(h), int(m), int(s)
    print('{:.0f}h {:.0f}m {:.0f}s'.format(h, m, s))
    
def amin(l):
    min_list = [np.amin(l[i]) for i in range(len(l))]
    return np.amin(min_list)

def amax(l):
    min_list = [np.amax(l[i]) for i in range(len(l))]
    return np.amax(min_list)

class ReconstructionNeuralNetwork():
    
    def __init__(self, t, x, rho, v, L, Tmax, N_f=1000, N_g=100):
        '''
        Initialize a neural network for density reconstruction

        Parameters
        ----------
        t : List of N numpy array of shape (?,)
            time coordinate of training points.
        x : list of N numpy array of shape (?,)
            space coordinate of training points.
        rho : list of N numpy array of shape (?,)
            density values at training points.
        v : list of N numpy array of shape (?,)
            velocity values at training points.
        L : float64
            Length of the spacial domain.
        Tmax : float64
            Length of the temporal domain.
        N_f : integer, optional
            Number of physical points for F. The default is 1000.
        N_g : integer, optional
            Number of physical points for G. The default is 100.

        Returns
        -------
        None.

        '''
        
        self.Nxi = len(x) # Number of agents
        
        self.rho = rho
        self.v = v
        
        num_hidden_layers = int(Tmax*8/100) 
        num_nodes_per_layer = int(20*L/7000) 
        layers = [2] # There are two inputs: space and time
        for _ in range(num_hidden_layers):
            layers.append(num_nodes_per_layer)
        layers.append(1)
        
        t_train, x_train, u_train, v_train, X_f_train, t_g_train = self.createTrainingDataset(t, x, rho, v, L, Tmax, N_f, N_g) # Creation of standardized training dataset
        
        self.neural_network = NeuralNetwork(t_train, x_train, u_train, v_train, X_f_train, t_g_train, layers_density=layers, 
                                            layers_trajectories=(1, 5, 5, 5, 5, 1), 
                                            layers_speed=(1, 5, 5, 1),) # Creation of the neural network
        self.train() # Training of the neural network
            
    def createTrainingDataset(self, t, x, rho, v, L, Tmax, N_f, N_g):       
        '''
        Standardize the dataset

        Parameters
        ----------
        t : list of N arrays of float64 (?,)
            Time coordinate of agents.
        x : list of N arrays of float64 (?,)
            Position of agents along time.
        rho : list of N arrays of float64 (?,)
            Density measurement from each agent.
        v : list of N arrays of float64 (?,)
            Velocity measurement from each agent.
        L : float
            Length of the road.
        Tmax : float
            Time-window.
        N_f : int
            Number of physical points for f.
        N_g : int
            Number of physical points for g.

        Returns
        -------
        t : list of N arrays of float64 (?,)
            Standardized time coordinate of agents.
        x : list of N arrays of float64 (?,)
            Standardized position of agents along time.
        u : list of N arrays of float64 (?,)
            Standardized density measurement from each agent.
        v : list of N arrays of float64 (?,)
            Standardized velocity measurement from each agent.
        X_f : 2D array of shape (N_f, 2)
            Standardized location of physical points for f.
        t_g : list of float64
            List of standardized phisical points for g.

        '''
        
        self.lb = np.array([amin(x), amin(t)])
        self.ub = np.array([amax(x), amax(t)])
        self.lb[0], self.lb[1] = 0, 0
        
        x = [2*(x_temp - self.lb[0])/(self.ub[0] - self.lb[0]) - 1 for x_temp in x]
        t = [2*(t_temp - self.lb[1])/(self.ub[1] - self.lb[1]) - 1 for t_temp in t]
        rho = [2*rho_temp-1 for rho_temp in rho]
        v = [v_temp*(self.ub[1] - self.lb[1]) / (self.ub[0] - self.lb[0]) for v_temp in v]
        
        X_f = np.array([2, 2])*lhs(2, samples=N_f)
        X_f = X_f - np.ones(X_f.shape)
        np.random.shuffle(X_f)
        
        t_g = []
        for i in range(self.Nxi):
            t_g.append(np.amin(t[i]) + lhs(1, samples=N_g)*(np.amax(t[i]) - np.amin(t[i])))
        
        return (t, x, rho, v, X_f, t_g)

    def train(self):
        '''
        Train the neural network

        Returns
        -------
        None.

        '''
        start = time()
        self.neural_network.train()
        hms(time() - start)
        
    def predict(self, t, x):
        '''
        Return the estimated density at (t, x)

        Parameters
        ----------
        t : numpy array (?, )
            time coordinate.
        x : numpy array (?, )
            space coordinate.

        Returns
        -------
        numpy array
            estimated density.

        '''
        
        x = 2*(x - self.lb[0])/(self.ub[0] - self.lb[0])-1
        t = 2*(t - self.lb[1])/(self.ub[1] - self.lb[1])-1
        
        return self.neural_network.predict(t, x)/2+0.5
    
    def predict_speed(self, rho):
        '''
        Return the estimated speed at rho

        Parameters
        ----------
        rho : numpy array (?, )
            density.

        Returns
        -------
        numpy array
            estimated speed.

        '''
        
        u = 2*rho-1
        
        return self.neural_network.predict_speed(u)*(self.ub[0] - self.lb[0]) / (self.ub[1] - self.lb[1])
    
    def predict_F(self, rho):
        '''
        Return the estimated characteristic speed at rho

        Parameters
        ----------
        rho : numpy array (?, )
            density.

        Returns
        -------
        numpy array
            estimated characteristic speed.

        '''
        
        u = 2*rho-1
        
        return self.neural_network.predict_F(u)*(self.ub[0] - self.lb[0]) / (self.ub[1] - self.lb[1])
    
    def predict_trajectories(self, t):
        '''
        Return the estimated agents' locations at t

        Parameters
        ----------
        t : list of N numpy arrays of size (?, )
            time coordinate.

        Returns
        -------
        list of N numpy arrays
            estimated agents location.

        '''
        
        t = [2*(t[i] - self.lb[1])/(self.ub[1] - self.lb[1])-1 for i in range(self.Nxi)]
        
        output = self.neural_network.predict_trajectories(t)
        output = [(output[i]+1)*(self.ub[0] - self.lb[0])/2 + self.lb[0] for i in range(self.Nxi)]
        return output
    
    
    def plot(self, axisPlot, rho):
        '''
        

        Parameters
        ----------
        axisPlot : tuple of two 1D-numpy arrays of shape (?,)
            Plot mesh.
        rho : 2D numpy array
            Values of the real density at axisPlot.

        Returns
        -------
        list of three Figures
            return the speed, reconstruction and error figures.

        '''
        
        x = axisPlot[0]
        t = axisPlot[1]

        Nx = len(x)
        Nt = len(t)
            
        XY_prediction = np.zeros((Nx * Nt, 2))
        k = 0
        for i in range(0, Nx):
            for j in range(0, Nt):
                XY_prediction[k] = np.array([t[j], x[i]])
                k = k + 1
        tstar = XY_prediction[:, 0:1]
        xstar = XY_prediction[:, 1:2]
        
        rho_prediction = self.predict(tstar, xstar).reshape(Nx, Nt)
        t_pred = [t.reshape(t.shape[0], 1)]*self.Nxi
        X_prediction = self.predict_trajectories(t_pred)
        rho_speed = np.linspace(0, 1).reshape(-1,1)
        v_prediction = self.predict_speed(rho_speed).reshape(-1,1)
        F_prediction = self.predict_F(rho_speed).reshape(-1,1)
        
        figSpeed = plt.figure(figsize=(7.5, 5))
        plt.plot(rho_speed, v_prediction, rasterized=True, label=r'NN approximation of $V$')
        plt.plot(rho_speed, F_prediction, rasterized=True, label=r'NN approximation of $F$')
        densityMeasurements = np.empty((0,1))
        speedMeasurements = np.empty((0,1))
        for i in range(self.Nxi):
            densityMeasurements = np.vstack((densityMeasurements, self.rho[i]))
            speedMeasurements = np.vstack((speedMeasurements, self.v[i]))
        plt.scatter(densityMeasurements, speedMeasurements, rasterized=True, c='black', s=1, label=r'Data')
        plt.xlabel(r'Normalized Density')
        plt.ylabel(r'Speed [km/h]')
        plt.ylim(-v_prediction[0], v_prediction[0])
        plt.xlim(0, 1)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        # plt.title('Reconstruction')
        plt.show()
        figSpeed.savefig('speed.eps', bbox_inches='tight')

        figReconstruction = plt.figure(figsize=(7.5, 5))
        X, Y = np.meshgrid(t, x)
        plt.pcolor(X, Y, rho_prediction, vmin=0.0, vmax=1.0, shading='auto', rasterized=True)
        for i in range(self.Nxi):
            plt.plot(t_pred[i], X_prediction[i], color="orange")
        plt.xlabel(r'Time [s]')
        plt.ylabel(r'Position [m]')
        plt.xlim(min(t), max(t))
        plt.ylim(min(x), max(x))
        plt.colorbar()
        plt.tight_layout()
        # plt.title('Reconstruction')
        plt.show()
        figReconstruction.savefig('reconstruction.eps', bbox_inches='tight')
        
        
        figError = plt.figure(figsize=(7.5, 5))
        X, Y = np.meshgrid(t, x)
        plt.pcolor(X, Y, np.abs(rho_prediction-rho), vmin=0.0, vmax=1.0, shading='auto', rasterized=True)
        for i in range(self.Nxi):
            plt.plot(t_pred[i], X_prediction[i], color="orange")
        plt.xlabel(r'Time [s]')
        plt.ylabel(r'Position [m]')
        plt.xlim(min(t), max(t))
        plt.ylim(min(x), max(x))
        plt.colorbar()
        plt.tight_layout()
        # plt.title('Absolute error')
        plt.show()
        figError.savefig('error.eps', bbox_inches='tight') 
        
        return [figSpeed, figReconstruction, figError]