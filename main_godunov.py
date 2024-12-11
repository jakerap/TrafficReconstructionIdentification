# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:17:25 2020

@author: barreau
"""

import numpy as np
np.random.seed(12345)
import godunov as g
# import reconstruction_neural_network as rn
import reconstruction_neural_network_pytorch as rn
from pyDOE import lhs
import matplotlib.pyplot as plt
import time

def plot_losses(loss_history):
    num_losses = len(loss_history)
    cols = 2  # You can adjust the number of columns based on your preference
    rows = num_losses // cols + (num_losses % cols > 0)

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    fig.suptitle('Loss History by Component')

    # Flatten the axes array for easy indexing
    axes_flat = axes.flatten()

    for i, (key, values) in enumerate(loss_history.items()):
        ax = axes_flat[i]
        ax.plot(values, label=key)
        ax.set_title(key)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)

    # Hide any unused subplots
    for ax in axes_flat[i+1:]:
        ax.set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to make room for the suptitle
    plt.show()


def plot_parameters(gamma_var_history, noise_rho_bar_history, lambdas_history):
    fig, ax = plt.subplots(3, 1, figsize=(15, 10))
    fig.suptitle('Parameter History')

    ax[0].plot(gamma_var_history)
    ax[0].set_title('Gamma Variance')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Variance')
    ax[0].grid(True)

    ax[1].plot(noise_rho_bar_history)
    ax[1].set_title('Noise Rho Bar')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Value')
    ax[1].grid(True)

    for key, values in lambdas_history.items():
        ax[2].plot(values, label=key)
    # ax[2].plot(lambdas_history)
    ax[2].set_title('lambdas_history')
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('Value')
    ax[2].grid(True)
    ax[2].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to make room for the suptitle
    plt.show()



#####################################
####     General parameters     #####
#####################################

Vf = 1.5 # Maximum car speed in km.min^-1
gamma = 0 # dissipativity coefficient (0 by default, discrepencies may occur if very small but not zero)
Tmax = 2 # simulation time in min
p = 1/20 # Probability that a car is a PV
L = 5 # Length of the road in km
rhoBar = 0.5 # Average density of cars on the road
rhoSigma = 0.45 # initial condition standard deviation
rhoMax = 120 # Number of vehicles per kilometer
noise = True # noise on the measurements and on the trajectories
greenshield = True # Type of flux function used for the numerical simulation
Ncar = rhoBar*rhoMax*L # Number of cars
Npv = int(Ncar*p) # Number of PV

# Initial position and time of probes vehicles
xiPos = L*lhs(1, samples=Npv).reshape((Npv,))
xiPos = np.flip(np.sort(xiPos))
xiT = np.array([0]*Npv)


# Godunov simulation of the PDE
simu_godunov = g.SimuGodunov(Vf, gamma, xiPos, xiT, L=L, Tmax=Tmax,
                             zMin=0, zMax=1, Nx=500, greenshield=greenshield,
                             rhoBar=rhoBar, rhoSigma=rhoSigma)

rho = simu_godunov.simulation()
simu_godunov.plot()
axisPlot = simu_godunov.getAxisPlot()

# collect data from PV
t_train, x_train, rho_train, v_train = simu_godunov.getMeasurements(selectedPacket=-1, totalPacket=-1, noise=noise)




trained_neural_network = rn.ReconstructionNeuralNetwork(t_train, x_train, rho_train, v_train,
                                                    L, Tmax, v_max=Vf, N_f=1000, N_g=50, N_v=30, opt=9)

loss_history, gamma_var_history, noise_rho_bar_history, lambdas_history = trained_neural_network.train()


plot_losses(loss_history)
plot_parameters(gamma_var_history, noise_rho_bar_history, lambdas_history)

# trained_neural_network.start() 
# trained_neural_network.train()

trained_neural_network.plot(axisPlot, rho)
simu_godunov.pv.plot()
# figError.savefig('error_godunov.png', bbox_inches='tight')

# plt.show()
# trained_neural_network.close()