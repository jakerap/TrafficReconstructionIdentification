# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:17:25 2020

@author: barreau
"""

import numpy as np
np.random.seed(12345)
# import reconstruction_neural_network as rn
import reconstruction_neural_network_pytorch as rn
import sumo as s
import matplotlib.pyplot as plt

#####################################
####     General parameters     #####
#####################################

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



scenario = 'highway'

sumo = s.Sumo(scenario)
Nx, Nt = sumo.Nx, sumo.Nt # Number of spatial and temporal points (250, 420)
L, Tmax = sumo.L, sumo.Tmax # Length of the road and maximum time (2.5, 7.0)

rho = sumo.getDensity()  # density(time, position) (250, 420)
# sumo.plotDensity()
t_train, x_train, rho_train, v_train = sumo.getMeasurements()
axisPlot = sumo.getAxisPlot()
Vf = np.amax(v_train[14]) # find the highest speed in the training data\

trained_neural_network = rn.ReconstructionNeuralNetwork(t_train, x_train, rho_train, v_train,
                                                    L, Tmax, N_f=500, N_g=100, opt=9, v_max=Vf)

loss_history, gamma_var_history, noise_rho_bar_history, lambdas_history = trained_neural_network.train()
plot_losses(loss_history)
plot_parameters(gamma_var_history, noise_rho_bar_history, lambdas_history)


trained_neural_network.plot(axisPlot, rho)
sumo.plotProbeVehicles()
plt.show()


